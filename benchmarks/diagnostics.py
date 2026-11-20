"""Descriptive diagnostics of saved forecasts; no fitting or benchmark selection."""
from statistics import NormalDist

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import acf, pacf


KEYS = ["origin_date", "target_date", "horizon"]


def historical_series(prices, calendar, end):
    """Reindex before differencing; missing sessions break changes and returns."""
    calendar = calendar[calendar <= pd.Timestamp(end)]
    price = prices.reindex(calendar)
    return pd.DataFrame({"price": price, "change": price.diff(),
                         "log_return": np.log(price).diff()})


def rolling_volatility(log_returns, window=20, ddof=1):
    return log_returns.rolling(window, min_periods=window).std(ddof=ddof)


def train_volatility_thresholds(volatility, train_start, train_end, quantiles=(1 / 3, 2 / 3)):
    train = volatility.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].dropna()
    if len(train) == 0:
        raise ValueError("No finite train volatility values")
    thresholds = train.quantile(quantiles, interpolation="linear").to_numpy()
    return thresholds, len(train)


def classify_volatility(volatility, thresholds):
    low, high = thresholds
    groups = pd.Series("unavailable", index=volatility.index)
    groups.loc[volatility.notna() & (volatility <= low)] = "low"
    groups.loc[(volatility > low) & (volatility <= high)] = "medium"
    groups.loc[volatility > high] = "high"
    return groups


def contiguous_segments(series):
    """Missing calendar slots separate segments; never call dropna across a hole."""
    valid = series.notna()
    runs = valid.ne(valid.shift()).cumsum()
    return [segment for _, segment in series[valid].groupby(runs[valid])]


def autocorrelation_values(series, max_lag=20, pacf_method="ywm"):
    values = np.asarray(series, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2 * max_lag + 2:
        raise ValueError("Need a finite contiguous segment longer than twice the maximum lag")
    return (acf(values, nlags=max_lag, fft=False, adjusted=False, missing="raise"),
            pacf(values, nlags=max_lag, method=pacf_method))


def serial_diagnostics(history, splits, config):
    correlations, tests, segments = [], [], []
    for split in ("train", "validation"):
        start, end = pd.to_datetime(splits[split])
        for name in ("change", "log_return", "abs_log_return"):
            series = history.log_return.abs() if name == "abs_log_return" else history[name]
            series = series.loc[start:end]
            for number, segment in enumerate(contiguous_segments(series), 1):
                usable = len(segment) >= max(config["minimum_contiguous_segment"], 2 * config["acf_max_lag"] + 2)
                meta = dict(split=split, series=name, segment=number, n=len(segment),
                            first_date=segment.index.min(), last_date=segment.index.max())
                segments.append({**meta, "status": "used" if usable else "too_short"})
                if not usable:
                    continue
                a, p = autocorrelation_values(segment, config["acf_max_lag"], config["pacf_method"])
                for lag, value in enumerate(a):
                    correlations.append({**meta, "lag": lag, "acf": value,
                                         "pacf": p[lag] if name == "change" else np.nan})
                lb = acorr_ljungbox(segment.to_numpy(), lags=config["ljung_box_lags"], model_df=0,
                                   auto_lag=False, return_df=True)
                for lag, row in lb.iterrows():
                    tests.append({**meta, "lag": lag, "lb_stat": row.lb_stat,
                                  "p_value": row.lb_pvalue})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = multipletests(tests.p_value, method="holm")[1]
    return pd.DataFrame(correlations), tests, pd.DataFrame(segments)


def saved_parameters(forecasts, validation_origins, config):
    rows = forecasts[forecasts.model_id.eq("B1") & forecasts.prediction_status.eq("available")].copy()
    for column in ("c", "phi", "n_pairs"):
        if rows.groupby("origin_date")[column].nunique().gt(1).any():
            raise ValueError("Saved parameters disagree across horizons")
    params = rows.drop_duplicates("origin_date").sort_values("origin_date")
    params = params[["origin_date", "train_end_date", "c", "phi", "n_pairs"]].reset_index(drop=True)
    if not params.train_end_date.eq(params.origin_date).all():
        raise ValueError("Unexpected training cutoff")
    params["used_for_validation"] = params.origin_date.isin(validation_origins)
    params["origin_split"] = np.where(params.origin_date <= pd.Timestamp(config["splits"]["train"][1]), "train", "validation")
    descriptions = []
    for scope, frame in (("all_available_train_validation_origins", params),
                         ("validation_forecast_origins", params[params.used_for_validation])):
        for coefficient in ("phi", "c"):
            values = frame[coefficient]
            nonzero_signs = np.sign(values[values.ne(0)])
            descriptions.append(dict(scope=scope, coefficient=coefficient, n=len(values),
                                     mean=values.mean(), median=values.median(), std=values.std(ddof=1),
                                     min=values.min(), max=values.max(), positive_share=values.gt(0).mean(),
                                     negative_share=values.lt(0).mean(),
                                     nonzero_sign_changes=int(nonzero_signs.diff().dropna().ne(0).sum())))
    return params, pd.DataFrame(descriptions)


def align_losses(scored):
    scored = scored[scored.split.eq("validation") & scored.scored_status.eq("scored")].copy()
    if scored.duplicated(["model_id", *KEYS]).any():
        raise ValueError("Duplicate model/forecast keys")
    columns = [*KEYS, "actual", "prediction", "error"]
    rw = scored.loc[scored.model_id.eq("B0"), columns]
    ar = scored.loc[scored.model_id.eq("B1"), columns]
    paired = rw.merge(ar, on=KEYS, suffixes=("_rw", "_ar"), how="outer", indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all():
        raise ValueError("RW and AR scored pair sets differ; establish a documented common set first")
    np.testing.assert_allclose(paired.actual_rw, paired.actual_ar, rtol=0, atol=1e-12)
    for model in ("rw", "ar"):
        np.testing.assert_allclose(paired[f"error_{model}"], paired[f"prediction_{model}"] - paired[f"actual_{model}"],
                                   rtol=0, atol=1e-12)
    paired = paired.drop(columns=["_merge", "actual_ar"]).rename(columns={"actual_rw": "actual"})
    paired = paired.sort_values(["horizon", "origin_date"]).reset_index(drop=True)
    paired["d"] = paired.error_ar ** 2 - paired.error_rw ** 2
    paired["cumulative_d"] = paired.groupby("horizon").d.cumsum()
    return paired


def hac_mean_interval(values, horizon, base_lag=20, confidence=0.95):
    values = np.asarray(values, dtype=float)
    lag = max(base_lag, horizon - 1)
    if not np.isfinite(values).all() or len(values) <= lag + 1:
        raise ValueError("Insufficient finite data for the fixed HAC lag")
    fit = OLS(values, np.ones((len(values), 1))).fit()
    variance = float(cov_hac(fit, nlags=lag, use_correction=True)[0, 0])
    se = np.sqrt(max(variance, 0.0))
    mean = float(np.mean(values))
    normal = NormalDist()
    z = normal.inv_cdf(0.5 + confidence / 2)
    p = 2 * (1 - normal.cdf(abs(mean / se))) if se else (1.0 if mean == 0 else 0.0)
    return dict(mean_d=mean, hac_lag=lag, hac_se=se, ci_lower=mean - z * se,
                ci_upper=mean + z * se, p_normal=p)


def summarize_losses(losses, calendar, config):
    rows = []
    for horizon, frame in losses.groupby("horizon"):
        positions = calendar.get_indexer(frame.origin_date)
        consecutive = bool((positions >= 0).all() and (np.diff(positions) == 1).all())
        inference = hac_mean_interval(frame.d, horizon, config["hac_base_lag"], config["confidence_level"]) if consecutive else {
            "mean_d": frame.d.mean(), "hac_lag": max(config["hac_base_lag"], horizon - 1),
            "hac_se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_normal": np.nan}
        absolute = frame.d.abs()
        rows.append(dict(horizon=horizon, n=len(frame), origin_min=frame.origin_date.min(),
                         origin_max=frame.origin_date.max(), target_min=frame.target_date.min(), target_max=frame.target_date.max(),
                         consecutive_origin_sessions=consecutive, **inference,
                         cumulative_d=frame.d.sum(), rw_better_share=frame.d.gt(0).mean(),
                         median_d=frame.d.median(), bias_rw=frame.error_rw.mean(), bias_ar=frame.error_ar.mean(),
                         error_correlation=frame.error_rw.corr(frame.error_ar),
                         top5_absolute_d_share=absolute.nlargest(5).sum() / absolute.sum() if absolute.sum() else 0.0,
                         top5_absolute_d_net=frame.loc[absolute.nlargest(5).index, "d"].sum()))
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    finite = result.p_normal.notna()
    result.loc[finite, "p_holm"] = multipletests(result.loc[finite, "p_normal"], method="holm")[1]
    return result


def volatility_diagnostics(losses, volatility, thresholds, horizons, small_n=30, calibration_end=None):
    result = losses.merge(volatility.rename("volatility_20"), left_on="origin_date", right_index=True, how="left", validate="many_to_one")
    result["regime"] = classify_volatility(result.volatility_20, thresholds)
    if calibration_end is not None:
        result["thresholds_available_from"] = pd.Timestamp(calibration_end)
        result.loc[result.origin_date < pd.Timestamp(calibration_end), "regime"] = "calibration_unavailable"
    rows = []
    for horizon in horizons:
        for regime in ("low", "medium", "high", "unavailable", "calibration_unavailable"):
            frame = result[result.horizon.eq(horizon) & result.regime.eq(regime)]
            excluded = regime in ("unavailable", "calibration_unavailable")
            if excluded and len(frame) == 0:
                continue
            rmse_rw = np.sqrt(np.mean(frame.error_rw ** 2)) if not excluded else np.nan
            rmse_ar = np.sqrt(np.mean(frame.error_ar ** 2)) if not excluded else np.nan
            rows.append(dict(horizon=horizon, regime=regime, n=len(frame), rmse_rw=rmse_rw,
                             rmse_ar=rmse_ar, rmse_ratio=rmse_ar / rmse_rw if rmse_rw > 0 else np.nan,
                             small_sample=len(frame) < small_n, excluded=excluded))
    return result, pd.DataFrame(rows)


def drift_components(c, phi, last_change, horizons=7):
    intercept_change, autoregressive_change = 0.0, float(last_change)
    intercept_level, autoregressive_level = 0.0, 0.0
    rows = []
    for horizon in range(1, horizons + 1):
        intercept_change = c + phi * intercept_change
        autoregressive_change = phi * autoregressive_change
        intercept_level += intercept_change
        autoregressive_level += autoregressive_change
        rows.append((horizon, intercept_level, autoregressive_level))
    return rows


def drift_decomposition(losses, parameters, history):
    components = []
    for parameter in parameters[parameters.used_for_validation].itertuples():
        for h, constant, autoregressive in drift_components(parameter.c, parameter.phi, history.loc[parameter.origin_date, "change"]):
            components.append(dict(origin_date=parameter.origin_date, horizon=h, c=parameter.c, phi=parameter.phi,
                                   constant_component=constant, autoregressive_component=autoregressive))
    result = losses.merge(pd.DataFrame(components), on=["origin_date", "horizon"], how="left", validate="one_to_one")
    if result[["constant_component", "autoregressive_component"]].isna().any().any():
        raise ValueError("Missing saved parameters for a validation forecast")
    result["origin_price"] = result.origin_date.map(history.price)
    result["last_change"] = result.origin_date.map(history.change)
    result["total_component"] = result.constant_component + result.autoregressive_component
    np.testing.assert_allclose(result.origin_price + result.total_component, result.prediction_ar, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.origin_price, result.prediction_rw, rtol=0, atol=1e-12)
    a, b, error = result.constant_component, result.autoregressive_component, result.error_rw
    # Split the cross term symmetrically: an algebraic allocation, not causal attribution.
    result["constant_loss_allocation"] = 2 * error * a + a ** 2 + a * b
    result["autoregressive_loss_allocation"] = 2 * error * b + b ** 2 + a * b
    np.testing.assert_allclose(result.constant_loss_allocation + result.autoregressive_loss_allocation, result.d, rtol=0, atol=1e-10)
    rows = []
    for horizon, frame in result.groupby("horizon"):
        rows.append(dict(horizon=horizon, n=len(frame), mean_constant=frame.constant_component.mean(),
                         mean_autoregressive=frame.autoregressive_component.mean(),
                         std_constant=frame.constant_component.std(), std_autoregressive=frame.autoregressive_component.std(),
                         mean_abs_constant=frame.constant_component.abs().mean(),
                         mean_abs_autoregressive=frame.autoregressive_component.abs().mean(),
                         constant_dominant_share=(frame.constant_component.abs() > frame.autoregressive_component.abs()).mean(),
                         mean_constant_loss_allocation=frame.constant_loss_allocation.mean(),
                         mean_autoregressive_loss_allocation=frame.autoregressive_loss_allocation.mean()))
    return result, pd.DataFrame(rows)
