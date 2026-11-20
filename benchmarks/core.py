"""Data checks, calendar-aware forecasting and separate scoring.

No candle filling, calendar inference from weekdays, or AR fallback is used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    split_bounds(config)
    if config["horizons"] != list(range(1, 8)):
        raise ValueError("Expected horizons 1..7")
    if config["min_ar_pairs"] < 2:
        raise ValueError("AR requires at least two pairs")
    if config["refit_schedule"] != "every_origin_expanding":
        raise ValueError("Unsupported refit schedule")
    return config


def split_bounds(config):
    bounds = [(name, *pd.to_datetime(config["splits"][name]))
              for name in ("train", "validation", "final_test")]
    previous = None
    for _, start, end in bounds:
        if start > end or (previous is not None and start <= previous):
            raise ValueError("Split bounds must be chronological and disjoint")
        previous = end
    return bounds


def target_split(date, config):
    if pd.isna(date):
        return "unknown"
    for name, start, end in split_bounds(config):
        if start <= date <= end:
            return name
    return "outside"


def make_splits(data, config):
    result = {name: data.loc[start:end].copy()
              for name, start, end in split_bounds(config)}
    indices = pd.DatetimeIndex([])
    for frame in result.values():
        if len(indices.intersection(frame.index)):
            raise ValueError("Overlapping split observations")
        indices = indices.union(frame.index)
    if len(indices) != len(data):
        raise ValueError("Observations outside configured splits")
    return result


def load_data(path):
    data = pd.read_csv(path)
    dates = pd.to_datetime(data["date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Missing or duplicate dates")
    if not dates.is_monotonic_increasing or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be ordered daily labels")
    data["date"] = dates
    original = data["price"]
    price = pd.to_numeric(original, errors="coerce")
    if (original.notna() & price.isna()).any():
        raise ValueError("Nonnumeric price")
    if ((price <= 0) | (price.notna() & ~np.isfinite(price))).any():
        raise ValueError("Nonpositive or infinite price")
    data["price"] = price
    return data.set_index("date")


def load_calendar(data, config, root):
    mode = config["calendar_mode"]
    if mode == "observed_session_proxy":
        if config.get("calendar_path"):
            raise ValueError("Proxy mode cannot declare an independent calendar")
        return data.index.copy()
    if mode != "independent" or not config.get("calendar_source"):
        raise ValueError("Independent calendar requires documented provenance")
    frame = pd.read_csv(Path(root) / config["calendar_path"])
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    validate_calendar(dates)
    if len(data.index.difference(dates)):
        raise ValueError("Observed date outside supplied calendar")
    return dates


def validate_calendar(calendar):
    if (len(calendar) == 0 or calendar.hasnans or calendar.has_duplicates
            or not calendar.is_monotonic_increasing
            or not calendar.equals(calendar.normalize())):
        raise ValueError("Calendar requires unique ordered daily dates")


def random_walk(history, horizons):
    return np.repeat(float(history.iloc[-1]), max(horizons)), {"n_pairs": 0}


def ar1(history, horizons, min_pairs=100):
    changes = history.diff()  # calendar holes remain NaN
    pairs = pd.concat([changes.shift(1), changes], axis=1).dropna()
    meta = {"n_pairs": len(pairs), "c": np.nan, "phi": np.nan}
    if len(pairs) < min_pairs:
        return None, {**meta, "reason": "insufficient_history"}
    if pd.isna(changes.iloc[-1]):
        return None, {**meta, "reason": "missing_last_change"}
    design = np.column_stack([np.ones(len(pairs)), pairs.iloc[:, 0]])
    params, _, rank, _ = np.linalg.lstsq(design, pairs.iloc[:, 1], rcond=None)
    if rank < 2:
        return None, {**meta, "reason": "singular_design"}
    c, phi = params
    meta.update(c=float(c), phi=float(phi))
    change, price = float(changes.iloc[-1]), float(history.iloc[-1])
    predictions = []
    for _ in range(max(horizons)):
        change = c + phi * change
        price += change
        predictions.append(price)
    if not np.isfinite(predictions).all():
        return None, {**meta, "reason": "nonfinite_forecast"}
    return np.asarray(predictions), meta


def generate_forecasts(data, calendar, config, models=None):
    """Models receive only calendar-indexed history through the current origin."""
    validate_calendar(calendar)
    if len(data.index.difference(calendar)):
        raise ValueError("Observed date outside calendar")
    series = data["price"].reindex(calendar)
    if models is None:
        models = {"B0": random_walk,
                  "B1": lambda hist, hs: ar1(hist, hs, config["min_ar_pairs"])}
    rows = []
    splits = {date: target_split(date, config) for date in calendar}
    for i, origin in enumerate(calendar):
        if pd.isna(series.iloc[i]):
            continue
        history = series.iloc[:i + 1].copy()
        for model_id, model in models.items():
            predictions, meta = model(history.copy(), config["horizons"])
            for h in config["horizons"]:
                target = calendar[i + h] if i + h < len(calendar) else pd.NaT
                rows.append({
                    "model_id": model_id, "model_version": "1.0",
                    "forecast_origin": pd.NaT, "origin_date": origin,
                    "forecast_origin_mode": config["forecast_origin_mode"],
                    "target_date": target, "horizon": h,
                    "prediction": predictions[h - 1] if predictions is not None else np.nan,
                    "prediction_status": "available" if predictions is not None else meta["reason"],
                    "scored_status": "unscored",
                    "target_date_status": "known" if pd.notna(target) else "calendar_unavailable",
                    "train_end_date": origin, "data_version": config["data_version"],
                    "source": data.loc[origin, "source"] if origin in data.index else None,
                    "pit_status": config["pit_status"],
                    "calendar_mode": config["calendar_mode"], "result_status": "preliminary",
                    "calendar_source": config.get("calendar_source"),
                    "split": splits.get(target, "unknown"), **meta,
                })
    return pd.DataFrame(rows)


def score_forecasts(predictions, data, allowed_splits=("validation",)):
    """Attach actuals after forecasting; final-test actuals are sealed by default."""
    scored = predictions.copy()
    permitted = scored["split"].isin(allowed_splits)
    scored["actual"] = np.nan
    scored.loc[permitted, "actual"] = scored.loc[permitted, "target_date"].map(data["price"])
    scored["scored_status"] = "unscored"
    valid = permitted & scored["actual"].notna() & scored["prediction"].notna()
    scored.loc[valid, "scored_status"] = "scored"
    scored["error"] = scored["prediction"] - scored["actual"]
    return scored


def metrics_by_horizon(scored):
    """Compare B0/B1 only on common origin/target pairs within each split/h."""
    keys = ["origin_date", "target_date", "horizon", "split"]
    valid = scored.loc[scored["scored_status"].eq("scored")]
    wide = valid.pivot(index=keys, columns="model_id", values="error")
    columns = ["split", "model_id", "horizon", "n", "mae", "rmse", "bias",
               "rmse_ratio_vs_rw", "origin_min", "origin_max", "target_min", "target_max"]
    if not {"B0", "B1"}.issubset(wide.columns):
        return pd.DataFrame(columns=columns)
    common = wide[["B0", "B1"]].dropna().reset_index()
    rows = []
    for (split, h), group in common.groupby(["split", "horizon"]):
        rw_rmse = np.sqrt(np.mean(group["B0"] ** 2))
        for model in ("B0", "B1"):
            error = group[model]
            rmse = np.sqrt(np.mean(error ** 2))
            rows.append(dict(split=split, model_id=model, horizon=h, n=len(group),
                             mae=np.mean(abs(error)), rmse=rmse, bias=np.mean(error),
                             rmse_ratio_vs_rw=rmse / rw_rmse if rw_rmse else np.nan,
                             origin_min=group.origin_date.min(), origin_max=group.origin_date.max(),
                             target_min=group.target_date.min(), target_max=group.target_date.max()))
    return pd.DataFrame(rows, columns=columns)
