import numpy as np
import pandas as pd
import pytest

from benchmarks.diagnostics import (align_losses, autocorrelation_values, classify_volatility,
                                    contiguous_segments, drift_components, drift_decomposition,
                                    hac_mean_interval, historical_series, rolling_volatility,
                                    serial_diagnostics, train_volatility_thresholds, volatility_diagnostics)


def test_acf_matches_direct_sample_autocovariance():
    values = np.random.default_rng(84).normal(size=200)
    a, p = autocorrelation_values(values)
    centered = values - values.mean()
    manual = [np.dot(centered[lag:], centered[:len(values) - lag]) / np.dot(centered, centered)
              for lag in range(21)]
    np.testing.assert_allclose(a, manual, atol=1e-14)
    assert p[0] == 1
    assert p[1] == pytest.approx(a[1])


def test_acf_pacf_recover_synthetic_ar1_dependence():
    noise = np.random.default_rng(81).normal(size=7000)
    series = np.zeros(len(noise))
    for i in range(1, len(series)):
        series[i] = 0.6 * series[i - 1] + noise[i]
    a, p = autocorrelation_values(series[200:])
    assert a[1] == pytest.approx(0.6, abs=0.04)
    assert a[2] == pytest.approx(0.36, abs=0.04)
    assert p[1] == pytest.approx(0.6, abs=0.04)
    assert abs(p[2]) < 0.04


def test_no_concatenation_of_changes_across_missing_session():
    calendar = pd.date_range("2025-01-01", periods=80)
    prices = pd.Series(100 + np.arange(80) * 0.1, index=calendar).drop(calendar[40])
    frame = historical_series(prices, calendar, calendar[-1])
    assert frame.loc[calendar[40:42], ["change", "log_return"]].isna().all().all()
    segments = contiguous_segments(frame.change)
    assert [len(segment) for segment in segments] == [39, 38]
    assert segments[0].index[-1] == calendar[39]
    assert segments[1].index[0] == calendar[42]
    with pytest.raises(ValueError, match="contiguous"):
        autocorrelation_values(frame.change.to_numpy())


def test_diagnostic_history_excludes_final_test_prices():
    calendar = pd.date_range("2025-12-01", periods=60)
    prices = pd.Series(100 + np.arange(60) * 0.1, index=calendar)
    changed = prices.copy()
    changed.loc["2026-01-01":] *= 10
    history = historical_series(prices, calendar, "2025-12-31")
    altered = historical_series(changed, calendar, "2025-12-31")
    pd.testing.assert_frame_equal(history, altered, check_exact=True)
    assert history.index.max() == pd.Timestamp("2025-12-31")


def test_serial_tests_keep_segments_separate():
    calendar = pd.date_range("2025-01-01", periods=130)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.001, 130))), index=calendar)
    prices.iloc[65] = np.nan
    frame = historical_series(prices, calendar, calendar[-1])
    config = dict(minimum_contiguous_segment=42, acf_max_lag=20, pacf_method="ywm", ljung_box_lags=[5, 10, 20])
    _, lb, segments = serial_diagnostics(frame, {"train": ["2025-01-01", "2025-05-31"],
                                                "validation": ["2025-06-01", "2025-12-31"]}, config)
    assert set(segments.n) == {64, 63}
    assert set(lb.segment) == {1, 2}
    assert lb.n.max() == 64
    assert len(lb) == 18


def test_rolling_volatility_never_reads_future_and_resets_at_hole():
    returns = pd.Series(np.random.default_rng(3).normal(size=100))
    changed = returns.copy()
    changed.loc[60:] *= 100
    original = rolling_volatility(returns)
    updated = rolling_volatility(changed)
    pd.testing.assert_series_equal(original.iloc[:60], updated.iloc[:60])
    assert original.iloc[30] == pytest.approx(returns.iloc[11:31].std(ddof=1))
    assert original.iloc[:19].isna().all()
    returns.iloc[40] = np.nan
    assert rolling_volatility(returns).iloc[40:60].isna().all()


def test_volatility_thresholds_are_train_only():
    dates = pd.date_range("2024-01-01", periods=730)
    volatility = pd.Series(np.arange(730) / 10000, index=dates)
    thresholds, n = train_volatility_thresholds(volatility, "2024-01-01", "2024-12-31")
    changed = volatility.copy()
    changed.loc["2025-01-01":] *= 10000
    altered, altered_n = train_volatility_thresholds(changed, "2024-01-01", "2024-12-31")
    np.testing.assert_array_equal(thresholds, altered)
    np.testing.assert_allclose(thresholds, volatility.loc[:"2024-12-31"].quantile([1 / 3, 2 / 3]).to_numpy())
    assert n == altered_n == 366
    assert classify_volatility(pd.Series([np.nan, thresholds[0], thresholds[1], thresholds[1] + 1]), thresholds).tolist() == [
        "unavailable", "low", "medium", "high"]


def _scored():
    rows = []
    for i in range(4):
        for model, error in (("B0", i - 1.0), ("B1", i - 0.5)):
            rows.append(dict(model_id=model, split="validation", scored_status="scored",
                             origin_date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                             target_date=pd.Timestamp("2025-01-02") + pd.Timedelta(days=i),
                             horizon=1, actual=100.0, prediction=100.0 + error, error=error))
    return pd.DataFrame(rows)


def test_loss_pair_alignment_and_signed_squared_difference():
    scored = _scored().sample(frac=1, random_state=3)
    paired = align_losses(scored)
    np.testing.assert_allclose(paired.d, paired.error_ar ** 2 - paired.error_rw ** 2)
    np.testing.assert_allclose(paired.cumulative_d, paired.d.cumsum())
    assert paired.origin_date.is_monotonic_increasing
    with pytest.raises(ValueError, match="pair sets"):
        align_losses(scored.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        align_losses(pd.concat([scored, scored.iloc[:1]]))


def test_volatility_groups_require_train_thresholds_already_available():
    losses = align_losses(_scored())
    volatility = pd.Series([0.01] * 4, index=pd.date_range("2025-01-01", periods=4))
    classified, summary = volatility_diagnostics(losses, volatility, [0.02, 0.03], [1],
                                                 calibration_end="2025-01-02")
    assert classified.iloc[0].regime == "calibration_unavailable"
    assert classified.iloc[1:].regime.eq("low").all()
    excluded = summary[summary.regime.eq("calibration_unavailable")].iloc[0]
    assert excluded.n == 1 and excluded.excluded
    assert pd.isna(excluded.rmse_ratio)
    assert summary.loc[summary.regime.eq("low"), "n"].iloc[0] == 3


@pytest.mark.parametrize("c,phi,last_change", [(0.2, 0.6, 0.7), (-0.1, -0.4, 1.1), (0.0, 0.0, 2.0), (0.2, 1.0, 0.7)])
def test_recursive_components_reconstruct_forecast(c, phi, last_change):
    decomposition = drift_components(c, phi, last_change)
    change, total = last_change, 0.0
    for h, constant, autoregressive in decomposition:
        change = c + phi * change
        total += change
        assert total == pytest.approx(constant + autoregressive, abs=1e-12)
        if phi == 1:
            assert constant == pytest.approx(c * h * (h + 1) / 2)
            assert autoregressive == pytest.approx(h * last_change)


def test_drift_allocation_sums_to_actual_loss_difference():
    date = pd.Timestamp("2025-01-03")
    parameters = pd.DataFrame([dict(origin_date=date, used_for_validation=True, c=0.2, phi=0.6)])
    history = pd.DataFrame({"price": [100.0], "change": [0.7]}, index=[date])
    rows = []
    for h, constant, ar in drift_components(0.2, 0.6, 0.7):
        actual = 99 + h * 0.1
        error_rw = 100 - actual
        error_ar = 100 + constant + ar - actual
        rows.append(dict(origin_date=date, target_date=date + pd.Timedelta(days=h), horizon=h,
                         prediction_rw=100., prediction_ar=100 + constant + ar, error_rw=error_rw, error_ar=error_ar,
                         d=error_ar ** 2 - error_rw ** 2))
    result, _ = drift_decomposition(pd.DataFrame(rows), parameters, history)
    np.testing.assert_allclose(result.constant_loss_allocation + result.autoregressive_loss_allocation, result.d,
                               rtol=0, atol=1e-10)


def test_hac_matches_manual_bartlett_variance_and_overlap_rule():
    values = np.random.default_rng(4).normal(size=100)
    result = hac_mean_interval(values, horizon=7)
    u = values - values.mean()
    meat = np.dot(u, u)
    for lag in range(1, 21):
        meat += 2 * (1 - lag / 21) * np.dot(u[lag:], u[:-lag])
    expected_variance = (100 / 99) * meat / 100 ** 2
    assert result["hac_lag"] == 20
    assert result["hac_se"] ** 2 == pytest.approx(expected_variance)
    assert result["ci_lower"] < result["mean_d"] < result["ci_upper"]
    assert hac_mean_interval(values, horizon=25)["hac_lag"] == 24
