import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks.calendar import KNOWN_SATURDAYS, audit_calendars
from benchmarks.compare import compare_runs
from benchmarks.core import generate_forecasts, load_calendar, load_config, load_data, score_forecasts
from benchmarks import run


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/processed/trading_calendar.csv"
SESSIONS = ROOT / "data/processed/usdrubf_trading_days.csv"
RAW = ROOT / "data/raw/USDRUBF_daily_2026-11-18.csv"


@pytest.fixture(scope="module")
def inputs():
    config = load_config(ROOT / "configs/benchmark_config.yaml")
    data = load_data(ROOT / config["data_path"])
    calendar = load_calendar(data, config, ROOT)
    return config, data, calendar


def test_real_calendars_match_claims_and_saturdays(inputs):
    config, data, calendar = inputs
    result = audit_calendars(FULL, SESSIONS, RAW)
    assert result["status"] == "passed"
    assert result["readme_claims_match"]
    assert all(result["checks"].values())
    assert result["counts"] == {"calendar_dates": 1668, "expected_trading_dates": 1160,
                                "listed_trading_dates": 1160, "raw_candles": 1160,
                                "missing_expected_candles": 0, "unexpected_candles": 0}
    assert set(result["known_saturdays"]) == set(KNOWN_SATURDAYS)
    assert all(result["known_saturdays"].values())
    assert calendar.equals(data.index)
    assert config["calendar_source"] == "reconstructed_moex_calendar_provisional"


@pytest.mark.parametrize("file_kind", ["full", "sessions"])
@pytest.mark.parametrize("corruption", ["duplicate", "unsorted", "invalid_date"])
def test_bad_calendar_dates_produce_diagnostics(tmp_path, file_kind, corruption):
    original = FULL if file_kind == "full" else SESSIONS
    frame = pd.read_csv(original)
    if corruption == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif corruption == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame.loc[0, "date"] = "2025-02-30"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    result = audit_calendars(path if file_kind == "full" else FULL,
                             path if file_kind == "sessions" else SESSIONS, RAW)
    assert result["status"] == "failed"
    assert result["errors"]


@pytest.mark.parametrize("column,value,check", [
    ("has_candle", 0, "has_candle_matches_raw"),
    ("is_trading_day_expected", 0, "sessions_equal_expected_dates"),
    ("weekday", "Sunday", "weekday_labels_match"),
    ("moex_evidence", "", "evidence_present"),
    ("has_candle", 2, None),
])
def test_calendar_flags_and_evidence_checked(tmp_path, column, value, check):
    full = pd.read_csv(FULL)
    full.loc[0, column] = value
    path = tmp_path / "full.csv"
    full.to_csv(path, index=False)
    before = path.read_bytes()
    result = audit_calendars(path, SESSIONS, RAW)
    assert result["status"] == "failed"
    if check:
        assert result["checks"][check] is False
    assert path.read_bytes() == before  # diagnostics never silently repair input


def test_full_calendar_must_cover_every_calendar_date(tmp_path):
    full = pd.read_csv(FULL).drop(index=4)  # remove a nontrading date, not a candle
    path = tmp_path / "full.csv"
    full.to_csv(path, index=False)
    result = audit_calendars(path, SESSIONS, RAW)
    assert result["status"] == "failed"
    assert not result["checks"]["full_calendar_contiguous"]


def test_missing_session_is_reported_and_preserved(inputs, tmp_path):
    config, data, calendar = inputs
    missing = pd.Timestamp("2025-01-09")
    raw = pd.read_csv(RAW)
    raw = raw.loc[~raw.begin.str.startswith(str(missing.date()))]
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    full = pd.read_csv(FULL)
    full.loc[full.date.eq(str(missing.date())), "has_candle"] = 0
    full_path = tmp_path / "full.csv"
    full.to_csv(full_path, index=False)
    result = audit_calendars(full_path, SESSIONS, raw_path)
    assert result["status"] == "gaps"
    assert result["missing_expected_dates"] == ["2025-01-09"]
    with_gap = data.loc[:"2025-01-15"].drop(missing)
    forecasts = generate_forecasts(with_gap, calendar, config)
    assert not forecasts.origin_date.eq(missing).any()
    rows = forecasts[forecasts.origin_date.eq("2025-01-08") & forecasts.horizon.eq(1)]
    assert rows.target_date.eq(missing).all() and len(rows) == 2
    scored = score_forecasts(forecasts, with_gap)
    assert scored.loc[scored.target_date.eq(missing), "scored_status"].eq("unscored").all()
    after_gap = forecasts[forecasts.origin_date.eq("2025-01-10") & forecasts.model_id.eq("B1")]
    assert after_gap.prediction_status.eq("missing_last_change").all()
    later = forecasts[forecasts.origin_date.eq("2025-01-15") & forecasts.model_id.eq("B1")].iloc[0]
    prices = with_gap.price.reindex(calendar).loc[:"2025-01-15"]
    # Exactly three AR rows touch the absent middle price. No bridging is allowed.
    assert later.n_pairs == len(prices) - 2 - 3


def test_seven_targets_include_working_saturday(inputs):
    config, data, calendar = inputs
    forecasts = generate_forecasts(data.loc[:"2024-12-27"], calendar, config)
    expected = pd.to_datetime(["2024-12-28", "2024-12-30", "2025-01-03", "2025-01-06",
                               "2025-01-08", "2025-01-09", "2025-01-10"])
    for model in ("B0", "B1"):
        rows = forecasts[forecasts.origin_date.eq("2024-12-27") & forecasts.model_id.eq(model)].sort_values("horizon")
        assert list(rows.target_date) == list(expected)


def test_missing_saturday_candle_does_not_remove_calendar_day(tmp_path):
    day = "2025-11-01"
    raw = pd.read_csv(RAW)
    path = tmp_path / "raw.csv"
    raw.loc[~raw.begin.str.startswith(day)].to_csv(path, index=False)
    full = pd.read_csv(FULL)
    full.loc[full.date.eq(day), "has_candle"] = 0
    full_path = tmp_path / "full.csv"
    full.to_csv(full_path, index=False)
    result = audit_calendars(full_path, SESSIONS, path)
    assert result["status"] == "gaps"
    assert result["known_saturdays"][day]
    assert not result["known_saturday_candles"][day]
    assert result["missing_expected_dates"] == [day]


def test_independent_calendar_no_lookahead(inputs):
    config, data, calendar = inputs
    restricted = data.loc[:"2025-01-20"]
    altered = restricted.copy()
    altered.loc["2025-01-10":, "price"] *= 3
    original = generate_forecasts(restricted, calendar, config)
    changed = generate_forecasts(altered, calendar, config)
    mask = original.origin_date < pd.Timestamp("2025-01-10")
    pd.testing.assert_frame_equal(original.loc[mask], changed.loc[mask], check_exact=True)


def test_default_runner_reproduces_validation_and_seals_test(inputs, tmp_path, monkeypatch):
    config, data, _ = inputs
    # Model-facing data must physically exclude final-test observations.
    seen = []
    original_generate = run.generate_forecasts

    def guarded_generate(frame, calendar, configuration):
        assert frame.index.max() == pd.Timestamp("2025-12-30")
        seen.append(True)
        return original_generate(frame, calendar, configuration)

    monkeypatch.setattr(run, "generate_forecasts", guarded_generate)
    monkeypatch.setattr(sys, "argv", ["benchmarks.run", "--output", str(tmp_path)])
    baseline = ROOT / "results/benchmark_run"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.iterdir() if p.is_file()}
    run.main()
    assert seen == [True]
    result, comparison = compare_runs(baseline, tmp_path)
    assert result["passed"]
    assert comparison.n_new.eq(254).all()
    assert len(comparison) == 14
    assert (comparison.filter(like="_delta").abs() <= 1e-12).all().all()
    assert hashes == result["previous_sha256"]
    for model in ("rw", "ar1"):
        assert result["forecast_counts"][model]["current_validation"] == 1778
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["scored_splits"] == ["validation"]
    assert not manifest["final_test_used_for_fitting_or_scoring"]
    assert config["calendar_path"].replace("/", "\\") in {key.replace("/", "\\") for key in manifest["sha256"]}
    assert config["full_calendar_path"].replace("/", "\\") in {key.replace("/", "\\") for key in manifest["sha256"]}
    assert "calendar.py" in " ".join(manifest["sha256"])
    for name in ("predictions_rw.parquet", "predictions_ar1.parquet"):
        frame = pd.read_parquet(tmp_path / name)
        assert len(frame) == 6559
        assert frame.scored_status.eq("unscored").all()
        assert frame.calendar_source.eq("reconstructed_moex_calendar_provisional").all()


def test_comparison_detects_pair_or_prediction_change(inputs, tmp_path, monkeypatch):
    # Reuse a small copy of baseline artifacts, then intentionally perturb one validation forecast.
    baseline = ROOT / "results/benchmark_run"
    for path in baseline.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["forecast_history_end"] = "2026-11-18"
    manifest_path.write_text(json.dumps(manifest))
    path = tmp_path / "predictions_ar1.parquet"
    frame = pd.read_parquet(path)
    index = frame.index[frame.split.eq("validation")][0]
    frame.loc[index, "prediction"] += 1
    frame.to_parquet(path, index=False)
    result, _ = compare_runs(baseline, tmp_path)
    assert not result["passed"]
    assert not result["checks"]["ar1_validation_predictions_equal"]
    assert result["checks"]["ar1_validation_pair_dates_equal"]


def test_original_run_directory_is_protected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmarks.run", "--output", "results/benchmark_run"])
    with pytest.raises(SystemExit, match="2"):
        run.main()
