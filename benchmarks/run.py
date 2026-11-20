"""Run with python -m benchmarks.run; all paths resolve from the repository."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from importlib.metadata import version

import numpy as np
import pandas as pd

from .core import (generate_forecasts, load_calendar, load_config, load_data,
                   make_splits, metrics_by_horizon, score_forecasts)
from .calendar import audit_calendars, calendar_report


def audit(data, raw_path, calendar, config):
    raw = pd.read_csv(raw_path)
    dates = pd.to_datetime(raw["begin"], errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Raw dates must be unique, present and ordered")
    raw.index = pd.DatetimeIndex(dates)
    ohlc = raw[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(ohlc) | (ohlc <= 0)).any(axis=1)
    integrity = ((ohlc.high < ohlc[["open", "close"]].max(axis=1))
                 | (ohlc.low > ohlc[["open", "close"]].min(axis=1))
                 | (ohlc.high < ohlc.low))
    same_dates = raw.index.equals(data.index)
    mismatch = not same_dates or not np.allclose(
        ohlc.close.to_numpy(), data.price.to_numpy(), rtol=0, atol=1e-10, equal_nan=True)
    volume_mismatch = not same_dates or not np.allclose(
        pd.to_numeric(raw.volume).to_numpy(), pd.to_numeric(data.volume).to_numpy(),
        rtol=0, atol=1e-10, equal_nan=True)
    returns = data.price.pct_change(fill_method=None)
    moves = returns[returns.abs() > config["large_move_threshold"]]
    lines = ["# Data quality report", "",
             "Status: PRELIMINARY; conditional EOD / pseudo-PIT.", "",
             f"- Observations: {len(data)} (expected claim: 1160).",
             f"- Range: {data.index.min().date()} — {data.index.max().date()}.",
             "- Dates: sorted, unique, nonmissing (loader rejects violations).",
             f"- Missing values by column: {data.isna().sum().to_dict()}.",
             "- Nonnumeric/nonpositive/infinite processed prices: 0 (loader rejects violations).",
             f"- Raw invalid/missing OHLC rows: {int(invalid.sum())}.",
             f"- Raw OHLC integrity failures: {int(integrity.sum())}.",
             f"- Raw/processed date sets match: {same_dates}.",
             f"- CLOSE/price mismatch: {mismatch}; volume mismatch: {volume_mismatch}.",
             "- No filling, replacement, adjustment or outlier removal performed.",
             "  Equality to raw checks derivation; upstream substitutions cannot be ruled out.", "",
             "## Chronological split (inclusive; forecast pairs use target dates)", "",
             "| Split | n | First observed | Last observed |", "|---|---:|---|---|"]
    for name, frame in make_splits(data, config).items():
        lines.append(f"| {name} | {len(frame)} | {frame.index.min()} | {frame.index.max()} |")
    lines += ["", "## Session calendar", "",
              f"Mode: `{config['calendar_mode']}`."]
    if config["calendar_mode"] == "independent":
        missing = calendar.difference(data.index[data.price.notna()])
        lines += [f"Source: {config['calendar_source']}",
                  f"Sessions without valid price: {len(missing)}: {list(missing.astype(str))}",
                  "The supplied calendar is a PROVISIONAL reconstruction from production-calendar",
                  "rules and MOEX notices; mode independent is a loader mode, not a verification claim.",
                  "See calendar_quality_report.md for full-calendar/expected-date/raw-date checks.",
                  "Completeness is assessed only relative to this reconstruction, not an independently",
                  "verified archive of every USDRUBF session. Known working Saturdays are retained."]
        for name, start, end in [(name, *pd.to_datetime(bounds)) for name, bounds in config["splits"].items()]:
            sessions = calendar[(calendar >= start) & (calendar <= end)]
            gaps_in_split = sessions.difference(data.index[data.price.notna()])
            lines.append(f"- {name}: {len(sessions)} expected sessions; {len(gaps_in_split)} without valid price.")
    else:
        lines += ["Repository inspection found only raw candles and a derived baseline;",
                  "neither supplies an independent calendar including sessions without candles.",
                  "The ISS candles endpoint in the README supplies candles, not independent sessions.",
                  "Calendar completeness is UNKNOWN, including in each split. Horizons are observed",
                  "session steps only; this is not a final seven-trading-day backtest.",
                  "No weekends-only calendar is synthesized. Unknown future target dates remain null."]
    gaps = data.index.to_series().diff().dt.days
    lines += ["", "Calendar-day gaps above 4 days (diagnostic, not evidence of missing sessions):",
              str({str(k.date()): int(v) for k, v in gaps[gaps > 4].items()}), "",
              f"## Absolute observed price moves above {config['large_move_threshold']:.0%}", "",
              "Threshold fixed in config; flagged observations retained.", ""]
    lines += [f"- {day.date()}: {move:.4%}" for day, move in moves.items()]
    lines += ["", "## Open semantics, provenance and protocol issues", "",
              "Per the A2 task agreement, target is the original MOEX ISS daily CANDLE_CLOSE,",
              "not CBR or clearing settlement. Raw/processed equality verifies the retained field.",
              "Existing protocol documents still condition semantics on external evidence; they are",
              "preserved. No new independent external field/session audit was performed in A2.",
              "Historical candle publication timestamps, end-label interpretation, and session-day",
              "attribution remain unverified. Data README documents atypical end labels and weekend",
              "session attribution; end is not assumed to be a precise last-trade or availability time.",
              "Q02 remains open; Q03 has a provisional reconstruction, not final independent closure.",
              "No exact forecast timestamp is fabricated; conditional EOD / pseudo-PIT remains.",
              "A1 bounds, minimum 100 AR pairs and every-origin expanding refit are fixed in config",
              "before validation scoring. Final-test metrics require explicit --include-final-test;",
              "final-test outcomes must never choose models, preprocessing or hyperparameters."]
    if invalid.any() or integrity.any() or mismatch or volume_mismatch:
        lines.append("\nBLOCKED: raw/processed integrity failure; modeling must not proceed.")
    return "\n".join(lines) + "\n", bool(invalid.any() or integrity.any() or mismatch or volume_mismatch)


def evaluation_data(data, config, split):
    """Audit may inspect full input integrity; forecasting/scoring receives no later prices."""
    return data.loc[:pd.Timestamp(config["splits"][split][1])].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--output", default="results/benchmark_calendar_v2")
    parser.add_argument("--evaluation-split", choices=["validation", "final_test"], default="validation")
    parser.add_argument("--include-final-test", action="store_true")
    args = parser.parse_args()
    if args.evaluation_split == "final_test" and not args.include_final_test:
        parser.error("final_test requires explicit --include-final-test")
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = load_config(config_path)
    git_status_at_start = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    data_path, raw_path = root / config["data_path"], root / config["raw_path"]
    output = root / args.output
    # Historical A1 artifacts are the immutable reference for A2 comparison.
    if output.resolve() == (root / "results/benchmark_run").resolve():
        parser.error("Choose a new output directory; results/benchmark_run is the A1 reference")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("full_calendar_path"):
        calendar_audit = audit_calendars(root / config["full_calendar_path"],
                                        root / config["calendar_path"], raw_path)
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports/calendar_quality_report.md").write_text(calendar_report(calendar_audit), encoding="utf-8")
        (output / "calendar_checks.json").write_text(json.dumps(calendar_audit, indent=2), encoding="utf-8")
        if calendar_audit["errors"]:
            raise ValueError("Calendar consistency failed; see reports/calendar_quality_report.md")
    data = load_data(data_path)
    calendar = load_calendar(data, config, root)
    report, blocked = audit(data, raw_path, calendar, config)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/data_quality_report.md").write_text(report, encoding="utf-8")
    if blocked:
        raise ValueError("Audit failed; see reports/data_quality_report.md")
    restricted = evaluation_data(data, config, args.evaluation_split)
    predictions = generate_forecasts(restricted, calendar, config)
    # Persist forecasts BEFORE attaching any future actuals.
    for model, name in (("B0", "rw"), ("B1", "ar1")):
        predictions.loc[predictions.model_id.eq(model)].to_parquet(
            output / f"predictions_{name}.parquet", index=False)
    allowed = [args.evaluation_split]
    scored = score_forecasts(predictions, restricted, allowed)
    scored.to_parquet(output / "scored_predictions.parquet", index=False)
    metrics = metrics_by_horizon(scored)
    metrics.to_csv(output / f"metrics_{args.evaluation_split}.csv", index=False)
    counts = scored.groupby(["model_id", "split", "horizon", "prediction_status", "scored_status"],
                            dropna=False).size().rename("n").reset_index()
    counts.to_csv(output / "pair_counts.csv", index=False)
    (output / "benchmark_config.yaml").write_bytes(config_path.read_bytes())
    sources = [data_path, raw_path, config_path, root / "requirements.txt", *sorted((root / "benchmarks").glob("*.py"))]
    if config.get("calendar_path"):
        sources.append(root / config["calendar_path"])
    if config.get("full_calendar_path"):
        sources.append(root / config["full_calendar_path"])
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": git_status_at_start,
        "implementation_state": "uncommitted_changes" if git_status_at_start else "committed",
        "reproducibility_note": "If uncommitted, hashes identify implementation; repeat after authorized commit to record its SHA.",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "PyYAML", "pyarrow")},
        "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "calendar_mode": config["calendar_mode"], "calendar_source": config.get("calendar_source"),
        "calendar_verification": "provisional", "pit_status": config["pit_status"],
        "forecast_history_end": str(restricted.index.max().date()),
        "final_test_used_for_fitting_or_scoring": args.evaluation_split == "final_test",
        "result_status": "preliminary", "scored_splits": allowed,
        "error_convention": "prediction - actual",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"PRELIMINARY {config['calendar_mode']}: {len(predictions)} forecasts; output: {output}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
