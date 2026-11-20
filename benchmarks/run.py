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
                  f"Sessions without valid price: {len(missing)}: {list(missing.astype(str))}"]
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
              "Processed README calls price original MOEX CANDLE_CLOSE and not settlement;",
              "research charter/benchmark spec leave this conditional on external verification.",
              "Raw/processed equality supports field provenance but does not independently prove",
              "market-close semantics or historical publication timing. No external confirmation",
              "was established in this implementation. Q01/Q02/Q03 remain open; primary experiment blocked.",
              "README mentions session-day attribution changing on 2026-03-23; historical availability",
              "and attribution require a source audit. No exact forecast timestamp is fabricated.",
              "Minimal clarification: retain conditional CLOSE semantics across documents until",
              "Data Engineer supplies field documentation, availability evidence and an independent calendar.",
              "A1 bounds, minimum 100 AR pairs and every-origin expanding refit are fixed in config",
              "before validation scoring. Final-test metrics require explicit --include-final-test;",
              "final-test outcomes must never choose models, preprocessing or hyperparameters."]
    if invalid.any() or integrity.any() or mismatch or volume_mismatch:
        lines.append("\nBLOCKED: raw/processed integrity failure; modeling must not proceed.")
    return "\n".join(lines) + "\n", bool(invalid.any() or integrity.any() or mismatch or volume_mismatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--output", default="results/benchmark_run")
    parser.add_argument("--evaluation-split", choices=["validation", "final_test"], default="validation")
    parser.add_argument("--include-final-test", action="store_true")
    args = parser.parse_args()
    if args.evaluation_split == "final_test" and not args.include_final_test:
        parser.error("final_test requires explicit --include-final-test")
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = load_config(config_path)
    data_path, raw_path = root / config["data_path"], root / config["raw_path"]
    data = load_data(data_path)
    calendar = load_calendar(data, config, root)
    report, blocked = audit(data, raw_path, calendar, config)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/data_quality_report.md").write_text(report, encoding="utf-8")
    if blocked:
        raise ValueError("Audit failed; see reports/data_quality_report.md")
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    predictions = generate_forecasts(data, calendar, config)
    # Persist forecasts BEFORE attaching any future actuals.
    for model, name in (("B0", "rw"), ("B1", "ar1")):
        predictions.loc[predictions.model_id.eq(model)].to_parquet(
            output / f"predictions_{name}.parquet", index=False)
    allowed = [args.evaluation_split]
    scored = score_forecasts(predictions, data, allowed)
    scored.to_parquet(output / "scored_predictions.parquet", index=False)
    metrics = metrics_by_horizon(scored)
    metrics.to_csv(output / f"metrics_{args.evaluation_split}.csv", index=False)
    counts = scored.groupby(["model_id", "split", "horizon", "prediction_status", "scored_status"],
                            dropna=False).size().rename("n").reset_index()
    counts.to_csv(output / "pair_counts.csv", index=False)
    (output / "benchmark_config.yaml").write_bytes(config_path.read_bytes())
    sources = [data_path, raw_path, config_path, root / "benchmarks/core.py", root / "benchmarks/run.py"]
    if config.get("calendar_path"):
        sources.append(root / config["calendar_path"])
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "PyYAML", "pyarrow")},
        "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "calendar_mode": config["calendar_mode"], "pit_status": config["pit_status"],
        "result_status": "preliminary", "scored_splits": allowed,
        "error_convention": "prediction - actual",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"PRELIMINARY {config['calendar_mode']}: {len(predictions)} forecasts; output: {output}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
