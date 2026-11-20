"""Compare A2 validation forecasts with immutable A1 artifacts, pair by pair."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _same_frame(old, new, columns, keys):
    old = old[columns].sort_values(keys).reset_index(drop=True)
    new = new[columns].sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        return True
    except AssertionError:
        return False


def compare_runs(previous, current):
    previous, current = Path(previous), Path(current)
    if previous.resolve() == current.resolve():
        raise ValueError("Previous and current runs must be separate directories")
    keys = ["model_id", "origin_date", "target_date", "horizon"]
    columns = keys + ["prediction", "prediction_status", "train_end_date", "n_pairs", "c", "phi"]
    result = {"previous_run": str(previous), "current_run": str(current),
              "checks": {}, "forecast_counts": {}, "previous_sha256": {}}
    old_manifest = json.loads((previous / "run_manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((current / "run_manifest.json").read_text(encoding="utf-8"))
    result["checks"]["previous_is_observed_session_proxy"] = old_manifest["calendar_mode"] == "observed_session_proxy"
    result["checks"]["current_validation_only"] = new_manifest["scored_splits"] == ["validation"]
    for model in ("rw", "ar1"):
        name = f"predictions_{model}.parquet"
        old, new = pd.read_parquet(previous / name), pd.read_parquet(current / name)
        result["forecast_counts"][model] = {"previous_all": len(old), "current_all": len(new)}
        old_validation = old[old.split.eq("validation")]
        new_validation = new[new.split.eq("validation")]
        result["checks"][f"{model}_validation_pair_dates_equal"] = _same_frame(old_validation, new_validation, keys, keys)
        result["checks"][f"{model}_validation_predictions_equal"] = _same_frame(old_validation, new_validation, columns, keys)
        # All retained train/validation-origin forecasts must also reproduce A1.
        retained = old[old.origin_date <= pd.Timestamp(new_manifest["forecast_history_end"])]
        result["checks"][f"{model}_all_retained_forecasts_equal"] = _same_frame(retained, new, columns, keys)
        result["checks"][f"{model}_no_final_test_origins"] = bool((new.origin_date < pd.Timestamp("2026-01-01")).all())
        result["checks"][f"{model}_no_actuals_before_scoring"] = "actual" not in new and "error" not in new
        result["forecast_counts"][model].update(previous_validation=len(old_validation), current_validation=len(new_validation))
    old_scored = pd.read_parquet(previous / "scored_predictions.parquet")
    new_scored = pd.read_parquet(current / "scored_predictions.parquet")
    scoring_columns = keys + ["actual", "error", "scored_status"]
    result["checks"]["validation_scoring_equal"] = _same_frame(
        old_scored[old_scored.split.eq("validation")], new_scored[new_scored.split.eq("validation")], scoring_columns, keys)
    final = new_scored.split.eq("final_test")
    result["checks"]["final_test_actuals_and_errors_sealed"] = bool(
        new_scored.loc[final, ["actual", "error"]].isna().all().all()
        and new_scored.loc[final, "scored_status"].eq("unscored").all())
    old_metrics = pd.read_csv(previous / "metrics_validation.csv")
    new_metrics = pd.read_csv(current / "metrics_validation.csv")
    metric_keys = ["model_id", "split", "horizon"]
    numeric = ["n", "mae", "rmse", "bias", "rmse_ratio_vs_rw"]
    dates = ["origin_min", "origin_max", "target_min", "target_max"]
    merged = old_metrics.merge(new_metrics, on=metric_keys, suffixes=("_old", "_new"), how="outer", indicator=True,
                               validate="one_to_one")
    result["checks"]["metric_groups_equal"] = bool(merged._merge.eq("both").all() and len(merged) == 14)
    for column in numeric:
        merged[f"{column}_delta"] = merged[f"{column}_new"] - merged[f"{column}_old"]
        result["checks"][f"{column}_equal"] = bool(np.allclose(
            merged[f"{column}_new"], merged[f"{column}_old"], rtol=1e-12, atol=1e-12))
    for column in dates:
        result["checks"][f"{column}_equal"] = bool(merged[f"{column}_new"].eq(merged[f"{column}_old"]).all())
    for path in sorted(previous.iterdir()):
        if path.is_file():
            result["previous_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["passed"] = all(result["checks"].values())
    result["metric_tolerance"] = {"rtol": 1e-12, "atol": 1e-12}
    return result, merged.drop(columns="_merge")


def comparison_report(result, metrics):
    lines = ["# A2 provisional calendar integration — validation comparison", "",
             f"All reproduction checks passed: **{result['passed']}**.", "",
             "## Data and procedure", "",
             "Uses the original MOEX ISS daily CANDLE_CLOSE and unchanged raw/processed prices.",
             "The full 1,668-date reconstruction provides 1,160 expected trading dates; all match",
             "raw candle dates, including 2024-04-27, 2024-11-02, 2024-12-28 and 2025-11-01.",
             "See calendar_quality_report.md and data_quality_report.md for detailed checks.",
             "Train/validation/final-test sizes remain 683/254/223. Splits, formulas, minimum 100 AR",
             "pairs, expanding refit and common-pair metric rules are unchanged; no validation tuning.", "",
             "## Pair-by-pair reproduction", "",
             "Origin/target/horizon keys and numerical predictions were compared exactly for every",
             "validation forecast, together with OLS parameters and training pair counts. Actuals",
             "and errors were compared separately after scoring. Metrics use rtol=atol=1e-12.",
             "The comparison also checks all retained train/validation-origin forecasts against A1.", "",
             "| Check | Passed |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in result["checks"].items()]
    lines += ["", "## Validation metrics (old → new)", "",
              "Each model has 254 common scored pairs per horizon, targets 2025-01-03–2025-12-30.",
              "Exact origin/target ranges, old/new MAE, RMSE, bias, ratio and deltas are saved in",
              "metrics_comparison.csv beside comparison.json in the new result directory.", "",
              "| h | n old/new | RW RMSE old/new | AR RMSE old/new | AR/RW old/new |", "|---:|---:|---:|---:|---:|"]
    for h in range(1, 8):
        group = metrics[metrics.horizon.eq(h)].set_index("model_id")
        rw, ar = group.loc["B0"], group.loc["B1"]
        lines.append(f"| {h} | {rw.n_old:.0f}/{rw.n_new:.0f} | {rw.rmse_old:.6f}/{rw.rmse_new:.6f} | "
                     f"{ar.rmse_old:.6f}/{ar.rmse_new:.6f} | {ar.rmse_ratio_vs_rw_old:.6f}/{ar.rmse_ratio_vs_rw_new:.6f} |")
    lines += ["", "AR does not improve diagnostic RMSE over RW. No statistical significance is claimed.", "",
              "## Scope, preservation and limitations", "",
              f"Forecast record counts: {result['forecast_counts']}.",
              "A2 model history stops at the end of validation (last observed origin 2025-12-30).",
              "A1 had 16,240 records including 2026-origin forecasts; A2 retains 13,118 records,",
              "6,559 per model. The removed 3,122 records are test-origin forecasts, not lost",
              "validation pairs. There are still 1,778 validation pairs per model (254 × 7).",
              "Test-period prices are used only for dataset integrity auditing, never passed to",
              "validation forecasting or scoring. Known future target dates can be retained unscored.",
              "Final-test fitting/evaluation was not run; actuals/errors for test targets remain null.",
              "Old results/benchmark_run files are preserved byte-for-byte; their hashes are recorded",
              "in comparison.json. New outputs are in results/benchmark_calendar_v2.",
              "The reconstructed calendar is PROVISIONAL. Loader mode independent does not certify",
              "a fully verified instrument-specific historical archive. Most dates use production-calendar",
              "rules and MOEX notices. Exact candle publication times and historical session attribution",
              "remain unresolved; conditional EOD / pseudo-PIT is retained. No exact timestamp is invented.",
              "The new manifest records hashes of both calendars, input data, config and all benchmark",
              "modules; Python/library versions, base Git SHA and working state are included. This run",
              "uses uncommitted A2 source: hashes identify it. Repeat after an authorized commit to record",
              "the implementation commit SHA. No A2 commit, push or merge has been performed."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="results/benchmark_run")
    parser.add_argument("--current", default="results/benchmark_calendar_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous, current = root / args.previous, root / args.current
    result, metrics = compare_runs(previous, current)
    (current / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics.to_csv(current / "metrics_comparison.csv", index=False)
    (root / "reports/benchmark_summary.md").write_text(comparison_report(result, metrics), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Reproduction mismatch; inspect comparison.json before drawing conclusions")


if __name__ == "__main__":
    main()
