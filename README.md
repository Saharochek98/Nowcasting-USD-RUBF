# USDRUBF benchmarks

Local implementation of A1, B0 Random Walk and B1 AR(1) on CLOSE differences.
The research protocol remains in `docs/`; no intervals or ML correction are implemented.

## Windows and VS Code

Open `D:\Nowcasting-USD-RUBF` as the VS Code folder. In its PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m benchmarks.run
git diff --stat
git diff
git status
```

Select `.venv\Scripts\python.exe` using VS Code's **Python: Select Interpreter**.
Activation is optional; explicit interpreter paths avoid PowerShell execution-policy changes.
The runner works without installing the project when invoked from the repository root.

## Evaluation and artifacts

Default: validation only. Current results are **PRELIMINARY**,
`observed_session_proxy`, `conditional EOD / pseudo-PIT`, rather than a verified
seven-trading-day market-close backtest. No independent session calendar or
historical candle publication timestamps are supplied. Exact origin timestamps
remain null; `origin_date` is a daily label, not proof of publication time.

`configs/benchmark_config.yaml` fixes inclusive splits by **target date**:
train 2022-04-26–2024-12-31, validation 2025-01-01–2025-12-31,
final test 2026-01-01–2026-11-18. AR OLS uses an intercept, at least 100 valid
adjacent change pairs, and expanding refit on every origin. These rules were
fixed before validation metrics. No clipping, filling or outlier removal is used.
Insufficient history, missing latest change or singular OLS design makes AR
unavailable, with an explicit reason; there is no RW substitution.

The runner writes `reports/data_quality_report.md` and, in `results/benchmark_run/`:

- `predictions_rw.parquet`, `predictions_ar1.parquet`: saved before scoring,
  including unscored forecasts and unavailable AR forecasts; no actuals or errors.
- `scored_predictions.parquet`: later scoring with actuals and `prediction - actual` errors.
  Actuals outside the selected evaluation split remain null.
- `metrics_validation.csv`: MAE, RMSE, bias, RMSE ratio on **common** B0/B1
  origin/target/horizon pairs, sample sizes and date ranges.
- `pair_counts.csv`: scored/unscored and prediction availability counts by split/horizon.
- `benchmark_config.yaml`, `run_manifest.json`: config snapshot, Git SHA and dirty
  status, source/data/config SHA256 hashes and exact Python/package versions.

Unknown future dates are null with `target_date_status=calendar_unavailable`;
forecasts are retained. Future target dates are never invented from weekdays.
For a documented independent calendar, set `calendar_mode: independent`,
`calendar_path` to a CSV containing unique sorted `date` labels and
`calendar_source` to its provenance. Include sessions without candles and known
future sessions; supplied dates are checked against observed dates. Missing
prices remain NaN, so differences and lagged AR pairs never bridge missing sessions.
Completeness outside a supplied calendar's coverage cannot be certified.

Only after model/preprocessing decisions are frozen, explicitly evaluate test:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run --evaluation-split final_test --include-final-test --output results/final_test
```

This command has **not** been run during development. Even with a calendar,
CLOSE semantics and historical availability still require independent confirmation.
RMSE ratio below one alone is not evidence of statistical significance.

## Compact code structure and extension

`benchmarks/core.py` contains loading, splits, calendar checks, models, walk-forward
and metrics. `benchmarks/run.py` provides the CLI, audit and persistence;
`tests/test_benchmarks.py` contains synthetic tests run by pytest.
Original raw/processed files and protocol documents are preserved.

Pass an additional model callable to `generate_forecasts(..., models={...})`.
Each callable accepts `(calendar_indexed_history_through_origin, horizons)` and
returns `(predictions_for_steps_1_through_max_horizon, metadata)`; on failure return
`(None, {"reason": "..."})`. Keep B0/B1 in the mapping for the current comparison
metrics. The engine passes only history through origin; scoring is a separate
function. Extend the metric comparison set explicitly for additional models.

## Safe Git workflow

Remote: `https://github.com/Saharochek98/Nowcasting-USD-RUBF.git`.
Work branch: `feature/a1-rw-ar1-baselines`, created from fetched `origin/main`.
Review changes against `origin/main` using `git diff origin/main...HEAD`.
Git identity is not invented or changed.

The implementation is organized into two commits:

1. `Implement chronological splits and calendar-aware RW/AR1 benchmarks`
2. `Add leakage tests and reproducible preliminary validation artifacts`

Then stage the reviewed files, commit with your configured identity and push only
the feature branch (`git push -u origin feature/a1-rw-ar1-baselines`). Open a Pull
Request with base `main`, describe validation and unresolved calendar/PIT issues,
and let the team review and merge it. No force push, reset, rebase or automatic
merge is needed. Commits and remote writes require your permission first.
