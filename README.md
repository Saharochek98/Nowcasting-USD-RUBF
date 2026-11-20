# USDRUBF benchmarks

Local implementation of A1, B0 Random Walk and B1 AR(1) on CLOSE differences.
The research protocol remains in `docs/`; no intervals or ML correction are implemented.

## Windows and VS Code

Open `D:\Nowcasting-USD-RUBF` as the VS Code folder. In its PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-diagnostics.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m benchmarks.run
.\.venv\Scripts\python.exe -m benchmarks.compare
git diff --stat
git diff
git status
```

Select `.venv\Scripts\python.exe` using VS Code's **Python: Select Interpreter**.
Activation is optional; explicit interpreter paths avoid PowerShell execution-policy changes.
The runner works without installing the project when invoked from the repository root.
The full pytest suite includes diagnostics; `requirements-diagnostics.txt` includes
the base requirements plus statsmodels and matplotlib.

## Evaluation and artifacts

Default: validation only. A2 results are **PRELIMINARY**, `calendar_mode=independent`,
`calendar_source=reconstructed_moex_calendar_provisional`, `conditional EOD / pseudo-PIT`.
The loader uses the supplied reconstructed session calendar, including working Saturdays;
its name does not imply a completed independent historical audit. Exact candle
publication timestamps are not supplied. Origin timestamps
remain null; `origin_date` is a daily label, not proof of publication time.

`configs/benchmark_config.yaml` fixes inclusive splits by **target date**:
train 2022-04-26–2024-12-31, validation 2025-01-01–2025-12-31,
final test 2026-01-01–2026-11-18. AR OLS uses an intercept, at least 100 valid
adjacent change pairs, and expanding refit on every origin. These rules were
fixed before validation metrics. No clipping, filling or outlier removal is used.
Insufficient history, missing latest change or singular OLS design makes AR
unavailable, with an explicit reason; there is no RW substitution.

The runner checks the full calendar against raw candle dates and the session list,
writes `reports/calendar_quality_report.md`, `reports/data_quality_report.md`
and new artifacts in `results/benchmark_calendar_v2/`:

- `predictions_rw.parquet`, `predictions_ar1.parquet`: saved before scoring,
  including unscored forecasts and unavailable AR forecasts; no actuals or errors.
- `scored_predictions.parquet`: later scoring with actuals and `prediction - actual` errors.
  Actuals outside the selected evaluation split remain null.
- `metrics_validation.csv`: MAE, RMSE, bias, RMSE ratio on **common** B0/B1
  origin/target/horizon pairs, sample sizes and date ranges.
- `pair_counts.csv`: scored/unscored and prediction availability counts by split/horizon.
- `benchmark_config.yaml`, `run_manifest.json`: config snapshot, Git SHA and dirty
  status, both calendar/input/code/config SHA256 hashes and exact Python/package versions.
- `calendar_checks.json`: date, flag, weekday, evidence and working-Saturday checks.

`python -m benchmarks.compare` writes `comparison.json`, `metrics_comparison.csv`
and updates `reports/benchmark_summary.md`. It checks every validation pair's dates,
forecasts, scoring and metrics against A1 `results/benchmark_run/` artifacts, including
old/new MAE, RMSE, bias, n, RMSE ratio and date ranges. Old files are preserved;
the runner rejects that directory as an output target.

For validation, only prices through 2025-12-31 reach forecasting/scoring; full input
integrity is audited separately. Test-origin fitting present in A1 artifacts is excluded,
while all 254 validation pairs per horizon are preserved. Later target dates can still
be retained unscored using calendar dates only.

Unknown future dates are null with `target_date_status=calendar_unavailable`;
forecasts are retained. Future target dates are never invented from weekdays.
The supplied configuration uses `calendar_mode: independent`,
`calendar_path: data/processed/usdrubf_trading_days.csv`,
`calendar_source: reconstructed_moex_calendar_provisional`, and
`full_calendar_path: data/processed/trading_calendar.csv`. The session file has unique
sorted date labels; the full file includes nontrading dates and evidence. Its 1,668
calendar dates and 1,160 expected dates match all 1,160 raw candle dates. This is
completeness relative to the reconstruction only. Include sessions without candles and known
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
`benchmarks/calendar.py` audits both calendars; `benchmarks/compare.py` compares A1/A2.
`tests/` contains original tests plus calendar and reproduction tests run by pytest.
Original raw/processed files and protocol documents are preserved.

Pass an additional model callable to `generate_forecasts(..., models={...})`.
Each callable accepts `(calendar_indexed_history_through_origin, horizons)` and
returns `(predictions_for_steps_1_through_max_horizon, metadata)`; on failure return
`(None, {"reason": "..."})`. Keep B0/B1 in the mapping for the current comparison
metrics. The engine passes only history through origin; scoring is a separate
function. Extend the metric comparison set explicitly for additional models.

## AR(1) research diagnostics

Install only the additional diagnostic dependencies, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-diagnostics.txt
.\.venv\Scripts\python.exe -m benchmarks.diagnostics_run
```

`configs/ar1_diagnostics.yaml` fixes lags, HAC bandwidth, rolling volatility and
train-only tertiles before calculation. The runner reads the saved A2 walk-forward
coefficients and validation forecasts, without fitting benchmark models or accessing
test outcomes for diagnostics. It writes `reports/ar1_diagnostics.md`, CSV tables,
PNG figures, a settings snapshot and a provenance manifest in `results/ar1_diagnostics/`.
ACF/PACF/Ljung–Box use contiguous calendar segments; HAC intervals account for
overlap. Volatility groups exclude origins before the train thresholds became
available; those forecasts remain in overall benchmark metrics. Drift decomposition reconstructs saved predictions, rather than creating
a tuned model. Base dependencies and benchmark artifacts remain unchanged.

## Safe Git workflow

Remote: `https://github.com/Saharochek98/Nowcasting-USD-RUBF.git`.
Work branch: `feature/a2-calendar-integration`, created from updated `main` after
confirming A1 was merged and fast-forwarding local `main` to fetched `origin/main`.
Review changes against `origin/main` using `git diff origin/main...HEAD`.
Git identity is not invented or changed.

Proposed A2 commit: `Integrate provisional USDRUBF trading calendar into benchmarks`.

Then stage the reviewed files, commit with your configured identity and push only
the feature branch (`git push -u origin feature/a2-calendar-integration`). Open a Pull
Request with base `main`, describe validation and unresolved calendar/PIT issues,
and let the team review and merge it. No force push, reset, rebase or automatic
merge is needed. Commits and remote writes require your permission first.
