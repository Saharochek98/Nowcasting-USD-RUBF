# Preliminary validation diagnostics — 2026-11-20

This run implements A1 and B0/B1 under `observed_session_proxy` and
`conditional EOD / pseudo-PIT`. It is **not** a confirmed seven-trading-day backtest.
See `data_quality_report.md` for audit evidence and unresolved issues.

## Data and fixed procedure

Raw and processed files both contain 1,160 observations, 2022-04-26–2026-11-18.
Train: 683 observations (2022-04-26–2024-12-30); validation: 254
(2025-01-03–2025-12-30); final test: 223 (2026-01-05–2026-11-18).
Configured inclusive boundaries and target-date assignment are in
`configs/benchmark_config.yaml`. No duplicate dates, invalid prices or OHLC
integrity failures were found. Processed prices/volumes match raw close/volume.

AR has an intercept and a minimum of 100 adjacent change pairs, recursively
accumulates predicted changes, and refits on each expanding history. Rules were
fixed before metrics. No imputation, scaling, clipping or model selection was
performed. Synthetic series appear only in tests.

## Results

254 common scored origin/target pairs per horizon, validation targets
2025-01-03–2025-12-30. Across horizons the earliest origin is 2024-12-23;
the latest is 2025-12-29. Exact per-horizon periods, MAE and bias are in
`results/benchmark_run/metrics_validation.csv`.

| Observed-session horizon | RW RMSE | AR(1) RMSE | AR/RW RMSE |
|---:|---:|---:|---:|
| 1 | 0.772440 | 0.774045 | 1.002078 |
| 2 | 1.135664 | 1.144051 | 1.007385 |
| 3 | 1.404904 | 1.422296 | 1.012380 |
| 4 | 1.617355 | 1.644489 | 1.016777 |
| 5 | 1.790585 | 1.827089 | 1.020387 |
| 6 | 1.965425 | 2.012490 | 1.023947 |
| 7 | 2.147724 | 2.207770 | 1.027958 |

AR did not improve RMSE over RW in this preliminary diagnostic. No statistical
significance claim is made; overlapping errors have not undergone inference.
No tuning followed these results. Final-test scoring was **not run**.

All 16,240 forecast records (8,120 per model) are retained, including unavailable
AR predictions and forecasts with unknown future target dates. Actuals/errors
are attached only in the separate scored artifact. Final-test actuals remain
null there; historical observations can still enter subsequent origin histories
under the fixed expanding-window rule.

## Remaining limits and Git handoff

An independent USDRUBF session calendar is absent; missing trading sessions
cannot be identified conclusively. CLOSE semantics, historical date attribution
and exact candle publication times require external verification. Q01–Q03 are
still open. Main experiment is blocked pending this evidence; diagnostic code
and synthetic calendar-hole tests remain usable.

`README.md` documents Windows setup, pytest, default validation, the separately
guarded final-test command, artifact contracts and model extension. Source files,
config snapshots, hashes and exact environment versions are recorded alongside
results. Git SHA identifies the base commit; source hashes identify uncommitted
implementation. Re-run after an authorized commit to record its new SHA.

Commit structure: implementation/config/dependencies; then tests, documentation
and preliminary artifacts. The user authorized commits, push and a Pull Request.
Publish only `feature/a1-rw-ar1-baselines` and open a Pull Request into `main`
for team review; merging remains the team's decision.
