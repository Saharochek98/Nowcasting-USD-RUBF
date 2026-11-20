# A2 provisional calendar integration — validation comparison

All reproduction checks passed: **True**.

## Data and procedure

Uses the original MOEX ISS daily CANDLE_CLOSE and unchanged raw/processed prices.
The full 1,668-date reconstruction provides 1,160 expected trading dates; all match
raw candle dates, including 2024-04-27, 2024-11-02, 2024-12-28 and 2025-11-01.
See calendar_quality_report.md and data_quality_report.md for detailed checks.
Train/validation/final-test sizes remain 683/254/223. Splits, formulas, minimum 100 AR
pairs, expanding refit and common-pair metric rules are unchanged; no validation tuning.

## Pair-by-pair reproduction

Origin/target/horizon keys and numerical predictions were compared exactly for every
validation forecast, together with OLS parameters and training pair counts. Actuals
and errors were compared separately after scoring. Metrics use rtol=atol=1e-12.
The comparison also checks all retained train/validation-origin forecasts against A1.

| Check | Passed |
|---|---|
| previous_is_observed_session_proxy | True |
| current_validation_only | True |
| rw_validation_pair_dates_equal | True |
| rw_validation_predictions_equal | True |
| rw_all_retained_forecasts_equal | True |
| rw_no_final_test_origins | True |
| rw_no_actuals_before_scoring | True |
| ar1_validation_pair_dates_equal | True |
| ar1_validation_predictions_equal | True |
| ar1_all_retained_forecasts_equal | True |
| ar1_no_final_test_origins | True |
| ar1_no_actuals_before_scoring | True |
| validation_scoring_equal | True |
| final_test_actuals_and_errors_sealed | True |
| metric_groups_equal | True |
| n_equal | True |
| mae_equal | True |
| rmse_equal | True |
| bias_equal | True |
| rmse_ratio_vs_rw_equal | True |
| origin_min_equal | True |
| origin_max_equal | True |
| target_min_equal | True |
| target_max_equal | True |

## Validation metrics (old → new)

Each model has 254 common scored pairs per horizon, targets 2025-01-03–2025-12-30.
Exact origin/target ranges, old/new MAE, RMSE, bias, ratio and deltas are saved in
metrics_comparison.csv beside comparison.json in the new result directory.

| h | n old/new | RW RMSE old/new | AR RMSE old/new | AR/RW old/new |
|---:|---:|---:|---:|---:|
| 1 | 254/254 | 0.772440/0.772440 | 0.774045/0.774045 | 1.002078/1.002078 |
| 2 | 254/254 | 1.135664/1.135664 | 1.144051/1.144051 | 1.007385/1.007385 |
| 3 | 254/254 | 1.404904/1.404904 | 1.422296/1.422296 | 1.012380/1.012380 |
| 4 | 254/254 | 1.617355/1.617355 | 1.644489/1.644489 | 1.016777/1.016777 |
| 5 | 254/254 | 1.790585/1.790585 | 1.827089/1.827089 | 1.020387/1.020387 |
| 6 | 254/254 | 1.965425/1.965425 | 2.012490/2.012490 | 1.023947/1.023947 |
| 7 | 254/254 | 2.147724/2.147724 | 2.207770/2.207770 | 1.027958/1.027958 |

AR does not improve diagnostic RMSE over RW. No statistical significance is claimed.

## Scope, preservation and limitations

Forecast record counts: {'rw': {'previous_all': 8120, 'current_all': 6559, 'previous_validation': 1778, 'current_validation': 1778}, 'ar1': {'previous_all': 8120, 'current_all': 6559, 'previous_validation': 1778, 'current_validation': 1778}}.
A2 model history stops at the end of validation (last observed origin 2025-12-30).
A1 had 16,240 records including 2026-origin forecasts; A2 retains 13,118 records,
6,559 per model. The removed 3,122 records are test-origin forecasts, not lost
validation pairs. There are still 1,778 validation pairs per model (254 × 7).
Test-period prices are used only for dataset integrity auditing, never passed to
validation forecasting or scoring. Known future target dates can be retained unscored.
Final-test fitting/evaluation was not run; actuals/errors for test targets remain null.
Old results/benchmark_run files are preserved byte-for-byte; their hashes are recorded
in comparison.json. New outputs are in results/benchmark_calendar_v2.
The reconstructed calendar is PROVISIONAL. Loader mode independent does not certify
a fully verified instrument-specific historical archive. Most dates use production-calendar
rules and MOEX notices. Exact candle publication times and historical session attribution
remain unresolved; conditional EOD / pseudo-PIT is retained. No exact timestamp is invented.
The new manifest records hashes of both calendars, input data, config and all benchmark
modules; Python/library versions, base Git SHA and working state are included. This run
uses uncommitted A2 source: hashes identify it. Repeat after an authorized commit to record
the implementation commit SHA. No A2 commit, push or merge has been performed.
