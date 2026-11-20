# Data quality report

Status: PRELIMINARY; conditional EOD / pseudo-PIT.

- Observations: 1160 (expected claim: 1160).
- Range: 2022-04-26 — 2026-11-18.
- Dates: sorted, unique, nonmissing (loader rejects violations).
- Missing values by column: {'price': 0, 'volume': 0, 'price_type': 0, 'source': 0}.
- Nonnumeric/nonpositive/infinite processed prices: 0 (loader rejects violations).
- Raw invalid/missing OHLC rows: 0.
- Raw OHLC integrity failures: 0.
- Raw/processed date sets match: True.
- CLOSE/price mismatch: False; volume mismatch: False.
- No filling, replacement, adjustment or outlier removal performed.
  Equality to raw checks derivation; upstream substitutions cannot be ruled out.

## Chronological split (inclusive; forecast pairs use target dates)

| Split | n | First observed | Last observed |
|---|---:|---|---|
| train | 683 | 2022-04-26 00:00:00 | 2024-12-30 00:00:00 |
| validation | 254 | 2025-01-03 00:00:00 | 2025-12-30 00:00:00 |
| final_test | 223 | 2026-01-05 00:00:00 | 2026-11-18 00:00:00 |

## Session calendar

Mode: `independent`.
Source: reconstructed_moex_calendar_provisional
Sessions without valid price: 0: []
The supplied calendar is a PROVISIONAL reconstruction from production-calendar
rules and MOEX notices; mode independent is a loader mode, not a verification claim.
See calendar_quality_report.md for full-calendar/expected-date/raw-date checks.
Completeness is assessed only relative to this reconstruction, not an independently
verified archive of every USDRUBF session. Known working Saturdays are retained.
- train: 683 expected sessions; 0 without valid price.
- validation: 254 expected sessions; 0 without valid price.
- final_test: 223 expected sessions; 0 without valid price.

Calendar-day gaps above 4 days (diagnostic, not evidence of missing sessions):
{'2022-05-04': 5, '2022-05-11': 5, '2024-01-03': 5, '2026-01-05': 6}

## Absolute observed price moves above 10%

Threshold fixed in config; flagged observations retained.


## Open semantics, provenance and protocol issues

Per the A2 task agreement, target is the original MOEX ISS daily CANDLE_CLOSE,
not CBR or clearing settlement. Raw/processed equality verifies the retained field.
Existing protocol documents still condition semantics on external evidence; they are
preserved. No new independent external field/session audit was performed in A2.
Historical candle publication timestamps, end-label interpretation, and session-day
attribution remain unverified. Data README documents atypical end labels and weekend
session attribution; end is not assumed to be a precise last-trade or availability time.
Q02 remains open; Q03 has a provisional reconstruction, not final independent closure.
No exact forecast timestamp is fabricated; conditional EOD / pseudo-PIT remains.
A1 bounds, minimum 100 AR pairs and every-origin expanding refit are fixed in config
before validation scoring. Final-test metrics require explicit --include-final-test;
final-test outcomes must never choose models, preprocessing or hyperparameters.
