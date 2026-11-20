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

Mode: `observed_session_proxy`.
Repository inspection found only raw candles and a derived baseline;
neither supplies an independent calendar including sessions without candles.
The ISS candles endpoint in the README supplies candles, not independent sessions.
Calendar completeness is UNKNOWN, including in each split. Horizons are observed
session steps only; this is not a final seven-trading-day backtest.
No weekends-only calendar is synthesized. Unknown future target dates remain null.

Calendar-day gaps above 4 days (diagnostic, not evidence of missing sessions):
{'2022-05-04': 5, '2022-05-11': 5, '2024-01-03': 5, '2026-01-05': 6}

## Absolute observed price moves above 10%

Threshold fixed in config; flagged observations retained.


## Open semantics, provenance and protocol issues

Processed README calls price original MOEX CANDLE_CLOSE and not settlement;
research charter/benchmark spec leave this conditional on external verification.
Raw/processed equality supports field provenance but does not independently prove
market-close semantics or historical publication timing. No external confirmation
was established in this implementation. Q01/Q02/Q03 remain open; primary experiment blocked.
README mentions session-day attribution changing on 2026-03-23; historical availability
and attribution require a source audit. No exact forecast timestamp is fabricated.
Minimal clarification: retain conditional CLOSE semantics across documents until
Data Engineer supplies field documentation, availability evidence and an independent calendar.
A1 bounds, minimum 100 AR pairs and every-origin expanding refit are fixed in config
before validation scoring. Final-test metrics require explicit --include-final-test;
final-test outcomes must never choose models, preprocessing or hyperparameters.
