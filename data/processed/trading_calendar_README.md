# USDRUBF trading calendar (preliminary)

This note describes the two calendar files in this directory. For OHLCV data, field definitions of `CANDLE_CLOSE` and candle `end` timestamps, see [README.md](README.md). No raw price series, baseline data, benchmark source or existing reports were modified.

## Scope and files

Period: **2022-04-26 through 2026-11-18**, inclusive.

- [`trading_calendar.csv`](trading_calendar.csv): **1,668 calendar dates**, including weekends, Russian public holidays and days with no daily candle; 5 columns.
- [`usdrubf_trading_days.csv`](usdrubf_trading_days.csv): **1,160 expected trading-date labels**, one `date` column, in chronological order. This is a derivative of the full calendar, not another independently maintained source.

The original prices are in [`../raw/USDRUBF_daily_2026-11-18.csv`](../raw/USDRUBF_daily_2026-11-18.csv). No missing candles were forward-filled or fabricated.

## Columns of trading_calendar.csv

| Field | Meaning |
|---|---|
| `date` | Calendar date in YYYY-MM-DD format; when a candle exists, also the MOEX daily candle's date label |
| `weekday` | Day of week in English (`Monday`–`Sunday`) |
| `is_trading_day_expected` | 1 if the **reconstructed market-date schedule** expects a distinct daily USDRUBF trading-date label; 0 otherwise |
| `has_candle` | 1 if `date` appears in the unchanged MOEX ISS daily-candle dataset; 0 otherwise |
| `moex_evidence` | Type of evidence or rule used to determine the expected market-date status |

`moex_evidence` values:
- `STANDARD_RULE_BASED_PROVISIONAL` — Russian five-day production calendar plus the conventional market schedule; **not individually instrument-verified**.
- `MOEX_HOLIDAY_TRADE_NOTICE` — additional trading date inferred from a published MOEX holiday schedule.
- `MOEX_UPDATED_CLOSURE_NOTICE` — subsequent MOEX closure notice overrides an earlier holiday schedule (notably 2022).
- `MOEX_DSVD_NO_SEPARATE_TRADE_DATE` — supplementary weekend/holiday session is not a separate daily trading-date label. Exact historical instrument eligibility still needs confirmation.

## Construction method and sources

The expected dates were reconstructed **without using the observed candle dates as a rule for whether trading was scheduled**. The method starts from the Russian five-day production calendar (federal holidays and transferred working days, 2022–2026), then applies published MOEX futures-market holiday sessions and subsequent changes. The `has_candle` flag is computed **separately** from the raw USDRUBF daily candles obtained from MOEX ISS (`interval=24`).

Production calendars:
- 2022: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2022/
- 2023: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2023/
- 2024: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2024b/
- 2025: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2025/
- 2026: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2026/

MOEX references for the futures-market schedule and exceptions:
- Trading calendar: https://www.moex.com/s3346
- Holiday schedules: https://www.moex.com/n36780 ; https://www.moex.com/n51887 ; https://www.moex.com/n64121 ; https://www.moex.com/n73701 ; https://www.moex.com/n96571
- Revised closures in 2022: https://www.moex.com/n47095 ; https://www.moex.com/n48076 ; https://www.moex.com/n51250
- Official futures-calendar API (full history for this instrument not independently retrieved): https://iss.moex.com/iss/calendars/futures.json

The exchange's accounting of evening and supplementary weekend sessions has changed over time. In particular, weekend supplementary sessions may belong to the **next exchange trading-date label** rather than generating separate Saturday/Sunday rows. Do not treat physical session dates and trading-date labels as interchangeable.

## Audit findings (provisional)

| Measure | Count |
|---|---:|
| Calendar dates in range | 1,668 |
| Dates expected to have a distinct daily candle | 1,160 |
| Observed distinct daily-candle dates | 1,160 |
| Expected date without a candle | **0** |
| Candle on a date not expected by the reconstructed calendar | **0** |

The counts above establish agreement **only with the reconstructed schedule**, not full proof that every historical USDRUBF trading date has been verified. Most dates were inferred using `STANDARD_RULE_BASED_PROVISIONAL`. Do not assert instrument-level complete coverage or a confirmed real-time / point-in-time trading calendar on this basis.

To review candidates for omissions, filter `is_trading_day_expected == 1` and `has_candle == 0`. For unexpected candles, filter `is_trading_day_expected == 0` and `has_candle == 1`. Both sets are empty in this version.

## Using the calendar in benchmarks

`benchmarks/core.py` expects a **sorted unique `date` column containing trading dates only**. Use `usdrubf_trading_days.csv`, **not** all rows of `trading_calendar.csv`, for a calendar-aware run.

**Do not change `configs/benchmark_config.yaml` or label the model's outcomes as fully verified without agreement from A.** The calendar is preliminary; the team's documented `conditional EOD / pseudo-PIT` limitations and absence of validated candle-publication timestamps remain in force. Do not use this finite historical file to invent future trading dates beyond 2026-11-18.

For a complete formal audit, confirm the historical instrument-level FORTS sessions against an official archive or equivalent independent MOEX records.
