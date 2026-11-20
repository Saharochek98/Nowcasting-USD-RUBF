# USDRUBF reconstructed calendar audit

Consistency status: **passed**.
Calendar status: **provisional**, not a fully verified independent session archive.

Inputs: `data/processed/trading_calendar.csv`, `usdrubf_trading_days.csv`,
and raw MOEX ISS candle `begin` dates. Expected dates come from the supplied
production-calendar/MOEX-notice reconstruction; has_candle is checked separately.

| Count | Actual |
|---|---:|
| calendar_dates | 1668 |
| expected_trading_dates | 1160 |
| listed_trading_dates | 1160 |
| raw_candles | 1160 |
| missing_expected_candles | 0 |
| unexpected_candles | 0 |

README numerical claims reproduced: True.

| Consistency check | Passed |
|---|---|
| dates_unique_sorted_valid | True |
| full_calendar_contiguous | True |
| weekday_labels_match | True |
| evidence_present | True |
| sessions_within_full_calendar | True |
| sessions_equal_expected_dates | True |
| has_candle_matches_raw | True |
| raw_dates_within_full_calendar | True |
| known_saturdays_preserved | True |

Known working Saturdays: {'2024-04-27': True, '2024-11-02': True, '2024-12-28': True, '2025-11-01': True}.
Candles on known working Saturdays: {'2024-04-27': True, '2024-11-02': True, '2024-12-28': True, '2025-11-01': True}.
Missing expected dates: [].
Unexpected candle dates: [].
Evidence label counts: {'STANDARD_RULE_BASED_PROVISIONAL': 1625, 'MOEX_HOLIDAY_TRADE_NOTICE': 28, 'MOEX_UPDATED_CLOSURE_NOTICE': 11, 'MOEX_DSVD_NO_SEPARATE_TRADE_DATE': 4}.
Errors: [].

Equality to the reconstruction does not prove an independent historical MOEX audit.
No prices, calendars or flags were repaired. The supplied calendar ends on 2026-11-18;
no later target dates are invented. Historical candle availability remains pseudo-PIT.
