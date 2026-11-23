# Покрытие features_daily_v1

Сгенерировано scripts/build_model_features.py без обучения и оценки моделей.

Режим: **conditional EOD / pseudo-PIT**. Сценарии согласованы A в задании B3.
Один store содержит объединение 12 значений; default core использует ровно 9.

Период: 2022-04-26 — 2026-11-18; строк: 1160.
Даты и пропуски сохранены; число признаков ниже включает известный close_t, исключает date.

## Scenario A - core

Строк: 1160; признаков: 9; полных строк: 1006.
Минимальная дата: 2022-04-26; максимальная: 2026-11-18.
Постоянные экономические столбцы (без пропусков): нет.
Сценарий чувствительности: нет.

| Признак | Пропуски | Доля |
|---|---:|---:|
| `open` | 0 | 0.000000% |
| `close` | 0 | 0.000000% |
| `high` | 0 | 0.000000% |
| `low` | 0 | 0.000000% |
| `volume` | 0 | 0.000000% |
| `cbr_usdrub_effective` | 0 | 0.000000% |
| `swaprate_prev_session` | 154 | 13.275862% |
| `key_rate_pct_effective` | 0 | 0.000000% |
| `spread_close_minus_cbr` | 0 | 0.000000% |

Ограничения: funding не заполнен; строки не удалены. Импутация/scaler/PCA только внутри train/walk-forward.

## Scenario B1 - core + RUONIA + Brent

Строк: 1160; признаков: 11; полных строк: 1006.
Минимальная дата: 2022-04-26; максимальная: 2026-11-18.
Постоянные экономические столбцы (без пропусков): нет.
Сценарий чувствительности: да.

| Признак | Пропуски | Доля |
|---|---:|---:|
| `open` | 0 | 0.000000% |
| `close` | 0 | 0.000000% |
| `high` | 0 | 0.000000% |
| `low` | 0 | 0.000000% |
| `volume` | 0 | 0.000000% |
| `cbr_usdrub_effective` | 0 | 0.000000% |
| `swaprate_prev_session` | 154 | 13.275862% |
| `key_rate_pct_effective` | 0 | 0.000000% |
| `spread_close_minus_cbr` | 0 | 0.000000% |
| `ruonia_pct_last_reported` | 0 | 0.000000% |
| `brent_eia_spot_usd_bbl_lag10d` | 0 | 0.000000% |

Ограничения: funding не заполнен; строки не удалены. Импутация/scaler/PCA только внутри train/walk-forward.

- RUONIA: DT < date and date(DateUpdate) <= date; DateUpdate is not proven first release; CBR working calendar and actual receipt unverified
- Brent: observation date <= date - 10 calendar days; historical first-release/vintage sufficiency of lag10 not proven

## Scenario B2 - core + RUONIA + Brent + USD/CNH

Строк: 1160; признаков: 12; полных строк: 1006.
Минимальная дата: 2022-04-26; максимальная: 2026-11-18.
Постоянные экономические столбцы (без пропусков): нет.
Сценарий чувствительности: да.

| Признак | Пропуски | Доля |
|---|---:|---:|
| `open` | 0 | 0.000000% |
| `close` | 0 | 0.000000% |
| `high` | 0 | 0.000000% |
| `low` | 0 | 0.000000% |
| `volume` | 0 | 0.000000% |
| `cbr_usdrub_effective` | 0 | 0.000000% |
| `swaprate_prev_session` | 154 | 13.275862% |
| `key_rate_pct_effective` | 0 | 0.000000% |
| `spread_close_minus_cbr` | 0 | 0.000000% |
| `ruonia_pct_last_reported` | 0 | 0.000000% |
| `brent_eia_spot_usd_bbl_lag10d` | 0 | 0.000000% |
| `usdcnh_investing_price_prev_date` | 0 | 0.000000% |

Ограничения: funding не заполнен; строки не удалены. Импутация/scaler/PCA только внутри train/walk-forward.

- RUONIA: DT < date and date(DateUpdate) <= date; first release and actual receipt unverified
- Brent: existing lag of 10 calendar days assumes timely release and acceptable vintage; not verified PIT
- USD/CNH: provider Date < date; historical Price type, daily cutoff/timezone, revisions and anomalies unresolved

## Контрольные SHA-256

| Артефакт | SHA-256 |
|---|---|
| source v2 | `da5b582e622ce5d3efb6fb619c0b9258dc15fc7d18b6d76860ce5fb021107103` |
| features_daily_v1.csv | `95ff7cc95f3cbd08db994bbc6e098b12d06019cfc5e8b593fd647cd9606df792` |
| features_daily_v1_metadata.csv | `3e398df07ee63852b43752082364a03082b81997bab217c6740b5c2eae24c92f` |
| scenario config (LF normalized) | `5511440a790aa2ac5dc7f8d4d7d5bf2aa477e3397c2af19c8747ba7effa92bc9` |

Исторические available_at не восстановлены. Проверки дат не доказывают строгий PIT.
Календарь остаётся предварительным; правила публикаций и ревизий наследуются из B1/B2.
Метрики прогнозов и выбор сценария по test не выполнялись.
