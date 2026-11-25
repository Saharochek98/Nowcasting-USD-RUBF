# Project Status — USDRUBF Nowcasting

**Актуально:** 2026-11-24, после merge Ridge v1/v2, RW with Drift, CatBoost и XGBoost в `main`. **Исследовательский статус: EXPLORATORY VALIDATION**, `conditional EOD / pseudo-PIT`; календарь `PROVISIONAL`. **Final test 2026 не открывался для выбора или оценки моделей.**

## Что выполнено

| Направление | Статус | Артефакт |
|---|---|---|
| Данные, календарь, PIT, funding, B1–B5 | DONE инженерно, historical PIT не verified | [Data handoff](data_handoff_to_modeling.md), [funding audit](../reports/funding_missingness_audit.md), [calendar audit](../reports/calendar_quality_report.md) |
| RW и AR(1), h=1…7 | IMPLEMENTED, VALIDATION_EVALUATED | [Benchmark summary](../reports/benchmark_summary.md) |
| Ridge price-only/Core v1, B1/B2, текущий funding R5; delta/log | IMPLEMENTED, VALIDATION_EVALUATED | [Ridge v1](../reports/ridge_experiments.md) |
| Ridge Core v2 (10 полей), prequential/frozen, no-spread и расширенные сценарии | IMPLEMENTED, VALIDATION_EVALUATED | [Ridge v2](../reports/ridge_experiments_v2.md) |
| RW with Drift | IMPLEMENTED, VALIDATION_EVALUATED | [Drift experiment](../reports/rw_drift_experiment.md) |
| CatBoost Core v2 | IMPLEMENTED, VALIDATION_EVALUATED | [CatBoost](../reports/catboost_core_v2.md) |
| XGBoost Core v2 | IMPLEMENTED, VALIDATION_EVALUATED | [XGBoost](../reports/xgboost_core_v2.md) |
| RW интервалы 90%/95%, PCA/DFM/Kalman, ML-коррекция DFM | **Нет подтверждения выполненных OOS-артефактов в данном срезе** | Не заявлять DONE без отчётов и прогнозов |

**Данные:** 1160 дневных свечей 2022-04-26…2026-11-18. Train до 2024-12-31, validation targets 2025 (254 общие пары на каждый h), финальный test 2026 (223 сессии) запечатан. Target — дневной `CANDLE_CLOSE` **USDRUBF**, не официальный курс ЦБ и не settlement.

**Разграничение сценариев:** frozen B3 `features_daily_v1.csv` и `feature_scenarios_v1.json` неизменны: `core` = 9 признаков, sensitivity = 11/12, `swaprate_current` исключён. Для Ridge v2/CatBoost/XGBoost введён **отдельный модельный Core v2 = 10 признаков**, добавляющий `swaprate_current[t]` к прежним девяти. Его доступность на origin **ASSUMED_AVAILABLE по решению команды**, не historical VERIFIED. Использовать версии и загрузчики из соответствующих отчетов, **не** `load_scenario('core')` как источник 10 признаков. В `swaprate_prev_session` 154 начальных missing, в `swaprate_current` 153; наблюдаемые нули не импутировать как missing.

## Итог validation 2025

RMSE (RUB/USD) на одинаковых прогнозных датах:

| Модель | h=1 | h=7 | RMSE ниже RW, горизонтов из 7 |
|---|---:|---:|---:|
| **RW** | **0.772440** | 2.147724 | эталон |
| AR(1) | 0.774045 | 2.207770 | 0 |
| RW with Drift | 0.776165 | 2.211516 | 0 |
| Ridge Core v2, prequential delta | 0.775978 | 2.212588 | 0 |
| CatBoost Core v2 | 0.773605 | **2.145415** | 3 (h=4,6,7) |
| XGBoost Core v2 | 0.782843 | 2.204440 | 0 |

**Вывод:** CatBoost превосходит Ridge Core v2 по RMSE на всех семи h, XGBoost не выигрывает у CatBoost или RW ни на одном. **Ни у одного из проведённых факторных экспериментов нет статистически подтверждённого преимущества над RW после заявленных HAC/Holm-проверок.** Не интерпретировать отсутствие значимости как эквивалентность. 2025 год многократно использовался для исследовательских решений, поэтому это **exploratory**, а не независимая оценка.

**Target-преобразования:** модели уже прогнозируют `C[t+h]−C[t]` и восстанавливают цену `C[t]+prediction`. Ridge отдельно проверял накопленную log-return; преимуществ не выявлено. **Прямой прогноз уровня `C[t+h]` против приращений при одинаковом протоколе ещё НЕ проверен.**

## Следующие решения и задания команде

| Участник | Приоритет | Задача |
|---|---|---|
| **A — Data Engineer** | P1 | Поддержка PIT и данных; при эксперименте с Open Interest отдельно проверить источник/историческую доступность. Не изменять frozen B3 без новой версии |
| **B — Research Lead & Backtesting** | P0 | Закрыть журнал экспериментов и гипотез; согласовать с командой **один** заранее описанный следующий experiment (direct level-vs-delta или relative features); контролировать общие forecast keys, maturity, отсутствие test leakage |
| **C — Models** | P1 | После согласования B: относительные признаки вместо/в дополнение к сырым OHLC, фиксированные горизонты/окна; отдельно, при достаточных ресурсах, DFM/Kalman |
| **D — Uncertainty** | P0 | RW-прогнозные интервалы 90%/95%, coverage, width, interval score по h и train-only/prequential calibration; предъявить артефакты |

**Новые гипотезы, не результаты:** (1) matched *direct level vs delta*; (2) относительные признаки: returns 1/2/3/7/14 торговых сессий, `(Close−SMA20)/SMA20`, RSI(14), ATR(14)/Close, funding change, Open Interest % change **только при проверенном источнике/PIT**. Сравнить frozen Core v2, Core v2 + относительные и замену сырых цен на относительные на одинаковых парах. Не рассматривать 2025 как свежий независимый validation после этих итераций.

**Открытые ограничения:** historical `available_at` свечей и funding не проверены; календарь provisional; внешние vintages не гарантированы; расхождение target USDRUBF и официального курса ЦБ в исходной постановке требует согласования. Final test 2026 — только после preregistration выбранного протокола и моделей.

История и детали: [Decision Log](decision_log.md), [Hypothesis Register](hypothesis_register.md), [Benchmark Specification](benchmark_spec.md), [текущие задания](two_day_plan.md).

## Последнее решение (25.11.2026): зафиксирован финальный двухдневный план

**Все исходные RW/AR/Ridge/CatBoost/XGBoost результаты, код и raw/frozen datasets без изменений.** После изучения результатов выбран один feature experiment **25 ноября H11-REL-01** и один target experiment **26 ноября H13-TRANSFORM-01**; ни один из них **пока не выполнен**. Исполнитель C, ревью и единый scoring — Research Lead B; A — поддержка данных при необходимости, D — отдельная работа с интервалами.

H11: matched CatBoost Core v2 против Core v2 + шесть relative features с одинаковыми train windows и старой по-h конфигурацией. H13: на **чистом Core v2** сравнить delta и log-return, не добавляя H11 признаки. New parameter tuning/XGBoost/DFM/H12 data expansion исключены из двухдневного плана. [Задачи/DoD](two_day_plan.md), [H11 config](../configs/h11_catboost_relative_v1.json), [H13 config](../configs/h13_catboost_target_v1.json).
