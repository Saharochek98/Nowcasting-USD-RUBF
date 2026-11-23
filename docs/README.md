# USDRUBF — Research Documentation

**Актуально:** 2026-11-23, после B1/B2/B3 merge.  
**Режим исследования:** `conditional EOD / pseudo-PIT`; финальный test 2026 закрыт для выбора моделей.

## С чего начать

1. [Project Status](project_status.md) — подтверждённые артефакты, ограничения, следующие задачи A/B/C/D.
2. [План ближайших двух дней](two_day_plan.md) — **актуальный** roadmap: B4 funding missingness, B5 календарь/handoff, параллельно A/C/D.
3. [Backtest Protocol](backtest_protocol.md) и [Research Charter](research_charter.md) — научные правила, target и оценки.
4. [Features v1 — инструкция C](features_daily_v1.md) — загрузка, сценарии и ограниченная PIT-достоверность.

## Источники, данные и B1–B3

- **Target:** рыночный `CANDLE_CLOSE` дневной свечи USDRUBF, MOEX ISS `interval=24`, **не** официальный USD/RUB ЦБ и **не** расчётная цена клиринга.
- **Горизонт:** h=1…7 следующих **торговых** дат USDRUBF. `forecast_origin` после доступности полной свечи, но исторический час неизвестен; время `end` не заменяет `available_at`.
- **Календарь:** 1160 сессий/свечей 26.04.2022–18.11.2026. Проверен на внутреннюю согласованность, **PROVISIONAL** до независимого архива.
- **B1:** [forecast origin / candle audit](forecast_origin_pit_proposal.md), [ручной монитор MOEX](moex_candle_monitoring.md); 35 свечей `end≈19:30` исследованы, но техническая причина меток и время финальной публикации не доказаны.
- **B2:** [полный словарь 30 столбцов](feature_data_dictionary.md), [каталог доступности](feature_availability_catalog.csv), [политика лагов](feature_lag_policy_proposal.md), [отчёт](../reports/feature_pit_audit.md). Research Lead A согласовал группы A/B/C; это **не** означает verified record-level PIT.
- **B3:** `data/processed/features_daily_v1.csv` — `date` + 12 экономических полей; `features_daily_v1_metadata.csv` — provenance. [Сценарии](../configs/feature_scenarios_v1.json): **core 9** (default), **core+RUONIA+Brent 11**, **core+RUONIA+Brent+USD/CNH 12** (последние два sensitivity only).
- **Пропуски:** `swaprate_prev_session` — 154/1160 (13.275862%); без подмены нулями или удаления строк. `swaprate_current` исключён.
- **Воспроизводимость:** `python scripts/build_model_features.py --compare`, [покрытие и контрольные суммы](../reports/features_daily_v1_coverage.md). Пользовательский прогон: **3 MATCH; 49 tests OK, 1 skipped** (полный rebuild v2 требует внешних frozen exports).

## Передача C

Использовать `load_scenario(scenario='core')` из `scripts/build_model_features.py`, не выбирать автоматически все 12 колонок из общего CSV. Предобработка (импутация/scaler/PCA/selection) — только внутри train/walk-forward на созревших метках. Test 2026 не использовать для оптимизации.

## Baselines и научные выводы

RW и AR(1) реализованы, validation 2025 оценена; AR(1) уступает RW по RMSE на всех горизонтах, но HAC+Holm не подтверждает семейную значимость различий. См. [benchmark summary](../reports/benchmark_summary.md), [AR diagnostics](../reports/ar1_diagnostics.md). Новых validation-результатов C/D после B3 этим документом не заявлено.

## Навигация по прочим правилам

- [Decision Log](decision_log.md) — история решений, включая дополнение о B1–B3.
- [Hypothesis Register](hypothesis_register.md) — H1–H6 до отдельного OOS-эксперимента **NOT_TESTED**.
- [Benchmark Specification](benchmark_spec.md) — RW/AR и вспомогательные модели.
- [Data quality](../reports/data_quality_report.md), [Calendar report](../reports/calendar_quality_report.md) — исходные проверки.

**Важно:** исходная официальная постановка `BG_2026_stud.pdf` описывает прогноз курса ЦБ, тогда как фактическая цель команды — цена USDRUBF. Расхождение фиксировать в исследовательском отчёте и согласовать с наставниками. Giannone et al. (2008) — методология dynamic factors/nowcasting, не эмпирическое доказательство качества модели USDRUBF.

Исторические материалы и старые manifests не перезаписывать. Каждый новый результат маркировать `IMPLEMENTED` / `VALIDATION_EVALUATED` / `TEST_CONFIRMED` только при наличии файла, периода и воспроизводимой процедуры.
