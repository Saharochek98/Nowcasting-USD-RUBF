# Project Status — USDRUBF Nowcasting

**Срез:** 2026-11-23, после слияния B1–B5 в `main` (проверен commit `619c8b3fe13652f618e50b97ee65095576535cbb` до этого обновления).  
**Научный статус:** `PRELIMINARY`, **conditional EOD / pseudo-PIT**. Исторические `available_at` не восстановлены. **Final test 2026 закрыт для оценки и выбора моделей.**

## Уже выполнено

| Направление | Статус | Воспроизводимые артефакты |
|---|---|---|
| USDRUBF daily `CANDLE_CLOSE` | DONE — определён источник MOEX ISS `interval=24`, не clearing/settlement | `data/raw/USDRUBF_daily_2026-11-18.csv`, `reports/data_quality_report.md` |
| Исторические данные и календарь | DONE для внутренней сверки; календарь **PROVISIONAL** | 1160 свечей (26.04.2022–18.11.2026), `data/processed/trading_calendar.csv`, `reports/calendar_quality_report.md` |
| RW и AR(1), h=1…7 | IMPLEMENTED; validation 2025 оценена | `benchmarks/`, `results/benchmark_calendar_v2/`, `reports/benchmark_summary.md` |
| Диагностика AR(1) | DONE exploratory; AR(1) хуже RW по RMSE на каждом h, без значимости после HAC+Holm | `reports/ar1_diagnostics.md` |
| **B1: forecast origin и аудит свечей** | **DONE как аудит, NOT verified historical PIT** | `docs/forecast_origin_pit_proposal.md`, `docs/moex_candle_monitoring.md`, `results/moex_minute_audit/comparison.csv` |
| **B2: PIT-каталог и data dictionary** | **DONE; группы/сценарии согласованы A**, статусы самих публикаций остаются assumed/unknown | `docs/feature_availability_catalog.csv`, `docs/feature_data_dictionary.md`, `docs/feature_lag_policy_proposal.md`, `reports/feature_pit_audit.md` |
| **B3: модельная панель** | **DONE, merged; структурная сверка и локальный тестовый прогон успешны** | `data/processed/features_daily_v1.csv`, `data/processed/features_daily_v1_metadata.csv`, `configs/feature_scenarios_v1.json`, `scripts/build_model_features.py`, `docs/features_daily_v1.md` |
| PCA/Ridge/DFM, интервалы, ML-коррекция | **Не подтверждены представленными артефактами main на данном срезе** | Ответственные C/D/A сообщают реализацию и OOS-результаты отдельно |

### B3 — контракт передачи моделистам

- `features_daily_v1.csv`: **1160 строк**, `date` + **12** экономических значений в общем store. `core` использует **9**, `core_ruonia_brent` — **11**, `core_ruonia_brent_usdcnh` — **12**. Только `core` — основной согласованный условный сценарий; остальные — проверки чувствительности с дополнительными PIT-предположениями.
- **154** пустых `swaprate_prev_session` (13.275862%), **1006** полных строк каждого сценария. Пропуски сохранены, нули не подставлены, строки не удалены. `swaprate_current` и постоянное `value` не передаются в модель.
- `close_t` — известный на условном origin уровень; `close_{t+h}` создаётся только бэктестом по следующему **торговому** календарю.
- Загружать данные через `load_scenario(scenario='core')` из `scripts/build_model_features.py`, **не** брать автоматически все столбцы общего store. Импутация, scaler/PCA/feature selection — **train-only в каждом walk-forward окне**.
- Локально сообщён результат: `python scripts/build_model_features.py --compare` — **3 MATCH**; `unittest discover -s tests -p 'test_*.py'` — **49 tests, OK, skipped=1**. Пропущен **полный rebuild v2 из исходных внешних выгрузок**, т.к. отсутствует `USDRUBF_SOURCE_DIR`; это не тест построения B3. Прямого независимого запуска полного тестового набора автором этого статуса не было.
- Замороженный источник v2: SHA-256 `da5b582e622ce5d3efb6fb619c0b9258dc15fc7d18b6d76860ce5fb021107103`. Контрольные SHA файлов B3 — в `reports/features_daily_v1_coverage.md`.

## Актуальные задачи после B5 (следующий рабочий день)

**B1–B5 завершены на уровне инженерного conditional EOD / pseudo-PIT MVP.** [Инструкция A/C/D](data_handoff_to_modeling.md), [аудит B4](../reports/funding_missingness_audit.md). Повторять аудит календаря или funding не требуется.

| Ответственный | Приоритет | Следующее действие | Подтверждение выполнения |
|---|---|---|---|
| **A — research/backtesting** | P0 | Чисто воспроизвести RW/AR из `main`, закрепить Git SHA; утвердить общий `(origin_date, target_date, h)`, train-only правила и неизменный validation scoring | Новый manifest + common-pair validation + leakage review |
| **C — factor models** | P0 | Начать с price-only и Ridge на `core` (9 признаков); S0/S1/S2 — только заранее определённая абляция с общими датами | OOS validation прогнозы h=1…7, RMSE/MAE/bias vs RW и правила missingness |
| **D — uncertainty** | P0 | RW-интервалы 90/95%; калибровать исключительно по созревшим ошибкам | Validation coverage, width и interval score по h |
| **A + C** | P1 | При наличии Ridge benchmark сравнить PCA+Ridge; DFM/Kalman — решение по ценности и срокам, не автоматический следующий шаг | Зафиксированное go/no-go решение, без открытия test |
| **B — данные** | По запросу | Передать A/C/D ссылки на B3/B4 и объяснить PIT, календарь, funding; поддержать **только обнаруженные интеграционные проблемы** | Передача подтверждена, новые исходные версии создаются только при необходимости и одобрении A |

**Командная синхронизация:** A подтверждает, что C использует ровно разрешённый сценарий, D — ту же ось торговых дат и созревшие таргеты; B отвечает на вопросы по coverage. Любая новая эмпирическая оценка — validation-only; H1–H6 не подтверждены одним наличием инфраструктуры.

## Незакрытые ограничения

- **P0:** точный исторический момент доступности финального daily `CLOSE` неизвестен; `end` и гипотеза 00:05 МСК не являются timestamp публикации. 15-минутная задержка бесплатных данных не доказывает финальность свечи.
- **P0:** `verified` для record-level PIT **нет ни у одного** из 30 столбцов; для `swaprate_prev_session`, действующих значений ЦБ и остальных показателей остаются документированные допущения.
- **P1:** календарь внутренне согласован, но не подтверждён независимым архивом; после 23.03.2026 изменилась структура торговой даты срочного рынка.
- **P1:** полная пересборка v2 из frozen external exports не воспроизведена в новой среде; исходные экспорты не включены в Git.
- **P1:** исходные manifests прошлых RW/AR запусков фиксируют `uncommitted_changes`; требуется clean rerun.
- **P1:** RUONIA/Brent/USD-CNH доступны в sensitivity-сценариях только по допущениям. Особенности `end≈19:30` исследованы на 35 днях: 9/35 полных совпадений OHLCV для минутного окна до 19:30, механизм не установлен.

## Известные validation-результаты (без изменений)

Train 683, validation 254, final test 223 дневные свечи. На validation 2025 RW имеет меньший RMSE, чем AR(1), на каждом h=1..7; для h=7: **RW 2.147724; AR(1) 2.207770** RUB/USD. HAC+Holm не подтверждает семейно-значимое различие. См. `reports/benchmark_summary.md` и `reports/ar1_diagnostics.md`.

**Не открывать final test для настройки** до регистрации окончательных моделей, pipeline, правил отбора и интервалов. История тестов и метрик не переписывается.

## Обновление направления B — 2026-11-23

- **B4 объединён в `main`:** [аудит funding](../reports/funding_missingness_audit.md), 153/154 начальных пропуска current/previous и по 198 наблюдаемых нулей, граница покрытия 05/06.12.2022. Исторические publication timestamps не проверены; 61 unittest (60 passed, 1 skipped) согласно локальному отчёту.
- **B5 — передача данных моделистам:** [краткая инструкция A/C/D](data_handoff_to_modeling.md), без повторного аудита и изменения данных.
- **Инженерное направление B закрыто для conditional EOD / pseudo-PIT MVP;** остаются отдельно открытыми независимая сертификация календаря и строгий historical PIT. Следующие модельные эксперименты принадлежат A/C/D; final test не используется для выбора.
