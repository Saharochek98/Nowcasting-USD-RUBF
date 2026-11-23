# Project Status — USDRUBF Nowcasting

**Срез:** 2026-11-23, после слияния B1–B3 в `main` (проверенный tree SHA `791964a5bbe12cdb686641fa7e6434d3d8413ca7`).  
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

## Следующие задачи / приоритеты

| ID | Владелец | Приоритет | Действие | Definition of Done |
|---|---|---|---|---|
| **B4** | B | P0 | Диагностика 154 пропусков funding **по месяцам/годам и непрерывным сегментам**, пересечение с режимами торгов; проверить источник, пропуск vs действительный ноль | Отчёт `reports/funding_missingness_audit.md`, таблица периодов, воспроизводимый read-only скрипт, список рисков для C; **не** импутировать и **не** выбирать модель |
| **B5** | B + A | P0 | Независимая проверка специальных торговых дат и реформы 23.03.2026, расхождений candle/contractresults; окончательный Data Handoff | `reports/calendar_final_audit.md` либо явно PROVISIONAL, список открытых дат, immutable hashes, handoff C с режимом pseudo-PIT |
| **A1** | A | P0 | Контрольный rerun RW/AR и diagnostics **из чистого main** с новым manifest и Git SHA; убедиться, что final test не используется для selection | Новый каталог результатов без перезаписи истории, common-pair validation keys, тесты |
| **A2** | A | P0 | Заморозить модельный API и сценарии, обеспечить сравнение на **общих** origin/target | Фиксированные train/validation keys, JSON конфиги, реестр экспериментов и leakage review |
| **C1** | C | P0 | Ridge на `core` (и price-only ablation), train-only preprocessing, метки только созревших h-step targets | OOS validation прогнозы h=1..7 + тесты и сравнение с RW; ничего не заключать по test |
| **C2** | C | P1 | PCA+Ridge как следующий этап; DFM/Kalman — только после сравнения с простыми моделями | train-only PCA, воспроизводимые OOS прогнозы и решение go/no-go для DFM |
| **D1** | D | P0 | RW 90%/95% интервалы и separate coverage, width, interval score | Validation по h, интервал калибруется только на доступных past targets/OOS errors |
| **D2** | D + A | P1 | Online-калибровка и/или variance shrinkage, затем возможная residual ML | Зарегистрированный эксперимент, без случайного доступа к validation/test в fit |

### Ближайшая синхронизация
1. **A → C/D:** зафиксировать `origin_date`, `target_date`, `h`, общий score set и правила созревания меток. Согласование feature groups A уже сообщено командой; дополнительные изменения делать отдельной версией.
2. **B → C:** передать B3 API, metadata, покрытие и список предположений, не выдавая `pseudo-PIT` за `verified`.
3. **A/C/D:** отдельно докладывать факт кода, факт validation и факт статистической значимости; не переносить статусы без артефактов.

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
