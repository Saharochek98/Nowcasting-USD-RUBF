# Decision Log / Open Questions — USDRUBF

**Версия:** 1.4 · 2026-11-24 · **Владелец:** B, Research Lead

## Принятые решения

| ID | Решение | Основание | Статус |
|---|---|---|---|
| D01 | Target — **рыночная цена USDRUBF**, не курс ЦБ | прямое уточнение задания | ACCEPTED |
| D02 | Random Walk — обязательный benchmark | исследовательский протокол | ACCEPTED |
| D03 | PIT, хронологический walk-forward, финальный test закрыт | научная строгость | ACCEPTED |
| D04 | Точечные и интервальные метрики отдельно; абляции и отрицательные результаты сохраняем | цели исследования | ACCEPTED |
| D05 | Основной горизонт `h=1…7` **торговых дней USDRUBF** | решение команды | ACCEPTED (письменную совместимость с PDF уточнить) |
| D06 | Имеются **только дневные OHLC**; внутридневных сделок и котировок нет | факт от команды | ACCEPTED |
| D07 | Основной target MVP — `CLOSE` дневной свечи USDRUBF **при подтверждении, что это рыночная цена** | решение команды | PROVISIONAL — source semantics OPEN |
| D08 | Прогноз — **после фактической доступности полной дневной свечи**; 23:51 МСК более не считается подтвержденным origin | ограничение данных | ACCEPTED; timestamp OPEN |
| D09 | Режим без timestamp публикации обозначать `conditional EOD / pseudo-PIT`; не выдавать за подтвержденный PIT | научная строгость | ACCEPTED |
| D10 | Официальная спецификация: приказ MOEX № МБ-П-2026-3630 от 08.09.2026 | спецификация USD/RUBF | ACCEPTED AS SOURCE |
| D11 | Временные пороги 5/15/30 минут, last-trade до 23:50 и 10-минутный VWAP **выведены из основного протокола** до появления intraday данных | несоответствие имеющимся данным | ACCEPTED |
| D12 | RW + AR(1) на дневных изменениях — обязательные benchmarks дня 2; drift и spread/funding — дополнительные | план второго дня | ACCEPTED |

## Подтвержденные регламентом особенности

- USDRUBF: RUB за 1 USD, лот 1000 USD, шаг 0,01 RUB, стоимость шага 10 RUB (приложение № 1).
- Расчетная цена для вариационной маржи — последний опубликованный курс ЦБ (п. 2.1.7); не target.
- SwapRate для USDRUBF использует сделки 10:00–15:30 и курс ЦБ, установленный в этот торговый день (п. 2.1.3).
- Ежедневная автопролонгация происходит в вечерней клиринговой сессии (п. 1.4).
- Текущее расписание срочного рынка MOEX: вечерняя сессия 19:00–23:50; mark-to-market 23:50–00:30. Проверять исторические изменения.

## Нерешенные вопросы / задачи аудита

| ID | Вопрос | Ответственный | Когда | Критерий закрытия |
|---|---|---|---|---|
| Q01 | Какой источник и точное определение исходного `CLOSE`? Дневная рыночная цена или settlement/clearing? | B + A | День 2 | документация API/выгрузки и пример исходных строк |
| Q02 | Когда полная дневная свеча фактически доступна? | B | День 2 | timestamp или документированный лаг; иначе conditional EOD |
| Q03 | Есть ли независимый календарь торгов USDRUBF (включая специальные выходные)? | B | День 2 | календарь, даты без свечей сохраняются |
| Q04 | Каковы пропуски, дубликаты, OHLC-ошибки и длина непрерывных сегментов? | B + A | День 2 | data_quality_report |
| Q05 | Даты train/validation/final test, минимальный train для AR(1), частота переобучения | A | День 2 | задокументированы до validation/model selection |
| Q06 | Когда доступны funding и курс ЦБ; как менялись правила? | B | День 2–3 | PIT-каталог и история изменений |
| Q07 | Уровни покрытия интервалов и процедура калибровки | A + D | День 2–3 | до настройки по validation |
| Q08 | Совместимость trading-day target с постановкой PDF о курсе ЦБ | A + наставники | День 2 | разъяснение для итогового отчета |

**История решения:** первоначально рассматривались последняя сделка до 23:50 и прогноз в 23:51 МСК. Они **не подтверждаются имеющимися дневными OHLC**, поэтому теперь являются архивной гипотезой, а не действующим определением таргета.

## Как добавлять запись

`YYYY-MM-DD | ID | Решение | Альтернативы | Источник | Кто согласовал | Влияние | Статус`

Не помечать Q01–Q04 как закрытые до первичного аудита фактических дневных данных.

## Status update — 2026-11-20 (после PR #1–#6)

| ID | Решение / состояние | Основание | Статус |
|---|---|---|---|
| D13 | Каноническая рабочая версия baseline — `results/benchmark_calendar_v2/`; предыдущую A1 оставить неизменной для replay | `reports/benchmark_summary.md`, `comparison.json` | ACCEPTED |
| D14 | B0 RW и B1 AR(1) реализованы и оценены на validation 2025, не final test | код + сохранённые метрики | DONE_VALIDATION |
| D15 | Календарь `reconstructed_moex_calendar_provisional` согласован со всеми 1160 raw-свечами, но не сертифицирован независимым архивом | `reports/calendar_quality_report.md` | PROVISIONAL |
| D16 | Эксперименты остаются `conditional EOD / pseudo-PIT`; точное историческое `available_at` дневного CLOSE не подтверждено | Data README / manifest | OPEN_LIMITATION |
| D17 | AR(1) уступает RW по validation RMSE для h=1…7; HAC+Holm не подтверждает семейно-значимого различия на 5% | `reports/ar1_diagnostics.md` | OBSERVED_VALIDATION |
| D18 | Перспективное направление — оценка неопределённости; волатильность демонстрирует зависимость, но интервальные модели пока не реализованы | ACF/Ljung–Box diagnostics | RESEARCH_HYPOTHESIS |
| D19 | Повторить `benchmark_calendar_v2` и diagnostics на чистом `main` с manifest Git SHA (прошлые запуски на uncommitted tree) | `run_manifest.json`, `diagnostics_manifest.json` | TODO |
| D20 | Финальный test 2026 не открывать до регистрации модели, факторной панели и calibration policy | предотвращение selection bias | ACCEPTED |

### Состояние прежних Q01–Q08

- **Q01 (CLOSE):** источник и исходное поле **идентифицированы**: MOEX ISS `candles interval=24`, `close` → `CANDLE_CLOSE`; raw/processed совпадают. **Момент появления свечи и точная репрезентативность как «закрытия» остаются ограничениями**, это не settlement-цена из отдельного поля.
- **Q02 (timestamp):** OPEN; `conditional EOD / pseudo-PIT`.
- **Q03 (календарь):** PROVISIONAL DONE — восстановленный список сессий, четыре рабочие субботы, 0 несовпадений относительно реконструкции; архивное подтверждение OPEN.
- **Q04 (качество OHLC):** DONE для внутреннего аудита, 1160 свечей, 0 OHLC-нарушений и дубликатов; отчёт в `reports/data_quality_report.md`.
- **Q05 (split, AR):** DONE: train/validation/test по target-дате, AR expanding, 100 pairs, `configs/benchmark_config.yaml`.
- **Q06 (CBR/funding):** OPEN — готовой PIT-панели нет.
- **Q07 (interval calibration):** OPEN — договориться об уровнях 90%/95% и истории созревших OOS-ошибок, не тестировать на финальном test.
- **Q08 (расхождение с PDF):** OPEN — согласовать с наставниками trading-day и смену официального курса на USDRUBF.

**Для распределения задач A/B/C/D на следующие два дня:** [two_day_plan.md](two_day_plan.md). Статус выполнения — [project_status.md](project_status.md).

## Status update — 2026-11-23 (после слияний B1/B2/B3)

Этот раздел **дополняет** решения D01–D20 и старые вопросы; не исправляет задним числом исторические статусы 20.11. Базовый `main` проверен после merge: tree SHA `791964a5bbe12cdb686641fa7e6434d3d8413ca7`.

| ID | Новое решение / наблюдение | Основание | Статус |
|---|---|---|---|
| D21 | Источник target идентифицирован как MOEX ISS USDRUBF `interval=24` / `CANDLE_CLOSE`; это не отдельное поле settlement. Точное фактическое время готовности свечи неизвестно | B1, `forecast_origin_pit_proposal.md`, raw candle | ACCEPTED SOURCE / HISTORICAL TIMESTAMP OPEN |
| D22 | ЕТС с 23.03.2026 изменила принадлежность вечерней сессии к торговому дню; необычный `end≈19:30` **не** является доказанным timestamp publication/last trade | B1, `results/moex_minute_audit/comparison.csv` | ACCEPTED CAVEAT |
| D23 | B2 описал все 30 полей и публикационные ограничения; A согласовал группы/сценарии, но **исторический record-level PIT не verified** | `feature_availability_catalog.csv`, `feature_lag_policy_proposal.md`, пользовательское подтверждение A | APPROVED CONDITIONAL POLICY |
| D24 | Версия `features_daily_v1` утверждена для инженерного использования: основной `core` 9, sensitivity RUONIA/Brent 11, sensitivity плюс USD/CNH 12. `swaprate_current` и `value` исключены | `configs/feature_scenarios_v1.json`, `docs/features_daily_v1.md` | ACCEPTED CONDITIONAL EOD |
| D25 | Не импутировать B3 funding, не удалять строки; 154 пропуска, fill/scale/PCA только train-only и по протоколу B/C | `reports/features_daily_v1_coverage.md` | ACCEPTED ENGINEERING POLICY |
| D26 | B3 локально проверен: 3 MATCH `--compare`; 49 unittest / 1 skipped (full source rebuild v2). Результат локального запуска сообщён пользователем; CI-clean rerun команды не заявлен | пользовательский терминальный журнал от 23.11.2026 | REPORTED_LOCAL_VERIFICATION |
| D27 | Следующие B задачи: **B4 funding missingness**, **B5 независимый календарный audit и handoff**; C/D могут использовать frozen B3 при принятом conditional EOD | `two_day_plan.md` | NEXT_ACTIONS |

### Состояние открытых вопросов на 23.11

- **Q01:** source идентифицирован и raw→panel `close` сохранён; экономическая интерпретация как биржевой candle close зафиксирована. Время публикации не установлено (**Q02 OPEN**).
- **Q03:** `PROVISIONAL`, нужен независимый архив USDRUBF и проверка особых торговых дат (**B5**).
- **Q04:** внутренний аудит качества завершён.
- **Q05:** split и baseline protocol зафиксированы; чистый replay с manifest всё ещё задача **A**.
- **Q06:** B2-каталог и лаговая политика **подготовлены и группы согласованы A**; исторические публикации funding, CBR, RUONIA, EIA, CNH недостаточно восстановлены для verified PIT.
- **Q07:** интервал 90%/95% и online calibration **задача D/A**; готовность не подтверждена.
- **Q08:** расхождение target USDRUBF vs официального USD/RUB в PDF **требует согласования с наставниками**.

Любое новое изменение разрешённых факторов, fixed forecast origin или lag policy оформлять отдельной версией и review A. Финальный test 2026 закрыт для model selection; до preregistration новых моделей никаких test-результатов.

## Решение 2026-11-24 — сокращение линейки точечных моделей

| ID | Решение | Обоснование | Статус |
|---|---|---|---|
| D21 | Приоритет текущего спринта: **M1 Ridge price-only, M2 Ridge core, M3 CatBoost core**, семь direct h-step моделей на `Δ_h CLOSE` | ограниченное время и 9 согласованных core-признаков, возможность двух чистых абляций | ACCEPTED; NOT_IMPLEMENTED на момент решения |
| D22 | `core` = 9 полей согласно `configs/feature_scenarios_v1.json`, из них 5 OHLCV и 4 внешних; 11/12 только sensitivity вне спринта | data handoff B1–B5 | ACCEPTED |
| D23 | PCA/DFM/Kalman, XGBoost, ML residual correction **не обязательны в текущем спринте** | сложность без достаточных вневыборочных оснований | DEFERRED |
| D24 | Сравнение M1/M2/M3 только по общим validation forecast keys; обучающие target созревают с лагом h, без test | защита от leakage и selection bias | ACCEPTED |
| D25 | Interval baseline RW и calibration остаются параллельным направлением D, B помогает только при запросе | ограниченный срок проекта | ACCEPTED |

**Остаточные ограничения:** conditional EOD/pseudo-PIT; непроверенное фактическое время доступности свечей; provisional calendar; 154 начальных funding missing, 198 наблюдаемых нулей; final test 2026 закрыт.

**Исторические задачи в старых разделах про DFM/PCA — архивная постановка, не текущий план.** Текущий список работ: [two_day_plan.md](two_day_plan.md).

## Исправление распределения ролей — 2026-11-24

**A = Data Engineer; B = Research Lead & Backtesting (руководитель исследования); C = Models; D = Uncertainty.** Все текущие назначения B0/B1 benchmarks, оценки, review, common-pair и go/no-go принадлежат **B**, а загрузка/качество/PIT-факторы — **A**. Старые подписи владельцев A/B в документах были ошибочной перестановкой ролей; скорректированы без изменения экспериментальных данных. Исторические ID B1–B5 и имена веток data work не переименовываем. При упоминании «группы A/B/C» в B2 речь о категориях признаков, не участниках команды.


## Решения после merge линейных и boosting-экспериментов — 2026-11-24

Эта запись **дополняет, но не переписывает** исторические решения D21–D25, которые были приняты до фактической реализации моделей.

| ID | Решение / наблюдение | Основание | Статус |
|---|---|---|---|
| D28 | Ridge v1/v2, RW with Drift, CatBoost Core v2 и XGBoost Core v2 реализованы и оценены на validation 2025, h=1…7; frozen B3 v1 остаётся без изменений | `reports/ridge_experiments*.md`, `reports/rw_drift_experiment.md`, `reports/catboost_core_v2.md`, `reports/xgboost_core_v2.md` | VALIDATION_EVALUATED_EXPLORATORY |
| D29 | Для новых моделей принят **Core v2 = 10** полей с `swaprate_current[t]`; доступность до origin — предположение команды, **ASSUMED_AVAILABLE**, не фактический historical verified timestamp | `reports/ridge_experiments_v2.md` и B4 PIT caveats | ACCEPTED_CONDITIONAL_ASSUMPTION |
| D30 | CatBoost имеет меньший RMSE против RW на h=4/6/7, но улучшение после HAC/Holm не подтверждено; XGBoost не лучше CatBoost ни на одном h. Не объявлять победу над RW | validation reports, общие 254 пары/h | OBSERVED_EXPLORATORY |
| D31 | Прогноз `C[t+h]−C[t]` уже используется во всех факторных моделях, в Ridge сравнен с log-return; **прямой level vs delta ещё не проверен** | Ridge/CatBoost/XGBoost reports | ACCEPTED_SCOPE_CLARIFICATION |
| D32 | Новые кандидаты — matched level-vs-delta и relative-features, без автоматического дальнейшего перебора бустингов; OI только после PIT-проверки; DFM/Kalman отдельный ресурсный выбор | [план](two_day_plan.md), [гипотезы](hypothesis_register.md) | PROPOSED_NOT_TESTED |
| D33 | Final test 2026 по-прежнему закрыт для любых исследовательских решений; repeated validation 2025 **не** независимое подтверждение | [протокол](backtest_protocol.md) | ACCEPTED |

**Роли:** A — Data Engineer; B — Research Lead & Backtesting; C — Models; D — Uncertainty. Существующие исторические решения и manifest не перезаписывать.

## Финальная предрегистрация спринта — 25.11.2026

| ID | Решение | Статус |
|---|---|---|
| D34 | **25.11 — только H11-REL-01:** matched CatBoost Core v2 10 (R0) vs Core v2 + **ровно 6** относительных признаков H11 P0 (R1); прежние h-specific параметры CatBoost, одинаковые train/validation keys | ACCEPTED / NOT_EXECUTED |
| D35 | **26.11 — только H13-TRANSFORM-01:** неизменный Core v2, CatBoost **delta vs log-return**; обе модели с ранее выбранными delta-параметрами, цена восстанавливается по соответствующей формуле | ACCEPTED / NOT_EXECUTED |
| D36 | H10 остаётся **direct level vs delta, NOT_TESTED**; H13 — другой контраст (delta vs log). H12 cross-asset и DFM не входят в спринт; конфликт исторических H12-ID подлежит отдельному исправлению, не трактовать как два результата одной гипотезы | ACCEPTED |
| D37 | Владелец B — Research Lead/backtest; C — реализация; A — проверка источников при запросе. Новый validation 2025 exploratory; **final test 2026 закрыт**, frozen источники/старые результаты не менять | ACCEPTED |

**Канонические задания:** [two_day_plan.md](two_day_plan.md). **Предрегистрация:** [H11 JSON](../configs/h11_catboost_relative_v1.json), [H13 JSON](../configs/h13_catboost_target_v1.json). Это решения, **не проведённые эксперименты**.
