# План команды на 24–25 ноября 2026 — модели M1/M2/M3

**Решение Research Lead:** в текущем спринте реализовать **M1 Ridge price-only**, **M2 Ridge core (9 признаков)** и **M3 CatBoost core (9 признаков)**. B0 RW и B1 AR(1) остаются обязательными эталонами. **PCA/DFM/XGBoost отложены**, ML-коррекция DFM не реализуется в этом цикле.

**Что уже готово:** B1–B5 data handoff, `features_daily_v1.csv`, `configs/feature_scenarios_v1.json`, RW/AR validation pipeline. Данные `conditional EOD / pseudo-PIT`, календарь PROVISIONAL, финальный test 2026 **не открывать**.

## Единая постановка M1/M2/M3

### Приоритетные модели

| ID | Класс | Признаки | Цель |
|---|---|---|---|
| **M1** | Ridge | `price-only`: `open, high, low, close, volume` из дневной свечи (и заранее объявленные lag/rolling-преобразования при необходимости) | Есть ли сигнал в истории USDRUBF сверх RW/AR? |
| **M2** | Ridge | **ровно `core` из `configs/feature_scenarios_v1.json`**: `open, close, high, low, volume, cbr_usdrub_effective, swaprate_prev_session, key_rate_pct_effective, spread_close_minus_cbr` | Есть ли добавочная ценность согласованных внешних индикаторов? |
| **M3** | CatBoostRegressor | Тот же **`core`**, что M2, без отдельной подборки признаков | Даёт ли нелинейная модель выигрыш при том же информационном множестве? |

`price-only` — название **набора признаков**, а не отдельного алгоритма. Чтобы выделить эффект внешних факторов, M1 и M2 должны использовать сопоставимые трансформации базовых OHLCV и одинаковые evaluation keys. Если у M1 есть дополнительные лаги/rolling-признаки, они должны быть явно перечислены и одинаково добавлены в M2/M3 либо вынесены в отдельную абляцию.

Для каждого `h=1,…,7` обучаем отдельную direct-модель с target `y_{t,h}=C_{t+h}-C_t`, затем `C_hat_{t+h|t}=C_t+y_hat_{t,h}`. **Созревание метки:** обучающая строка `t` доступна для нового fit только после появления `C_{t+h}` на текущем origin. Все признаковые преобразования и подбор Ridge alpha/параметров CatBoost — только по прошлому, без final test.

**Сравнения:**
1. B0 RW и B1 AR(1) — обязательные существующие benchmarks.
2. M1 vs B0/B1 — price-only модель.
3. M2 vs M1 — добавленная ценность внешних факторов (`core`).
4. M3 vs M2 — добавленная ценность нелинейности на одинаковом `core`.
5. Для каждого сравнения: **одинаковые `(origin_date, target_date, horizon)`**, сравнивать MAE/RMSE/bias и число пар; не выбирать модель по final test.

**Funding:** в `swaprate_prev_session` есть 154 начальных missing; 198 наблюдаемых нулей — реальные нули, не пропуски. Допустимые treatment/маски пропусков должны быть зафиксированы до validation-выбора; импутация/масштабирование только train-only. Если M1 имеет более длинную историю, для сравнения с M2/M3 пересчитывать метрики на общих origin/target.

**Статус:** `PLANNED / NOT IMPLEMENTED` на дату принятия решения; не приписывать M1–M3 численные результаты. Статус `conditional EOD / pseudo-PIT`, предварительный календарь и закрытый final test остаются неизменными.

**Не входят в обязательный этап:** PCA+Ridge, DFM/Kalman, XGBoost и ML-коррекция остатков DFM. Остаются в разделе возможных расширений после оценки M1–M3; статью Giannone et al. (2008) используем как методологическое обоснование nowcasting/PIT, **не** как воспроизведённую архитектуру.

## 24 ноября — реализация и контроль

| Участник | Приоритет | Задачи | Definition of Done |
|---|---|---|---|
| **A — Research Lead** | P0 | Проверить clean rerun RW/AR и тесты; согласовать 7 direct targets `Δ_h CLOSE`, созревание h-шаговых меток, общий prediction schema, неизменные validation keys; утвердить заранее ограниченные гиперпараметры и missing policy | Общая конфигурация экспериментов, тесты no-leakage, валидная RW/AR контрольная таблица с git SHA |
| **C — Models** | P0 | Реализовать **M1 Ridge price-only** и **M2 Ridge core** с train-only scaling/imputation, expanding/direct h=1..7, OOS validation прогнозами; подготовить минимальный M3 CatBoost core | PR с кодом и тестами; `predictions_m1/m2`; общий список признаков; метрики vs RW/AR на одинаковых парах |
| **D — Uncertainty** | P0 | RW интервалы 90%/95% из созревших исторических h-шаговых ошибок; отдельно interval coverage, width, interval score; no-leakage тесты | Интервальный baseline и validation таблицы по 7 h и двум уровням |
| **B — Data** | По необходимости | Ответить на вопросы по `load_scenario('core')`, 154 missing `swaprate_prev_session` и 198 фактических нулях; PIT-lag policy; новых сборов без блока не начинать | Подтверждение data contract, correction PR только если найден баг |

## 25 ноября — нелинейность и абляции

| Участник | Приоритет | Задачи | Definition of Done |
|---|---|---|---|
| **A** | P0 | Сверить clean validation результаты для B0/B1/M1/M2/M3, common-sample сравнения M1↔M2 и M2↔M3, даты, n, метрики и leakage; обновить реестр гипотез, журнал экспериментов | Таблица сравнения по h и обоснованный вывод без final test |
| **C** | P0 | **M3 CatBoost core** на ровно тех же признаках, origin/target, правилах missing и периоде, что M2; маленькая заранее зафиксированная конфигурация; seed и deterministic run где возможно | `predictions_m3`, config, воспроизводимый run, абляции M1/M2/M3 |
| **D** | P1 | Проверить интервалы RW по h, при времени — калибровка на уже созревших OOS-ошибках, исследовать связь ошибки с прошлой волатильностью | Отдельный interval report; не подбирать по final test |
| **B** | Поддержка | Устранить только найденные проблемы lag policy/source; сохранить frozen feature panel | Не менять silently старый data snapshot |

## Методологические требования / стоп-сигналы

- `core` — **ровно 9**, а не все 12 столбцов общего store. 11/12 остаются sensitivity-only и **не входят** в этот спринт.
- M1 использовать только допустимые price/volume на доступной полной дневной свече; признаковые лаги и rolling рассчитываются на истории до origin. Все дополнительные трансформации регистрируются до сравнения.
- M2 и M3 используют одинаковые входные признаки, календарь и evaluation keys; missing funding не означает 0.
- `train-only` — включая scaler, imputers, выбор `alpha`, CatBoost параметры и созданные лаговые статистики с обучаемыми порогами. Для direct h-labeled обучений дата target каждой обучающей строки уже должна быть известна.
- M3: маленькая модель без широкой сетки; пример стартовой предрегистрации `depth=3, learning_rate=0.05, iterations=300, l2_leaf_reg=5, loss_function=RMSE, random_seed=42`. Это **стартовый выбор, не оптимум**.
- Final test 2026 закрыт. Основной отчёт — validation 2025. Нельзя объявлять M1–M3 `VALIDATION_EVALUATED`, пока нет сохранённых прогнозов и метрик.
- Если не хватает времени: приоритет **M1/M2 и честная абляция**, затем M3; не форсировать DFM или XGBoost.

## Передача и review

C отдаёт A `model_id,origin_date,target_date,horizon,prediction,train_end_date,feature_scenario,model_version,pit_status`; фактические значения `actual` добавляются только при scoring. D использует те же календарные ключи. A документирует `n` и правила common-sample, сохраняет старые RW/AR artifacts.

Данные: [Data handoff](data_handoff_to_modeling.md). Статус: [Project status](project_status.md). Метрики и правила: [Backtest protocol](backtest_protocol.md).
