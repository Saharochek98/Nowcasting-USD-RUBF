# Финальный двухдневный план: H11 → CatBoost delta vs log-return

**Принято Research Lead B:** 2026-11-25. **Дни:** 25 ноября (H11) и 26 ноября (target transform). **Статус:** ПЛАН / PRE-REGISTERED, не исполненные эксперименты.

**Цель:** два ограниченных, воспроизводимых сравнительных эксперимента CatBoost Core v2 без смены базовых данных, расширения сетки алгоритма и без обращения к final test 2026 для выбора.

**Роли:** A — Data Engineer (консультация и аудит derived features), **B — Research Lead & Backtesting (владелец протокола)**, **C — Models (реализация и прогон)**, D — Uncertainty (параллельная работа над интервалами; вне критического пути). Исторические data-этапы B1–B5 — не обозначения ролей людей.

## Неизменная база

- Instrument/target price: дневной MOEX ISS **USDRUBF CANDLE_CLOSE**, не CBR official rate и не settlement; forecast origin после условной доступности свечи, `conditional EOD / pseudo-PIT`.
- `h=1,…,7` **торговых** дней по provisional календарю.
- Frozen dataset: `data/processed/model_dataset_usdrubf_all_features_v2.csv` + loader `benchmarks/ridge_v2.py`; Core v2 **10** числовых полей, включая `swaprate_current` только на допущении `ASSUMED_AVAILABLE`. Frozen B3 core **9** — другая версия, её не подменять.
- CatBoost: `iterations=100, learning_rate=0.03, loss_function=RMSE, bootstrap_type=No, random_seed=240112, thread_count=1`; по каждому h фиксируются **ранее выбранные** пары depth/L2: h1 (2,30), h2 (3,30), h3 (2,30), h4 (3,30), h5 (2,30), h6 (2,30), h7 (2,30). **Никакой повторной настройки** / outer early stopping.
- Expanding/prequential fit на каждом origin и **только созревших h-step labels** (`target_date <= origin_date` при условном EOD); train-only медианная обработка missing, наблюдаемый ноль funding сохраняется; all-missing train-column = ошибка.
- Outer validation: target dates 2025, исходно 254 пары на h. Реальное `n` и совпадение троек `(origin_date,target_date,h)` проверять, а не предполагать; final test 2026 не используется для модельного выбора.
- Старые `configs/catboost_experiment.json`, `results/catboost_core_v2/`, старые таблицы/файлы источников **не изменять**. Новые прогоны — только в собственных `results/` с manifest.

## День 1 — 25 ноября: H11-REL-01 (фичи, delta target)

**Зарегистрированный единственный контраст:** `R0_MATCHED = CatBoost Core v2 (10)` **vs** `R1 = CatBoost Core v2 + ровно шесть P0 relative features`. R0 повторно обучается на тех же допустимых train rows, что R1; нельзя считать новым контролем старые метрики с отличающимся warm-up.

Шесть фичей из [предложения A](feature_candidates_H11_2026-11-25.md): `logret_1=ln(C_t/C_{t-1})`, `logret_3=ln(C_t/C_{t-3})`, `logret_7=ln(C_t/C_{t-7})`, `range_rel=(High_t-Low_t)/Open_t`, `volume_ratio_20=V_t/SMA20(V)_t`, `distance_sma20=(C_t-SMA20(C)_t)/SMA20(C)_t`. Rolling включает доступную свечу t, ось — торговые дни; знаменатель <=0 / неполные окна → missing. **Никаких RSI/ATR/дополнительных окон, R2 и OI** в этом эксперименте.

**B — Research Lead & Backtesting (P0, ориентир 2–3 ч):**
1. Зафиксировать схему из `configs/h11_catboost_relative_v1.json` и убедиться, что оба run используют одинаковый target `delta=C_{t+h}-C_t` и старые выбранные CatBoost параметры без нового tune.
2. Утвердить правило общего warm-up: первые 19 сессий исключаются из **обеих** обучающих выборок (и любые незавершённые rolling-окна обрабатываются одинаково); сравнить идентичность допустимых train windows, validation keys и source SHA.
3. Выполнить review отсутствия будущей информации: perturb future OHLCV / target не изменяет прежних признаков и прогнозов; `target_date` строго фиксируется календарём; imputer только train; current funding `ASSUMED_AVAILABLE`.
4. Оценить парные RMSE/MAE/bias (`prediction-actual`), RMSE/RW, `n` и HAC-Newey–West разность MSE для семи горизонтов с `L=max(20,h-1)` и **одной Holm-семьёй из 7** для R1−R0. Другие comparisons — явно exploratory; не делать вид, что новый Holm исправляет все прошлые итерации.
5. Написать краткое решение по H11: наблюдаемое улучшение/ухудшение/неопределённо, без заявлений о test.

**C — Models (P0, ориентир 4–6 ч):**
1. Реализовать отдельный feature builder из frozen OHLCV, не менять исходную панель.
2. Реализовать R0_MATCHED и R1 с одинаковыми обучающими origin, h-maturity и imputation; 7 direct CatBoost моделей, **только сохранённые h-specific параметры**.
3. Записать `predictions_r0.csv`, `predictions_r1.csv` (до scoring без actual), `metrics_h11.csv`, `hac_h11.csv`, `manifest.json`, tests `test_h11_no_leakage.py` и `reports/h11_results.md` (только после расчёта).
4. Отдельно проверить математически границы warm-up и запрет включения будущих значений.

**Definition of Done:** код + тесты + frozen config + manifest Git SHA/source hashes, 7× сопоставимых пар/сроков, summary with `n`, статистическое/экономическое значение результата и ограничения; результаты на validation маркируются **exploratory**. Если к концу дня H11 не прошёл эти проверки — не запускать второй эксперимент, закончить воспроизводимость H11.

## День 2 — 26 ноября: H13-TRANSFORM-01 (delta vs log-return)

**Зарегистрированный единственный контраст:** `T_DELTA` **vs** `T_LOG` на **одном и том же неизменном Core v2 (10)**, тех же train/validation origin и **тех же h-specific CatBoost параметрах**. H11-признаки **не добавлять ни в один вариант**, иначе одновременно изменятся входы и target.

- `y_delta(t,h)=C_{t+h}-C_t`; `C_hat_delta=C_t+yhat_delta`.
- `y_log(t,h)=ln(C_{t+h}/C_t)`; `C_hat_log=C_t*exp(yhat_log)`.
- Обе модели оценивать на **ценовой** шкале RMSE/MAE/bias; `C>0`. Экспонента без Jensen/smearing correction в первичном сравнении — это **геометрическая** точка, не несмещённая оценка `E[C_future | X]` для price RMSE.
- Параметры фиксированы по ранее проведённому **delta** inner tuning; это честный дешёвый sensitivity, **не** доказательство оптимальных параметров для log-loss. Никакой новой сетки.
- `H13` — отдельный ID, поскольку `H10` уже означает direct **level vs delta**, а `H12` конфликтует между DFM и cross-asset предложением; H12 в этот спринт не входит.

**B — Research Lead & Backtesting (P0, 2–3 ч):**
1. Утвердить `configs/h13_catboost_target_v1.json` **до** нового scoring, проконтролировать одни и те же `X`, `train_pairs`, календарь, 7 h и отсутствие re-tuning.
2. Проверить инвариантность прошлых прогнозов к изменениям будущих цен и верность восстановления цены из обеих шкал.
3. Проверить `RMSE_h, MAE_h, bias_h, n_h`, matched `T_LOG - T_DELTA` и сравнение с RW на общих origin/target/h; HAC/Newey–West, `L=max(20,h-1)`, Holm по **7 первичным** контрастам `T_LOG - T_DELTA` (RW — контекст).
4. Сформулировать окончательный вывод: нужно ли в последующей работе менять target (не по одному минимальному RMSE), ограничения log→price, статус exploratory.

**C — Models (P0, 4–6 ч):**
1. Реализовать и запустить две версии CatBoost на одном `Core v2`, без H11 фичей, с фиксированными выбранными параметрами.
2. Обеспечить одинаковые таблицы `train_origin,target_date,available_label`, train-only imputer и `validation` forecast keys; никакого учёта будущей метки для обучения.
3. Сохранить `predictions_delta.csv`, `predictions_log.csv`, `metrics_h13.csv`, `hac_h13.csv`, `manifest.json`, tests `test_h13_target_alignment.py`, `reports/h13_results.md`.

**Definition of Done:** сравнимые ценовые ошибки по h 1…7, инвариантность данных/прогнозов до origin, проверка экспоненциального восстановления, сохранённые таблицы и версии, вывод без test claims.

## Общие решения / stop conditions

- **Нет времени/не прошёл H11?** День 2 отдаём завершению H11 и подготовке защиты. Отрицательный H11 — тоже допустимый завершённый результат.
- **Нет PIT доказательств?** Не повышаем статус `ASSUMED_AVAILABLE / pseudo-PIT` и не трактуем CatBoost как применимую в конкретную минуту реального времени.
- **2025 уже много раз анализировался:** новые HAC+Holm корректируют только заявленное узкое семейство, не всю историю model selection; научные выводы exploratory.
- **2026 final test** не используется для выбора признаков/targets/thresholds; полноценный independent test требует отдельной фиксации протокола и оценки степени прежнего косвенного использования.
- **Не делать**: R2 (замена OHLC), H12 cross-assets, SOFR/RGBI/OI, дополнительные глубины/сетки, XGBoost, DFM.
- Участники **A** и **D** не получают новый обязательный deliverable: A консультирует по source/calendar/missing, D продолжает интервалы независимо.

## Артефакты-предпосылки

- [H11 review](feature_candidates_H11_2026-11-25.md)
- [CatBoost Core v2 results](../reports/catboost_core_v2.md)
- [Benchmark/Backtesting protocol](backtest_protocol.md)
- [Hypothesis register](hypothesis_register.md)
- Конфигурации: [H11](../configs/h11_catboost_relative_v1.json), [H13](../configs/h13_catboost_target_v1.json).
