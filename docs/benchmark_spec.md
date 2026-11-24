# Benchmark Specification — USDRUBF

**Версия:** 1.4 · 2026-11-19 · **Ответственный:** B, Research Lead

## Единые соглашения

`C_d=CLOSE_d` дневной OHLC-свечи USDRUBF, **после подтверждения семантики поля CLOSE** как рыночной цены. `forecast_origin` — после готовности дневной свечи; точное время не установлено, режим `conditional EOD / pseudo-PIT`. Горизонты `h=1,…,7` относятся к следующим торговым дням по независимому календарю.

Расчетная цена клиринга, официальный курс ЦБ и фандинг не являются target. Если CLOSE отсутствует, origin пропускается; если отсутствует будущая целевая цена, прогноз сохраняется как `unscored`, но дата цели не передвигается. Нет внутридневных данных для реконструкции последней сделки в 23:50.

## Таблица моделей

| ID | Модель | Формула/идея | Данные |
|---|---|---|---|
| B0 | Random Walk | `C_hat(d+h|d)=C_d` | дневной CLOSE |
| B1 | AR(1) на разностях | `ΔC_d=c+φΔC_{d-1}+ε_d`; рекурсивное прогнозирование разностей и суммирование | последовательные дневные изменения CLOSE |
| B2 | RW with Drift (дополнительно) | `C_d+h*mu_hat_d` | доступные последовательные дневные изменения |
| B3 | Spread/funding regression (позже) | изменения CLOSE на лаги `C_d−S_CBR_asof` и опубликованный `SwapRate_asof` | подтвержденные PIT-факторы |
| M1 | PCA + прогнозная регрессия | статические факторы | PIT-панель |
| M2 | DFM + Kalman | динамические факторы | PIT-панель |
| M3 | DFM+ML | коррекция реальных OOS-ошибок | созревшие метки |
| M4 | Interval calibration | усадка/интервалы | train/validation |

Для B1 по умолчанию expanding-window OLS с обновлением на каждом origin; минимум исторических наблюдений определяется до validation, а не по final test. AR обучается только на **соседних** торговых днях с валидными CLOSE (без замыкания разрывов).

## Процедура B0 / B1

1. Аудит OHLC, семантики поля CLOSE и независимого торгового календаря.
2. Определить допустимый origin после дневной свечи и PIT-статус данных.
3. B0 прогнозирует `C_d` на каждый из следующих семи торговых дней.
4. B1 на каждом origin обучает AR(1) на известных последовательных дневных разностях и рекурсивно прогнозирует семь цен.
5. Сохранять прогнозы до расчёта метрик. Будущие CLOSE используются **только** для последующего scoring.
6. Сравнивать RMSE, MAE, bias по горизонтам **на пересечении** допустимых origin/target двух моделей.

## Артефакты

- `predictions_rw.parquet`, `predictions_ar1.parquet`: `model_id, model_version, forecast_origin, origin_date, target_date, horizon, prediction, train_end_date, data_version, pit_status`.
- `metrics_by_horizon.csv`: `model_id, horizon, n, mae, rmse, bias, rmse_ratio_vs_rw`.
- `benchmark_config.yaml`: `target_field=CLOSE`, `frequency=daily`, `forecast_origin_mode=conditional_EOD`, календарь, train/validation/test, минимум AR-history, re-fit schedule.
- `data_quality_report.md`: дубликаты, пропуски, проверки OHLC, корректность календаря, неразрешенные предположения.

Никакие эмпирические результаты пока не заявлены.

## Статус выполнения (20.11.2026)

**B0 RW и B1 AR(1) — IMPLEMENTED + VALIDATION EVALUATED**, не просто план. Код `benchmarks/core.py`, CLI `benchmarks/run.py`, результаты `results/benchmark_calendar_v2/`. Параметры: 100 минимальных valid AR-пар, expanding-window OLS, обновление на каждом origin; 2025 validation по 254 общие пары на каждый h. Сравнения A1→A2 повторены по парам с `passed=true` (см. `reports/benchmark_summary.md`).

| h | RW RMSE | AR RMSE | AR/RW |
|---:|---:|---:|---:|
| 1 | 0.772440 | 0.774045 | 1.002078 |
| 2 | 1.135664 | 1.144051 | 1.007385 |
| 3 | 1.404904 | 1.422296 | 1.012380 |
| 4 | 1.617355 | 1.644489 | 1.016777 |
| 5 | 1.790585 | 1.827089 | 1.020387 |
| 6 | 1.965425 | 2.012490 | 1.023947 |
| 7 | 2.147724 | 2.207770 | 1.027958 |

AR не выигрывает по RMSE на validation; семейно-значимого отличия по HAC+Holm на 5% не обнаружено. Это **не** итоговый результат на final test. B2 Drift и B3 Spread/Funding пока **NOT IMPLEMENTED**. План расширения: `docs/two_day_plan.md`.

## Решение 24.11.2026: M1/M2/M3 — обязательная ветка точечного прогноза

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

**Реализация:** использовать `scripts.build_model_features.load_scenario(scenario="core")`, брать разрешённые поля по имени, а не все поля датасета. Новых полученных результатов M1–M3 пока **нет**. B0/B1 уже validation evaluated, их код и исходные отчёты не менять. DFM/Kalman, PCA+Ridge, XGBoost — deferred.


## Актуализация 24.11.2026 — фактически выполненные модели

Предыдущие таблицы M1/M2/M3 и статус `PLANNED` выше относятся к **исторической версии плана**, не к текущему состоянию. Сведения ниже имеют приоритет для текущей модели. Не переименовывать старые ID без миграции.

- **RW, AR(1)** — базовые validation-эталоны, [результаты](../reports/benchmark_summary.md).
- **RW with Drift** — `C[t]+h*mean(ΔC)`, expanding/prequential; [результат](../reports/rw_drift_experiment.md): хуже RW по RMSE на всех h.
- **Ridge v1/v2** — price-only (5), старый B3 core (9), расширенные B1/B2, Core v2 (10, включая `swaprate_current`), prequential/frozen, no-spread; [v1](../reports/ridge_experiments.md), [v2](../reports/ridge_experiments_v2.md).
- **CatBoost Core v2** — 10 признаков; [результат](../reports/catboost_core_v2.md): RMSE ниже RW на h=4,6,7, без значимости после HAC/Holm.
- **XGBoost Core v2** — те же 10 признаков; [результат](../reports/xgboost_core_v2.md): не улучшил RW или CatBoost ни на одном h.

Все прямые факторные модели предсказывают `Δ_h C=C[t+h]-C[t]` и восстанавливают цену как `C[t]+predicted_delta`; Ridge также проверил `log(C[t+h]/C[t])`. **Парное сравнение с direct target уровня `C[t+h]` ещё не выполнялось**. На validation 2025 у каждого h 254 одинаковых прогнозных ключа. Результаты exploratory; final test 2026 не оценивался. Историческая доступность `swaprate_current` в Core v2 — **ASSUMED_AVAILABLE**, а не VERIFIED; frozen B3 core=9 не заменён Core v2=10.

**Новые гипотезы (NOT_TESTED):** direct level vs delta при равных условиях; относительные returns/SMA/RSI/ATR/funding features вместо или вместе с сырыми ценовыми уровнями; Open Interest — только после PIT-аудита источника. Детали — [текущий план](two_day_plan.md) и [реестр](hypothesis_register.md). PCA/DFM/Kalman/ML residual и интервальные калибровки не обозначать реализованными без подтверждающих артефактов.
