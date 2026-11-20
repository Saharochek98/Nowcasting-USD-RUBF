# Benchmark Specification — USDRUBF

**Версия:** 1.3 · 2026-11-19 · **Ответственный:** A

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
