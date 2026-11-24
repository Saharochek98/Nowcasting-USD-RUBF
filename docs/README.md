# USDRUBF — Research Documentation

**Актуально:** 2026-11-24, после merge Ridge v1/v2, RW Drift, CatBoost и XGBoost. Режим — **conditional EOD / pseudo-PIT**. Final test 2026 пока закрыт.

## Начать здесь

1. [Project Status](project_status.md) — выполненные исследования, таблица основных RMSE, открытые ограничения.
2. [План команды](two_day_plan.md) — актуальные задачи A/B/C/D и следующие гипотезы.
3. [Backtest Protocol](backtest_protocol.md), [Benchmark Specification](benchmark_spec.md), [Hypothesis Register](hypothesis_register.md), [Decision Log](decision_log.md) — методология, модели и история решений.
4. [Data handoff](data_handoff_to_modeling.md), [Features B3](features_daily_v1.md) — frozen данные, сценарии и PIT-ограничения.

## Результаты validation 2025

| Эксперимент | Отчёт | Итог |
|---|---|---|
| RW / AR(1) | [Benchmark](../reports/benchmark_summary.md) | RW — обязательный эталон |
| Ridge v1 (price-only/core/B1/B2/R5, delta/log) | [Ridge v1](../reports/ridge_experiments.md) | Нет подтверждённого выигрыша над RW |
| Ridge v2 (Core v2, frozen/prequential, no-spread) | [Ridge v2](../reports/ridge_experiments_v2.md) | Prequential лучше frozen; RW по RMSE лучше на каждом h |
| RW with Drift | [Drift](../reports/rw_drift_experiment.md) | RW с дрейфом не улучшил RW |
| CatBoost Core v2 | [CatBoost](../reports/catboost_core_v2.md) | Ниже RMSE, чем RW, на h=4,6,7; HAC/Holm выигрыш не подтвердил |
| XGBoost Core v2 | [XGBoost](../reports/xgboost_core_v2.md) | Не превзошёл RW или CatBoost ни на одном h |

Все указанные результаты относятся к повторно использованной **exploratory validation 2025**, 254 общие пары на горизонт, h=1…7 следующих торговых сессий. Это прогноз `CANDLE_CLOSE` фьючерса USDRUBF, **не** официального курса ЦБ и **не** цены клиринга.

## Версии признаков

- **Frozen B3 v1:** `configs/feature_scenarios_v1.json`; core **9** полей, sensitivity 11/12; `swaprate_current` не входит.
- **Модельный Core v2:** **10** полей, включая `swaprate_current[t]` из frozen full source v2. В модели его доступность — **ASSUMED_AVAILABLE** после закрытия t, не исторически VERIFIED. Отдельные конфигурация/loader описаны в [Ridge v2](../reports/ridge_experiments_v2.md); B3 v1 не перезаписана.
- Текущий/предыдущий funding: 153/154 начальных missing соответственно; числовые нули остаются наблюдениями. Календарь **PROVISIONAL**.

## Следующие гипотезы

- **Level vs delta:** модели уже прогнозируют `C[t+h]−C[t]`; равноправное прямое предсказание `C[t+h]` **ещё не проверяли**. Отдельный Ridge log-return эксперимент выполнен.
- **Относительные признаки:** returns 1/2/3/7/14, `(Close−SMA20)/SMA20`, RSI14, ATR14/Close, изменение funding и (если источник/PIT подтверждены) Open Interest %.
- **DFM + Kalman** — отдельная не реализованная пока проверка динамических скрытых факторов; **RW intervals 90/95%** — задача D.

**Команда:** A — Data Engineer; B — Research Lead & Backtesting; C — Models; D — Uncertainty. Код/метрики новых тестов не заменяют исходные результаты. Final test 2026 открывается только по заранее зарегистрированному конечному протоколу. Расхождение между официальным USD/RUB в первоначальной постановке и USDRUBF требует согласования с наставниками.
