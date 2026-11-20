# Project status — USDRUBF Nowcasting

**Срез проекта:** 2026-11-20 · **База:** `main`, коммит `84cc4a5` (до обновления документации)  
**Режим выводов:** `PRELIMINARY`, `conditional EOD / pseudo-PIT`; финальный test 2026 **не оценивался**.

Этот документ — краткий командный dashboard. Полные результаты и методология находятся в [reports/benchmark_summary.md](../reports/benchmark_summary.md), [reports/ar1_diagnostics.md](../reports/ar1_diagnostics.md), [reports/calendar_quality_report.md](../reports/calendar_quality_report.md), [reports/data_quality_report.md](../reports/data_quality_report.md). Код и конфигурация — в [benchmarks/](../benchmarks/) и [configs/benchmark_config.yaml](../configs/benchmark_config.yaml).

## Выполнено и подтверждено сохранёнными артефактами

| Блок | Состояние на 20.11.2026 | Подтверждение |
|---|---|---|
| Research Charter, backtest protocol, decision log | DONE для текущей **условной EOD** постановки; документы обновлены в текущем изменении | `docs/` |
| Дневные OHLCV USDRUBF | DONE, 1 160 свечей за 2022-04-26—2026-11-18; raw/processed | `data/raw/`, `data/processed/`, `reports/data_quality_report.md` |
| Качество и согласованность данных | DONE для первичного аудита: без дубликатов, неположительных цен и нарушений OHLC | `reports/data_quality_report.md` |
| Торговый календарь | **PROVISIONAL**: реконструкция 1 668 календарных дат и 1 160 ожидаемых торговых дат, равенство с raw 1 160/1 160; 4 рабочие субботы | `reports/calendar_quality_report.md` |
| Baseline B0 Random Walk | IMPLEMENTED + validation evaluated | `benchmarks/core.py`, `results/benchmark_calendar_v2/` |
| Baseline B1 AR(1) на `ΔCLOSE` | IMPLEMENTED + validation evaluated; expanding OLS, минимум 100 валидных пар | те же источники |
| Универсальный walk-forward и метрики | IMPLEMENTED для B0/B1; отдельное формирование прогноза и scoring | `benchmarks/core.py`, `benchmarks/run.py` |
| Тесты и контроль отсутствия look-ahead | IMPLEMENTED; тесты присутствуют в репозитории, сохранённая проверка воспроизводимости A1/A2 = passed | `tests/`, `results/benchmark_calendar_v2/comparison.json` |
| AR diagnostics | DONE, исследовательская диагностика train/validation без настройки baseline | `reports/ar1_diagnostics.md`, `results/ar1_diagnostics/` |
| Final test 2026 | **SEALED** для оценки; в текущих результатах не открыт | `results/benchmark_calendar_v2/run_manifest.json` |
| PIT-факторы, PCA/DFM, ML residual correction, интервалы | NOT IMPLEMENTED в просмотренной ветке `main` | пока нет соответствующих артефактов |

«DONE» означает наличие конкретной реализации/сохранённого результата, **не** полное независимое подтверждение протокола. Исполнение `pytest` из чистого checkout текущего `main` в рамках этого документа не выполнялось.

## Эмпирические результаты: validation 2025

Train 683 торговых свечи (до 2024-12-30), validation 254 свечи (2025-01-03—2025-12-30), final test 223 свечи (2026-01-05—2026-11-18; **не оценивался**). 254 общих scored forecast pairs на каждом горизонте у обеих моделей.

| h (торговые дни) | RW RMSE | AR(1) RMSE | AR/RW RMSE |
|---:|---:|---:|---:|
| 1 | 0.772440 | 0.774045 | 1.002078 |
| 2 | 1.135664 | 1.144051 | 1.007385 |
| 3 | 1.404904 | 1.422296 | 1.012380 |
| 4 | 1.617355 | 1.644489 | 1.016777 |
| 5 | 1.790585 | 1.827089 | 1.020387 |
| 6 | 1.965425 | 2.012490 | 1.023947 |
| 7 | 2.147724 | 2.207770 | 1.027958 |

**Вывод по validation:** AR(1) имеет худший RMSE, чем RW, на всех семи горизонтах. Это **не** означает семейно-значимое различие: после Holm-поправки семи HAC-сравнений ни один горизонт не имеет `p_adj < 0.05` (см. `results/ar1_diagnostics/loss_summary_hac.csv`).

**Исследовательские инсайты:**
1. Lag-1 ACF дневных изменений: ~0.050 на train и ~0.061 на validation; Ljung–Box на заранее выбранных лагах не отвергает отсутствие серийной корреляции signed changes. Не утверждать полное отсутствие прогнозируемости.
2. Зависимость абсолютных доходностей сильнее зависимости направления: для validation `abs(log return)` Holm-adjusted Ljung–Box на лаге 20 `p≈0.027`. Это мотивирует тестирование **прогнозной дисперсии и интервалов**, но не доказывает преимущества какой-либо модели.
3. У AR(1) прогнозный дрейф константы накапливается к h=7. По сохранённым прогнозам средний вклад константы ≈+0.1003 RUB, AR-части ≈−0.0059 RUB. Это **алгебраическая диагностика**, а не эксперимент с моделью без константы.
4. A1→A2: 254 validation-пары для каждого h, прогнозы, OLS-параметры и ошибки воспроизведены; прежние файлы сохранены.

## Блокеры и ограничения

- **P0 — external feature panel отсутствует:** для Ridge/PCA/DFM нет готового PIT-совместимого набора внешних признаков; funding и курс ЦБ требуют дат публикации или заранее оговорённых консервативных лагов.
- **P0 — доступность дневного CLOSE:** `CANDLE_CLOSE` получен из MOEX ISS `interval=24` и отделён от биржевой расчётной цены по происхождению поля, однако историческое время готовности свечи не подтверждено. Нельзя называть прогноз «в 23:51».
- **P1 — календарь:** `reconstructed_moex_calendar_provisional` внутренне согласован с ценами, но ещё не проверен полностью по независимому историческому архиву USDRUBF. Совпадение всех дат с реконструкцией не равняется независимому аудиту.
- **P1 — reproducibility manifest:** A2 и AR diagnostics сохранены при `implementation_state=uncommitted_changes`; их код позднее вошёл в `main`, поэтому желателен контрольный rerun из чистого checkout с верным SHA текущего кода.
- **P1 — интервалы:** нет point forecast interval benchmark и калибровки; метрики покрытия и ширины должны считаться отдельно от RMSE.
- **P1 — закрытый test:** не открывать до регистрации выбора моделей, доступных признаков, правил калибровки и финального протокола.

## Ближайшие 2 дня

Детальные задания, зависимости и definition of done: [two_day_plan.md](two_day_plan.md).

**A — Research Lead/Backtesting:** чистый rerun RW/AR; реестр экспериментов, leakage review, API новых моделей; затем интеграция и common-pair OOS-сравнение.

**B — Data Engineer/PIT:** data dictionary с датами публикаций, 2–3 внешних экономически мотивированных индикатора, сверка funding/CBR и окончательная пометка календарных ограничений; затем PIT-панель и её покрытие.

**C — Quant Modeling:** спецификация факторных групп и простой Ridge baseline; затем PCA + регрессия **только при достаточной панели** и совместимость с benchmark API.

**D — ML/Uncertainty:** RW-интервалы по семи горизонтам и interval metrics; затем causal/online calibration без test и анализ роли прошлой волатильности.

## Правило обновления статуса

Изменяйте статус на DONE только при наличии **пути к артефакту + периода + точного критерия проверки**. Данные и метрики финального test не использовать для выбора моделей. После новых результатов обновлять статус, [hypothesis_register.md](hypothesis_register.md) и [decision_log.md](decision_log.md). Не переписывать историю validation задним числом.
