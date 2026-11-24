# CatBoost Core v2: exploratory validation 2025

Эксперимент выполнен 2026-11-24 в `feature/direct-ridge-experiments`. **2025 год уже использовался в Ridge v1/v2 и drift; результаты exploratory и требуют независимого test.** Новых признаков и расширения сетки после просмотра validation нет. 2026 не использован для обучения, настройки или оценки.

## Результаты на общих датах

RMSE в RUB/USD; на каждом h ровно 254 одинаковые тройки `(origin_date, target_date, horizon)` у всех пяти моделей. Targets: 2025-01-03…2025-12-30; первые origins лежат в декабре 2024 года.

| h | RW | AR(1) | RW drift | Ridge Core v2 | CatBoost Core v2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.772440 | 0.774045 | 0.776165 | 0.775978 | 0.773605 |
| 2 | 1.135664 | 1.144051 | 1.145666 | 1.144966 | 1.141679 |
| 3 | 1.404904 | 1.422296 | 1.422904 | 1.421418 | 1.414397 |
| 4 | 1.617355 | 1.644489 | 1.644651 | 1.642514 | 1.614577 |
| 5 | 1.790585 | 1.827089 | 1.828883 | 1.827823 | 1.791839 |
| 6 | 1.965425 | 2.012490 | 2.015911 | 2.016073 | 1.958003 |
| 7 | 2.147724 | 2.207770 | 2.211516 | 2.212588 | 2.145415 |

CatBoost: остальные ценовые метрики (`bias = prediction − actual`):

| h | MAE | bias | RMSE/RW | n |
| --- | --- | --- | --- | --- |
| 1 | 0.572362 | +0.100161 | 1.001508 | 254 |
| 2 | 0.841060 | +0.178777 | 1.005297 | 254 |
| 3 | 1.050673 | +0.299348 | 1.006757 | 254 |
| 4 | 1.197323 | +0.377026 | 0.998283 | 254 |
| 5 | 1.329281 | +0.453181 | 1.000701 | 254 |
| 6 | 1.428680 | +0.500179 | 0.996224 | 254 |
| 7 | 1.539292 | +0.567508 | 0.998925 | 254 |

По RMSE CatBoost лучше RW на h: **4, 6, 7**, лучше Ridge Core v2 на h: **1, 2, 3, 4, 5, 6, 7**. Это описательное сравнение. Полные RMSE/MAE/bias/RMSE-RW/n и границы дат для всех моделей: [metrics.csv](../results/catboost_core_v2/metrics.csv). Сохранённые baseline-прогнозы не переобучались и не перезаписывались.

## HAC/Holm

До запуска зафиксировано одно семейство 21 двустороннего сравнения: CatBoost против RW, drift и Ridge Core v2 × семь h. Потери — квадрат ценовой ошибки, знак разности — CatBoost минус comparator. Newey–West/Bartlett, lag=max(20,h−1), поправка n/(n−1), normal approximation. Holm применяется совместно ко всем 21; 95% CI в таблице нескорректированы.

- Против RW: значимых улучшений нет; значимых ухудшений нет.
- Против RW_DRIFT: значимых улучшений нет; значимых ухудшений нет.
- Против Ridge_Core_v2: значимых улучшений нет; значимых ухудшений нет.

Полная таблица SE/CI/p/Holm: [hac_comparisons.csv](../results/catboost_core_v2/hac_comparisons.csv). Отсутствие значимости не доказывает эквивалентность. Одного повторно используемого validation-года, перекрывающихся горизонтов и асимптотической процедуры недостаточно для сильных выводов. Это семейство не исправляет множественность всей истории предыдущих исследовательских итераций.

## Модель, данные и настройки

Core v2 — те же 10 числовых признаков: `open, high, low, close, volume, cbr_usdrub_effective, swaprate_prev_session, swaprate_current, key_rate_pct_effective, spread_close_minus_cbr`. Использован неизменённый frozen-v2 loader с проверкой источника/дат/SHA256 и неизменённый торговый календарь. Горизонты — торговые сессии с сохранением календарных отверстий.

Финальный `swaprate_current[t]` считается известным после закрытия t только по допущению команды **ASSUMED_AVAILABLE**, не VERIFIED. Другие ограничения conditional EOD / pseudo-PIT и внешних vintages остаются. Предыдущий funding имеет 154 начальных NaN, текущий — 153; медиана обучается заново на допустимых train-парах, реальные нули сохранены. Масштабирование не вводилось; all-missing train-колонка вызывает ошибку.

Для каждого h — отдельная direct CatBoostRegressor, target `C[t+h]−C[t]`, прогноз цены `C[t]+prediction`. На каждом origin expanding/prequential train содержит только пары с `target_date ≤ origin_date`, включая уже созревшие исходы 2025. Imputer использует те же строки. Никаких новых лагов/rolling, отбора признаков или новых моделей.

Зафиксированные параметры: iterations=100, learning_rate=0.03, random_seed=240112, thread_count=1, bootstrap_type=No, loss_function=RMSE. Сетка: depth={2,3}, l2_leaf_reg={10,30}, ровно четыре кандидата. Настройка — только на внутренних origins I/II полугодий 2023 и I полугодия 2024: prequential **внутри** каждого fold, все tuning-targets строго раньше первого внешнего origin. Каждый кандидат имеет одинаковые 378 inner-пары на h. Выбор по pooled ценовому RMSE, tie-break по исходному порядку сетки.

Все семь выборов параметров сохранены до первого внешнего прогноза. Внешние прогнозы сохраняются без actual до расчёта метрик. Основной validation не используется ни для выбора, ни для early stopping: `fit(..., use_best_model=False)` вызывается без eval_set. См. [официальное описание fit](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit).

Выбор параметров по h:

| h | depth | l2_leaf_reg | Inner RMSE | Inner n |
| --- | --- | --- | --- | --- |
| 1 | 2 | 30 | 0.795532 | 378 |
| 2 | 3 | 30 | 1.137965 | 378 |
| 3 | 2 | 30 | 1.415449 | 378 |
| 4 | 3 | 30 | 1.672641 | 378 |
| 5 | 2 | 30 | 1.890794 | 378 |
| 6 | 2 | 30 | 2.099952 | 378 |
| 7 | 2 | 30 | 2.304646 | 378 |

## Feature importance

Сохранены PredictionValuesChange для каждого внешнего окна и среднее/std/min/max/rank по h. Они вычислены из обученных деревьев с train leaf weights, без использования validation для отбора признаков. Это **не permutation importance, не статистическая значимость и не причинный эффект**. OHLC коррелированы, а spread точно связан с close/CBR; деревья могут распределять важность между взаимозаменяемыми признаками. Важность сама по себе не доказывает добавочный OOS-сигнал. [Официальное описание](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_feature_importance).

Два первых признака по средней importance каждого h:

| h | feature | Mean importance | Std по origins |
| --- | --- | --- | --- |
| 1 | cbr_usdrub_effective | 17.9653 | 2.5562 |
| 1 | volume | 17.8369 | 2.8524 |
| 2 | volume | 21.3406 | 2.3665 |
| 2 | key_rate_pct_effective | 19.1476 | 1.6973 |
| 3 | volume | 21.7642 | 5.4470 |
| 3 | key_rate_pct_effective | 18.7142 | 4.1397 |
| 4 | key_rate_pct_effective | 19.7326 | 2.1631 |
| 4 | swaprate_prev_session | 19.5717 | 3.9519 |
| 5 | swaprate_prev_session | 19.1418 | 5.5429 |
| 5 | key_rate_pct_effective | 18.3857 | 1.9298 |
| 6 | key_rate_pct_effective | 21.0253 | 2.9124 |
| 6 | swaprate_prev_session | 18.5116 | 4.7837 |
| 7 | key_rate_pct_effective | 26.1077 | 2.6517 |
| 7 | swaprate_current | 14.6341 | 3.9895 |

## Артефакты, тесты и воспроизведение

`results/catboost_core_v2/`: root OOS/scored predictions, все метрики и HAC, feature importance, design и manifest. В `h1`…`h7`: все inner predictions/metrics, selection, OOS predictions, training windows с медианами, importance по origins, финальная модель `.cbm` с feature-order/imputer/resolved parameters и runtime. Сохраняется последняя модель каждого h; все промежуточные fit воспроизводятся по коду, выбранным параметрам и ledger. Веса сериализованных моделей могут содержать служебные метаданные; воспроизводимость проверяется по численным прогнозам, а не по совпадению бинарного model_guid.

Код до запуска зафиксирован коммитом `7b80cd3`; CatBoost `1.2.10`. Manifest сохраняет source hashes, версии пакетов, SHA256 всех защищённых данных/старых результатов и SHA256 новых артефактов. Исторические Core/Ridge/RW/AR/drift/proposal/backtest protocol не изменялись. [Verification](../results/catboost_core_v2/verification.json) проверяет все 1778 train-окон и повторяет first/middle/last origin каждого h (21 refit), включая medians/importance и загрузку семи `.cbm`.

10 новых CatBoost-тестов прошли; полный unittest — 102 теста, 101 прошёл, 1 пропущен. Проверяются maturity h=1…7, future invariance, train-only imputation/zeros, tuning до validation, common keys, отсутствие eval_set/early stopping, seed и model roundtrip. B3/B4 compare и git diff --check прошли.

Воспроизведение (новый пустой output-root; заранее установить `requirements-catboost.txt`):

```powershell
python scripts/run_catboost_experiment.py --stage init --output-root results/catboost_reproduction
1..7 | ForEach-Object { python scripts/run_catboost_experiment.py --stage tune --horizon $_ --output-root results/catboost_reproduction }
1..7 | ForEach-Object { python scripts/run_catboost_experiment.py --stage forecast --horizon $_ --output-root results/catboost_reproduction }
python scripts/run_catboost_experiment.py --stage summarize --output-root results/catboost_reproduction
python scripts/verify_catboost_experiment.py
python -m unittest discover -s tests -p test_catboost_direct.py -v
```

Tuning и forecast разных h могут исполняться параллельно отдельными процессами с thread_count=1; порядок вычисления h не меняет процедуру. Verifier проверяет канонический сохранённый запуск. Отчёт заново собирается `python scripts/build_catboost_report.py` без переобучения.

## Решение после эксперимента

В этом пилоте нелинейная модель снизила RMSE относительно Ridge на всех h, но после Holm статистически подтверждённого преимущества нет. Против RW снижения только на h=4,6,7 и составляют менее 0.4%; заявлять победу над RW оснований нет. Это не доказывает отсутствия нелинейного сигнала вообще: проверена только заранее объявленная небольшая спецификация.

Результат относится к повторно используемому validation 2025. Даже снижение RMSE или малый adjusted p здесь не являются независимым подтверждением для применения модели. Независимый test не использовался для моделирования. Сетка после просмотра результатов не расширяется; дальнейший подбор и новые модели автоматически не запускаются.
