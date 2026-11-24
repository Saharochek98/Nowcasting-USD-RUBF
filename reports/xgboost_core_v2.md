# XGBoost Core v2 — один исследовательский эксперимент

Дата: 2026-11-24. Ветка: `feature/direct-ridge-experiments`. **Exploratory validation:** 2025 год уже использовался для Ridge, drift и CatBoost. Любой положительный результат здесь требует независимой оценки; final test 2026 не использовался для обучения, настройки или оценки.

## Результаты

RMSE в RUB/USD; для всех моделей на каждом h — ровно 254 одинаковые тройки `(origin_date, target_date, horizon)`, targets 2025-01-03…2025-12-30. Первые origins лежат в декабре 2024 года.

| h | RW | AR(1) | RW drift | Ridge Core v2 | CatBoost | XGBoost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.772440 | 0.774045 | 0.776165 | 0.775978 | 0.773605 | 0.782843 |
| 2 | 1.135664 | 1.144051 | 1.145666 | 1.144966 | 1.141679 | 1.152077 |
| 3 | 1.404904 | 1.422296 | 1.422904 | 1.421418 | 1.414397 | 1.444125 |
| 4 | 1.617355 | 1.644489 | 1.644651 | 1.642514 | 1.614577 | 1.646834 |
| 5 | 1.790585 | 1.827089 | 1.828883 | 1.827823 | 1.791839 | 1.834601 |
| 6 | 1.965425 | 2.012490 | 2.015911 | 2.016073 | 1.958003 | 1.998761 |
| 7 | 2.147724 | 2.207770 | 2.211516 | 2.212588 | 2.145415 | 2.204440 |

Остальные метрики XGBoost (bias = prediction − actual):

| h | MAE | bias | RMSE/RW | n |
| --- | --- | --- | --- | --- |
| 1 | 0.573664 | +0.099069 | 1.013468 | 254 |
| 2 | 0.848287 | +0.194518 | 1.014453 | 254 |
| 3 | 1.071289 | +0.324533 | 1.027917 | 254 |
| 4 | 1.224443 | +0.399737 | 1.018227 | 254 |
| 5 | 1.361894 | +0.488167 | 1.024582 | 254 |
| 6 | 1.452257 | +0.536801 | 1.016961 | 254 |
| 7 | 1.569599 | +0.637070 | 1.026408 | 254 |

Полные метрики всех шести моделей: [metrics.csv](../results/xgboost_core_v2/metrics.csv). Сохранённые baseline-прогнозы не переобучались и не перезаписывались.

- Меньший RMSE против RW: ни на одном h.
- Меньший RMSE против Ridge_Core_v2: 6, 7.
- Меньший RMSE против CatBoost_Core_v2: ни на одном h.


## HAC/Newey–West и Holm

До validation объявлено одно семейство **21** двустороннего сравнения: XGBoost против RW, Ridge Core v2 и CatBoost × семь h. Потери — квадраты ценовых ошибок, разность XGBoost−comparator. Newey–West/Bartlett, lag=max(20,h−1), поправка n/(n−1), normal approximation. Holm совместно для всех 21; CSV содержит нескорректированные 95% CI и adjusted p.

- Против RW: подтверждённого улучшения нет; подтверждённого ухудшения нет.
- Против Ridge_Core_v2: подтверждённого улучшения нет; подтверждённого ухудшения нет.
- Против CatBoost_Core_v2: подтверждённого улучшения нет; подтверждённого ухудшения нет.

Полная таблица: [hac_comparisons.csv](../results/xgboost_core_v2/hac_comparisons.csv). Отсутствие значимости не доказывает эквивалентность. Один повторно использованный validation-год, перекрытие горизонтов и асимптотические SE ограничивают выводы. Семейство не учитывает всю множественность предыдущих исследовательских итераций.

## Протокол и неизменность информации

Те же 10 Core v2 признаков и frozen-v2 loader, что у CatBoost: OHLCV, CBR, предыдущий и текущий funding, key rate и spread. `swaprate_current[t]` предполагается известным после закрытия t: **ASSUMED_AVAILABLE**, не VERIFIED. Сохраняются ограничения conditional EOD / pseudo-PIT. Данные, старые Core, результаты Ridge/CatBoost/RW/AR/drift и backtest protocol не изменены; проверены SHA256 всех защищённых файлов.

Семь отдельных XGBRegressor прогнозируют `C[t+h]−C[t]`; цена восстанавливается добавлением известного C[t]. Expanding/prequential на каждом origin, включая созревшие исходы 2025, но всегда `target_date ≤ origin_date`. Медиана заново fit на тех же допустимых train-парах; настоящие нули не заменяются NaN, all-missing train-колонка отклоняется. Предыдущий funding имеет 154 начальных пропуска, текущий — 153. Масштабирование, новые признаки/лаги/rolling и отбор переменных не вводились.

Календарь тот же реконструированный provisional; позиции не сжимаются при пропусках. Протокол использует неизменённые общие функции CatBoost для загрузки, календаря и maturity; совпадение всех inner-даты/train bounds программно проверено против сохранённого CatBoost ledger, без использования его ошибок для tuning.

Четыре кандидата: max_depth={2,3}, reg_lambda={10,30}. Фиксированы n_estimators=100, learning_rate=0.03, min_child_weight=5, subsample=colsample_bytree=1, objective=reg:squarederror, tree_method=hist, random_state=240112, n_jobs=1. CPU; остальные native defaults сохранены в resolved booster config. Сетка не расширялась. [Описание параметров XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

Те же три внутренних окна origins: I/II полугодия 2023 и I полугодие 2024, prequential внутри fold, по 378 пар для каждого кандидата/h. Все tuning-targets строго раньше первого внешнего origin. Минимум 100 созревших direct-пар. Выбор — pooled ценовой RMSE, tie-break по порядку сетки. Все семь selections сохранены до первого внешнего прогноза; fit не получает eval_set или early stopping. Основной validation не используется для настройки. Все OOS сохраняются без actual до расчёта метрик.

Выбранные параметры:

| h | max_depth | reg_lambda | Inner RMSE | Inner n |
| --- | --- | --- | --- | --- |
| 1 | 2 | 30 | 0.796938 | 378 |
| 2 | 2 | 30 | 1.147096 | 378 |
| 3 | 2 | 30 | 1.421713 | 378 |
| 4 | 2 | 30 | 1.681469 | 378 |
| 5 | 2 | 30 | 1.885619 | 378 |
| 6 | 2 | 30 | 2.099512 | 378 |
| 7 | 2 | 30 | 2.342264 | 378 |

## Gain и сравнение признаков с CatBoost

Для каждого origin сохранён **raw gain** — среднее улучшение objective по split данного признака; неиспользованный признак имеет ноль. Также сохранены доля gain в сумме по признакам и rank в каждом окне. Агрегаты содержат mean/std raw gain и доли, mean/std rank, частоту положительного gain. [Определение importance в API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

CatBoost использует PredictionValuesChange; XGBoost — gain. Их абсолютные величины **не сопоставимы**. Ни один из методов не является статистической значимостью или причинным влиянием. OHLC коррелированы, spread точно зависит от close/CBR; важность может распределяться между заменяющими друг друга признаками и зависит от разбиений. Эти train-based диагностики не использовались для отбора признаков.

Ведущие признаки по средней нормализованной gain XGBoost и по средней PredictionValuesChange CatBoost:

| h | XGBoost top-3 | CatBoost top-3 | Общих из трёх |
| --- | --- | --- | --- |
| 1 | volume, key_rate_pct_effective, cbr_usdrub_effective | cbr_usdrub_effective, volume, key_rate_pct_effective | 3 |
| 2 | volume, high, key_rate_pct_effective | volume, key_rate_pct_effective, high | 3 |
| 3 | high, key_rate_pct_effective, volume | volume, key_rate_pct_effective, close | 2 |
| 4 | high, close, key_rate_pct_effective | key_rate_pct_effective, swaprate_prev_session, low | 1 |
| 5 | high, swaprate_prev_session, key_rate_pct_effective | swaprate_prev_session, key_rate_pct_effective, close | 2 |
| 6 | swaprate_prev_session, high, key_rate_pct_effective | key_rate_pct_effective, swaprate_prev_session, close | 2 |
| 7 | swaprate_current, key_rate_pct_effective, high | key_rate_pct_effective, swaprate_current, close | 2 |

Все окна — `h*/gain_windows.csv`; агрегаты — [gain_importance.csv](../results/xgboost_core_v2/gain_importance.csv); сопоставление всех признаков — [importance_vs_catboost.csv](../results/xgboost_core_v2/importance_vs_catboost.csv).

## Проверки, артефакты и воспроизводимость

До расчёта зафиксирован код `c22b366`, XGBoost 3.2.0, scikit-learn 1.9.1; другие версии и source hashes сохранены в design/manifest. Зависимости: `requirements-xgboost.txt`. В root сохранены OOS/scored predictions, metrics, HAC/Holm, gain/сравнения, manifest; в h1…h7 — inner predictions/metrics, selection, OOS, training windows с медианами, gain по окнам, runtime и последняя модель UBJ с imputer/schema/resolved config.

10 новых XGBoost-тестов прошли; полный unittest: 112 тестов, 111 прошли, 1 пропущен. Проверены leakage/maturity h=1…7, future invariance, train-only median/zeros, tuning до validation, точные common keys, seeded determinism, gain unused features и UBJ roundtrip. B3/B4 compare и diff check прошли.

[Verifier](../scripts/verify_xgboost_experiment.py) сверяет все старые хеши и новые артефакты, пересчитывает selections, проверяет все 1778 train-окон и повторяет 21 first/middle/last origin по семи h с точным совпадением predictions/imputers/gain. Отдельно проверяется загрузка семи сохранённых моделей. [verification.json](../results/xgboost_core_v2/verification.json).

```powershell
python -m pip install -r requirements-xgboost.txt
python scripts/run_xgboost_experiment.py --stage init --output-root results/xgboost_reproduction
1..7 | ForEach-Object { python scripts/run_xgboost_experiment.py --stage tune --horizon $_ --output-root results/xgboost_reproduction }
1..7 | ForEach-Object { python scripts/run_xgboost_experiment.py --stage forecast --horizon $_ --output-root results/xgboost_reproduction }
python scripts/run_xgboost_experiment.py --stage summarize --output-root results/xgboost_reproduction
python scripts/verify_xgboost_experiment.py
python -m unittest discover -s tests -p test_xgboost_direct.py -v
```

Каждый output-root должен быть пустым перед init. Горизонты могут считаться параллельно отдельными процессами, каждый fit имеет n_jobs=1; порядок h не меняет процедуру. Verifier проверяет канонический сохранённый запуск; отчёт пересобирается `python scripts/build_xgboost_report.py` без обучения.

## Научный вывод

XGBoost не снизил RMSE относительно Random Walk ни на одном горизонте. XGBoost не снизил RMSE относительно CatBoost ни на одном горизонте. В этом ограниченном эксперименте меньший RMSE относительно Ridge получен на h=6, 7. После Holm подтверждённых улучшений нет. Результат не является независимым test-подтверждением и не обобщается на весь класс нелинейных моделей или другие сетки. Дальнейший подбор, расширение сетки и другие модели не запускаются; final test остаётся отдельным будущим решением.
