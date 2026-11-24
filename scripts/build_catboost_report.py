"""Build the fixed experiment report from saved results; no training or tuning."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def horizons(values):
    return ', '.join(map(str,values)) if len(values) else 'нет'


def main():
    output = ROOT/'results/catboost_core_v2'
    metrics = pd.read_csv(output/'metrics.csv')
    hac = pd.read_csv(output/'hac_comparisons.csv')
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    cb = metrics[metrics.model_id.eq('CatBoost_Core_v2')].sort_values('horizon')
    importance = pd.read_csv(output/'feature_importance.csv')
    models = ['RW','AR1','RW_DRIFT','Ridge_Core_v2','CatBoost_Core_v2']
    rows = []
    for h in range(1,8):
        group = metrics[metrics.horizon.eq(h)].set_index('model_id')
        rows.append([h]+[f'{group.loc[m,"rmse"]:.6f}' for m in models])
    report = '''# CatBoost Core v2: exploratory validation 2025

Эксперимент выполнен 2026-11-24 в `feature/direct-ridge-experiments`. **2025 год уже использовался в Ridge v1/v2 и drift; результаты exploratory и требуют независимого test.** Новых признаков и расширения сетки после просмотра validation нет. 2026 не использован для обучения, настройки или оценки.

## Результаты на общих датах

RMSE в RUB/USD; на каждом h ровно 254 одинаковые тройки `(origin_date, target_date, horizon)` у всех пяти моделей. Targets: 2025-01-03…2025-12-30; первые origins лежат в декабре 2024 года.

'''
    report += table(['h','RW','AR(1)','RW drift','Ridge Core v2','CatBoost Core v2'],rows)
    report += '\n\nCatBoost: остальные ценовые метрики (`bias = prediction − actual`):\n\n'
    report += table(['h','MAE','bias','RMSE/RW','n'],[[int(r.horizon),f'{r.mae:.6f}',f'{r.bias:+.6f}',
        f'{r.rmse_ratio_vs_rw:.6f}',int(r.n)] for r in cb.itertuples()])
    rw = metrics[metrics.model_id.eq('RW')].set_index('horizon')
    ridge = metrics[metrics.model_id.eq('Ridge_Core_v2')].set_index('horizon')
    wins_rw = [int(r.horizon) for r in cb.itertuples() if r.rmse < rw.loc[r.horizon,'rmse']]
    wins_ridge = [int(r.horizon) for r in cb.itertuples() if r.rmse < ridge.loc[r.horizon,'rmse']]
    report += f'\n\nПо RMSE CatBoost лучше RW на h: **{horizons(wins_rw)}**, лучше Ridge Core v2 на h: **{horizons(wins_ridge)}**. '
    report += 'Это описательное сравнение. Полные RMSE/MAE/bias/RMSE-RW/n и границы дат для всех моделей: [metrics.csv](../results/catboost_core_v2/metrics.csv). Сохранённые baseline-прогнозы не переобучались и не перезаписывались.\n\n'
    report += '## HAC/Holm\n\nДо запуска зафиксировано одно семейство 21 двустороннего сравнения: CatBoost против RW, drift и Ridge Core v2 × семь h. Потери — квадрат ценовой ошибки, знак разности — CatBoost минус comparator. Newey–West/Bartlett, lag=max(20,h−1), поправка n/(n−1), normal approximation. Holm применяется совместно ко всем 21; 95% CI в таблице нескорректированы.\n\n'
    for name in ['RW','RW_DRIFT','Ridge_Core_v2']:
        group = hac[hac.versus.eq(name)]
        better = group[(group.mean_squared_loss_difference < 0)&(group.holm_p < .05)].horizon.tolist()
        worse = group[(group.mean_squared_loss_difference > 0)&(group.holm_p < .05)].horizon.tolist()
        better_text = f'подтверждённое улучшение на h={horizons(better)}' if better else 'значимых улучшений нет'
        worse_text = f'подтверждённое ухудшение на h={horizons(worse)}' if worse else 'значимых ухудшений нет'
        report += f'- Против {name}: {better_text}; {worse_text}.\n'
    report += '\nПолная таблица SE/CI/p/Holm: [hac_comparisons.csv](../results/catboost_core_v2/hac_comparisons.csv). Отсутствие значимости не доказывает эквивалентность. Одного повторно используемого validation-года, перекрывающихся горизонтов и асимптотической процедуры недостаточно для сильных выводов. Это семейство не исправляет множественность всей истории предыдущих исследовательских итераций.\n\n'
    report += '''## Модель, данные и настройки

Core v2 — те же 10 числовых признаков: `open, high, low, close, volume, cbr_usdrub_effective, swaprate_prev_session, swaprate_current, key_rate_pct_effective, spread_close_minus_cbr`. Использован неизменённый frozen-v2 loader с проверкой источника/дат/SHA256 и неизменённый торговый календарь. Горизонты — торговые сессии с сохранением календарных отверстий.

Финальный `swaprate_current[t]` считается известным после закрытия t только по допущению команды **ASSUMED_AVAILABLE**, не VERIFIED. Другие ограничения conditional EOD / pseudo-PIT и внешних vintages остаются. Предыдущий funding имеет 154 начальных NaN, текущий — 153; медиана обучается заново на допустимых train-парах, реальные нули сохранены. Масштабирование не вводилось; all-missing train-колонка вызывает ошибку.

Для каждого h — отдельная direct CatBoostRegressor, target `C[t+h]−C[t]`, прогноз цены `C[t]+prediction`. На каждом origin expanding/prequential train содержит только пары с `target_date ≤ origin_date`, включая уже созревшие исходы 2025. Imputer использует те же строки. Никаких новых лагов/rolling, отбора признаков или новых моделей.

Зафиксированные параметры: iterations=100, learning_rate=0.03, random_seed=240112, thread_count=1, bootstrap_type=No, loss_function=RMSE. Сетка: depth={2,3}, l2_leaf_reg={10,30}, ровно четыре кандидата. Настройка — только на внутренних origins I/II полугодий 2023 и I полугодия 2024: prequential **внутри** каждого fold, все tuning-targets строго раньше первого внешнего origin. Каждый кандидат имеет одинаковые 378 inner-пары на h. Выбор по pooled ценовому RMSE, tie-break по исходному порядку сетки.

Все семь выборов параметров сохранены до первого внешнего прогноза. Внешние прогнозы сохраняются без actual до расчёта метрик. Основной validation не используется ни для выбора, ни для early stopping: `fit(..., use_best_model=False)` вызывается без eval_set. См. [официальное описание fit](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit).

Выбор параметров по h:

'''
    selections = []
    for h in range(1,8):
        selected = manifest['selected_parameters'][str(h)]['parameters']
        inner = pd.read_csv(output/f'h{h}'/'inner_metrics.csv')
        winner = inner[inner.selected].iloc[0]
        selections.append([h,selected['depth'],selected['l2_leaf_reg'],f'{winner.rmse:.6f}',int(winner.n)])
    report += table(['h','depth','l2_leaf_reg','Inner RMSE','Inner n'],selections)
    report += '''

## Feature importance

Сохранены PredictionValuesChange для каждого внешнего окна и среднее/std/min/max/rank по h. Они вычислены из обученных деревьев с train leaf weights, без использования validation для отбора признаков. Это **не permutation importance, не статистическая значимость и не причинный эффект**. OHLC коррелированы, а spread точно связан с close/CBR; деревья могут распределять важность между взаимозаменяемыми признаками. Важность сама по себе не доказывает добавочный OOS-сигнал. [Официальное описание](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_feature_importance).

Два первых признака по средней importance каждого h:

'''
    report += table(['h','feature','Mean importance','Std по origins'],[[int(r.horizon),r.feature,f'{r.mean:.4f}',f'{r.std:.4f}']
        for r in importance.sort_values(['horizon','mean'],ascending=[True,False]).groupby('horizon').head(2).itertuples()])
    report += '''

## Артефакты, тесты и воспроизведение

`results/catboost_core_v2/`: root OOS/scored predictions, все метрики и HAC, feature importance, design и manifest. В `h1`…`h7`: все inner predictions/metrics, selection, OOS predictions, training windows с медианами, importance по origins, финальная модель `.cbm` с feature-order/imputer/resolved parameters и runtime. Сохраняется последняя модель каждого h; все промежуточные fit воспроизводятся по коду, выбранным параметрам и ledger. Веса сериализованных моделей могут содержать служебные метаданные; воспроизводимость проверяется по численным прогнозам, а не по совпадению бинарного model_guid.

'''
    report += f"Код до запуска зафиксирован коммитом `{manifest['git_sha'][:7]}`; CatBoost `{manifest['packages']['catboost']}`. Manifest сохраняет source hashes, версии пакетов, SHA256 всех защищённых данных/старых результатов и SHA256 новых артефактов. Исторические Core/Ridge/RW/AR/drift/proposal/backtest protocol не изменялись. [Verification](../results/catboost_core_v2/verification.json) проверяет все 1778 train-окон и повторяет first/middle/last origin каждого h (21 refit), включая medians/importance и загрузку семи `.cbm`.\n\n"
    report += '''10 новых CatBoost-тестов прошли; полный unittest — 102 теста, 101 прошёл, 1 пропущен. Проверяются maturity h=1…7, future invariance, train-only imputation/zeros, tuning до validation, common keys, отсутствие eval_set/early stopping, seed и model roundtrip. B3/B4 compare и git diff --check прошли.

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
'''
    (ROOT/'reports/catboost_core_v2.md').write_text(report,encoding='utf-8',newline='\n')


if __name__ == '__main__': main()
