"""Rebuild the XGBoost report from immutable experiment tables, without training."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    output = ROOT/'results/xgboost_core_v2'
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(output/'metrics.csv')
    hac = pd.read_csv(output/'hac_comparisons.csv')
    xgb = metrics[metrics.model_id.eq('XGBoost_Core_v2')].sort_values('horizon')
    models = ['RW','AR1','RW_DRIFT','Ridge_Core_v2','CatBoost_Core_v2','XGBoost_Core_v2']
    report = '''# XGBoost Core v2 — один исследовательский эксперимент

Дата: 2026-11-24. Ветка: `feature/direct-ridge-experiments`. **Exploratory validation:** 2025 год уже использовался для Ridge, drift и CatBoost. Любой положительный результат здесь требует независимой оценки; final test 2026 не использовался для обучения, настройки или оценки.

## Результаты

RMSE в RUB/USD; для всех моделей на каждом h — ровно 254 одинаковые тройки `(origin_date, target_date, horizon)`, targets 2025-01-03…2025-12-30. Первые origins лежат в декабре 2024 года.

'''
    rows = []
    for h in range(1,8):
        group = metrics[metrics.horizon.eq(h)].set_index('model_id')
        rows.append([h]+[f'{group.loc[m,"rmse"]:.6f}' for m in models])
    report += table(['h','RW','AR(1)','RW drift','Ridge Core v2','CatBoost','XGBoost'],rows)
    report += '\n\nОстальные метрики XGBoost (bias = prediction − actual):\n\n'
    report += table(['h','MAE','bias','RMSE/RW','n'],[[int(r.horizon),f'{r.mae:.6f}',f'{r.bias:+.6f}',
        f'{r.rmse_ratio_vs_rw:.6f}',int(r.n)] for r in xgb.itertuples()])
    report += '\n\nПолные метрики всех шести моделей: [metrics.csv](../results/xgboost_core_v2/metrics.csv). Сохранённые baseline-прогнозы не переобучались и не перезаписывались.\n\n'
    wins = {}
    for versus in ['RW','Ridge_Core_v2','CatBoost_Core_v2']:
        base = metrics[metrics.model_id.eq(versus)].set_index('horizon')
        wins[versus] = [int(r.horizon) for r in xgb.itertuples() if r.rmse < base.loc[r.horizon,'rmse']]
        report += f'- Меньший RMSE против {versus}: '+(', '.join(map(str,wins[versus])) if wins[versus] else 'ни на одном h')+'.\n'
    report += '''

## HAC/Newey–West и Holm

До validation объявлено одно семейство **21** двустороннего сравнения: XGBoost против RW, Ridge Core v2 и CatBoost × семь h. Потери — квадраты ценовых ошибок, разность XGBoost−comparator. Newey–West/Bartlett, lag=max(20,h−1), поправка n/(n−1), normal approximation. Holm совместно для всех 21; CSV содержит нескорректированные 95% CI и adjusted p.

'''
    for versus in ['RW','Ridge_Core_v2','CatBoost_Core_v2']:
        group = hac[hac.versus.eq(versus)]
        better = group[(group.mean_squared_loss_difference < 0)&(group.holm_p < .05)].horizon.tolist()
        worse = group[(group.mean_squared_loss_difference > 0)&(group.holm_p < .05)].horizon.tolist()
        better_text = 'подтверждённое улучшение h='+', '.join(map(str,better)) if better else 'подтверждённого улучшения нет'
        worse_text = 'подтверждённое ухудшение h='+', '.join(map(str,worse)) if worse else 'подтверждённого ухудшения нет'
        report += f'- Против {versus}: {better_text}; {worse_text}.\n'
    report += '''
Полная таблица: [hac_comparisons.csv](../results/xgboost_core_v2/hac_comparisons.csv). Отсутствие значимости не доказывает эквивалентность. Один повторно использованный validation-год, перекрытие горизонтов и асимптотические SE ограничивают выводы. Семейство не учитывает всю множественность предыдущих исследовательских итераций.

## Протокол и неизменность информации

Те же 10 Core v2 признаков и frozen-v2 loader, что у CatBoost: OHLCV, CBR, предыдущий и текущий funding, key rate и spread. `swaprate_current[t]` предполагается известным после закрытия t: **ASSUMED_AVAILABLE**, не VERIFIED. Сохраняются ограничения conditional EOD / pseudo-PIT. Данные, старые Core, результаты Ridge/CatBoost/RW/AR/drift и backtest protocol не изменены; проверены SHA256 всех защищённых файлов.

Семь отдельных XGBRegressor прогнозируют `C[t+h]−C[t]`; цена восстанавливается добавлением известного C[t]. Expanding/prequential на каждом origin, включая созревшие исходы 2025, но всегда `target_date ≤ origin_date`. Медиана заново fit на тех же допустимых train-парах; настоящие нули не заменяются NaN, all-missing train-колонка отклоняется. Предыдущий funding имеет 154 начальных пропуска, текущий — 153. Масштабирование, новые признаки/лаги/rolling и отбор переменных не вводились.

Календарь тот же реконструированный provisional; позиции не сжимаются при пропусках. Протокол использует неизменённые общие функции CatBoost для загрузки, календаря и maturity; совпадение всех inner-даты/train bounds программно проверено против сохранённого CatBoost ledger, без использования его ошибок для tuning.

Четыре кандидата: max_depth={2,3}, reg_lambda={10,30}. Фиксированы n_estimators=100, learning_rate=0.03, min_child_weight=5, subsample=colsample_bytree=1, objective=reg:squarederror, tree_method=hist, random_state=240112, n_jobs=1. CPU; остальные native defaults сохранены в resolved booster config. Сетка не расширялась. [Описание параметров XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

Те же три внутренних окна origins: I/II полугодия 2023 и I полугодие 2024, prequential внутри fold, по 378 пар для каждого кандидата/h. Все tuning-targets строго раньше первого внешнего origin. Минимум 100 созревших direct-пар. Выбор — pooled ценовой RMSE, tie-break по порядку сетки. Все семь selections сохранены до первого внешнего прогноза; fit не получает eval_set или early stopping. Основной validation не используется для настройки. Все OOS сохраняются без actual до расчёта метрик.

Выбранные параметры:

'''
    rows = []
    for h in range(1,8):
        selected = manifest['selected_parameters'][str(h)]['parameters']
        inner = pd.read_csv(output/f'h{h}'/'inner_metrics.csv')
        winner = inner[inner.selected].iloc[0]
        rows.append([h,selected['max_depth'],selected['reg_lambda'],f'{winner.rmse:.6f}',int(winner.n)])
    report += table(['h','max_depth','reg_lambda','Inner RMSE','Inner n'],rows)
    report += '''

## Gain и сравнение признаков с CatBoost

Для каждого origin сохранён **raw gain** — среднее улучшение objective по split данного признака; неиспользованный признак имеет ноль. Также сохранены доля gain в сумме по признакам и rank в каждом окне. Агрегаты содержат mean/std raw gain и доли, mean/std rank, частоту положительного gain. [Определение importance в API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

CatBoost использует PredictionValuesChange; XGBoost — gain. Их абсолютные величины **не сопоставимы**. Ни один из методов не является статистической значимостью или причинным влиянием. OHLC коррелированы, spread точно зависит от close/CBR; важность может распределяться между заменяющими друг друга признаками и зависит от разбиений. Эти train-based диагностики не использовались для отбора признаков.

Ведущие признаки по средней нормализованной gain XGBoost и по средней PredictionValuesChange CatBoost:

'''
    importance = pd.read_csv(output/'importance_vs_catboost.csv')
    rows = []
    for h,group in importance.groupby('horizon'):
        a = group.sort_values(['gain_share_mean','feature'],ascending=[False,True]).head(3).feature.tolist()
        b = group.sort_values(['catboost_pvc_mean','feature'],ascending=[False,True]).head(3).feature.tolist()
        rows.append([h,', '.join(a),', '.join(b),len(set(a)&set(b))])
    report += table(['h','XGBoost top-3','CatBoost top-3','Общих из трёх'],rows)
    report += '''

Все окна — `h*/gain_windows.csv`; агрегаты — [gain_importance.csv](../results/xgboost_core_v2/gain_importance.csv); сопоставление всех признаков — [importance_vs_catboost.csv](../results/xgboost_core_v2/importance_vs_catboost.csv).

## Проверки, артефакты и воспроизводимость

'''
    report += f"До расчёта зафиксирован код `{manifest['git_sha'][:7]}`, XGBoost {manifest['packages']['xgboost']}, scikit-learn {manifest['packages']['scikit-learn']}; другие версии и source hashes сохранены в design/manifest. Зависимости: `requirements-xgboost.txt`. В root сохранены OOS/scored predictions, metrics, HAC/Holm, gain/сравнения, manifest; в h1…h7 — inner predictions/metrics, selection, OOS, training windows с медианами, gain по окнам, runtime и последняя модель UBJ с imputer/schema/resolved config.\n\n"
    report += '''10 новых XGBoost-тестов прошли; полный unittest: 112 тестов, 111 прошли, 1 пропущен. Проверены leakage/maturity h=1…7, future invariance, train-only median/zeros, tuning до validation, точные common keys, seeded determinism, gain unused features и UBJ roundtrip. B3/B4 compare и diff check прошли.

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

'''
    for versus,label in [('RW','Random Walk'),('CatBoost_Core_v2','CatBoost'),('Ridge_Core_v2','Ridge')]:
        hlist = wins[versus]
        report += (f'В этом ограниченном эксперименте меньший RMSE относительно {label} получен на h='+', '.join(map(str,hlist))+'. '
                   if hlist else f'XGBoost не снизил RMSE относительно {label} ни на одном горизонте. ')
    significant = hac[(hac.mean_squared_loss_difference < 0)&(hac.holm_p < .05)]
    report += ('После Holm подтверждённых улучшений нет. ' if significant.empty else 'Подтверждённые улучшения после Holm перечислены выше. ')
    report += 'Результат не является независимым test-подтверждением и не обобщается на весь класс нелинейных моделей или другие сетки. Дальнейший подбор, расширение сетки и другие модели не запускаются; final test остаётся отдельным будущим решением.\n'
    (ROOT/'reports/xgboost_core_v2.md').write_text(report,encoding='utf-8',newline='\n')


if __name__ == '__main__': main()
