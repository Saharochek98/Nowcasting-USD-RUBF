"""Refresh numerical report tables from saved validation results, without refits."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    output = ROOT/'results/ridge_experiments_v2'
    metrics = pd.read_csv(output/'metrics_all.csv')
    ridge = metrics[metrics.comparator.eq('ridge')]
    models = ['price_only', 'core_v2', 'core_v2_no_spread', 'B1_v2', 'B2_v2']
    base = metrics[(metrics.model_id == 'core_v2') & (metrics.regime == 'prequential') &
                   (metrics['transform'] == 'delta')]
    rows = []
    for h in range(1, 8):
        group = base[base.horizon.eq(h)]
        values = [group[group.comparator.eq(c)].rmse.iloc[0] for c in ['B0', 'B1']]
        values += [ridge[(ridge.model_id == model) & (ridge.regime == 'prequential') &
                         (ridge['transform'] == 'delta') & ridge.horizon.eq(h)].rmse.iloc[0]
                   for model in models]
        rows.append([h]+[f'{v:.5f}' for v in values])
    text = '**Prequential delta: RMSE в RUB/USD**, n=254 для каждой строки/модели.\n\n'
    text += table(['h', 'RW', 'AR(1)']+models, rows)
    text += '''

Core v2 имеет MAE от 0.57338 до 1.59281 и положительный bias от +0.10843 до +0.76846 (h=1→7). Core немного лучше price-only, но уступает RW на всех h; против AR(1) RMSE ниже только на h=3–4. Удаление spread чуть ухудшает RMSE на всех h; добавление B1/B2 даёт очень малые, разнонаправленные изменения. Ни одно такое различие не подтверждено после Holm. Полные MAE/bias/RMSE/RW всех моделей и обоих режимов: [metrics_all.csv](../results/ridge_experiments_v2/metrics_all.csv).

**Core v2: сопоставление режимов и targets, RMSE/RW.**

'''
    rows = []
    for regime in ['frozen', 'prequential']:
        for transform in ['delta', 'log_return']:
            group = ridge[(ridge.model_id == 'core_v2') & (ridge.regime == regime) &
                          (ridge['transform'] == transform)].sort_values('horizon')
            rows.append([regime, transform]+[f'{v:.5f}' for v in group.rmse_ratio_vs_rw])
    text += table(['Режим', 'Target']+[f'h={h}' for h in range(1, 8)], rows)
    text += '''

Prequential улучшает frozen RMSE для каждого сценария/target/h, но лог-доходности хуже delta во всех таких сопоставлениях. Это описательные результаты; ни в основном семействе 98 тестов (минимальный Holm p=0.17550), ни в исследовательском семействе 434 тестов нет adjusted p<0.05. Отсутствие значимости не означает эквивалентность моделей. Полная таблица с общими n, RMSE-разностями, HAC/CI/p: [comparisons_hac_holm.csv](../results/ridge_experiments_v2/comparisons_hac_holm.csv).

Во всех 70 комбинациях сценарий×h×target выбрано **alpha=100000 — снова верхняя граница сетки**. Это не доказательство оптимальности такого штрафа; сетка не расширялась после просмотра v2 validation. Сильное сжатие делает различия признаковых сценариев очень малыми. Сравнение v1 с v2 смешивает изменение grid, Core и режима; для эффекта режима следует использовать matched frozen/prequential v2, а не старый R2 против нового Core.

На фиксированном первом окне Core delta при alpha=100000 RMS полной коррекции равен 0.04408 на h=1 и 0.33223 на h=7; RMS только slope — 0.000745 и 0.006229. В пределе alpha→∞ остаётся drift +0.043695 и +0.337627 соответственно. Таким образом, наклоны почти обнулены, но модель **не сходится к точному RW с нулевой коррекцией** из-за свободного intercept. Для log-return наблюдается тот же эффект; [regularization_path.csv](../results/ridge_experiments_v2/regularization_path.csv) содержит все h и alpha, включая предел.

**Диагностика prequential delta, h=7**: признак с наименьшим средним rank |β| и отдельно максимальная permutation ΔMSE. β усреднён по origins; доля отрицательного знака относится к указанному коэффициенту.

'''
    coefficients = pd.read_csv(output/'prequential_delta/coefficient_stability.csv')
    importance = pd.read_csv(output/'prequential_delta/permutation_importance.csv')
    rows = []
    for model in models:
        a = coefficients[coefficients.model_id.eq(model) & coefficients.horizon.eq(7)].nsmallest(1, 'mean_abs_rank').iloc[0]
        b = importance[importance.model_id.eq(model) & importance.horizon.eq(7)].nlargest(1, 'mse_increase_mean').iloc[0]
        rows.append([model, a.feature, f'{a.coefficient_mean:+.7f}', f'{a.negative_share:.3f}',
                     f'{a.rank_std:.3f}', b.feature, f'{b.mse_increase_mean:.6f}'])
    text += table(['Сценарий', 'Коэффициент', 'Средний β', 'Доля β<0', 'Std rank',
                   'Permutation top', 'ΔMSE'], rows)
    text += '''

Для других признаков знаки менялись: суммарно по всем признакам и h зарегистрировано 17 смен в price-only, 24 в Core, 22 в no_spread, 24 в B1 и 30 в B2. Эти суммы описывают динамику, а не число независимых свидетельств. Для каждого h сохранены полные coefficients/stability/importance, а не только приведённые примеры. Текущий funding имеет высокую диагностическую importance на h=7, но это не выделяет его добавочный причинный эффект и остаётся условным на ASSUMED_AVAILABLE.

Всего сохранено **35 560 OOS-прогнозов**: 5 сценариев × 7 h × 2 режима × 2 target × 254 пары.

'''
    report = ROOT/'reports/ridge_experiments_v2.md'
    original = report.read_text(encoding='utf-8')
    start = original.index('## Численные результаты')
    end = original.index('## Интерпретация и диагностика')
    report.write_text(original[:start]+'## Численные результаты\n\n'+text+original[end:], encoding='utf-8')


if __name__ == '__main__':
    main()
