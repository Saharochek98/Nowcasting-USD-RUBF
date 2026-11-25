"""Run the fixed H13 ablation, with a mandatory all-origin M0 gate before M1."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, catboost_target_ablation as ab, ridge_v2

CONFIG = ROOT / 'configs/h13_catboost_logreturn.json'


def json_write(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def write(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g', date_format='%Y-%m-%d', lineterminator='\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    names = subprocess.check_output(['git', 'ls-files', '--', 'data', 'results', 'configs'], cwd=ROOT, text=True).splitlines()
    return {name: digest(ROOT/name) for name in names
            if not name.startswith('results/h13_catboost_logreturn/') and name != 'configs/h13_catboost_logreturn.json'}


def read_predictions(path):
    return pd.read_csv(path, parse_dates=['origin_date', 'target_date'], float_precision='round_trip')


def check_original_windows(windows, source):
    for h, group in windows.groupby('horizon'):
        original = pd.read_csv(source/f'h{h}/training_windows.csv',
                               parse_dates=['origin_date', 'train_origin_start', 'train_origin_end', 'train_label_end'],
                               float_precision='round_trip')
        cols = ['origin_date', 'train_n', 'train_origin_start', 'train_origin_end', 'train_label_end']
        cols += [c+'_median' for c in ridge_v2.CORE]
        pd.testing.assert_frame_equal(group[cols].reset_index(drop=True), original[cols], check_exact=True)


def plots(paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for h in [1, 3, 5, 7]:
        group = paired[paired.horizon.eq(h)]
        actual_delta = group.actual-group.RW
        changes = [group.prediction-group.RW, group.prediction_reference-group.RW]
        limits = [min(actual_delta.min(), *(d.min() for d in changes)),
                  max(actual_delta.max(), *(d.max() for d in changes))]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        for ax, model, delta in zip(axes, ab.MODELS, changes):
            ax.scatter(actual_delta, delta, alpha=.65, s=13)
            ax.plot(limits, limits, '--', color='black', label='Ideal y=x')
            ax.set(xlabel='Actual delta (RUB/USD)', ylabel='Predicted delta (RUB/USD)',
                   title=f'{model}, h={h}', xlim=limits, ylim=limits, aspect='equal')
            ax.legend()
        fig.tight_layout()
        fig.savefig(output/f'scatter_h{h}.png', dpi=150, metadata={'Software': 'H13 CatBoost target ablation'})
        plt.close(fig)


def markdown_table(frame):
    def cell(value):
        return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] +
                     ['| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None)])


def report(tables, manifest, output):
    metrics, hac, behavior = tables['metrics'], tables['hac_holm'], tables['behavior']
    rows = []
    for h in range(1, 8):
        g = metrics[metrics.horizon.eq(h)].set_index('model_id')
        rows.append(dict(h=h, n=int(g.loc['M0_DELTA', 'n']), RMSE_M0=g.loc['M0_DELTA', 'rmse'],
                         RMSE_M1=g.loc['M1_LOGRETURN', 'rmse'], RMSE_RW=g.loc['RW', 'rmse'],
                         M1_over_M0=g.loc['M1_LOGRETURN', 'rmse']/g.loc['M0_DELTA', 'rmse'],
                         M1_over_RW=g.loc['M1_LOGRETURN', 'rmse_ratio_vs_rw']))
    text = '''# H13 — CatBoost Delta vs Log-return

Сравнивается только target при неизменных 10 признаках Core v2, frozen v2 dataset, provisional trading calendar, origins и обучающих индексах. Validation 2025 многократно использовалась: все выводы exploratory, независимое преимущество не подтверждается. 2026 не используется для обучения, настройки и оценки.

M0_DELTA: `C[t+h]-C[t]`, восстановление `C[t]+prediction_target`. M1_LOGRETURN: `log(C[t+h]/C[t])`, восстановление `C[t]*exp(prediction_target)`. Jensen correction не применяется. Target M1 минимизирует RMSE лог-доходности при обучении; оценка обеих моделей проводится по восстановленной цене.

## Контроль и протокол

'''
    text += f"Все 1778 прогнозов M0 (254 на h) совпали с оригинальным CatBoost при atol=1e-8, rtol=1e-10. Максимальное отличие цены: {manifest['m0_max_abs_difference']:.12g}. Все оригинальные окна, n и 10 train-only медиан воспроизведены точно. До завершения контроля всех h обучение M1 не начиналось.\n\n"
    text += '''Expanding/prequential refit на каждом origin; только метки с target_date≤origin_date. Train начинается с исходной первой сессии: прогрев на 19 сессий и шестёрка H11 не применяются. Индексы M0/M1 пересчитываются независимо и сравниваются целиком; сохраняются SHA256 массивов int64 little-endian, границы и n. Медианы обеих моделей сравниваются точно. Никаких eval_set, early stopping или нового подбора параметров.

Фиксировано: iterations=100, learning_rate=0.03, bootstrap_type=No, random_seed=240112, thread_count=1, loss_function=RMSE. Depth и l2_leaf_reg взяты из оригинальных h*/selection.json для каждого h и одинаковы между M0/M1. Все параметры и порядок 10 признаков приведены в manifest.

Правило невалидных значений зафиксировано до расчёта: при любой неположительной или нечисловой CLOSE, нечисловом доступном target или восстановленном прогнозе весь эксперимент останавливается. Для восстановленной log-return цены также требуется строгое значение >0 (включая контроль underflow). Нет clipping, fallback или удаления прогнозов по факту ошибки. Терминальные h недоступных меток не входят в обучение по календарной зрелости.

## OOS качество восстановленной цены

'''
    text += markdown_table(pd.DataFrame(rows))+'\n\n'
    wins = metrics.pivot(index='horizon', columns='model_id', values='rmse')
    text += f"M1 имеет меньший RMSE, чем M0, на h={wins.index[wins.M1_LOGRETURN < wins.M0_DELTA].tolist()}; меньший RMSE, чем RW, на h={wins.index[wins.M1_LOGRETURN < wins.RW].tolist()}. Это описательное сравнение; MAE, bias=prediction−actual, RMSE/RW и n всех трёх моделей находятся в [metrics.csv](../results/h13_catboost_logreturn/metrics.csv).\n\n"
    text += 'Изменение RMSE M1 относительно M0 по h=1…7, %: ' + ', '.join(
        f'{100*(row["M1_over_M0"]-1):+.4f}' for row in rows) + '.\n\n'
    text += '''## Основной контраст HAC/Holm

`d=(M1−actual)^2−(M0−actual)^2`; отрицательное среднее благоприятно для M1. Newey–West, Bartlett, lag=max(20,h−1), поправка n/(n−1), двусторонняя нормальная аппроксимация. Holm применяется совместно к ровно семи заранее объявленным сравнениям h=1…7. Приведённые 95% CI не скорректированы за множественность. Сравнение с RW описательное и не добавляет тестов в это семейство.

'''
    text += markdown_table(hac[['horizon', 'mean_squared_loss_difference', 'hac_se', 'ci_low', 'ci_high', 'p_two_sided', 'holm_p']])+'\n\n'
    significant = hac[hac.holm_p < .05]
    text += f"После Holm при alpha=0.05 значимые горизонты: {significant.horizon.tolist()}. Отсутствие значимости не доказывает эквивалентность.\n\n"
    nominal = hac[hac.p_two_sided < .05]
    text += f"Без поправки p<0.05 на h={nominal.horizon.tolist()}; эти результаты нельзя считать значимыми после Holm, если скорректированное p≥0.05.\n\n"
    text += '## Амплитуда, bias и направление\n\n'
    text += markdown_table(behavior[behavior.model_id.ne('RW')][['model_id', 'horizon', 'predicted_delta_mean',
                            'predicted_delta_std', 'actual_delta_mean', 'actual_delta_std', 'bias', 'std_ratio',
                            'correlation', 'direction_accuracy']])+'\n\n'
    b0 = behavior[behavior.model_id.eq('M0_DELTA')].set_index('horizon')
    b1 = behavior[behavior.model_id.eq('M1_LOGRETURN')].set_index('horizon')
    smaller = b1.index[b1.predicted_delta_std < b0.predicted_delta_std].tolist()
    text += f"Std прогнозируемого изменения M1 ниже M0 на h={smaller}. Для M1 отношение std(predicted_delta)/std(actual_delta) составляет {100*b1.std_ratio.min():.2f}–{100*b1.std_ratio.max():.2f}%, для M0 — {100*b0.std_ratio.min():.2f}–{100*b0.std_ratio.max():.2f}%. Смена target не устраняет ранее обнаруженную малую амплитуду. Корреляция M1 отрицательна на h={b1.index[b1.correlation < 0].tolist()}. Большая дисперсия прогноза не используется как критерий выбора.\n\n"
    text += '''Std рассчитывается с ddof=1. Направление восстановленной цены определяется по `sign(prediction−C[t])`, фактическое — по `sign(actual−C[t])`; ноль — точное равенство нулю, без подбора порога. Нулевые фактические и прогнозные изменения, верные нули и точность без нулевых actual приведены отдельно в behavior.csv. Полные 3×3 матрицы для каждой модели и h — в direction_confusion.csv. RW всегда прогнозирует ноль; его корреляция не определена. Основной критерий — OOS RMSE цены: увеличение дисперсии прогноза само по себе не означает улучшения.

'''
    for h in [1, 3, 5, 7]:
        text += f'![M0 и M1, h={h}](../results/h13_catboost_logreturn/scatter_h{h}.png)\n\n'
    text += '''## Артефакты и воспроизведение

```powershell
python scripts/run_h13_catboost_logreturn.py
python -m unittest discover -s tests -p "test_h13_catboost_logreturn.py" -v
python -m unittest discover -s tests -p "test_*.py" -q
git diff --check
python scripts/run_h13_catboost_logreturn.py --report-only
```

Полный запуск требует пустого output-каталога, чтобы не смешивать эксперименты. Скрипт никогда не перезаписывает исторические результаты. Автоматически сформированный отчёт сохраняется в results и reports; `--report-only` пересобирает его из сохранённых таблиц и validation.json без обучения. Прогнозы до scoring: predictions_oos.csv. Оценённые прогнозы: scored_predictions.csv, парное сопоставление: paired_predictions.csv (`prediction`=M0, `prediction_reference`=M1, `d`=SSE M1−SSE M0 для строки). Контроль оригинала: m0_reproduction.csv; окна обеих моделей: training_windows.csv. В manifest зафиксированы SHA исходного Git, версии библиотек, параметры, SHA256 входов и защищённых данных, отсутствие 2026 в числовых данных и фактические диапазоны дат. Для новых артефактов закреплён LF в .gitattributes, чтобы SHA256 сохранялись при Windows checkout.

Протокол conditional EOD / pseudo-PIT / ASSUMED_AVAILABLE и provisional-календарь унаследованы от Core v2. Многодневные ошибки перекрываются, HAC учитывает автокорреляцию с фиксированным лагом, но один повторно используемый validation-год ограничивает выводы. Статистическое различие здесь не доказывает независимую воспроизводимость или торговую доходность.
'''
    validation_path = output/'validation.json'
    text += '\n## Проверки\n\n'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        for label, result in validation['tests'].items():
            text += f"{label}: выполнено {result['run']}, прошли {result['passed']}, пропущено {result['skipped']}, failures={result['failures']}, errors={result['errors']}.\n\n"
        text += f"git diff --check: {validation['git_diff_check']}. Воспроизведены все 1778 M0 и 21 M1 (первый/средний/последний origin каждого h); эти 21 M1 совпали побитно как float64, включая prediction_target. Дополнительно проверена повторяемость обеих моделей и неизменность прогноза после изменения данных позже origin на синтетической панели.\n"
    else:
        text += 'Результаты запуска тестов ещё не внесены в validation.json.\n'
    (output/'report.md').write_text(text.replace('../results/h13_catboost_logreturn/', ''), encoding='utf-8', newline='\n')
    (ROOT/'reports/h13_catboost_logreturn.md').write_text(text, encoding='utf-8', newline='\n')


def report_only():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    output = ROOT/cfg['output_root']
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    ab.require(manifest['m0_all_1778_verified'], 'M0 gate has not passed')
    tables = {name: pd.read_csv(output/f'{name}.csv', float_precision='round_trip')
              for name in ['metrics', 'hac_holm', 'behavior']}
    report(tables, manifest, output)
    for name in list(manifest['implementation_sha256_lf']) + ['tests/test_h13_catboost_logreturn.py', '.gitattributes']:
        manifest['implementation_sha256_lf'][name] = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    manifest['output_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir())
                                 if p.is_file() and p.name != 'manifest.json'}
    if (output/'validation.json').exists():
        manifest['validation'] = json.loads((output/'validation.json').read_text(encoding='utf-8'))
    json_write(manifest, output/'manifest.json')
    print('H13 report rebuilt from saved tables; no fitting')


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    _, proposal, features = cb.config()
    ab.require(cfg['fixed_parameters'] == proposal['fixed_parameters'] and cfg['models'] == ab.MODELS,
               'H13 fixed models/parameters changed')
    ab.require(cfg['horizons'] == list(range(1, 8)) and not cfg['final_test_access'] and not cfg['jensen_correction'],
               'Unsafe H13 specification')
    output, source = ROOT/cfg['output_root'], ROOT/cfg['original_results']
    ab.require(not output.exists() or not any(output.iterdir()), 'Use an empty H13 output directory')
    protected = protected_hashes()
    source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frame, calendar = cb.load_inputs(features)
    ab.require(frame.index.max() < pd.Timestamp('2026-01-01') and frame.index.equals(calendar), '2026/calendar violation')
    saved = read_predictions(source/'scored_common_predictions.csv')
    ab.require(saved.model_id.eq('CatBoost_Core_v2').all() and saved.split.eq('validation').all(), 'Unexpected reference rows')
    for column, dates in [('actual', saved.target_date), ('RW', saved.origin_date)]:
        ab.require(np.isclose(saved[column], frame.close.reindex(dates).to_numpy(), atol=1e-8, rtol=1e-10).all(),
                   f'Reference {column} differs from exact calendar CLOSE')
    selections = {h: json.loads((source/f'h{h}/selection.json').read_text())['parameters'] for h in cfg['horizons']}
    for parameters in selections.values():
        ab.require(set(parameters) == {'depth', 'l2_leaf_reg'}, 'Unexpected selected parameters')
    output.mkdir(parents=True, exist_ok=True)
    all_predictions, all_windows = {}, {}
    for model_id in ab.MODELS:
        predictions, windows = [], []
        for h in cfg['horizons']:
            pred, window = ab.forecast(frame, h, selections[h], proposal, features, model_id,
                                       progress=lambda message: print(message, flush=True))
            predictions.append(pred)
            windows.append(window)
        all_predictions[model_id] = pd.concat(predictions, ignore_index=True)
        all_windows[model_id] = pd.concat(windows, ignore_index=True)
        if model_id == 'M0_DELTA':
            controls = ab.check_m0(all_predictions[model_id], saved, cfg['comparison_atol'], cfg['comparison_rtol'])
            check_original_windows(all_windows[model_id], source)
            write(controls, output/'m0_reproduction.csv')
            print('M0 GATE PASSED: all 1778 predictions and all original windows; M1 may start', flush=True)
    window_check = ab.check_windows(all_windows['M0_DELTA'], all_windows['M1_LOGRETURN'])
    paired = ab.exact_pairs(all_predictions['M0_DELTA'], all_predictions['M1_LOGRETURN'], ['prediction', 'prediction_target'])
    paired = ab.exact_pairs(paired, saved, ['actual', 'RW'])
    positions = calendar.get_indexer(paired.origin_date)
    target_positions = calendar.get_indexer(paired.target_date)
    ab.require(np.array_equal(target_positions-positions, paired.horizon), 'Trading horizon mismatch')
    for _, g in paired.groupby('horizon'):
        ab.require((np.diff(calendar.get_indexer(g.origin_date)) == 1).all(), 'Nonconsecutive HAC origins')
    tables = ab.summarize(paired, cfg)
    unscored = pd.concat(list(all_predictions.values()), ignore_index=True)
    scored = unscored.merge(saved[ab.KEYS+['actual', 'RW']], on=ab.KEYS, validate='many_to_one')
    scored['predicted_delta'] = scored.prediction-scored.RW
    scored['actual_delta'] = scored.actual-scored.RW
    scored['error'] = scored.prediction-scored.actual
    paired['d'] = (paired.prediction_reference-paired.actual)**2-(paired.prediction-paired.actual)**2
    tables.update(predictions_oos=unscored, scored_predictions=scored, paired_predictions=paired,
                  training_windows=pd.concat(list(all_windows.values()), ignore_index=True))
    for name, table in tables.items():
        write(table, output/f'{name}.csv')
    plots(paired, output)
    inputs = [CONFIG, ROOT/'configs/catboost_experiment.json', ROOT/'configs/catboost_core_v2_proposal.json',
              ROOT/'configs/ridge_experiments_v2.json', ROOT/'configs/feature_scenarios_v1.json']
    inputs += [ROOT/features[name] for name in ['v2_path', 'feature_path', 'metadata_path', 'calendar_path']]
    inputs += [source/'scored_common_predictions.csv', source/'metrics.csv']
    inputs += [source/f'h{h}/{name}.json' for h in cfg['horizons'] for name in ['selection']]
    inputs += [source/f'h{h}/training_windows.csv' for h in cfg['horizons']]
    implementation = [CONFIG, ROOT/'benchmarks/catboost_target_ablation.py', Path(__file__),
                      ROOT/'benchmarks/catboost_direct.py', ROOT/'benchmarks/ridge.py', ROOT/'benchmarks/ridge_v2.py']
    ab.require(protected == protected_hashes(), 'Protected historical inputs/results changed')
    manifest = dict(source_git_sha=source_sha, python=platform.python_version(),
                    packages={name: version(name) for name in ['catboost', 'numpy', 'pandas', 'scipy', 'matplotlib', 'pyarrow']},
                    config=cfg, features=ridge_v2.CORE, selected_parameters=selections,
                    input_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
                    implementation_sha256_lf={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in implementation},
                    protected_sha256=protected, historical_files_unchanged=True,
                    m0_max_abs_difference=float(controls.max_abs_prediction_difference.max()), m0_all_1778_verified=True,
                    window_check=window_check, no_2026_numeric_data=True, final_test_access=False,
                    data_bounds=dict(frame=[str(frame.index.min().date()), str(frame.index.max().date())],
                                     origin=[str(paired.origin_date.min().date()), str(paired.origin_date.max().date())],
                                     target=[str(paired.target_date.min().date()), str(paired.target_date.max().date())]),
                    no_2026_policy='Date-filter CSV rows before numeric parsing; whole-file hashes include 2026 for provenance only',
                    n_fits=3556, elapsed_seconds=time.perf_counter()-started)
    report(tables, manifest, output)
    manifest['output_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    json_write(manifest, output/'manifest.json')
    print('H13 COMPLETE: 3556 fits, 1778 pairs per model, seven primary HAC/Holm contrasts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Rebuild report from completed saved results, without fitting')
    args = parser.parse_args()
    report_only() if args.report_only else run()
