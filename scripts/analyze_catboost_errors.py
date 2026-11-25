"""Diagnose saved validation forecasts only. Run from the repository root."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/catboost_error_diagnostics'
KEY = ['origin_date', 'target_date', 'horizon']
MODEL = 'CatBoost_Core_v2'
RTOL, ATOL = 1e-10, 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(np.isclose(a, b, rtol=RTOL, atol=ATOL).all(), message)


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def dated(frame, columns):
    frame = frame.copy()
    for col in columns:
        require(frame[col].astype('string').str.fullmatch(r'\d{4}-\d{2}-\d{2}').fillna(False).all(),
                f'Invalid or missing date: {col}')
        frame[col] = pd.to_datetime(frame[col], format='%Y-%m-%d', errors='raise')
    return frame


def validate_pairs(predictions, history, calendar, config):
    """Exact joins only; accept saved RW column or separate RW model rows."""
    required = set(KEY + ['model_id', 'prediction', 'prediction_target', 'actual', 'split'])
    require(required <= set(predictions), f'Missing forecast columns: {required - set(predictions)}')
    p = dated(predictions, KEY[:2])
    require(not p.duplicated(['model_id'] + KEY).any(), 'Duplicate model/forecast keys')
    start, end = map(pd.Timestamp, config['splits']['validation'])
    require(start.year == end.year == 2025, 'Expected validation 2025')
    p = p[p.split.eq('validation') & p.target_date.between(start, end) & p.horizon.isin(range(1, 8))]
    cb = p[p.model_id.eq(MODEL)].copy()
    require(set(cb.horizon) == set(range(1, 8)), 'Missing horizons 1..7')
    if p.model_id.eq('RW').any():
        rw = p[p.model_id.eq('RW')][KEY + ['prediction', 'actual']]
        joined = cb.merge(rw, on=KEY, how='outer', validate='one_to_one', indicator=True,
                          suffixes=('', '_rw'))
        require(joined._merge.eq('both').all(), 'RW/CatBoost key sets differ')
        close(joined.actual, joined.actual_rw, 'RW/CatBoost actual mismatch')
        if 'RW' in joined:
            close(joined.RW, joined.prediction_rw, 'RW row/column mismatch')
        cb = joined.drop(columns=['_merge', 'actual_rw'])
    else:
        require('RW' in cb, 'Neither RW column nor RW model rows found')
        cb['prediction_rw'] = cb.RW
    counts = cb.groupby('horizon').size()
    require(counts.eq(254).all(), f'Expected 254 common pairs per h; got {counts.to_dict()}')
    history = dated(history, ['date'])
    history = history[history.date <= end].copy()
    require(not history.date.duplicated().any(), 'Duplicate history dates')
    require(history.price_type.eq('CANDLE_CLOSE').all(), 'Baseline price must be CANDLE_CLOSE')
    sessions = dated(calendar, ['date']).date
    sessions = pd.DatetimeIndex(sessions[sessions <= end])
    require(sessions.is_unique and sessions.is_monotonic_increasing, 'Calendar must be unique and sorted')
    prices = history.set_index('date').price.reindex(sessions)
    require(np.isfinite(prices).all() and prices.gt(0).all(), 'Missing/nonpositive calendar CLOSE prices')
    origin = sessions.get_indexer(cb.origin_date)
    target = sessions.get_indexer(cb.target_date)
    require((origin >= 0).all() and (target >= 0).all(), 'Forecast date absent from calendar')
    require(np.array_equal(target - origin, cb.horizon.to_numpy()), 'Invalid trading-session horizon')
    require(np.isfinite(cb[['prediction', 'prediction_target', 'prediction_rw', 'actual']]).all().all(),
            'Missing/nonfinite forecast or actual price')
    close(cb.actual, prices.reindex(cb.target_date).to_numpy(), 'Actual differs from target CLOSE')
    close(cb.prediction_rw, prices.reindex(cb.origin_date).to_numpy(), 'RW differs from origin CLOSE')
    close(cb.prediction, cb.prediction_rw + cb.prediction_target, 'prediction != RW + prediction_target')
    cb = cb.rename(columns={'prediction': 'prediction_catboost', 'prediction_target': 'predicted_delta'})
    cb['error_catboost'] = cb.prediction_catboost - cb.actual
    cb['error_rw'] = cb.prediction_rw - cb.actual
    cb['actual_delta'] = cb.actual - cb.prediction_rw
    return cb.sort_values(['horizon', 'target_date', 'origin_date']).reset_index(drop=True), prices


def metrics(frame):
    out = {'n': len(frame)}
    for model in ['catboost', 'rw']:
        error = frame['error_' + model].to_numpy()
        out.update({f'rmse_{model}': float(np.sqrt(np.mean(error ** 2))),
                    f'mae_{model}': float(np.mean(np.abs(error))), f'bias_{model}': float(np.mean(error))})
    out['rmse_ratio_vs_rw'] = out['rmse_catboost'] / out['rmse_rw'] if out['rmse_rw'] else np.nan
    out['catboost_worse_fraction'] = float((frame.error_catboost.abs() > frame.error_rw.abs()).mean())
    return out


def grouped(frame, columns):
    return pd.DataFrame([dict(zip(columns, key if isinstance(key, tuple) else (key,)), **metrics(g))
                         for key, g in frame.groupby(columns, observed=True, sort=True)])


def check_metrics(frame, saved):
    require(not saved.duplicated(['model_id', 'horizon']).any(), 'Duplicate saved metric keys')
    result = grouped(frame, ['horizon'])
    for row in result.to_dict('records'):
        for model, suffix in [(MODEL, 'catboost'), ('RW', 'rw')]:
            match = saved[saved.model_id.eq(model) & saved.horizon.eq(row['horizon'])]
            require(len(match) == 1, f'Missing control metric {model}, h={row["horizon"]}')
            require(match.iloc[0]['n'] == row['n'], f'Control n mismatch: {model}')
            for metric in ['rmse', 'mae', 'bias']:
                close(row[f'{metric}_{suffix}'], match.iloc[0][metric],
                      f'Control {metric} mismatch: {model}, h={row["horizon"]}')
    return result


def volatility(prices, train_bounds):
    """20 returns ending at origin close; sample std, no annualization."""
    vol = np.log(prices).diff().rolling(20, min_periods=20).std(ddof=1)
    start, end = map(pd.Timestamp, train_bounds)
    require(end < pd.Timestamp('2025-01-01'), 'Volatility thresholds require pre-2025 train')
    train = vol.loc[start:end].dropna()
    require(len(train) > 0, 'No complete train volatility windows')
    thresholds = train.quantile([1 / 3, 2 / 3]).to_numpy()
    require(thresholds[0] < thresholds[1], 'Degenerate volatility thresholds')
    return vol, thresholds, train


def direction(values):
    """Exact zero, fixed before results; no fitted deadband."""
    return np.where(np.asarray(values) == 0, 'zero', np.where(np.asarray(values) > 0, 'up', 'down'))


def analyze(frame, prices, config):
    frame = frame.copy()
    vol, thresholds, train = volatility(prices, config['splits']['train'])
    frame['volatility_20'] = vol.reindex(frame.origin_date).to_numpy()
    require(np.isfinite(frame.volatility_20).all(), 'Missing origin volatility')
    frame['volatility_regime'] = np.where(frame.volatility_20 <= thresholds[0], 'low',
                                          np.where(frame.volatility_20 <= thresholds[1], 'medium', 'high'))
    frame['quarter'] = frame.target_date.dt.to_period('Q').astype(str)
    frame['actual_direction'] = direction(frame.actual_delta)
    frame['predicted_direction'] = direction(frame.predicted_delta)
    tables = {'quarterly_metrics': grouped(frame, ['horizon', 'quarter']),
              'volatility_metrics': grouped(frame, ['horizon', 'volatility_regime']),
              'direction_metrics': grouped(frame, ['horizon', 'actual_direction'])}
    accuracy, confusion, amplitude, top = [], [], [], []
    for h, g in frame.groupby('horizon', sort=True):
        correct = g.actual_direction.eq(g.predicted_direction)
        nonzero = g.actual_direction.ne('zero')
        accuracy.append(dict(horizon=h, n=len(g), correct=int(correct.sum()), accuracy=correct.mean(),
                             n_nonzero=int(nonzero.sum()), n_zero=int((~nonzero).sum()),
                             accuracy_nonzero=correct[nonzero].mean()))
        for actual in ['down', 'zero', 'up']:
            for predicted in ['down', 'zero', 'up']:
                confusion.append(dict(horizon=h, actual_direction=actual, predicted_direction=predicted,
                                      n=int((g.actual_direction.eq(actual) & g.predicted_direction.eq(predicted)).sum())))
        if h not in [1, 3, 5, 7]:
            continue
        row = dict(horizon=h, n=len(g), correlation=g.actual_delta.corr(g.predicted_delta),
                   std_ratio=g.predicted_delta.std(ddof=1) / g.actual_delta.std(ddof=1))
        for name in ['actual_delta', 'predicted_delta']:
            row[name + '_mean'] = g[name].mean()
            row[name + '_std'] = g[name].std(ddof=1)
            for q in [0, .05, .25, .5, .75, .95, 1]:
                row[f'{name}_q{q:g}'] = g[name].quantile(q)
        amplitude.append(row)
        worst = g.assign(abs_error=g.error_catboost.abs()).sort_values(
            ['abs_error', 'target_date', 'origin_date'], ascending=[False, True, True]).head(10).copy()
        worst['sse_share'] = worst.error_catboost ** 2 / (g.error_catboost ** 2).sum()
        worst['rank'] = np.arange(1, len(worst) + 1)
        top.append(worst)
    tables.update(direction_accuracy=pd.DataFrame(accuracy), direction_confusion=pd.DataFrame(confusion),
                  amplitude_summary=pd.DataFrame(amplitude), top_errors=pd.concat(top, ignore_index=True),
                  diagnostic_pairs=frame)
    return tables, thresholds, train


def write_tables(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False, float_format='%.17g',
                     date_format='%Y-%m-%d', lineterminator='\n')


def plots(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for h in [1, 3, 5, 7]:
        g = frame[frame.horizon.eq(h)]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(g.actual_delta, g.predicted_delta, s=15, alpha=.6)
        limits = [min(g.actual_delta.min(), g.predicted_delta.min()),
                  max(g.actual_delta.max(), g.predicted_delta.max())]
        ax.plot(limits, limits, '--', color='black', label='Ideal y=x')
        ax.set(xlabel='Actual delta (RUB/USD)', ylabel='Predicted delta (RUB/USD)',
               title=f'CatBoost Core v2, validation 2025, h={h}', xlim=limits, ylim=limits)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'amplitude_h{h}.png', dpi=150, metadata={'Software': 'CatBoost error diagnostics'})
        plt.close(fig)


def run(output=OUTPUT):
    paths = {'predictions': ROOT / 'results/catboost_core_v2/scored_common_predictions.csv',
             'metrics': ROOT / 'results/catboost_core_v2/metrics.csv',
             'baseline': ROOT / 'data/processed/USDRUBF_baseline_2026-11-18.csv',
             'config': ROOT / 'configs/benchmark_config.yaml'}
    config = yaml.safe_load(paths['config'].read_text(encoding='utf-8'))
    paths['calendar'] = ROOT / config['calendar_path']
    inputs = {key: read(path) for key, path in paths.items() if key != 'config'}
    schema = {key: {col: str(dtype) for col, dtype in data.dtypes.items()} for key, data in inputs.items()}
    frame, prices = validate_pairs(inputs['predictions'], inputs['baseline'], inputs['calendar'], config)
    controls = check_metrics(frame, inputs['metrics'])
    tables, thresholds, train = analyze(frame, prices, config)
    tables['horizon_metrics'] = controls
    # No output is written until all alignment and metric controls have passed.
    write_tables(tables, output)
    plots(frame, output)
    manifest = dict(source_git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'PyYAML', 'matplotlib']},
                    input_sha256={str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in paths.values()}, input_schema=schema,
                    parameters=dict(horizons=list(range(1, 8)), expected_pairs=254, rtol=RTOL, atol=ATOL,
                                    error='prediction - actual', volatility_window_returns=20, volatility_ddof=1,
                                    volatility_annualized=False, origin_close_included=True,
                                    threshold_quantiles=[1/3, 2/3], threshold_interpolation='linear',
                                    regime_rule='low <= q1; q1 < medium <= q2; high > q2',
                                    zero_rule='delta == 0 exactly, for actual and prediction', amplitude_std_ddof=1,
                                    no_training=True, no_outlier_exclusion=True),
                    bounds=dict(config_train=config['splits']['train'], config_validation=config['splits']['validation'],
                                price_history=[str(prices.index.min().date()), str(prices.index.max().date())],
                                threshold_volatility_dates=[str(train.index.min().date()), str(train.index.max().date())],
                                threshold_n=len(train), origin=[str(frame.origin_date.min().date()), str(frame.origin_date.max().date())],
                                target=[str(frame.target_date.min().date()), str(frame.target_date.max().date())]),
                    volatility_thresholds=dict(low_medium=float(thresholds[0]), medium_high=float(thresholds[1])),
                    calendar_status=config['calendar_source'], checks='all alignment and saved metric controls passed',
                    artifacts=sorted([f'{name}.csv' for name in tables] + [f'amplitude_h{h}.png' for h in [1,3,5,7]] + ['manifest.json']))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('PASS: 7 horizons x 254 pairs; exact dates/prices and saved metrics verified; no training; no 2026 calculations')
    return tables


if __name__ == '__main__':
    run()
