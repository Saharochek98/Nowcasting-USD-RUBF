"""Validation-only Ridge v2; immutable v1 evidence and prespecified comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from benchmarks import ridge, ridge_v2
from scripts.run_ridge_experiments import stability, write


def protected_hashes(cfg):
    paths = [ROOT/cfg[k] for k in ['feature_path', 'metadata_path', 'v2_path', 'calendar_path',
                                 'scenario_config', 'baseline_path']]
    for folder in ['results/ridge_experiments', 'results/benchmark_calendar_v2',
                   'results/funding_missingness_audit']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    paths += [ROOT/p for p in ['configs/ridge_experiments.json', 'benchmarks/ridge.py',
                              'scripts/run_ridge_experiments.py', 'docs/feature_availability_catalog.csv']]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def compare_pair(left, right, cfg, identity, primary=False):
    pair = left.merge(right, on=ridge.KEYS, suffixes=('_a', '_b'), validate='one_to_one')
    if len(pair) != len(left) or len(pair) != len(right):
        raise ValueError('Comparison date sets differ')
    np.testing.assert_allclose(pair.actual_a, pair.actual_b, atol=1e-10, rtol=0)
    losses = pd.DataFrame({'prediction': pair.prediction_a, 'actual': pair.actual_a,
                           'rw_prediction': pair.prediction_b})
    ea, eb = pair.prediction_a-pair.actual_a, pair.prediction_b-pair.actual_b
    return dict(**identity, n=len(pair), primary=primary,
                rmse_difference=float(np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2))),
                **ridge.loss_hac(losses, int(pair.horizon.iloc[0]), cfg))


def comparisons(scored, cfg):
    rows = []
    increments = cfg['primary_hac_family']['incremental_pairs']
    for (regime, transform, h), group in scored.groupby(['scope', 'transform', 'horizon']):
        primary = regime == 'prequential' and transform == 'delta'
        for model, a in group.groupby('model_id'):
            for label, col in [('RW', 'rw_prediction'), ('AR1', 'ar_prediction')]:
                b = a.copy()
                b['prediction'] = b[col]
                rows.append(compare_pair(a, b, cfg, dict(kind='baseline', model=model, versus=label,
                            regime=regime, transform=transform, horizon=h), primary))
        for model, versus in increments:
            rows.append(compare_pair(group[group.model_id.eq(model)], group[group.model_id.eq(versus)], cfg,
                        dict(kind='features', model=model, versus=versus, regime=regime,
                             transform=transform, horizon=h), primary))
    for (model, transform, h), group in scored.groupby(['model_id', 'transform', 'horizon']):
        rows.append(compare_pair(group[group.scope.eq('prequential')], group[group.scope.eq('frozen')], cfg,
                    dict(kind='training_policy', model=model, versus='frozen', regime='prequential',
                         transform=transform, horizon=h)))
    for (model, regime, h), group in scored.groupby(['model_id', 'scope', 'horizon']):
        rows.append(compare_pair(group[group['transform'].eq('log_return')], group[group['transform'].eq('delta')], cfg,
                    dict(kind='target_transform', model=model, versus='delta', regime=regime,
                         transform='log_return', horizon=h)))
    result = pd.DataFrame(rows)
    assert result.primary.sum() == cfg['primary_hac_family']['size']
    result['holm_p'] = np.nan
    for primary, group in result.groupby('primary'):
        result.loc[group.index, 'holm_p'] = ridge.holm(group.p_two_sided.to_numpy())
    return result


def regularization_path(frame, cfg):
    """Fixed first-origin fit; diagnose shrinkage, never tune on outer validation."""
    rows = []
    cols = ridge_v2.CORE
    for transform in cfg['transforms']:
        for h in cfg['horizons']:
            y = ridge.supervised(frame, h, transform)
            origins = np.arange(len(frame)-h)
            origins = origins[(frame.index[origins+h] >= cfg['validation_start']) &
                              (frame.index[origins+h] <= cfg['validation_end'])]
            train = ridge.training_indices(frame, y, h, int(origins[0]), cfg)
            anchor = frame.close.iloc[origins].to_numpy()
            for alpha in cfg['alpha_grid'] + [float('inf')]:
                fit = ridge.fit_ridge(frame[cols].iloc[train], y[train], alpha if np.isfinite(alpha) else 1.)
                if not np.isfinite(alpha):
                    fit['beta'][:] = 0.
                slope = ridge.standardized(fit, frame[cols].iloc[origins]) @ fit['beta']
                correction = ridge.price_from_change(anchor, fit['intercept']+slope, transform)-anchor
                drift = ridge.price_from_change(anchor, np.full(len(anchor), fit['intercept']), transform)-anchor
                rows.append(dict(model_id='core_v2', transform=transform, horizon=h,
                    alpha=alpha, train_n=len(train), train_label_end=frame.index[train[-1]+h],
                    first_origin=frame.index[origins[0]], coefficient_l2=np.linalg.norm(fit['beta']),
                    slope_target_rms=np.sqrt(np.mean(slope**2)), intercept_target=fit['intercept'],
                    price_correction_rms=np.sqrt(np.mean(correction**2)),
                    drift_limit_price_rms=np.sqrt(np.mean(drift**2)),
                    distance_to_drift_limit_rms=np.sqrt(np.mean((correction-drift)**2)),
                    n=len(origins), diagnostic_only=True))
    return pd.DataFrame(rows)


def run(output, cfg):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output root; saved runs are immutable')
    output.mkdir(parents=True, exist_ok=True)
    before = protected_hashes(cfg)
    frame, calendar, columns = ridge_v2.load_inputs(cfg)
    baseline = ridge.load_baselines(cfg)
    # Save family declaration and config before evaluating any outer predictions.
    (output/'design.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    all_scores, all_metrics, tunings = [], [], []
    expected_keys = None
    for regime in cfg['regimes']:
        for transform in cfg['transforms']:
            dest = output/f'{regime}_{transform}'
            dest.mkdir()
            preds, coefs, jobs = [], [], []
            for model, cols in columns.items():
                for h in cfg['horizons']:
                    pred, coef, tuning, origins, fits = ridge_v2.walk_forward(frame, cols, model, h, transform, cfg, regime)
                    keys = set(pred[ridge.KEYS].itertuples(index=False, name=None))
                    if expected_keys is None:
                        expected_keys = {}
                    if h in expected_keys and keys != expected_keys[h]:
                        raise ValueError('Models/policies/transforms have different dates')
                    expected_keys[h] = keys
                    preds.append(pred)
                    coefs.append(coef)
                    tunings.extend(dict(model_id=model, horizon=h, transform=transform, regime=regime, **r) for r in tuning)
                    jobs.append((model, h, pred, origins, fits))
                    print(regime, transform, model, h, 'n', len(pred), 'alpha', pred.alpha.iloc[0], flush=True)
            pred = pd.concat(preds, ignore_index=True)
            coef = pd.concat(coefs, ignore_index=True)
            write(pred, dest/'predictions_oos.csv')
            write(coef, dest/'coefficients_windows.csv')
            write(stability(coef, pred), dest/'coefficient_stability.csv')
            importance = []
            for model, h, pred, origins, fits in jobs:
                scores, metrics, pairs = ridge.common_comparison(pred, baseline, frame)
                all_scores.append(pairs)
                all_metrics.extend(dict(model_id=model, horizon=h, transform=transform, regime=regime,
                                        pit_status=pred.pit_status.iloc[0], **m) for m in metrics)
                importance.extend(dict(model_id=model, horizon=h, transform=transform, regime=regime,
                                       pit_status=pred.pit_status.iloc[0], **r)
                                  for r in ridge.permutation(frame, columns[model], pred, origins, fits, cfg))
            write(pd.DataFrame(importance), dest/'permutation_importance.csv')
    scores = pd.concat(all_scores, ignore_index=True)
    write(scores, output/'scored_common_predictions.csv')
    write(pd.DataFrame(all_metrics), output/'metrics_all.csv')
    tuning = pd.DataFrame(tunings)
    write(tuning, output/'inner_tuning.csv')
    # Policies must have identical tuning evidence and alpha selections.
    a = tuning[tuning.regime.eq('frozen')].drop(columns='regime').reset_index(drop=True)
    b = tuning[tuning.regime.eq('prequential')].drop(columns='regime').reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    finalize(output, cfg, frame, calendar, before)


def finalize(output, cfg, frame, calendar, before, resumed=False):
    scores = pd.read_csv(output/'scored_common_predictions.csv', float_precision='round_trip')
    if len(scores) != 35560 or not scores.groupby(['model_id', 'scope', 'transform', 'horizon']).size().eq(254).all():
        raise ValueError('Incomplete persisted forecasts')
    for regime in cfg['regimes']:
        for transform in cfg['transforms']:
            dest = output/f'{regime}_{transform}'
            forecasts = pd.read_csv(dest/'predictions_oos.csv', float_precision='round_trip')
            saved = scores[scores.scope.eq(regime) & scores['transform'].eq(transform)]
            assert forecasts.scored_status.eq('unscored').all() and saved.scored_status.eq('scored').all()
            check_columns = list(forecasts.columns.drop('scored_status'))
            pd.testing.assert_frame_equal(forecasts[check_columns].reset_index(drop=True),
                                          saved[check_columns].reset_index(drop=True), check_exact=True)
            coefficients = pd.read_csv(dest/'coefficients_windows.csv', usecols=['window_id', 'feature'])
            assert set(forecasts.window_id) == set(coefficients.window_id)
            for model, columns in cfg['scenarios'].items():
                windows = forecasts.loc[forecasts.model_id.eq(model), 'window_id']
                assert coefficients[coefficients.window_id.isin(windows)].groupby('window_id').size().eq(len(columns)).all()
    write(comparisons(scores, cfg), output/'comparisons_hac_holm.csv')
    write(regularization_path(frame, cfg), output/'regularization_path.csv')
    assert before == protected_hashes(cfg), 'Protected v1/input/baseline evidence changed'
    implementation = ['benchmarks/ridge_v2.py', 'scripts/run_ridge_experiments_v2.py',
                      'configs/ridge_experiments_v2.json', 'benchmarks/ridge.py', 'scripts/run_ridge_experiments.py']
    manifest = dict(config=cfg, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                    implementation_state='uncommitted; exact implementation identified by source hashes',
                    python=platform.python_version(), packages={p: version(p) for p in ['numpy','pandas','pyarrow']},
                    protected_sha256=before, protected_files_unchanged=True,
                    implementation_sha256_lf={p: hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in implementation},
                    prediction_count=len(scores), each_model_horizon_n=254, postprocessing_resumed=resumed,
                    data_cutoff=str(calendar[-1].date()), final_test_used_for_modeling_or_scoring=False,
                    research_after_validation_v1=True, funding_status='ASSUMED_AVAILABLE')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/ridge_experiments_v2')
    parser.add_argument('--finalize-existing', action='store_true',
                        help='Complete postprocessing of a fully persisted run without a manifest; never refit or alter forecasts')
    args = parser.parse_args()
    cfg = ridge_v2.config()
    if args.finalize_existing:
        if (args.output_root/'manifest.json').exists():
            raise ValueError('Completed runs are immutable')
        if json.loads((args.output_root/'design.json').read_text()) != cfg:
            raise ValueError('Persisted design differs from configuration')
        before = protected_hashes(cfg)
        frame, calendar, _ = ridge_v2.load_inputs(cfg)
        finalize(args.output_root, cfg, frame, calendar, before, resumed=True)
    else:
        run(args.output_root, cfg)


if __name__ == '__main__':
    main()
