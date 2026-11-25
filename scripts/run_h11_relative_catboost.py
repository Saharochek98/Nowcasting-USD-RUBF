"""Run H11-REL-01 end to end: rerun CatBoost Core v2, R0_MATCHED vs R1, scoring, HAC, figures.

Validation 2025 only; the final test 2026 is never loaded.  Writes into an empty
results/h11_relative_catboost_v1/ and refuses to overwrite an existing run.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, h11_plots, h11_relative as H, ridge
from benchmarks.new_models import parallel_map
from scripts.run_ridge_experiments import write

SOURCES = ['benchmarks/h11_relative.py', 'benchmarks/h11_plots.py', 'scripts/run_h11_relative_catboost.py',
           'benchmarks/catboost_direct.py', 'benchmarks/ridge.py', 'benchmarks/ridge_v2.py',
           'configs/h11_catboost_relative_v1.json', 'tests/test_h11_no_leakage.py']
PROTECTED_DIRS = ['data', 'configs', 'reports', 'results', 'docs']
SAVED = ROOT/'results/catboost_core_v2/predictions_oos.csv'
COMMON = ROOT/'results/rw_drift_experiment/common_pairs.csv'


def json_write(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')


def protected(output):
    files = {}
    for folder in PROTECTED_DIRS:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and output not in p.parents and p.name != '.DS_Store':
                files[p.relative_to(ROOT).as_posix()] = H.sha256(p)
    return files


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def check_calendar(pred):
    calendar = pd.to_datetime(pd.read_csv(ROOT/'data/processed/usdrubf_trading_days.csv').date)
    calendar = pd.DatetimeIndex(calendar[calendar <= '2025-12-31'])
    position = pd.Series(np.arange(len(calendar)), index=calendar)
    ok = (position.loc[pred.target_date].to_numpy()-position.loc[pred.origin_date].to_numpy()) == pred.horizon.to_numpy()
    return bool(ok.all())


def score(predictions, frame):
    common = pd.read_csv(COMMON, parse_dates=['origin_date', 'target_date'], float_precision='round_trip')
    keys = set(common[ridge.KEYS].itertuples(index=False, name=None))
    wide = common[ridge.KEYS+['actual', 'RW']].copy()
    for name, pred in predictions.items():
        if pred.duplicated(ridge.KEYS).any() or set(pred[ridge.KEYS].itertuples(index=False, name=None)) != keys:
            raise ValueError(f'{name}: forecast keys differ from the common validation pairs')
        wide = wide.merge(pred[ridge.KEYS+['prediction']].rename(columns={'prediction': name}),
                          on=ridge.KEYS, validate='one_to_one')
    np.testing.assert_allclose(wide.actual, wide.target_date.map(frame.close), atol=1e-10, rtol=0)
    np.testing.assert_allclose(wide.RW, wide.origin_date.map(frame.close), atol=1e-10, rtol=0)
    if not np.isfinite(wide.drop(columns=ridge.KEYS).to_numpy(float)).all():
        raise ValueError('Nonfinite scored value')
    wide = wide.sort_values(['horizon', 'origin_date']).reset_index(drop=True)
    metrics = []
    for h, g in wide.groupby('horizon'):
        rw_rmse = np.sqrt(np.mean((g.RW-g.actual)**2))
        for model in ['RW', *predictions]:
            e = g[model]-g.actual
            metrics.append(dict(model_id=model, horizon=h, n=len(g), rmse=np.sqrt(np.mean(e**2)), mae=np.mean(np.abs(e)),
                bias=np.mean(e), rmse_ratio_vs_rw=np.sqrt(np.mean(e**2))/rw_rmse,
                origin_min=g.origin_date.min(), origin_max=g.origin_date.max(),
                target_min=g.target_date.min(), target_max=g.target_date.max()))
    return wide, pd.DataFrame(metrics)


def hac_tests(wide):
    """Primary family (preregistered): R1 vs R0_MATCHED x 7 h, Holm over 7.  Others exploratory."""
    contrasts = [('primary', 'R1', 'R0_MATCHED'), ('exploratory', 'R1', 'RW'), ('exploratory', 'R0_MATCHED', 'RW')]
    rows = []
    for family, model, reference in contrasts:
        block = []
        for h, g in wide.groupby('horizon'):
            result = H.paired_hac(g[model]-g.actual, g[reference]-g.actual, h)
            block.append(dict(family=family, model=model, versus=reference, horizon=h, n=len(g),
                              rmse_model=np.sqrt(np.mean((g[model]-g.actual)**2)),
                              rmse_versus=np.sqrt(np.mean((g[reference]-g.actual)**2)), **result))
        block = pd.DataFrame(block)
        block['holm_p'] = ridge.holm(block.p_two_sided.to_numpy())
        block['holm_family'] = f'{model} vs {reference}, 7 horizons'
        block['significant_5pct_holm'] = block.holm_p < .05
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/h11_relative_catboost_v1')
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'{output} is not empty; a completed run is immutable')
    started = time.perf_counter()
    h11, cfg, proposal, features = H.load_config()
    protected_before = protected(output)
    frame, calendar = H.load_panel(features)
    (output/'figures').mkdir(parents=True, exist_ok=True)

    design = dict(experiment_id=h11['experiment_id'], config_path='configs/h11_catboost_relative_v1.json',
        config_sha256=H.sha256(H.CONFIG_PATH), config_git_blob=git('hash-object', 'configs/h11_catboost_relative_v1.json'),
        config_matches_origin_main=git('hash-object', 'configs/h11_catboost_relative_v1.json')
            == git('rev-parse', 'origin/main:configs/h11_catboost_relative_v1.json'),
        preregistered_config=h11, arms={k: v for k, v in H.ARMS.items()}, added_feature_formulas=H.FORMULAS,
        warmup_sessions_excluded=H.WARMUP_SESSIONS,
        warmup_rule='training positions 0..18 of the trading calendar (2022-04-26..2022-05-26) excluded for both arms',
        catboost_fixed=proposal['fixed_parameters'],
        parameters_by_h={h: H.parameters(h11, h) for h in h11['horizons']},
        training_rule='prequential expanding: refit every origin on pairs with s+h <= origin and target <= 2025-12-31',
        imputation='median of eligible training rows per origin; observed zeros kept',
        funding_status='ASSUMED_AVAILABLE (swaprate_current)', pit_status='conditional EOD / pseudo-PIT',
        rw_source=COMMON.relative_to(ROOT).as_posix(), reproduction_reference=SAVED.relative_to(ROOT).as_posix(),
        hac='d = e_model^2 - e_reference^2, Bartlett lag max(20,h-1), n/(n-1), two-sided normal p',
        primary_family='R1 vs R0_MATCHED, h=1..7, Holm over 7', exploratory_contrasts=['R1 vs RW', 'R0_MATCHED vs RW'],
        final_test_access=False, data_end=str(frame.index.max().date()))
    json_write(design, output/'config_resolved.json')
    features_out = frame[H.ADDED].reset_index().rename(columns={'index': 'date'})
    write(features_out, output/'features_h11.csv')

    tasks = [(arm, h) for arm in [H.ORIGINAL, 'R0_MATCHED', 'R1'] for h in h11['horizons']]
    print(f'{len(tasks)} tasks on {args.jobs} processes', flush=True)
    results = parallel_map(H.run_task, tasks, args.jobs)
    pred = {arm: pd.concat([p for a, _, p, _ in results if a == arm], ignore_index=True) for arm in dict(tasks)}
    importance = pd.concat([i for a, _, _, i in results if i is not None], ignore_index=True)
    forecast_seconds = time.perf_counter()-started

    # 1. Reproduction of the saved CatBoost Core v2 run.
    saved = pd.read_csv(SAVED, parse_dates=['origin_date', 'target_date'], float_precision='round_trip')
    merged = saved[ridge.KEYS+['prediction']].merge(pred[H.ORIGINAL][ridge.KEYS+['prediction']], on=ridge.KEYS,
                                                    suffixes=('_saved', '_rerun'), validate='one_to_one')
    merged['abs_diff'] = np.abs(merged.prediction_saved-merged.prediction_rerun)
    reproduction = merged.groupby('horizon').abs_diff.agg(n='size', max_abs_diff='max').reset_index()
    reproduction['within_1e-6'] = reproduction.max_abs_diff <= 1e-6
    write(reproduction, output/'reproduction_catboost_core_v2.csv')
    write(pred[H.ORIGINAL], output/'predictions_catboost_core_v2_rerun.csv')

    # 2. Matched design: identical keys and training windows for R0 and R1.
    r0, r1 = pred['R0_MATCHED'], pred['R1']
    window_cols = ridge.KEYS+['train_n', 'train_origin_start', 'train_origin_end', 'train_label_end']
    matched = r0[window_cols].reset_index(drop=True).equals(r1[window_cols].reset_index(drop=True))
    write(r0, output/'predictions_r0.csv')
    write(r1, output/'predictions_r1.csv')

    # 3. Scoring on the common 254 pairs per h.
    wide, metrics = score({'CatBoost_Core_v2_rerun': pred[H.ORIGINAL], 'R0_MATCHED': r0, 'R1': r1}, frame)
    hac = hac_tests(wide)
    write(wide, output/'scored_h11.csv')
    write(metrics, output/'metrics_h11.csv')
    write(hac, output/'hac_h11.csv')
    agg = importance.groupby(['model_id', 'horizon', 'feature']).importance.agg(['mean', 'std', 'min', 'max']).reset_index()
    agg['mean_rank'] = agg.groupby(['model_id', 'horizon'])['mean'].rank(ascending=False, method='min')
    write(agg, output/'feature_importance_h11.csv')

    # 4. Figures (2025 targets only).
    fig = output/'figures'
    for h in h11['horizons']:
        h11_plots.forecast_figure(wide, h, metrics, fig/f'forecast_h{h}.png')
    h11_plots.forecast_panels(wide, [1, 3, 7], metrics, fig/'forecast_h1_3_7.png')
    h11_plots.rmse_ratio_bars(metrics, fig/'rmse_ratio_vs_rw.png')
    h11_plots.hac_difference(hac, fig/'mse_difference_r1_minus_r0_hac.png')
    h11_plots.importance_bars(agg, H.ADDED, fig/'feature_importance_r1.png')

    # 5. Checks and manifest.
    all_pred = pd.concat(pred.values(), ignore_index=True)
    protected_after = protected(output)
    checks = dict(
        reproduction_within_1e_6=bool(reproduction['within_1e-6'].all()),
        reproduction_max_abs_diff=float(reproduction.max_abs_diff.max()),
        r0_r1_identical_keys_and_training_windows=bool(matched),
        n_per_horizon_254=bool(metrics.groupby(['model_id', 'horizon']).n.first().eq(254).all()),
        warmup_excluded_both_arms=bool(r0.train_origin_start.min() == r1.train_origin_start.min() == frame.index[H.WARMUP_SESSIONS]),
        horizons_match_trading_calendar=check_calendar(all_pred),
        no_2026_in_predictions=bool((all_pred.target_date < '2026-01-01').all() and (all_pred.origin_date < '2026-01-01').all()),
        no_2026_in_panel=bool(frame.index.max() < pd.Timestamp('2026-01-01')),
        labels_matured=bool((r0.train_label_end <= r0.origin_date).all() and (r1.train_label_end <= r1.origin_date).all()
                            and (pred[H.ORIGINAL].train_end_date <= pred[H.ORIGINAL].origin_date).all()),
        no_missing_origin_features_r1=bool(r1.origin_features_missing.eq(0).all()),
        protected_files_unchanged=protected_before == protected_after,
        primary_family_size=int((hac.family == 'primary').sum()))
    failed = [k for k, v in checks.items() if v is False]
    manifest = dict(design_file='config_resolved.json', checks=checks, failed_checks=failed,
        git_head=git('rev-parse', 'HEAD'), git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
        git_status_porcelain=git('status', '--porcelain').splitlines(),
        data_sha256={p: H.sha256(ROOT/p) for p in [features['v2_path'], features['feature_path'], features['calendar_path']]},
        code_sha256_lf={p: hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                        for p in SOURCES if (ROOT/p).exists()},
        packages={p: version(p) for p in ['catboost', 'numpy', 'pandas', 'matplotlib', 'scipy']},
        python=platform.python_version(), platform=platform.platform(), n_jobs=args.jobs,
        seconds_forecasts=round(forecast_seconds, 1), seconds_total=round(time.perf_counter()-started, 1),
        n_fits=int(len(all_pred)), final_test_used=False, result_status='exploratory_validation_2025',
        protected_files_checked=len(protected_before),
        output_sha256={p.relative_to(output).as_posix(): H.sha256(p) for p in sorted(output.rglob('*'))
                       if p.is_file() and p.name != 'manifest.json'})
    json_write(manifest, output/'manifest.json')
    print(metrics.pivot(index='model_id', columns='horizon', values='rmse_ratio_vs_rw').round(4).to_string())
    print(hac[['family', 'model', 'versus', 'horizon', 'mean_squared_loss_difference', 'p_two_sided', 'holm_p']].round(4).to_string())
    print('checks', checks)
    if failed:
        raise SystemExit(f'Failed checks: {failed}')


if __name__ == '__main__':
    main()
