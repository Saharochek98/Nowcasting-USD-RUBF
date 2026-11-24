"""Validation-only RW drift and common-date comparisons with immutable baselines."""
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
from benchmarks import ridge, rw_drift
from scripts.run_ridge_experiments import write
from scripts.run_ridge_experiments_v2 import protected_hashes

MODELS = ['RW', 'AR1', 'RW_DRIFT', 'Ridge_Price', 'Ridge_Core_v2']


def assemble(predictions, baseline, saved_ridge, data):
    keys = ridge.KEYS
    pair = predictions[keys+['prediction', 'mu', 'n_changes', 'n_pairs']].rename(columns={'prediction':'RW_DRIFT'})
    pair['actual'] = pair.target_date.map(data.price)
    expected = set(pair[keys].itertuples(index=False, name=None))
    for model, source, identity in [('RW', baseline, 'B0'), ('AR1', baseline, 'B1'),
                                     ('Ridge_Price', saved_ridge, 'price_only'),
                                     ('Ridge_Core_v2', saved_ridge, 'core_v2')]:
        selected = source[source.model_id.eq(identity)].copy()
        if selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f'Common forecast keys differ for {model}')
        if 'actual' in selected:
            actual = selected.target_date.map(data.price)
            np.testing.assert_allclose(selected.actual, actual, atol=1e-10, rtol=0)
        pair = pair.merge(selected[keys+['prediction']].rename(columns={'prediction':model}), on=keys, validate='one_to_one')
    if not np.isfinite(pair[['actual']+MODELS]).all().all():
        raise ValueError('Nonfinite common predictions/actuals')
    return pair.sort_values(['horizon', 'origin_date']).reset_index(drop=True)


def summaries(pair, cfg):
    metrics, hac = [], []
    for h, group in pair.groupby('horizon', sort=True):
        rw_rmse = np.sqrt(np.mean((group.RW-group.actual)**2))
        for model in MODELS:
            error = group[model]-group.actual
            metrics.append(dict(model_id=model, horizon=h, n=len(group), rmse=np.sqrt(np.mean(error**2)),
                mae=np.mean(abs(error)), bias=np.mean(error), rmse_ratio_vs_rw=np.sqrt(np.mean(error**2))/rw_rmse,
                origin_min=group.origin_date.min(), origin_max=group.origin_date.max(),
                target_min=group.target_date.min(), target_max=group.target_date.max()))
        for versus in cfg['hac_family']['right']:
            losses = pd.DataFrame(dict(prediction=group.RW_DRIFT, actual=group.actual, rw_prediction=group[versus]))
            hac.append(dict(model='RW_DRIFT', versus=versus, horizon=h, n=len(group),
                            **ridge.loss_hac(losses, h, cfg)))
    hac = pd.DataFrame(hac)
    assert len(hac) == cfg['hac_family']['size']
    hac['holm_p'] = ridge.holm(hac.p_two_sided.to_numpy())
    return pd.DataFrame(metrics), hac


def diagnostics(pair, predictions, saved_ridge, data, cfg):
    core = saved_ridge[saved_ridge.model_id.eq('core_v2')]
    path = ROOT/Path(cfg['ridge_predictions']).parent/'coefficients_windows.csv'
    coefs = pd.read_csv(path, usecols=['model_id','window_id','intercept_target'], float_precision='round_trip')
    coefs = coefs[coefs.model_id.eq('core_v2')]
    assert coefs.groupby('window_id').intercept_target.nunique().eq(1).all()
    intercept = coefs.drop_duplicates('window_id').set_index('window_id').intercept_target
    detail = pair.merge(core[ridge.KEYS+['window_id']], on=ridge.KEYS, validate='one_to_one')
    detail['anchor_close'] = detail.origin_date.map(data.price)
    detail['ridge_direct_intercept'] = detail.window_id.map(intercept)
    detail['drift_correction'] = detail.horizon*detail.mu
    detail['ridge_correction'] = detail.Ridge_Core_v2-detail.anchor_close
    detail['ridge_slope_correction'] = detail.ridge_correction-detail.ridge_direct_intercept
    detail['drift_minus_ridge_price'] = detail.RW_DRIFT-detail.Ridge_Core_v2
    rows = []
    for h, group in detail.groupby('horizon'):
        rms = lambda x: float(np.sqrt(np.mean(x**2)))
        rows.append(dict(horizon=h, n=len(group), forecast_difference_rms=rms(group.drift_minus_ridge_price),
            forecast_difference_mae=float(group.drift_minus_ridge_price.abs().mean()),
            drift_correction_rms=rms(group.drift_correction), direct_ridge_intercept_rms=rms(group.ridge_direct_intercept),
            ridge_slope_correction_rms=rms(group.ridge_slope_correction),
            drift_minus_direct_intercept_rms=rms(group.drift_correction-group.ridge_direct_intercept)))
    return detail, pd.DataFrame(rows)


def run(output, cfg, canonical):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; completed results are immutable')
    ridge_cfg = ridge.config()
    before = protected_hashes(ridge_cfg)
    for path in (ROOT/'results/ridge_experiments_v2').rglob('*'):
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for p in ['data/processed/USDRUBF_baseline_2026-11-18.csv', 'configs/benchmark_config.yaml', 'docs/backtest_protocol.md']:
        before[p] = hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output/'design.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8', newline='\n')
    data, calendar = rw_drift.load_inputs(cfg)
    predictions = rw_drift.generate(data, calendar, cfg, canonical)
    write(predictions, output/'predictions_oos.csv')  # No actuals before scoring.
    baseline = ridge.load_baselines(cfg)
    saved = pd.read_csv(ROOT/cfg['ridge_predictions'], parse_dates=['origin_date','target_date'], float_precision='round_trip')
    pair = assemble(predictions, baseline, saved, data)
    assert len(pair) == 1778 and pair.groupby('horizon').size().eq(254).all()
    metrics, hac = summaries(pair, cfg)
    detail, diagnostic = diagnostics(pair, predictions, saved, data, cfg)
    write(pair, output/'common_pairs.csv')
    write(metrics, output/'metrics.csv')
    write(hac, output/'hac_comparisons.csv')
    write(detail, output/'drift_vs_ridge_pairs.csv')
    write(diagnostic, output/'drift_vs_ridge_diagnostics.csv')
    for name, digest in before.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    sources = ['benchmarks/rw_drift.py', 'scripts/run_rw_drift_experiment.py', 'configs/rw_drift_experiment.json']
    manifest = dict(config=cfg, git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],text=True),
        python=platform.python_version(), packages={p:version(p) for p in ['numpy','pandas','pyarrow']},
        protected_sha256=before, protected_files_unchanged=True,
        implementation_sha256_lf={p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources},
        prediction_count=len(predictions), common_pairs_per_horizon=254,
        final_test_used_for_modeling_or_scoring=False, data_cutoff=str(calendar[-1].date()),
        research_status=cfg['research_status'], catboost_executed=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(metrics.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/rw_drift_experiment')
    args = parser.parse_args()
    cfg, canonical = rw_drift.config()
    run(args.output_root, cfg, canonical)


if __name__ == '__main__':
    main()
