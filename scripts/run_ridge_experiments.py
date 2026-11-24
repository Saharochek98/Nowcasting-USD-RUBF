"""Validation-only staged direct Ridge. Never overwrites benchmark artifacts."""
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
from benchmarks import ridge

STAGES = {'core': (['R1', 'R2'], 'delta', 'all'),
          'extended': (['R3', 'R4', 'R5'], 'delta', 'all'),
          'log': (['R1', 'R2', 'R3', 'R4', 'R5'], 'log_return', 'all'),
          's2': (['R1', 'R2'], 'delta', 'S2')}


def write(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g', date_format='%Y-%m-%d', lineterminator='\n')


def stability(coefficients, predictions):
    expanded = predictions[['window_id', 'origin_date']].merge(coefficients, on='window_id', validate='many_to_many')
    rows = []
    for keys, group in expanded.groupby(['model_id', 'horizon', 'transform', 'scope', 'feature']):
        group = group.sort_values('origin_date')
        coef = group.coefficient_standardized
        rows.append(dict(zip(['model_id', 'horizon', 'transform', 'scope', 'feature'], keys)) |
            dict(n_origins=len(group), positive_share=float((coef > 0).mean()),
                 negative_share=float((coef < 0).mean()), zero_share=float((coef == 0).mean()),
                 coefficient_mean=coef.mean(), coefficient_std=coef.std(ddof=0),
                 coefficient_min=coef.min(), coefficient_max=coef.max(),
                 mean_abs_rank=group.abs_rank.mean(), rank_std=group.abs_rank.std(ddof=0),
                 sign_changes=int((np.diff(np.sign(coef.to_numpy())) != 0).sum()),
                 unique_windows=group.window_id.nunique()))
    return pd.DataFrame(rows)


def run(stage, output, cfg):
    models, transform, scope = STAGES[stage]
    frame, calendar, columns = ridge.load_inputs(cfg, include_current='R5' in models)
    base = ridge.load_baselines(cfg)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; previous experiments are immutable')
    output.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / cfg[key] for key in ('feature_path', 'metadata_path', 'v2_path',
                  'calendar_path', 'scenario_config', 'baseline_path')]
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    jobs, predictions, coeffs, tuning = [], [], [], []
    for model in models:
        for h in cfg['horizons']:
            pred, coef, inner, origins, fits = ridge.walk_forward(frame, columns[model], model, h, transform, cfg, scope)
            predictions.append(pred)
            coeffs.append(coef)
            tuning.extend([dict(model_id=model, horizon=h, transform=transform, scope=scope, **r) for r in inner])
            jobs.append((model, h, pred, origins, fits))
            print(stage, model, h, 'pairs', len(pred), 'alpha', pred.alpha.iloc[0], flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    coefficients = pd.concat(coeffs, ignore_index=True)
    # Persist forecasts without validation actuals before scoring or importance.
    write(predictions, output / 'predictions_oos.csv')
    write(coefficients, output / 'coefficients_windows.csv')
    write(pd.DataFrame(tuning), output / 'inner_tuning.csv')
    scores, metrics, pairs, importances, hac = [], [], [], [], []
    for model, h, pred, origins, fits in jobs:
        scored, measures, paired = ridge.common_comparison(pred, base, frame)
        scores.append(scored)
        metrics.extend([dict(model_id=model, horizon=h, transform=transform, scope=scope,
                             primary_eligible=model != 'R5', **m) for m in measures])
        pairs.append(paired)
        importance = ridge.permutation(frame, columns[model], pred, origins, fits, cfg)
        importances.extend([dict(model_id=model, horizon=h, transform=transform, scope=scope,
                                 pit_status=pred.pit_status.iloc[0], **row) for row in importance])
        hac.append(dict(model_id=model, horizon=h, transform=transform, scope=scope,
                        primary_eligible=model != 'R5', **ridge.loss_hac(paired, h, cfg)))
    scored = pd.concat(scores, ignore_index=True)
    paired = pd.concat(pairs, ignore_index=True)
    # All requested scenario pairs must share exactly the same forecast keys.
    expected = None
    for model, group in predictions.groupby('model_id'):
        keys = set(group[ridge.KEYS].itertuples(index=False, name=None))
        if expected is not None and keys != expected:
            raise ValueError('Scenario scored keys differ')
        expected = keys
    write(scored, output / 'scored_validation.csv')
    write(pd.DataFrame(metrics), output / 'metrics.csv')
    write(paired[ridge.KEYS+['model_id', 'transform', 'scope', 'prediction', 'actual', 'rw_prediction', 'ar_prediction']], output / 'common_pairs.csv')
    write(stability(coefficients, predictions), output / 'coefficient_stability.csv')
    write(pd.DataFrame(importances), output / 'permutation_importance.csv')
    # Holm correction is performed across complete prespecified families by summarize().
    write(pd.DataFrame(hac), output / 'hac_loss_comparison.csv')
    manifest = dict(stage=stage, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                    python=platform.python_version(), packages={p: version(p) for p in ['numpy', 'pandas', 'pyarrow']},
                    config=cfg, models=models, columns={m: columns[m] for m in models},
                    transform=transform, scope=scope, final_test_loaded_for_modeling=False,
                    final_test_scored=False, validation_labels_used_in_fit=False,
                    baseline_policy='saved canonical prequential RW/AR; AR may use matured validation history unlike strict Ridge train cap',
                    data_cutoff=str(calendar[-1].date()), input_sha256=fingerprints,
                    prediction_count=len(predictions), comparison_passed=True,
                    implementation_state='uncommitted changes; source hashes identify implementation')
    manifest['implementation_sha256_lf'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in [ROOT / 'benchmarks/ridge.py', ROOT / 'scripts/run_ridge_experiments.py', ROOT / 'configs/ridge_experiments.json']}
    for path, digest in fingerprints.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Protected input changed: {path}')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def summarize(root):
    metrics = pd.concat([pd.read_csv(root / s / 'metrics.csv') for s in STAGES], ignore_index=True)
    write(metrics, root / 'metrics_all.csv')
    hac = pd.concat([pd.read_csv(root / s / 'hac_loss_comparison.csv') for s in STAGES], ignore_index=True)
    hac['holm_p'] = np.nan
    for (transform, scope), group in hac[hac.primary_eligible].groupby(['transform', 'scope']):
        hac.loc[group.index, 'holm_p'] = ridge.holm(group.p_two_sided.to_numpy())
    write(hac, root / 'hac_comparisons_all.csv')
    pairs = pd.concat([pd.read_csv(root / s / 'common_pairs.csv') for s in STAGES], ignore_index=True)
    increments = []
    for (transform, scope, h), group in pairs.groupby(['transform', 'scope', 'horizon']):
        for left, right in [('R2', 'R1'), ('R3', 'R2'), ('R4', 'R3'), ('R5', 'R2')]:
            a, b = [group[group.model_id.eq(m)] for m in [left, right]]
            if a.empty or b.empty:
                continue
            common = a.merge(b, on=ridge.KEYS, suffixes=('_a', '_b'), validate='one_to_one')
            if len(common) != len(a) or len(common) != len(b):
                raise ValueError('Incremental comparison key mismatch')
            ea, eb = common.prediction_a-common.actual_a, common.prediction_b-common.actual_b
            increments.append(dict(transform=transform, scope=scope, horizon=h, model=left, versus=right,
                n=len(common), rmse_difference=np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2)),
                mean_squared_loss_difference=np.mean(ea**2-eb**2), diagnostic_only=left == 'R5'))
    write(pd.DataFrame(increments), root / 'incremental_comparisons.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=list(STAGES), required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/ridge_experiments')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    cfg = ridge.config()
    # Stage prerequisites enforce the requested implementation/experiment sequence.
    prerequisites = {'extended': ['core'], 'log': ['core', 'extended'], 's2': ['core']}
    for previous in prerequisites.get(args.stage, []):
        if not (args.output_root / previous / 'manifest.json').exists():
            raise ValueError(f'Complete and check {previous} first')
    run(args.stage, args.output_root / args.stage, cfg)
    if args.summarize:
        summarize(args.output_root)


if __name__ == '__main__':
    main()
