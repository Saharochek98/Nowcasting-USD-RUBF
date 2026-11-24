"""Immutable stages: init -> all inner tuning -> all outer forecasts -> scoring."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from benchmarks import catboost_direct as cb, ridge, ridge_v2
from scripts.run_ridge_experiments import write

SOURCES = ['benchmarks/catboost_direct.py','scripts/run_catboost_experiment.py',
           'configs/catboost_experiment.json','requirements-catboost.txt']


def json_write(value,path):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')


def protected():
    paths = [p for p in (ROOT/'data').rglob('*.csv') if p.is_file()]
    for folder in ['ridge_experiments','ridge_experiments_v2','rw_drift_experiment',
                   'benchmark_calendar_v2','funding_missingness_audit']:
        paths.extend(p for p in (ROOT/'results'/folder).rglob('*') if p.is_file())
    paths += [ROOT/p for p in ['configs/catboost_core_v2_proposal.json','configs/feature_scenarios_v1.json',
        'configs/ridge_experiments.json','configs/ridge_experiments_v2.json','docs/backtest_protocol.md',
        'docs/feature_availability_catalog.csv','benchmarks/ridge.py','benchmarks/ridge_v2.py',
        'reports/catboost_go_no_go.md']]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def initialize(output,cfg,proposal,features):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output root')
    output.mkdir(parents=True,exist_ok=True)
    design = dict(config=cfg,proposal=proposal,features=ridge_v2.CORE,training_config=features,
        authorization='Explicit user request to execute the historical proposal',
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        protected_sha256=protected(),implementation_sha256_lf={p:hashlib.sha256(
            (ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES},
        packages={p:version(p) for p in ['catboost','numpy','pandas','pyarrow','scipy']},
        python=platform.python_version(),platform=platform.platform(),final_test_access=False)
    json_write(design,output/'design.json')


def check_design(output,cfg,proposal,features):
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    if (design['config'] != cfg or design['proposal'] != proposal or design['training_config'] != features
            or design['protected_sha256'] != protected()):
        raise ValueError('Frozen design/protected evidence changed')
    for name,digest in design['implementation_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError('Implementation changed during experiment')
    return design


def tune(output,h,cfg,proposal,features):
    check_design(output,cfg,proposal,features)
    folder = output/f'h{h}'
    if folder.exists():
        raise ValueError('Horizon tuning output already exists')
    folder.mkdir()
    frame,_ = cb.load_inputs(features)
    started = time.perf_counter()
    results = []
    for i,parameters in enumerate(cb.candidates(proposal)):
        result = cb.inner_evaluate(frame,h,parameters,i,proposal,features,
                                  progress=lambda msg:print(msg,flush=True))
        results.append(result)
    inner = pd.concat(results,ignore_index=True)
    parameters,metrics = cb.choose_parameters(inner,proposal)
    write(inner,folder/'inner_predictions.csv')
    write(metrics,folder/'inner_metrics.csv')
    json_write(dict(horizon=h,parameters=parameters,tuning_targets_before_first_outer=True,
                    validation_used_for_selection=False),folder/'selection.json')
    json_write(dict(seconds=time.perf_counter()-started,n_fits=len(inner)),folder/'runtime_tuning.json')
    print('SELECTED',h,parameters,flush=True)


def forecast(output,h,cfg,proposal,features):
    check_design(output,cfg,proposal,features)
    # All seven selections are sealed before the first outer forecast is made.
    selections = {str(k):json.loads((output/f'h{k}'/'selection.json').read_text()) for k in proposal['horizons']}
    folder = output/f'h{h}'
    if (folder/'predictions_oos.csv').exists():
        raise ValueError('Outer predictions already exist')
    frame,_ = cb.load_inputs(features)
    started = time.perf_counter()
    pred,windows,importance,model,median = cb.outer_forecasts(frame,h,selections[str(h)]['parameters'],
        proposal,features,progress=lambda msg:print(msg,flush=True))
    assert len(pred) == 254 and 'actual' not in pred
    write(pred,folder/'predictions_oos.csv')
    write(windows,folder/'training_windows.csv')
    write(importance,folder/'feature_importance_windows.csv')
    model.save_model(str(folder/'final_model.cbm'))
    json_write(dict(feature_order=ridge_v2.CORE,imputer_median=median.tolist(),
        origin_date=str(pred.origin_date.iloc[-1].date()),train_label_end=str(pred.train_end_date.iloc[-1].date()),
        resolved_catboost_parameters=model.get_all_params(),selections_before_outer=selections),folder/'final_model.json')
    json_write(dict(seconds=time.perf_counter()-started,n_fits=len(pred)),folder/'runtime_outer.json')


def common_pairs(predictions,baseline,frame):
    if predictions.duplicated(ridge.KEYS).any() or baseline.duplicated(ridge.KEYS).any():
        raise ValueError('Duplicate comparison keys')
    if set(predictions[ridge.KEYS].itertuples(index=False,name=None)) != set(baseline[ridge.KEYS].itertuples(index=False,name=None)):
        raise ValueError('Comparison dates differ')
    pair = predictions.merge(baseline[ridge.KEYS+['actual','RW','AR1','RW_DRIFT','Ridge_Core_v2']],
                             on=ridge.KEYS,validate='one_to_one')
    np.testing.assert_allclose(pair.actual,pair.target_date.map(frame.close),atol=1e-10,rtol=0)
    if not np.isfinite(pair[['prediction','actual','RW','AR1','RW_DRIFT','Ridge_Core_v2']]).all().all():
        raise ValueError('Nonfinite evaluation value')
    return pair.sort_values(['horizon','origin_date']).reset_index(drop=True)


def summarize(output,cfg,proposal,features):
    design = check_design(output,cfg,proposal,features)
    if (output/'manifest.json').exists():
        raise ValueError('Completed experiment is immutable')
    pred = pd.concat([pd.read_csv(output/f'h{h}'/'predictions_oos.csv',
        parse_dates=['origin_date','target_date'],float_precision='round_trip') for h in proposal['horizons']],ignore_index=True)
    assert len(pred) == 1778 and pred.groupby('horizon').size().eq(254).all()
    write(pred,output/'predictions_oos.csv')
    baseline = pd.read_csv(ROOT/cfg['comparison_path'],parse_dates=['origin_date','target_date'],float_precision='round_trip')
    frame,_ = cb.load_inputs(features)
    pair = common_pairs(pred,baseline,frame)
    metrics,hac = [],[]
    for h,group in pair.groupby('horizon'):
        rw_rmse = np.sqrt(np.mean((group.RW-group.actual)**2))
        for model,column in [('RW','RW'),('AR1','AR1'),('RW_DRIFT','RW_DRIFT'),
                             ('Ridge_Core_v2','Ridge_Core_v2'),('CatBoost_Core_v2','prediction')]:
            error = group[column]-group.actual
            metrics.append(dict(model_id=model,horizon=h,n=len(group),rmse=np.sqrt(np.mean(error**2)),
                mae=np.mean(abs(error)),bias=np.mean(error),rmse_ratio_vs_rw=np.sqrt(np.mean(error**2))/rw_rmse,
                origin_min=group.origin_date.min(),origin_max=group.origin_date.max(),
                target_min=group.target_date.min(),target_max=group.target_date.max()))
        for versus in proposal['hac_family']['versus']:
            losses = pd.DataFrame(dict(prediction=group.prediction,actual=group.actual,rw_prediction=group[versus]))
            hac.append(dict(model='CatBoost_Core_v2',versus=versus,horizon=h,n=len(group),
                            **ridge.loss_hac(losses,h,cfg)))
    hac = pd.DataFrame(hac)
    assert len(hac) == proposal['hac_family']['size']
    hac['holm_p'] = ridge.holm(hac.p_two_sided.to_numpy())
    write(pair,output/'scored_common_predictions.csv')
    write(pd.DataFrame(metrics),output/'metrics.csv')
    write(hac,output/'hac_comparisons.csv')
    importance = pd.concat([pd.read_csv(output/f'h{h}'/'feature_importance_windows.csv') for h in proposal['horizons']],ignore_index=True)
    aggregate = importance.groupby(['horizon','feature']).importance.agg(['mean','std','min','max']).reset_index()
    aggregate['mean_rank'] = aggregate.groupby('horizon')['mean'].rank(ascending=False,method='min')
    write(aggregate,output/'feature_importance.csv')
    selections = {str(h):json.loads((output/f'h{h}'/'selection.json').read_text()) for h in proposal['horizons']}
    manifest = dict(design,selected_parameters=selections,prediction_count=len(pred),common_pairs_per_horizon=254,
        protected_files_unchanged=design['protected_sha256']==protected(),
        data_cutoff=str(frame.index[-1].date()),funding_status='ASSUMED_AVAILABLE',
        result_status=cfg['research_status'],early_stopping=False,validation_used_for_selection=False,
        final_test_used_for_modeling_or_scoring=False,
        output_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob('*') if p.is_file() and 'runtime_' not in p.name})
    assert manifest['protected_files_unchanged']
    json_write(manifest,output/'manifest.json')
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','tune','forecast','summarize'])
    parser.add_argument('--horizon',type=int,choices=range(1,8))
    parser.add_argument('--output-root',type=Path,default=ROOT/'results/catboost_core_v2')
    args = parser.parse_args()
    cfg,proposal,features = cb.config()
    if args.stage == 'init': initialize(args.output_root,cfg,proposal,features)
    elif args.stage == 'summarize': summarize(args.output_root,cfg,proposal,features)
    else:
        if args.horizon is None: parser.error('--horizon is required')
        (tune if args.stage == 'tune' else forecast)(args.output_root,args.horizon,cfg,proposal,features)


if __name__ == '__main__': main()
