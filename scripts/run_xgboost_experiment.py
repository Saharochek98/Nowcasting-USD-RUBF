"""Sealed inner tuning, prequential forecasts, then one common-date evaluation."""
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
from benchmarks import xgboost_direct as xgb, ridge, ridge_v2
from scripts.run_catboost_experiment import protected as historical_protected, json_write
from scripts.run_ridge_experiments import write

SOURCES = ['benchmarks/xgboost_direct.py','scripts/run_xgboost_experiment.py','configs/xgboost_core_v2.json',
           'requirements-xgboost.txt','benchmarks/catboost_direct.py','benchmarks/ridge.py','benchmarks/ridge_v2.py']


def protected():
    hashes = historical_protected()
    for path in (ROOT/'results/catboost_core_v2').rglob('*'):
        if path.is_file(): hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['configs/catboost_experiment.json','scripts/run_catboost_experiment.py','reports/catboost_core_v2.md']:
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return hashes


def initialize(output,cfg,features):
    if output.exists() and any(output.iterdir()): raise ValueError('Use an empty output root')
    output.mkdir(parents=True,exist_ok=True)
    json_write(dict(config=cfg,training_config=features,features=ridge_v2.CORE,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        protected_sha256=protected(),implementation_sha256_lf={p:hashlib.sha256(
            (ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES},
        packages={p:version(p) for p in ['xgboost','scikit-learn','numpy','pandas','scipy','pyarrow']},
        python=platform.python_version(),platform=platform.platform(),final_test_access=False),output/'design.json')


def check_design(output,cfg,features):
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    if design['config'] != cfg or design['training_config'] != features or design['protected_sha256'] != protected():
        raise ValueError('Frozen design or prior evidence changed')
    for name,digest in design['implementation_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError('Implementation changed after initialization')
    for name,package_version in design['packages'].items():
        if version(name) != package_version: raise ValueError('Library version changed during experiment')
    return design


def tune(output,h,cfg,features):
    check_design(output,cfg,features)
    folder = output/f'h{h}'
    if folder.exists(): raise ValueError('Tuning already exists')
    folder.mkdir()
    frame,_ = xgb.load_inputs(features)
    started = time.perf_counter()
    inner = pd.concat([xgb.inner_evaluate(frame,h,p,i,cfg,features,progress=lambda s:print(s,flush=True))
                       for i,p in enumerate(xgb.candidates(cfg))],ignore_index=True)
    # Match CatBoost's inner schedule, without using its candidate losses.
    columns = ['candidate_id','horizon','fold','origin_date','target_date','train_n','train_label_end']
    old = pd.read_csv(ROOT/f'results/catboost_core_v2/h{h}/inner_predictions.csv',usecols=columns,
                     parse_dates=['origin_date','target_date','train_label_end'])
    pd.testing.assert_frame_equal(inner[columns].reset_index(drop=True),old[columns].reset_index(drop=True))
    parameters,metrics = xgb.choose_parameters(inner,cfg)
    write(inner,folder/'inner_predictions.csv')
    write(metrics,folder/'inner_metrics.csv')
    json_write(dict(horizon=h,parameters=parameters,validation_used_for_selection=False,
                    catboost_inner_schedule_identical=True),folder/'selection.json')
    json_write(dict(seconds=time.perf_counter()-started,n_fits=len(inner)),folder/'runtime_tuning.json')
    print('SELECTED',h,parameters,flush=True)


def forecast(output,h,cfg,features):
    check_design(output,cfg,features)
    selections = {str(k):json.loads((output/f'h{k}'/'selection.json').read_text()) for k in cfg['horizons']}
    folder = output/f'h{h}'
    if (folder/'predictions_oos.csv').exists(): raise ValueError('Outer predictions already exist')
    frame,_ = xgb.load_inputs(features)
    started = time.perf_counter()
    pred,windows,importance,model,median = xgb.outer_forecasts(frame,h,selections[str(h)]['parameters'],cfg,features,
                                                           progress=lambda s:print(s,flush=True))
    assert len(pred)==254 and 'actual' not in pred
    write(pred,folder/'predictions_oos.csv')
    write(windows,folder/'training_windows.csv')
    write(importance,folder/'gain_windows.csv')
    model.save_model(str(folder/'final_model.ubj'))
    json_write(dict(feature_order=ridge_v2.CORE,imputer_median=median.tolist(),
        origin_date=str(pred.origin_date.iloc[-1].date()),resolved_booster_config=json.loads(model.get_booster().save_config()),
        selections_before_outer=selections),folder/'final_model.json')
    json_write(dict(seconds=time.perf_counter()-started,n_fits=len(pred)),folder/'runtime_outer.json')


def common_pairs(predictions,baseline,frame):
    columns = ['actual','RW','AR1','RW_DRIFT','Ridge_Core_v2','CatBoost_Core_v2']
    if predictions.duplicated(ridge.KEYS).any() or baseline.duplicated(ridge.KEYS).any(): raise ValueError('Duplicate keys')
    if set(predictions[ridge.KEYS].itertuples(index=False,name=None)) != set(baseline[ridge.KEYS].itertuples(index=False,name=None)):
        raise ValueError('Comparison dates differ')
    pairs = predictions.merge(baseline[ridge.KEYS+columns],on=ridge.KEYS,validate='one_to_one')
    np.testing.assert_allclose(pairs.actual,pairs.target_date.map(frame.close),atol=1e-10,rtol=0)
    if not np.isfinite(pairs[['prediction']+columns]).all().all(): raise ValueError('Nonfinite scoring values')
    return pairs.sort_values(['horizon','origin_date']).reset_index(drop=True)


def summarize(output,cfg,features):
    design = check_design(output,cfg,features)
    if (output/'manifest.json').exists(): raise ValueError('Completed experiments are immutable')
    pred = pd.concat([pd.read_csv(output/f'h{h}'/'predictions_oos.csv',parse_dates=['origin_date','target_date'],
                                  float_precision='round_trip') for h in cfg['horizons']],ignore_index=True)
    assert len(pred)==1778 and pred.groupby('horizon').size().eq(254).all()
    write(pred,output/'predictions_oos.csv')
    old = pd.read_csv(ROOT/cfg['comparison_path'],parse_dates=['origin_date','target_date'],float_precision='round_trip')
    # Historical CatBoost stores its forecast in the generic prediction column.
    old = old.rename(columns={'prediction':'CatBoost_Core_v2'})
    frame,_ = xgb.load_inputs(features)
    pairs = common_pairs(pred,old,frame)
    metrics,hac = [],[]
    models = [('RW','RW'),('AR1','AR1'),('RW_DRIFT','RW_DRIFT'),('Ridge_Core_v2','Ridge_Core_v2'),
              ('CatBoost_Core_v2','CatBoost_Core_v2'),('XGBoost_Core_v2','prediction')]
    for h,group in pairs.groupby('horizon'):
        rw_rmse = np.sqrt(np.mean((group.RW-group.actual)**2))
        for model,column in models:
            e = group[column]-group.actual
            metrics.append(dict(model_id=model,horizon=h,n=len(group),rmse=np.sqrt(np.mean(e**2)),mae=np.mean(abs(e)),
                bias=np.mean(e),rmse_ratio_vs_rw=np.sqrt(np.mean(e**2))/rw_rmse,
                origin_min=group.origin_date.min(),origin_max=group.origin_date.max(),
                target_min=group.target_date.min(),target_max=group.target_date.max()))
        for versus in cfg['hac_family']['versus']:
            loss = pd.DataFrame(dict(prediction=group.prediction,actual=group.actual,rw_prediction=group[versus]))
            hac.append(dict(model='XGBoost_Core_v2',versus=versus,horizon=h,n=len(group),**ridge.loss_hac(loss,h,cfg)))
    hac = pd.DataFrame(hac)
    assert len(hac)==cfg['hac_family']['size']
    hac['holm_p'] = ridge.holm(hac.p_two_sided.to_numpy())
    write(pairs,output/'scored_common_predictions.csv')
    write(pd.DataFrame(metrics),output/'metrics.csv')
    write(hac,output/'hac_comparisons.csv')
    gain = pd.concat([pd.read_csv(output/f'h{h}'/'gain_windows.csv') for h in cfg['horizons']],ignore_index=True)
    aggregate = gain.groupby(['horizon','feature']).agg(gain_mean=('gain','mean'),gain_std=('gain','std'),
        gain_share_mean=('gain_share','mean'),gain_share_std=('gain_share','std'),mean_rank=('gain_rank','mean'),
        rank_std=('gain_rank','std'),positive_gain_share=('gain',lambda s:np.mean(s>0))).reset_index()
    aggregate['aggregate_rank'] = aggregate.groupby('horizon').gain_share_mean.rank(ascending=False,method='min')
    write(aggregate,output/'gain_importance.csv')
    cat = pd.read_csv(ROOT/cfg['catboost_importance'])[['horizon','feature','mean','mean_rank']]
    cat = cat.rename(columns={'mean':'catboost_pvc_mean','mean_rank':'catboost_rank'})
    comparison = aggregate.merge(cat,on=['horizon','feature'],validate='one_to_one')
    write(comparison,output/'importance_vs_catboost.csv')
    manifest = dict(design,selected_parameters={str(h):json.loads((output/f'h{h}'/'selection.json').read_text()) for h in cfg['horizons']},
        prediction_count=len(pred),common_pairs_per_horizon=254,protected_files_unchanged=design['protected_sha256']==protected(),
        result_status=cfg['research_status'],funding_status='ASSUMED_AVAILABLE',data_cutoff=str(frame.index[-1].date()),
        early_stopping=False,validation_used_for_selection=False,final_test_used_for_modeling_or_scoring=False,
        output_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob('*') if p.is_file() and 'runtime_' not in p.name})
    assert manifest['protected_files_unchanged']
    json_write(manifest,output/'manifest.json')
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','tune','forecast','summarize'])
    parser.add_argument('--horizon',type=int,choices=range(1,8))
    parser.add_argument('--output-root',type=Path,default=ROOT/'results/xgboost_core_v2')
    args = parser.parse_args()
    cfg,features = xgb.config()
    if args.stage=='init': initialize(args.output_root,cfg,features)
    elif args.stage=='summarize': summarize(args.output_root,cfg,features)
    else:
        if args.horizon is None: parser.error('--horizon required')
        (tune if args.stage=='tune' else forecast)(args.output_root,args.horizon,cfg,features)


if __name__ == '__main__': main()
