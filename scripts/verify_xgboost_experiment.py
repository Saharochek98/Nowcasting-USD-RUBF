"""Verify all train windows and repeat first/middle/last forecast of every h."""
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from benchmarks import xgboost_direct as xgb, ridge, ridge_v2
from scripts.run_xgboost_experiment import check_design
from scripts.run_catboost_experiment import json_write
from scripts.run_ridge_experiments import write


def normalized(frame):
    buffer = io.StringIO()
    write(frame,buffer)
    return pd.read_csv(io.StringIO(buffer.getvalue()),float_precision='round_trip')


def main():
    cfg,features = xgb.config()
    output = ROOT/cfg['output_root']
    design = check_design(output,cfg,features)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    for name,digest in design['implementation_sha256_lf'].items():
        committed = subprocess.check_output(['git','show',design['git_sha']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).hexdigest()==digest,name
    frame,_ = xgb.load_inputs(features)
    records = []
    for h in cfg['horizons']:
        folder = output/f'h{h}'
        pred = pd.read_csv(folder/'predictions_oos.csv',float_precision='round_trip')
        windows = pd.read_csv(folder/'training_windows.csv',float_precision='round_trip')
        assert len(pred)==254 and 'actual' not in pred and pred.scored_status.eq('unscored').all()
        assert (pred.train_end_date <= pred.origin_date).all()
        assert pred.target_date.between('2025-01-01','2025-12-31').all()
        assert pred.pit_status.eq('ASSUMED_AVAILABLE').all()
        inner = pd.read_csv(folder/'inner_predictions.csv',parse_dates=['origin_date','target_date','train_label_end'],float_precision='round_trip')
        assert (inner.train_label_end <= inner.origin_date).all()
        assert (inner.target_date < pd.Timestamp(pred.origin_date.min())).all()
        parameters,_ = xgb.choose_parameters(inner,cfg)
        assert parameters == manifest['selected_parameters'][str(h)]['parameters']
        old_keys = ['candidate_id','horizon','fold','origin_date','target_date','train_n','train_label_end']
        old = pd.read_csv(ROOT/f'results/catboost_core_v2/h{h}/inner_predictions.csv',usecols=old_keys,
                          parse_dates=['origin_date','target_date','train_label_end'])
        pd.testing.assert_frame_equal(inner[old_keys],old[old_keys])
        y = ridge.supervised(frame,h,'delta')
        for row in windows.itertuples():
            origin = frame.index.get_loc(pd.Timestamp(row.origin_date))
            train = xgb.training(frame,y,h,origin,features)
            assert len(train)==row.train_n and frame.index[train[-1]+h] <= frame.index[origin]
            np.testing.assert_array_equal(np.nanmedian(frame.iloc[train].to_numpy(float),axis=0),
                                          [getattr(row,c+'_median') for c in ridge_v2.CORE])
        selected = pred.iloc[[0,len(pred)//2,len(pred)-1]].reset_index(drop=True)
        origins = frame.index.get_indexer(pd.to_datetime(selected.origin_date))
        p,w,g,_,_ = xgb.outer_forecasts(frame,h,parameters,cfg,features,origins=origins)
        pd.testing.assert_frame_equal(normalized(p),selected,check_exact=True)
        pd.testing.assert_frame_equal(normalized(w),windows[windows.window_id.isin(selected.window_id)].reset_index(drop=True),
                                      check_exact=True,check_dtype=False)
        saved_gain = pd.read_csv(folder/'gain_windows.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(normalized(g),saved_gain[saved_gain.window_id.isin(selected.window_id)].reset_index(drop=True),
                                      check_exact=True,check_dtype=False)
        metadata = json.loads((folder/'final_model.json').read_text())
        model = XGBRegressor(n_jobs=1)
        model.load_model(folder/'final_model.ubj')
        last = pred.iloc[-1]
        x = frame.loc[[pd.Timestamp(last.origin_date)]].to_numpy(float)
        correction = float(xgb.predict(model,np.array(metadata['imputer_median']),x)[0])
        assert float(frame.loc[pd.Timestamp(last.origin_date),'close']+correction)==last.prediction
        records.append(dict(horizon=h,origins=selected.origin_date.tolist(),predictions_exact=True,
                            imputers_exact=True,gains_exact=True,saved_model_prediction_exact=True))
    hac = pd.read_csv(output/'hac_comparisons.csv')
    assert len(hac)==21 and hac.n.eq(254).all()
    json_write(dict(committed_source_verified=True,all_previous_data_and_results_unchanged=True,
        all_output_hashes_verified=True,all_1778_training_windows_medians_and_maturity_verified=True,
        catboost_inner_schedule_identical=True,inner_selection_recomputed=True,
        exact_real_data_refits=records,numeric_reproduction_scope='21 first/middle/last refits plus seven saved model reloads'),
        output/'verification.json')
    print('PASS: all protected files; CatBoost-identical inner schedule; 1778 train windows; 21 exact refits; 7 model reloads')


if __name__ == '__main__': main()
