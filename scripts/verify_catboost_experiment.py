"""Check immutable evidence and reproduce first/middle/last origin for every h."""
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
from catboost import CatBoostRegressor
from benchmarks import catboost_direct as cb, ridge, ridge_v2
from scripts.run_catboost_experiment import check_design, json_write
from scripts.run_ridge_experiments import write


def normalized(frame):
    buffer = io.StringIO()
    write(frame,buffer)
    return pd.read_csv(io.StringIO(buffer.getvalue()),float_precision='round_trip')


def main():
    cfg,proposal,features = cb.config()
    output = ROOT/cfg['output_root']
    design = check_design(output,cfg,proposal,features)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest,name
    for name,digest in design['implementation_sha256_lf'].items():
        committed = subprocess.check_output(['git','show',design['git_sha']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).hexdigest() == digest,name
    frame,_ = cb.load_inputs(features)
    records = []
    for h in proposal['horizons']:
        folder = output/f'h{h}'
        saved = pd.read_csv(folder/'predictions_oos.csv',float_precision='round_trip')
        windows = pd.read_csv(folder/'training_windows.csv',float_precision='round_trip')
        assert len(saved)==254 and saved.scored_status.eq('unscored').all() and 'actual' not in saved
        assert (saved.train_end_date <= saved.origin_date).all()
        assert saved.target_date.between('2025-01-01','2025-12-31').all()
        assert saved.pit_status.eq('ASSUMED_AVAILABLE').all()
        inner = pd.read_csv(folder/'inner_predictions.csv',parse_dates=['origin_date','target_date','train_label_end'],float_precision='round_trip')
        assert (inner.train_label_end <= inner.origin_date).all()
        assert (inner.target_date < pd.Timestamp(saved.origin_date.min())).all()
        parameters,_ = cb.choose_parameters(inner,proposal)
        assert parameters == manifest['selected_parameters'][str(h)]['parameters']
        y = ridge.supervised(frame,h,'delta')
        for row in windows.itertuples():
            origin = frame.index.get_loc(pd.Timestamp(row.origin_date))
            train = cb.training(frame,y,h,origin,features)
            assert len(train)==row.train_n and frame.index[train[-1]+h] <= frame.index[origin]
            np.testing.assert_array_equal(np.nanmedian(frame.iloc[train].to_numpy(float),axis=0),
                [getattr(row,c+'_median') for c in ridge_v2.CORE])
        positions = [0,len(saved)//2,len(saved)-1]
        selected = saved.iloc[positions].reset_index(drop=True)
        origins = frame.index.get_indexer(pd.to_datetime(selected.origin_date))
        pred,reproduced_windows,importance,_,_ = cb.outer_forecasts(frame,h,parameters,proposal,features,origins=origins)
        pd.testing.assert_frame_equal(normalized(pred),selected,check_exact=True)
        window_subset = windows[windows.window_id.isin(selected.window_id)].reset_index(drop=True)
        # A three-row CSV can infer integer medians where the full file has floats.
        pd.testing.assert_frame_equal(normalized(reproduced_windows),window_subset,check_exact=True,check_dtype=False)
        fi = pd.read_csv(folder/'feature_importance_windows.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(normalized(importance),fi[fi.window_id.isin(selected.window_id)].reset_index(drop=True),check_exact=True)
        metadata = json.loads((folder/'final_model.json').read_text())
        final = CatBoostRegressor().load_model(str(folder/'final_model.cbm'))
        last = saved.iloc[-1]
        x = frame.loc[[pd.Timestamp(last.origin_date)]].to_numpy(float)
        correction = cb.predict(final,np.array(metadata['imputer_median']),x)[0]
        assert float(frame.loc[pd.Timestamp(last.origin_date),'close']+correction) == last.prediction
        records.append(dict(horizon=h,reproduced_origins=selected.origin_date.tolist(),
                            predictions_exact=True,imputers_exact=True,importance_exact=True,final_model_roundtrip_exact=True))
    hac = pd.read_csv(output/'hac_comparisons.csv')
    assert len(hac)==21 and hac.n.eq(254).all()
    json_write(dict(committed_source_verified=True,all_protected_files_unchanged=True,
        all_output_hashes_verified=True,all_training_windows_medians_and_maturity_verified=True,
        inner_selection_recomputed=True,real_data_reproduction=records,
        numeric_reproduction_scope='21 first/middle/last origin refits across all h, plus all seven saved final models',
        binary_models='Byte identity across retraining is not asserted; serialized model predictions are checked'),output/'verification.json')
    print('PASS: committed sources; immutable inputs; 1778 training windows; 21 exact refits; 7 model roundtrips')


if __name__ == '__main__': main()
