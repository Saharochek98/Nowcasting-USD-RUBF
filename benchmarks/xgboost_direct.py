"""XGBRegressor, preserving CatBoost's matured-label/calendar/imputation protocol."""
import itertools
import json

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as protocol, ridge, ridge_v2

ROOT = ridge.ROOT
load_inputs = protocol.load_inputs
training = protocol.training
origins_for_validation = protocol.origins_for_validation


def config():
    cfg = json.loads((ROOT/'configs/xgboost_core_v2.json').read_text())
    features = ridge_v2.config(ROOT/cfg['feature_config'])
    fixed = dict(n_estimators=100,learning_rate=.03,min_child_weight=5,subsample=1.,colsample_bytree=1.,
        objective='reg:squarederror',tree_method='hist',random_state=240112,n_jobs=1)
    if (cfg['fixed_parameters'] != fixed or cfg['parameter_grid'] != {'max_depth':[2,3],'reg_lambda':[10,30]}
            or cfg['scenario'] != 'core_v2' or cfg['funding_status'] != 'ASSUMED_AVAILABLE'
            or cfg['horizons'] != list(range(1,8)) or cfg['validation_targets'] != ['2025-01-01','2025-12-31']
            or cfg['inner_origin_windows'] != features['inner_origin_windows']
            or cfg['minimum_train_pairs'] != features['minimum_train_pairs']
            or cfg['early_stopping'] is not False or cfg['final_test_access'] is not False
            or cfg['hac_family'] != dict(versus=['RW','Ridge_Core_v2','CatBoost_Core_v2'],size=21,adjustment='Holm jointly')):
        raise ValueError('XGBoost fixed design changed')
    return cfg,ridge_v2.policy_config(features,'prequential')


def candidates(cfg):
    return [dict(max_depth=d,reg_lambda=l) for d,l in itertools.product(
        cfg['parameter_grid']['max_depth'],cfg['parameter_grid']['reg_lambda'])]


def fit_model(x,y,parameters,cfg):
    from xgboost import XGBRegressor
    x,y = np.asarray(x,float),np.asarray(y,float)
    if not np.isfinite(y).all() or np.isinf(x).any() or np.isnan(x).all(axis=0).any():
        raise ValueError('Invalid eligible training feature/target')
    median = np.nanmedian(x,axis=0)
    model = XGBRegressor(**cfg['fixed_parameters'],**parameters)
    model.fit(np.where(np.isnan(x),median,x),y,verbose=False)  # No eval_set/early stopping.
    return model,median


def predict(model,median,x):
    x = np.asarray(x,float)
    result = np.asarray(model.predict(np.where(np.isnan(x),median,x)))
    if not np.isfinite(result).all(): raise ValueError('Nonfinite predicted correction')
    return result


def gain(model,n_features):
    mapping = model.get_booster().get_score(importance_type='gain')
    raw = np.array([mapping.get(f'f{k}',0.) for k in range(n_features)])
    normalized = raw/raw.sum() if raw.sum() else np.zeros(n_features)
    return raw,normalized


def inner_evaluate(frame,h,parameters,candidate_id,cfg,features,progress=None):
    first_outer = int(origins_for_validation(frame,h,features)[0])
    y = ridge.supervised(frame,h,'delta')
    x = frame[ridge_v2.CORE].to_numpy(float)
    rows = []
    for fold,(start,end) in enumerate(cfg['inner_origin_windows']):
        origins = np.flatnonzero((frame.index >= start)&(frame.index <= end))
        origins = origins[(origins+h < len(frame))&(origins+h < first_outer)]
        for origin in origins:
            train = training(frame,y,h,origin,features)
            model,median = fit_model(x[train],y[train],parameters,cfg)
            prediction = float(frame.close.iloc[origin]+float(predict(model,median,x[[origin]])[0]))
            actual = float(frame.close.iloc[origin+h])
            rows.append(dict(candidate_id=candidate_id,horizon=h,fold=fold,**parameters,
                origin_date=frame.index[origin],target_date=frame.index[origin+h],train_n=len(train),
                train_label_end=frame.index[train[-1]+h],prediction=prediction,actual=actual,error=prediction-actual))
        if progress: progress(f'h={h} candidate={candidate_id} fold={fold} n={len(origins)} complete')
    if not rows: raise ValueError('No eligible inner origins')
    return pd.DataFrame(rows)


def choose_parameters(inner,cfg):
    expected = None
    for _,group in inner.groupby('candidate_id'):
        keys = set(group[ridge.KEYS].itertuples(index=False,name=None))
        if group.duplicated(ridge.KEYS).any() or (expected is not None and expected != keys):
            raise ValueError('Inner candidate keys differ')
        expected = keys
    table = inner.groupby('candidate_id',sort=True).agg(n=('error','size'),mse=('error',lambda e:np.mean(e**2))).reset_index()
    if len(table) != len(candidates(cfg)) or not table.n.eq(table.n.iloc[0]).all():
        raise ValueError('Incomplete candidates')
    table['rmse'] = np.sqrt(table.mse)
    winner = int(table.sort_values(['mse','candidate_id'],kind='stable').candidate_id.iloc[0])
    table['selected'] = table.candidate_id.eq(winner)
    return candidates(cfg)[winner],table


def outer_forecasts(frame,h,parameters,cfg,features,origins=None,progress=None):
    y = ridge.supervised(frame,h,'delta')
    x = frame[ridge_v2.CORE].to_numpy(float)
    if origins is None: origins = origins_for_validation(frame,h,features)
    predictions,windows,importance = [],[],[]
    for j,origin in enumerate(origins):
        train = training(frame,y,h,origin,features)
        model,median = fit_model(x[train],y[train],parameters,cfg)
        correction = float(predict(model,median,x[[origin]])[0])
        window = f'xgboost_h{h}_{frame.index[origin].date()}'
        predictions.append(dict(model_id='XGBoost_Core_v2',model_version=cfg['version'],
            origin_date=frame.index[origin],target_date=frame.index[origin+h],horizon=h,
            prediction=float(frame.close.iloc[origin]+correction),prediction_target=correction,
            window_id=window,train_n=len(train),train_end_date=frame.index[train[-1]+h],train_origin_end=frame.index[train[-1]],
            split='validation',scored_status='unscored',forecast_origin='',forecast_origin_mode='conditional_EOD',
            pit_status='ASSUMED_AVAILABLE',result_status=cfg['research_status'],data_version='frozen_v2_core_v2',**parameters))
        windows.append(dict(window_id=window,horizon=h,origin_date=frame.index[origin],train_n=len(train),
            train_origin_start=frame.index[train[0]],train_origin_end=frame.index[train[-1]],train_label_end=frame.index[train[-1]+h],
            **{feature+'_median':median[k] for k,feature in enumerate(ridge_v2.CORE)}))
        raw,share = gain(model,len(ridge_v2.CORE))
        ranks = pd.Series(raw).rank(ascending=False,method='min').to_numpy()
        importance.extend(dict(window_id=window,horizon=h,origin_date=frame.index[origin],feature=f,
            gain=float(raw[k]),gain_share=float(share[k]),gain_rank=float(ranks[k])) for k,f in enumerate(ridge_v2.CORE))
        if progress and (j+1)%50==0: progress(f'h={h} outer {j+1}/{len(origins)}')
    return pd.DataFrame(predictions),pd.DataFrame(windows),pd.DataFrame(importance),model,median
