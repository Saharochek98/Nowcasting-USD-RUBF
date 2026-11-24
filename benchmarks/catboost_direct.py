"""Direct delta CatBoost with explicit matured labels and train-only imputation."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import ridge, ridge_v2

ROOT = ridge.ROOT


def config():
    cfg = json.loads((ROOT/'configs/catboost_experiment.json').read_text())
    proposal = json.loads((ROOT/cfg['proposal']).read_text())
    features = ridge_v2.config(ROOT/cfg['feature_config'])
    if (proposal['parameter_grid'] != {'depth':[2,3], 'l2_leaf_reg':[10,30]}
            or proposal['fixed_parameters'] != dict(iterations=100, learning_rate=.03, loss_function='RMSE',
                random_seed=240112, thread_count=1, allow_writing_files=False, bootstrap_type='No', verbose=False)
            or proposal['early_stopping'] is not False or proposal['final_test_access'] is not False
            or proposal['horizons'] != list(range(1,8)) or proposal['scenario'] != 'core_v2'
            or proposal['validation_targets'] != ['2025-01-01','2025-12-31']
            or proposal['funding_status'] != 'ASSUMED_AVAILABLE'
            or proposal['hac_family'] != dict(versus=['RW','RW_DRIFT','Ridge_Core_v2'],size=21,adjustment='Holm jointly')):
        raise ValueError('CatBoost proposal changed outside the fixed experiment')
    if proposal['inner_origin_windows'] != features['inner_origin_windows']:
        raise ValueError('Inner windows differ from the frozen design')
    return cfg, proposal, ridge_v2.policy_config(features,'prequential')


def load_inputs(features):
    frame, calendar, columns = ridge_v2.load_inputs(features)
    assert columns['core_v2'] == ridge_v2.CORE
    return frame[ridge_v2.CORE].copy(), calendar


def candidates(proposal):
    return [dict(depth=d,l2_leaf_reg=l) for d,l in itertools.product(
        proposal['parameter_grid']['depth'],proposal['parameter_grid']['l2_leaf_reg'])]


def fit_model(x, y, parameters, proposal):
    from catboost import CatBoostRegressor
    x, y = np.asarray(x,float), np.asarray(y,float)
    if not np.isfinite(y).all() or np.isinf(x).any() or np.isnan(x).all(axis=0).any():
        raise ValueError('Invalid or all-missing eligible training feature/target')
    median = np.nanmedian(x,axis=0)
    filled = np.where(np.isnan(x),median,x)
    model = CatBoostRegressor(**proposal['fixed_parameters'],**parameters)
    model.fit(filled,y,use_best_model=False)  # No eval_set or outer early stopping.
    return model, median


def predict(model, median, x):
    x = np.asarray(x,float)
    result = np.asarray(model.predict(np.where(np.isnan(x),median,x),thread_count=1))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite CatBoost correction')
    return result


def origins_for_validation(frame,h,features):
    origins = np.arange(len(frame)-h)
    return origins[(frame.index[origins+h] >= features['validation_start']) &
                   (frame.index[origins+h] <= features['validation_end'])]


def training(frame,y,h,origin,features):
    indices = ridge.training_indices(frame,y,h,int(origin),features)
    if len(indices) < features['minimum_train_pairs']:
        raise ValueError('Insufficient matured training labels')
    return indices


def inner_evaluate(frame,h,parameters,candidate_id,proposal,features,progress=None):
    """Prequential *within* folds; only targets strictly before first outer origin."""
    first_outer = int(origins_for_validation(frame,h,features)[0])
    y = ridge.supervised(frame,h,'delta')
    x = frame[ridge_v2.CORE].to_numpy(float)
    rows = []
    for fold,(start,end) in enumerate(proposal['inner_origin_windows']):
        origins = np.flatnonzero((frame.index >= start)&(frame.index <= end))
        origins = origins[(origins+h < len(frame))&(origins+h < first_outer)]
        for origin in origins:
            train = training(frame,y,h,origin,features)
            model,median = fit_model(x[train],y[train],parameters,proposal)
            correction = float(predict(model,median,x[[origin]])[0])
            prediction = float(frame.close.iloc[origin]+correction)
            actual = float(frame.close.iloc[origin+h])
            rows.append(dict(candidate_id=candidate_id,horizon=h,fold=fold,**parameters,
                origin_date=frame.index[origin],target_date=frame.index[origin+h],
                train_n=len(train),train_label_end=frame.index[train[-1]+h],
                prediction=prediction,actual=actual,error=prediction-actual))
        if progress:
            progress(f'h={h} candidate={candidate_id} fold={fold} complete; n={len(origins)}')
    if not rows:
        raise ValueError('No chronological inner validation pairs')
    return pd.DataFrame(rows)


def choose_parameters(inner,proposal):
    table = inner.groupby('candidate_id',sort=True).agg(n=('error','size'),mse=('error',lambda x: np.mean(x**2))).reset_index()
    if len(table) != len(candidates(proposal)) or not table.n.eq(table.n.iloc[0]).all():
        raise ValueError('Unequal or incomplete inner candidate dates')
    expected = None
    for _,group in inner.groupby('candidate_id'):
        keys = set(group[ridge.KEYS].itertuples(index=False,name=None))
        if group.duplicated(ridge.KEYS).any() or (expected is not None and keys != expected):
            raise ValueError('Inner candidate keys differ')
        expected = keys
    table['rmse'] = np.sqrt(table.mse)
    winner = int(table.sort_values(['mse','candidate_id'],kind='stable').candidate_id.iloc[0])
    table['selected'] = table.candidate_id.eq(winner)
    return candidates(proposal)[winner],table


def outer_forecasts(frame,h,parameters,proposal,features,origins=None,progress=None):
    y = ridge.supervised(frame,h,'delta')
    x = frame[ridge_v2.CORE].to_numpy(float)
    if origins is None:
        origins = origins_for_validation(frame,h,features)
    predictions,windows,importance = [],[],[]
    for j,origin in enumerate(origins):
        train = training(frame,y,h,origin,features)
        model,median = fit_model(x[train],y[train],parameters,proposal)
        correction = float(predict(model,median,x[[origin]])[0])
        window_id = f'catboost_h{h}_{frame.index[origin].date()}'
        predictions.append(dict(model_id='CatBoost_Core_v2',model_version='catboost_core_v2_direct_v1',
            origin_date=frame.index[origin],target_date=frame.index[origin+h],horizon=h,
            prediction=float(frame.close.iloc[origin]+correction),prediction_target=correction,
            window_id=window_id,train_n=len(train),train_end_date=frame.index[train[-1]+h],
            train_origin_end=frame.index[train[-1]],split='validation',scored_status='unscored',
            forecast_origin='',forecast_origin_mode='conditional_EOD',pit_status='ASSUMED_AVAILABLE',
            result_status='exploratory_after_ridge_and_drift',data_version='frozen_v2_core_v2',**parameters))
        windows.append(dict(window_id=window_id,horizon=h,origin_date=frame.index[origin],train_n=len(train),
            train_origin_start=frame.index[train[0]],train_origin_end=frame.index[train[-1]],
            train_label_end=frame.index[train[-1]+h],
            **{feature+'_median':median[k] for k,feature in enumerate(ridge_v2.CORE)}))
        values = model.get_feature_importance(type='PredictionValuesChange',thread_count=1)
        importance.extend(dict(window_id=window_id,horizon=h,origin_date=frame.index[origin],
                               feature=feature,importance=float(values[k]))
                          for k,feature in enumerate(ridge_v2.CORE))
        if progress and (j+1)%50 == 0:
            progress(f'h={h} outer origins {j+1}/{len(origins)}')
    return pd.DataFrame(predictions),pd.DataFrame(windows),pd.DataFrame(importance),model,median
