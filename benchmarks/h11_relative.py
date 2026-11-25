"""H11-REL-01: CatBoost Core v2 (R0_MATCHED) vs Core v2 + six relative features (R1).

Everything is fixed by configs/h11_catboost_relative_v1.json (preregistered on main):
same delta target C[t+h]-C[t], the CatBoost parameters already selected per h for
results/catboost_core_v2, the same prequential expanding training rule, and train-only
median imputation.  The only differences between the arms are the six added columns.
Both arms drop the first 19 calendar sessions from training so that every training row
has complete 20-session rolling windows.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, ridge, ridge_v2

ROOT = ridge.ROOT
CONFIG_PATH = ROOT/'configs/h11_catboost_relative_v1.json'
ADDED = ['logret_1', 'logret_3', 'logret_7', 'range_rel', 'volume_ratio_20', 'distance_sma20']
WARMUP_SESSIONS = 19  # positions 0..18; first complete SMA20 window ends at position 19
ROLLING = 20
ARMS = {'R0_MATCHED': ridge_v2.CORE, 'R1': ridge_v2.CORE + ADDED}
ORIGINAL = 'CatBoost_Core_v2_rerun'
FORMULAS = {
    'logret_1': 'ln(close[t]/close[t-1])',
    'logret_3': 'ln(close[t]/close[t-3])',
    'logret_7': 'ln(close[t]/close[t-7])',
    'range_rel': '(high[t]-low[t])/open[t]',
    'volume_ratio_20': 'volume[t]/mean(volume[t-19:t])',
    'distance_sma20': '(close[t]-mean(close[t-19:t]))/mean(close[t-19:t])'}


def load_config():
    """Preregistered H11 config plus the frozen CatBoost/feature configs it points to."""
    h11 = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    cfg, proposal, features = cb.config()
    fb = h11['feature_building']
    checks = {
        'experiment_id': h11['experiment_id'] == 'H11-REL-01',
        'base_columns': fb['base_columns'] == ridge_v2.CORE,
        'added_features': fb['added_features'] == FORMULAS,
        'rolling_includes_origin_bar': fb['rolling_includes_origin_bar'] is True,
        'horizons': h11['horizons'] == list(range(1, 8)),
        'target': h11['target_transform'] == 'delta',
        'catboost_fixed': h11['catboost_fixed'] == proposal['fixed_parameters'],
        'no_search': h11['new_hyperparameter_search'] is False and h11['outer_early_stopping'] is False,
        'no_test': h11['final_test_access'] is False,
        'funding': h11['funding_status'] == 'ASSUMED_AVAILABLE',
        'split': h11['target_date_split'] == {'train_end': '2024-12-31', 'validation_start': features['validation_start'],
                                              'validation_end': features['validation_end']},
        'min_train': h11['minimum_train_pairs'] == features['minimum_train_pairs'],
        'hac_lag': h11['hac']['bandwidth'] == 'max(20,h-1)' and h11['hac']['primary_family_size'] == 7,
        'arms': set(h11['arms']) == set(ARMS),
        'prequential': h11['refit'] == 'every_origin_expanding_prequential'
                       and features['train_target_end'] == features['validation_end'],
    }
    for h in range(1, 8):
        saved = json.loads((ROOT/f'results/catboost_core_v2/h{h}/selection.json').read_text())['parameters']
        checks[f'params_h{h}'] = saved == h11['by_h_previously_selected'][str(h)]
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f'H11 config deviates from the frozen design: {failed}')
    return h11, cfg, proposal, features


def parameters(h11, h):
    return dict(h11['by_h_previously_selected'][str(h)])


def relative_features(frame):
    """Six P0 features; backward-looking on the session axis, the bar t itself included.

    Incomplete windows and non-positive denominators stay NaN (never backfilled).
    """
    def positive(series):
        return series.where(series > 0)

    close = positive(frame.close.astype(float))
    out = pd.DataFrame(index=frame.index)
    for k in (1, 3, 7):
        out[f'logret_{k}'] = np.log(close/close.shift(k))
    out['range_rel'] = (frame.high.astype(float)-frame.low.astype(float))/positive(frame.open.astype(float))
    volume_mean = positive(frame.volume.astype(float).rolling(ROLLING, min_periods=ROLLING).mean())
    out['volume_ratio_20'] = frame.volume.astype(float)/volume_mean
    close_mean = positive(close.rolling(ROLLING, min_periods=ROLLING).mean())
    out['distance_sma20'] = (close-close_mean)/close_mean
    return out[ADDED].replace([np.inf, -np.inf], np.nan)


def load_panel(features):
    frame, calendar = cb.load_inputs(features)
    if not frame.index.equals(calendar) or frame.index.max() > pd.Timestamp(features['validation_end']):
        raise ValueError('Panel must be the full trading calendar through validation end only')
    return pd.concat([frame, relative_features(frame)], axis=1), calendar


def eligible(frame, y, h, origin, features, warmup=WARMUP_SESSIONS):
    """Matured direct pairs (s+h <= origin, target <= validation end), positions >= warmup."""
    indices = ridge.training_indices(frame, y, h, int(origin), features)
    indices = indices[indices >= warmup]
    if len(indices) < features['minimum_train_pairs']:
        raise ValueError('Insufficient matured training labels')
    return indices


def forecast_arm(frame, arm, h, params, proposal, features, origins=None, warmup=WARMUP_SESSIONS):
    columns = ARMS[arm]
    y = ridge.supervised(frame, h, 'delta')
    x = frame[columns].to_numpy(float)
    close = frame.close.to_numpy(float)
    if origins is None:
        origins = cb.origins_for_validation(frame, h, features)
    rows, importance = [], []
    for origin in origins:
        train = eligible(frame, y, h, origin, features, warmup)
        model, median = cb.fit_model(x[train], y[train], params, proposal)
        correction = float(cb.predict(model, median, x[[origin]])[0])
        rows.append(dict(model_id=arm, experiment_id='H11-REL-01', origin_date=frame.index[origin],
            target_date=frame.index[origin+h], horizon=h, prediction=close[origin]+correction,
            prediction_target=correction, train_n=len(train), train_origin_start=frame.index[train[0]],
            train_origin_end=frame.index[train[-1]], train_label_end=frame.index[train[-1]+h],
            warmup_sessions_excluded=warmup, n_features=len(columns),
            origin_features_missing=int(np.isnan(x[origin]).sum()), split='validation',
            forecast_origin_mode='conditional_EOD', pit_status='conditional_EOD_pseudo_PIT',
            funding_status='ASSUMED_AVAILABLE', target_transform='delta',
            result_status='exploratory_validation_2025', **params))
        values = model.get_feature_importance(type='PredictionValuesChange', thread_count=1)
        importance.extend(dict(model_id=arm, horizon=h, origin_date=frame.index[origin], feature=f,
                               importance=float(values[k])) for k, f in enumerate(columns))
    return pd.DataFrame(rows), pd.DataFrame(importance)


def run_task(task):
    """Worker entry point (spawn-safe): one (arm, h) cell, single-threaded CatBoost."""
    arm, h = task
    h11, _, proposal, features = load_config()
    params = parameters(h11, h)
    if arm == ORIGINAL:
        frame, _ = cb.load_inputs(features)
        pred, *_ = cb.outer_forecasts(frame, h, params, proposal, features)
        return arm, h, pred, None
    frame, _ = load_panel(features)
    pred, importance = forecast_arm(frame, arm, h, params, proposal, features)
    return arm, h, pred, importance


def paired_hac(errors_model, errors_reference, h, base_lag=20):
    """Newey-West test on d = e_model^2 - e_reference^2 (negative mean: model better)."""
    actual = np.zeros(len(errors_model))
    frame = pd.DataFrame(dict(prediction=np.asarray(errors_model, float), actual=actual,
                              rw_prediction=np.asarray(errors_reference, float)))
    return ridge.loss_hac(frame, h, {'hac_base_lag': base_lag})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
