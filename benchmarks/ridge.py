"""Direct Ridge with explicit matured train-only labels; no final-test access.

Objective: sum((y - intercept - standardized_X @ beta)**2) + alpha*sum(beta**2).
Intercept is unpenalized. NumPy implementation avoids a new sklearn dependency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['origin_date', 'target_date', 'horizon']
PRICE = ['open', 'high', 'low', 'close', 'volume']


def config(path=None):
    cfg = json.loads(Path(path or ROOT / 'configs/ridge_experiments.json').read_text())
    if (cfg['validation_labels_in_fit'] is not False or cfg['train_target_end'] >= cfg['validation_start']
            or cfg['validation_end'] >= '2026-01-01' or cfg['horizons'] != list(range(1, 8))
            or any(a <= 0 for a in cfg['alpha_grid']) or cfg['minimum_train_pairs'] < 2):
        raise ValueError('Unsafe Ridge split/regularization configuration')
    return cfg


def read_through(path, end):
    """Filter dates before numeric parsing; no 2026 feature/target enters memory."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['date'] <= end]
    frame = pd.DataFrame(rows)
    if frame.empty or frame.date.duplicated().any() or list(frame.date) != sorted(frame.date):
        raise ValueError('Invalid ordered unique feature dates')
    return frame.set_index(pd.to_datetime(frame.pop('date')))


def load_inputs(cfg, include_current=False):
    scenario = json.loads((ROOT / cfg['scenario_config']).read_text(encoding='utf-8-sig'))
    scenarios = scenario['scenarios']
    columns = {'R1': PRICE, 'R2': scenarios['core']['features'],
               'R3': scenarios['core_ruonia_brent']['features'],
               'R4': scenarios['core_ruonia_brent_usdcnh']['features']}
    frame = read_through(ROOT / cfg['feature_path'], cfg['validation_end']).apply(pd.to_numeric, errors='raise')
    with (ROOT / cfg['calendar_path']).open(encoding='utf-8-sig') as f:
        calendar = pd.DatetimeIndex([r['date'] for r in csv.DictReader(f) if r['date'] <= cfg['validation_end']])
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or not frame.index.isin(calendar).all():
        raise ValueError('Calendar mismatch')
    frame = frame.reindex(calendar)  # calendar holes retained, never compressed
    if include_current:
        # Whole-file hashes establish frozen source identity, not 2026 model data.
        digest = hashlib.sha256((ROOT / cfg['v2_path']).read_bytes()).hexdigest()
        if digest != scenario['source_sha256']:
            raise ValueError('R5 frozen v2 source hash mismatch')
        source = read_through(ROOT / cfg['v2_path'], cfg['validation_end'])
        meta = read_through(ROOT / cfg['metadata_path'], cfg['validation_end'])
        if not source.index.equals(frame.index) or not meta.index.equals(frame.index):
            raise ValueError('R5 date/source alignment mismatch')
        if not (source.begin.str[:10].to_numpy() == source.index.strftime('%Y-%m-%d')).all():
            raise ValueError('R5 current TRADEDATE/candle label mismatch')
        if not meta.source_sha256.eq(digest).all() or not meta.source_dataset.eq(cfg['v2_path']).all():
            raise ValueError('R5 metadata provenance mismatch')
        for col in ['open', 'high', 'low', 'close', 'volume', 'swaprate_prev_session']:
            np.testing.assert_allclose(pd.to_numeric(source[col]).to_numpy(), frame[col].to_numpy(), equal_nan=True)
        frame['swaprate_current'] = pd.to_numeric(source.swaprate_current)
        columns['R5'] = columns['R2'] + ['swaprate_current']
    return frame, calendar, columns


def supervised(frame, horizon, transform):
    current = frame.close.to_numpy(float)
    future = np.full(len(frame), np.nan)
    future[:-horizon] = current[horizon:]
    with np.errstate(divide='ignore', invalid='ignore'):
        y = future - current if transform == 'delta' else np.log(future / current)
    return y


def training_indices(frame, y, horizon, origin_i, cfg, scope='all'):
    positions = np.arange(len(frame))
    target_i = positions + horizon
    valid = (target_i <= origin_i) & (target_i < len(frame)) & np.isfinite(y)
    target_dates = frame.index[np.minimum(target_i, len(frame)-1)]
    valid &= (target_dates <= pd.Timestamp(cfg['train_target_end'])) & (frame.index >= cfg['train_start'])
    if scope == 'S2':
        valid &= (frame.index >= cfg['s2_start']) & frame.swaprate_prev_session.notna().to_numpy()
    return positions[valid]


def fit_ridge(x, y, alpha):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError('Nonfinite target or infinite input')
    if np.isnan(x).all(axis=0).any():
        raise ValueError('All-missing feature in eligible training window')
    median = np.nanmedian(x, axis=0)
    filled = np.where(np.isnan(x), median, x)
    mean, scale = filled.mean(axis=0), filled.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1., scale)
    z = (filled - mean) / scale
    intercept = y.mean()
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y-intercept))
    return {'median': median, 'mean': mean, 'scale': scale, 'beta': beta, 'intercept': intercept}


def standardized(fit, x):
    x = np.asarray(x, float)
    return (np.where(np.isnan(x), fit['median'], x) - fit['mean']) / fit['scale']


def predict_change(fit, x):
    return fit['intercept'] + standardized(fit, x) @ fit['beta']


def price_from_change(anchor, prediction, transform):
    result = anchor + prediction if transform == 'delta' else anchor * np.exp(prediction)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite reconstructed price')
    return result


def choose_alpha(frame, cols, h, transform, cfg, first_outer, scope='all'):
    y = supervised(frame, h, transform)
    candidates, folds = [], []
    for alpha in cfg['alpha_grid']:
        errors = []
        for fold, (start, end) in enumerate(cfg['inner_origin_windows']):
            origins = np.flatnonzero((frame.index >= start) & (frame.index <= end))
            origins = origins[(origins+h < len(frame)) & (origins+h < first_outer)]
            if scope == 'S2':
                origins = origins[(frame.index[origins] >= cfg['s2_start']) & frame.swaprate_prev_session.iloc[origins].notna().to_numpy()]
            if not len(origins):
                continue
            train = training_indices(frame, y, h, int(origins[0]), cfg, scope)
            if len(train) < cfg['minimum_train_pairs']:
                continue
            fit = fit_ridge(frame[cols].iloc[train], y[train], alpha)
            pred = price_from_change(frame.close.iloc[origins].to_numpy(), predict_change(fit, frame[cols].iloc[origins]), transform)
            err = pred - frame.close.iloc[origins+h].to_numpy()
            errors.extend(err)
            folds.append(dict(alpha=alpha, fold=fold, n=len(err), train_n=len(train),
                              train_label_end=frame.index[train[-1]+h], fold_origin_start=frame.index[origins[0]],
                              fold_target_end=frame.index[origins[-1]+h], rmse=np.sqrt(np.mean(err**2))))
        if errors:
            candidates.append((float(np.sqrt(np.mean(np.square(errors)))), alpha, len(errors)))
    if not candidates:
        raise ValueError('No eligible chronological inner folds')
    _, alpha, _ = min(candidates)
    return alpha, folds


def walk_forward(frame, cols, model, h, transform, cfg, scope='all'):
    y = supervised(frame, h, transform)
    origins = np.flatnonzero((frame.index + pd.Timedelta(0) >= pd.Timestamp(cfg['train_start'])))
    origins = origins[origins+h < len(frame)]
    origins = origins[(frame.index[origins+h] >= cfg['validation_start']) & (frame.index[origins+h] <= cfg['validation_end'])]
    if scope == 'S2':
        origins = origins[(frame.index[origins] >= cfg['s2_start']) & frame.swaprate_prev_session.iloc[origins].notna().to_numpy()]
    alpha, tuning = choose_alpha(frame, cols, h, transform, cfg, int(origins[0]), scope)
    rows, coefficients, fits, cache = [], [], [], {}
    for i in origins:
        train = training_indices(frame, y, h, int(i), cfg, scope)
        if len(train) < cfg['minimum_train_pairs']:
            continue
        cache_key = tuple(train)
        if cache_key not in cache:
            fit = fit_ridge(frame[cols].iloc[train], y[train], alpha)
            window_id = f'{model}_{transform}_{scope}_h{h}_n{len(train)}'
            cache[cache_key] = fit, window_id
            rank = pd.Series(np.abs(fit['beta'])).rank(ascending=False, method='min').to_numpy()
            for j, col in enumerate(cols):
                coefficients.append(dict(model_id=model, horizon=h, transform=transform, scope=scope,
                    window_id=window_id, first_used_origin=frame.index[i], train_n=len(train),
                    train_origin_start=frame.index[train[0]], train_origin_end=frame.index[train[-1]],
                    train_label_end=frame.index[train[-1]+h], alpha=alpha, feature=col,
                    coefficient_standardized=fit['beta'][j], sign=int(np.sign(fit['beta'][j])), abs_rank=rank[j],
                    imputer_median=fit['median'][j], scaler_mean=fit['mean'][j], scaler_scale=fit['scale'][j],
                    intercept_target=fit['intercept']))
        fit, window_id = cache[cache_key]
        delta = float(predict_change(fit, frame[cols].iloc[[i]])[0])
        prediction = float(price_from_change(frame.close.iloc[i], delta, transform))
        rows.append(dict(model_id=model, model_version=cfg['version'], origin_date=frame.index[i],
            target_date=frame.index[i+h], horizon=h, prediction=prediction, prediction_target=delta,
            forecast_origin='', forecast_origin_mode='conditional_EOD', split='validation',
            prediction_status='available', target_date_status='known', scored_status='unscored',
            transform=transform, scope=scope, alpha=alpha, train_n=len(train), window_id=window_id,
            train_end_date=frame.index[train[-1]+h], train_origin_end=frame.index[train[-1]],
            data_version='features_daily_v1', source=cfg['r5_source'] if model == 'R5' else cfg['feature_path'],
            pit_status='PIT_UNVERIFIED' if model == 'R5' else cfg['pit_status'],
            result_status='diagnostic_only' if model == 'R5' else 'preliminary',
            primary_eligible=model != 'R5', calendar_source='reconstructed_moex_calendar_provisional'))
        fits.append(fit)
    return pd.DataFrame(rows), pd.DataFrame(coefficients), tuning, origins, fits


def permutation(frame, cols, predictions, origins, fits, cfg):
    """Blocked validation permutation diagnostic; never refits or changes forecasts."""
    original = frame[cols].iloc[origins].to_numpy(float)
    anchor = frame.close.iloc[origins].to_numpy()
    actual = frame.close.iloc[origins+int(predictions.horizon.iloc[0])].to_numpy()
    base_mse = np.mean((predictions.prediction.to_numpy()-actual)**2)
    rng = np.random.default_rng(cfg['random_seed'])
    blocks = [np.arange(i, min(i+cfg['permutation_block_sessions'], len(origins)))
              for i in range(0, len(origins), cfg['permutation_block_sessions'])]
    rows = []
    medians = np.stack([fit['median'] for fit in fits])
    means = np.stack([fit['mean'] for fit in fits])
    scales = np.stack([fit['scale'] for fit in fits])
    betas = np.stack([fit['beta'] for fit in fits])
    z_original = (np.where(np.isnan(original), medians, original)-means)/scales
    for j, col in enumerate(cols):
        changes = []
        for _ in range(cfg['permutation_repeats']):
            order = np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])
            values = original[order, j]
            new = (np.where(np.isnan(values), medians[:, j], values)-means[:, j])/scales[:, j]
            adjusted = predictions.prediction_target.to_numpy() + betas[:, j]*(new-z_original[:, j])
            perturbed = price_from_change(anchor, adjusted, predictions['transform'].iloc[0])
            changes.append(float(np.mean((perturbed-actual)**2)-base_mse))
        rows.append(dict(feature=col, mse_increase_mean=np.mean(changes), mse_increase_std=np.std(changes),
                         repeats=cfg['permutation_repeats'], block_sessions=cfg['permutation_block_sessions'],
                         n=len(origins), diagnostic_only=True))
    return rows


def load_baselines(cfg):
    # Parquet filters prevent loading sealed final-test rows/actuals.
    base = pd.read_parquet(ROOT / cfg['baseline_path'], filters=[('split', '==', 'validation')])
    base = base.loc[base.model_id.isin(['B0', 'B1']) & base.scored_status.eq('scored')].copy()
    if base.duplicated(['model_id', *KEYS]).any() or (base.target_date > cfg['validation_end']).any():
        raise ValueError('Invalid saved baseline validation keys')
    return base


def common_comparison(predictions, base, frame):
    predictions = predictions.copy()
    predictions['actual'] = predictions.target_date.map(frame.close)
    predictions['error'] = predictions.prediction - predictions.actual
    predictions['scored_status'] = 'scored'
    rw = base.loc[base.model_id.eq('B0'), KEYS+['prediction', 'actual']].rename(columns={'prediction':'rw_prediction', 'actual':'rw_actual'})
    ar = base.loc[base.model_id.eq('B1'), KEYS+['prediction', 'actual']].rename(columns={'prediction':'ar_prediction', 'actual':'ar_actual'})
    paired = predictions.merge(rw, on=KEYS, validate='one_to_one').merge(ar, on=KEYS, validate='one_to_one')
    if len(paired) != len(predictions):
        raise ValueError('Ridge keys not fully covered by saved RW/AR')
    np.testing.assert_allclose(paired.actual, paired.rw_actual, atol=1e-10, rtol=0)
    np.testing.assert_allclose(paired.actual, paired.ar_actual, atol=1e-10, rtol=0)
    rows = []
    rw_rmse = float(np.sqrt(np.mean((paired.rw_prediction-paired.actual)**2)))
    for label, col in [('ridge', 'prediction'), ('B0', 'rw_prediction'), ('B1', 'ar_prediction')]:
        err = paired[col]-paired.actual
        rows.append(dict(comparator=label, n=len(paired), rmse=float(np.sqrt(np.mean(err**2))),
                         mae=float(np.mean(abs(err))), bias=float(err.mean()),
                         rmse_ratio_vs_rw=float(np.sqrt(np.mean(err**2)))/rw_rmse,
                         origin_min=paired.origin_date.min(), origin_max=paired.origin_date.max(),
                         target_min=paired.target_date.min(), target_max=paired.target_date.max()))
    return predictions, rows, paired


def loss_hac(paired, horizon, cfg):
    d = ((paired.prediction-paired.actual)**2 - (paired.rw_prediction-paired.actual)**2).to_numpy()
    n, mean = len(d), float(np.mean(d))
    centered = d-mean
    lag = max(cfg['hac_base_lag'], horizon-1)
    variance = np.dot(centered, centered)/n
    for k in range(1, min(lag+1, n)):
        variance += 2*(1-k/(lag+1))*np.dot(centered[k:], centered[:-k])/n
    se = math.sqrt(max(variance, 0)/n*n/(n-1))
    p = math.erfc(abs(mean/se)/math.sqrt(2)) if se else (1. if mean == 0 else 0.)
    return dict(mean_squared_loss_difference=mean, hac_se=se, hac_lag=lag,
                ci_low=mean-1.95996398454*se, ci_high=mean+1.95996398454*se, p_two_sided=p)


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values))
    result[order] = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return result
