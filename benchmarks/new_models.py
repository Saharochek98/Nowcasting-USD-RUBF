"""New USDRUBF models and evaluation tools for notebooks/model_comparison_2025.ipynb.

Safety rules shared by every function in this module:

* inputs come from the existing project loaders, which drop rows after 2025-12-31 before
  numeric parsing, so the sealed 2026 final test never enters memory;
* a forecast made at origin position ``i`` uses predictor rows ``<= i`` only and supervised
  labels ``C[s+h]`` with ``s + h <= i`` (labels matured by the origin);
* missing predictors are imputed with the median of the eligible training rows; observed zero
  funding values are left untouched;
* hyperparameters are chosen on the project's inner chronological folds (2023H1, 2023H2,
  2024H1) whose targets all precede the first validation origin; validation 2025 is never
  used for selection or early stopping.

Existing project modules are imported and reused, never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb
from benchmarks import core, ridge, ridge_v2
from benchmarks import xgboost_direct as xg

ROOT = ridge.ROOT
KEYS = ['origin_date', 'target_date', 'horizon']
SEED = 240112
HORIZONS = tuple(range(1, 8))
VAL_START, VAL_END = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
FINAL_TEST_START = pd.Timestamp('2026-01-01')
# First target date of the out-of-sample forecast history that conformal calibration may use.
CALIB_TARGET_START = pd.Timestamp('2023-07-01')
# Target block used only to choose the conformal calibration window (before validation).
CONFORMAL_TUNE_BLOCK = (pd.Timestamp('2024-07-01'), pd.Timestamp('2024-12-31'))
LEVELS = (0.90, 0.95)
QUANTILES = (0.025, 0.05, 0.5, 0.95, 0.975)
LEVEL_QUANTILES = {0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}
MIN_TRAIN = 100
MIN_AR_PAIRS = 100
MIN_CAL = 30
HAC_BASE_LAG = 20
GARCH_SIMULATIONS = 2000
ARMA_ORDERS = sorted(((p, q) for p in range(3) for q in range(3)), key=lambda o: (o[0] + o[1], o[0]))
Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}
RELATIVE = ['ret_1', 'ret_2', 'ret_3', 'ret_7', 'ret_14', 'sma20_gap', 'rsi14', 'atr14_rel', 'd_funding']
OHLC = ['open', 'high', 'low', 'close']
LGBM_FIXED = dict(objective='quantile', min_child_samples=20, reg_lambda=10.0, subsample=1.0,
                  subsample_freq=0, colsample_bytree=1.0, random_state=SEED, n_jobs=1,
                  deterministic=True, force_col_wise=True, verbose=-1)


# --------------------------------------------------------------------------- data and keys

def load_panel():
    """Core v2 panel through 2025-12-30 via the unchanged CatBoost/Ridge v2 loader."""
    _, proposal, features = cb.config()
    frame, calendar = cb.load_inputs(features)
    if frame.index.max() >= FINAL_TEST_START or calendar.max() >= FINAL_TEST_START:
        raise ValueError('Final-test dates reached the modelling panel')
    return frame, proposal, features


def origins_by_target(index, h, start, end):
    positions = np.arange(len(index) - h)
    targets = index[positions + h]
    return positions[(targets >= start) & (targets <= end)]


def validation_origins(index, h):
    return origins_by_target(index, h, VAL_START, VAL_END)


def extended_origins(index, h):
    """Pre-validation calibration history (targets from 2023-07-01) plus validation."""
    return origins_by_target(index, h, CALIB_TARGET_START, VAL_END)


def first_validation_origin(index, h):
    return int(validation_origins(index, h)[0])


def inner_origins(index, h, windows):
    """Same inner folds as the project: origins in each window, targets before first outer origin."""
    first_outer = first_validation_origin(index, h)
    folds = []
    for start, end in windows:
        origins = np.flatnonzero((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)))
        folds.append(origins[(origins + h < len(index)) & (origins + h < first_outer)])
    return folds


def eligible_rows(n_rows, h, origin, y, window=None):
    """Training rows s with matured label C[s+h] (s + h <= origin); optional rolling window."""
    rows = np.arange(n_rows)
    rows = rows[(rows + h <= origin) & np.isfinite(y)]
    if window is not None:
        rows = rows[-int(window):]
    return rows


def direct_target(close, h, kind):
    close = np.asarray(close, float)
    future = np.full(len(close), np.nan)
    future[:-h] = close[h:]
    if kind == 'delta':
        return future - close
    if kind == 'level':
        return future
    raise ValueError(f'Unknown target {kind}')


def to_price(anchor, value, kind):
    return anchor + value if kind == 'delta' else value


# ------------------- relative features (exploratory notebook set; NOT the preregistered H11-REL-01)

def relative_features(frame):
    """Backward-looking features in trading sessions; row t uses rows <= t only."""
    close, high, low = frame['close'], frame['high'], frame['low']
    out = pd.DataFrame(index=frame.index)
    log_close = np.log(close)
    for k in (1, 2, 3, 7, 14):
        out[f'ret_{k}'] = log_close - log_close.shift(k)
    sma = close.rolling(20, min_periods=20).mean()
    out['sma20_gap'] = (close - sma) / sma
    change = close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    out['rsi14'] = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    previous = close.shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    out['atr14_rel'] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    out['d_funding'] = frame['swaprate_current'] - frame['swaprate_prev_session']
    return out[RELATIVE]


def feature_sets(frame):
    core_v2 = frame[ridge_v2.CORE]
    relative = relative_features(frame)
    return {'core_v2': core_v2,
            'core_v2_plus_rel': pd.concat([core_v2, relative], axis=1),
            'rel_replaces_ohlc': pd.concat([core_v2.drop(columns=OHLC), relative], axis=1)}


# ----------------------------------------------------------------------- direct regressors

def _median_fill(x_train, x_test):
    x_train, x_test = np.asarray(x_train, float), np.asarray(x_test, float)
    if np.isinf(x_train).any() or np.isnan(x_train).all(axis=0).any():
        raise ValueError('Infinite or all-missing eligible training feature')
    median = np.nanmedian(x_train, axis=0)
    return np.where(np.isnan(x_train), median, x_train), np.where(np.isnan(x_test), median, x_test)


def fit_lgbm_quantiles(x_train, y_train, params, quantiles=QUANTILES):
    from lightgbm import LGBMRegressor
    models = {}
    for q in quantiles:
        model = LGBMRegressor(**LGBM_FIXED, alpha=q, max_depth=params['max_depth'],
                              num_leaves=2 ** params['max_depth'], learning_rate=params['learning_rate'],
                              n_estimators=params['n_estimators'])
        models[q] = model.fit(x_train, y_train)
    return models


def fit_predict(kind, params, x_train, y_train, x_test, model_cfg=None):
    """Fit on eligible rows only and predict ``x_test``; returns {name: array}."""
    y_train = np.asarray(y_train, float)
    if not np.isfinite(y_train).all():
        raise ValueError('Nonfinite training target')
    if kind == 'catboost':
        model, median = cb.fit_model(x_train, y_train, params, model_cfg)
        return {'point': cb.predict(model, median, x_test)}
    if kind == 'xgboost':
        model, median = xg.fit_model(x_train, y_train, params, model_cfg)
        return {'point': xg.predict(model, median, x_test)}
    if kind == 'ridge':
        fit = ridge.fit_ridge(x_train, y_train, params['alpha'])
        return {'point': ridge.predict_change(fit, x_test)}
    if kind == 'lgbm_median':  # tuning: only the 0.5 quantile model drives point accuracy
        filled_train, filled_test = _median_fill(x_train, x_test)
        model = fit_lgbm_quantiles(filled_train, y_train, params, quantiles=(0.5,))[0.5]
        return {'point': model.predict(filled_test)}
    if kind == 'lgbm_quantile':
        filled_train, filled_test = _median_fill(x_train, x_test)
        models = fit_lgbm_quantiles(filled_train, y_train, params)
        values = np.column_stack([models[q].predict(filled_test) for q in QUANTILES])
        values = np.sort(values, axis=1)  # monotone rearrangement against quantile crossing
        return {f'q{q}': values[:, j] for j, q in enumerate(QUANTILES)}
    raise ValueError(f'Unknown model kind {kind}')


def run_direct(task):
    """Walk-forward direct h-step model. ``task`` is a plain dict so it can be sent to workers."""
    x, close, index, h = task['X'], np.asarray(task['close'], float), task['index'], task['h']
    target = task['target']
    y = direct_target(close, h, target)
    rows = []
    for origin in task['origins']:
        origin = int(origin)
        train = eligible_rows(len(close), h, origin, y, task.get('window'))
        row = dict(model_id=task['model_id'], origin_date=index[origin], target_date=index[origin + h],
                   horizon=h, train_n=len(train), candidate_id=task.get('candidate_id'))
        if len(train) < MIN_TRAIN:
            rows.append(dict(row, prediction=np.nan, status='insufficient_history'))
            continue
        out = fit_predict(task['kind'], task['params'], x[train], y[train], x[[origin]], task.get('model_cfg'))
        row.update(train_origin_first=index[train[0]], train_origin_last=index[train[-1]],
                   train_label_end=index[train[-1] + h], status='ok')
        if 'point' in out:
            row['prediction_target'] = float(out['point'][0])
            row['prediction'] = float(to_price(close[origin], out['point'][0], target))
        else:
            for q in QUANTILES:
                row[f'q{q}'] = float(to_price(close[origin], out[f'q{q}'][0], target))
            row['prediction'] = row['q0.5']
            for level, (lo, hi) in LEVEL_QUANTILES.items():
                row[f'lower_{int(level * 100)}'] = row[f'q{lo}']
                row[f'upper_{int(level * 100)}'] = row[f'q{hi}']
        rows.append(row)
    return pd.DataFrame(rows)


def _worker_init():
    warnings.filterwarnings('ignore')


def parallel_map(function, tasks, n_jobs=1):
    """Deterministic map; every task is independent and single-threaded."""
    tasks = list(tasks)
    if n_jobs <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks)), mp_context=context,
                             initializer=_worker_init) as executor:
        return list(executor.map(function, tasks))


def select_best(table, score='score'):
    """Smallest score among complete candidates; ties go to the simpler (earlier) candidate."""
    ok = table.loc[table['status'].eq('ok')]
    if ok.empty:
        raise ValueError('No complete tuning candidate')
    best = ok[score].min()
    tied = ok.loc[ok[score] <= best * (1 + 1e-12)]
    return int(tied['candidate_id'].min())


def _candidate_status(group, expected_keys):
    keys = set(group.loc[np.isfinite(group['prediction']), KEYS].itertuples(index=False, name=None))
    return 'ok' if keys == expected_keys else 'incomplete'


def tune_direct(spec, x, close, index, windows, n_jobs=1):
    """Prequential-within-fold tuning (CatBoost/XGBoost project procedure), selection per h."""
    tasks = []
    for h in HORIZONS:
        origins = np.concatenate(inner_origins(index, h, windows))
        for cid, candidate in enumerate(spec['candidates']):
            tasks.append(dict(model_id=spec['model_id'], kind=spec.get('tune_kind', spec['kind']),
                              target=spec['target'], X=x, close=close, index=index, h=h, origins=origins,
                              candidate_id=cid,
                              params=candidate['params'], window=candidate.get('window'),
                              model_cfg=spec.get('model_cfg')))
    inner = pd.concat(parallel_map(run_direct, tasks, n_jobs), ignore_index=True)
    inner['actual'] = inner['target_date'].map(pd.Series(close, index=index))
    inner['error'] = inner['prediction'] - inner['actual']
    rows, selections = [], {}
    for h, group_h in inner.groupby('horizon'):
        expected = set(group_h[KEYS].drop_duplicates().itertuples(index=False, name=None))
        table = []
        for cid, group in group_h.groupby('candidate_id'):
            candidate = spec['candidates'][cid]
            err = group['error'].to_numpy()
            table.append(dict(model_id=spec['model_id'], horizon=h, candidate_id=cid,
                              params=json.dumps(candidate['params'], sort_keys=True),
                              window=candidate.get('window') or 'expanding', n=int(np.isfinite(err).sum()),
                              score=float(np.sqrt(np.nanmean(err ** 2))), criterion='pooled inner price RMSE',
                              status=_candidate_status(group, expected)))
        table = pd.DataFrame(table)
        winner = select_best(table)
        table['selected'] = table['candidate_id'].eq(winner)
        selections[h] = winner
        rows.append(table)
    return pd.concat(rows, ignore_index=True), selections, inner


def forecast_direct(spec, x, close, index, selections, n_jobs=1, origins_fn=extended_origins):
    tasks = []
    for h in HORIZONS:
        candidate = spec['candidates'][selections[h]]
        tasks.append(dict(model_id=spec['model_id'], kind=spec['kind'], target=spec['target'], X=x,
                          close=close, index=index, h=h, origins=origins_fn(index, h),
                          candidate_id=selections[h], params=candidate['params'],
                          window=candidate.get('window'), model_cfg=spec.get('model_cfg')))
    return pd.concat(parallel_map(run_direct, tasks, n_jobs), ignore_index=True)


# ------------------------------------- parallel replay of the project CatBoost/XGBoost runs

def _cb_inner_task(args):
    frame, h, params, cid, proposal, features = args
    return cb.inner_evaluate(frame, h, params, cid, proposal, features)


def _cb_outer_task(args):
    frame, h, params, proposal, features, origins = args
    return cb.outer_forecasts(frame, h, params, proposal, features, origins=origins)[0]


def _xgb_inner_task(args):
    frame, h, params, cid, cfg, features = args
    return xg.inner_evaluate(frame, h, params, cid, cfg, features)


def _xgb_outer_task(args):
    frame, h, params, cfg, features, origins = args
    return xg.outer_forecasts(frame, h, params, cfg, features, origins=origins)[0]


def replay_boosting(name, frame, model_cfg, features, n_jobs=1):
    """Unchanged project tuning + forecasting functions, parallel over (candidate, h) only."""
    module, inner_fn, outer_fn = ((cb, _cb_inner_task, _cb_outer_task) if name == 'catboost'
                                  else (xg, _xgb_inner_task, _xgb_outer_task))
    grid = module.candidates(model_cfg)
    tasks = [(frame, h, p, cid, model_cfg, features) for h in HORIZONS for cid, p in enumerate(grid)]
    inner = pd.concat(parallel_map(inner_fn, tasks, n_jobs), ignore_index=True)
    selections, tables = {}, []
    for h in HORIZONS:
        params, table = module.choose_parameters(inner.loc[inner.horizon.eq(h)], model_cfg)
        selections[h] = params
        tables.append(table.assign(horizon=h, params=[json.dumps(g, sort_keys=True) for g in grid]))
    outer = [(frame, h, selections[h], model_cfg, features, extended_origins(frame.index, h)) for h in HORIZONS]
    predictions = pd.concat(parallel_map(outer_fn, outer, n_jobs), ignore_index=True)
    return pd.concat(tables, ignore_index=True), selections, predictions


# ------------------------------------------------------- Ridge with a generic direct target

def ridge_choose_alpha(frame, cols, h, target, alpha_grid, windows, min_train=MIN_TRAIN):
    """Project Ridge alpha procedure (fixed fit at each inner fold start) for delta or level."""
    close = frame['close'].to_numpy(float)
    x = frame[cols].to_numpy(float)
    y = direct_target(close, h, target)
    candidates, folds = [], []
    for alpha in alpha_grid:
        errors = []
        for fold, origins in enumerate(inner_origins(frame.index, h, windows)):
            if not len(origins):
                continue
            train = eligible_rows(len(close), h, int(origins[0]), y)
            if len(train) < min_train:
                continue
            fit = ridge.fit_ridge(x[train], y[train], alpha)
            pred = to_price(close[origins], ridge.predict_change(fit, x[origins]), target)
            err = pred - close[origins + h]
            errors.extend(err)
            folds.append(dict(alpha=alpha, fold=fold, n=len(err), train_n=len(train),
                              rmse=float(np.sqrt(np.mean(err ** 2)))))
        if errors:
            candidates.append((float(np.sqrt(np.mean(np.square(errors)))), alpha, len(errors)))
    if not candidates:
        raise ValueError('No eligible chronological inner folds')
    _, alpha, _ = min(candidates)
    return alpha, pd.DataFrame(folds), candidates


def ridge_forecasts(frame, cols, h, target, alpha, origins, train_target_end=None, model_id='ridge'):
    """Expanding Ridge forecasts; optional cap on training label dates (frozen regime)."""
    close = frame['close'].to_numpy(float)
    x = frame[cols].to_numpy(float)
    y = direct_target(close, h, target)
    rows = []
    for origin in origins:
        origin = int(origin)
        train = eligible_rows(len(close), h, origin, y)
        if train_target_end is not None:
            train = train[frame.index[train + h] <= pd.Timestamp(train_target_end)]
        if len(train) < MIN_TRAIN:
            rows.append(dict(model_id=model_id, origin_date=frame.index[origin],
                             target_date=frame.index[origin + h], horizon=h, prediction=np.nan,
                             status='insufficient_history'))
            continue
        fit = ridge.fit_ridge(x[train], y[train], alpha)
        value = float(ridge.predict_change(fit, x[[origin]])[0])
        rows.append(dict(model_id=model_id, origin_date=frame.index[origin], target_date=frame.index[origin + h],
                         horizon=h, prediction=float(to_price(close[origin], value, target)),
                         prediction_target=value, alpha=alpha, train_n=len(train),
                         train_label_end=frame.index[train[-1] + h], status='ok'))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ univariate models

def ar1_window_forecast(price, origin, window=None):
    start = 0 if window is None else max(0, origin - int(window) + 1)
    history = pd.Series(price[start:origin + 1])
    predictions, meta = core.ar1(history, list(HORIZONS), MIN_AR_PAIRS)
    return predictions, meta


def drift_window_forecast(price, origin, window=None):
    """Mean of the last ``window`` adjacent changes (all changes when None)."""
    history = pd.Series(price[:origin + 1])
    changes = history.diff()
    lagged = pd.concat([changes.shift(1), changes], axis=1).dropna()
    if len(lagged) < MIN_AR_PAIRS or not np.isfinite(changes.iloc[-1]):
        return None, {'reason': 'insufficient_history'}
    available = changes[np.isfinite(changes)]
    if window is not None:
        available = available.iloc[-int(window):]
    mu = float(available.mean())
    return float(price[origin]) + np.arange(1, len(HORIZONS) + 1) * mu, {'mu': mu}


def _quiet_fit(model, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = model.fit(disp=False, **kwargs)
    converged = bool(result.mle_retvals.get('converged', True)) if hasattr(result, 'mle_retvals') else True
    return result, converged


def arma_fit(dx, order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p, q = order
    return _quiet_fit(SARIMAX(np.asarray(dx, float), order=(p, 0, q), trend='c'), method='lbfgs', maxiter=500)


def select_arma_order(dx):
    """BIC over ARMA(p,q), p,q <= 2, with constant, on the supplied (training) changes only."""
    table = []
    for order in ARMA_ORDERS:
        try:
            result, converged = arma_fit(dx, order)
            table.append(dict(p=order[0], q=order[1], bic=float(result.bic), converged=converged))
        except Exception as exc:  # recorded, never silently replaced
            table.append(dict(p=order[0], q=order[1], bic=np.inf, converged=False, error=str(exc)))
    table = pd.DataFrame(table)
    best = table.loc[np.isfinite(table['bic'])].sort_values('bic', kind='stable').iloc[0]
    return (int(best.p), int(best.q)), table


def arma_price_forecast(result, order, last_price, steps=len(HORIZONS)):
    """Price mean and analytic forecast s.d. of the cumulative ARMA change (psi weights)."""
    from statsmodels.tsa.arima_process import arma2ma
    p, q = order
    names = result.model.param_names
    params = np.asarray(result.params)
    ar = np.r_[1.0, -params[[names.index(f'ar.L{k}') for k in range(1, p + 1)]]] if p else np.r_[1.0]
    ma = np.r_[1.0, params[[names.index(f'ma.L{k}') for k in range(1, q + 1)]]] if q else np.r_[1.0]
    sigma2 = params[names.index('sigma2')]
    mean_dx = np.asarray(result.forecast(steps))
    psi_cum = np.cumsum(arma2ma(ar, ma, lags=steps))
    variance = sigma2 * np.cumsum(psi_cum ** 2)
    return last_price + np.cumsum(mean_dx), np.sqrt(variance)


def _window_start(origin, window):
    return 0 if window is None else max(0, origin - int(window) + 1)


def _normal_bands(row, mean, sd):
    for level in LEVELS:
        row[f'lower_{int(level * 100)}'] = mean - Z[level] * sd
        row[f'upper_{int(level * 100)}'] = mean + Z[level] * sd


def run_univariate(task):
    """One model configuration over ``task['origins']``; returns all h with target in calendar."""
    price, index, kind = np.asarray(task['price'], float), task['index'], task['kind']
    params = task.get('params', {})
    window = params.get('window')
    checkpoints = sorted(task.get('order_checkpoints', []))
    orders = {}
    if kind == 'arima':
        for cp in checkpoints:
            dx = np.diff(price[_window_start(cp, window):cp + 1])
            orders[cp], _ = select_arma_order(dx[np.isfinite(dx)])
    rows = []
    for origin in task['origins']:
        origin = int(origin)
        base = dict(model_id=task['model_id'], origin_date=index[origin], candidate_id=task.get('candidate_id'))
        status, mean, sd, extra, quant = 'ok', None, None, {}, None
        try:
            if kind in ('ar1', 'drift'):
                fn = ar1_window_forecast if kind == 'ar1' else drift_window_forecast
                mean, meta = fn(price, origin, window)
                if mean is None:
                    status = meta['reason']
            elif kind == 'arima':
                cp = max(c for c in checkpoints if c <= origin)
                dx = np.diff(price[_window_start(origin, window):origin + 1])
                result, converged = arma_fit(dx, orders[cp])
                mean, sd = arma_price_forecast(result, orders[cp], price[origin])
                extra = dict(order=f'{orders[cp]}', converged=converged)
            elif kind == 'uc':
                from statsmodels.tsa.statespace.structural import UnobservedComponents
                y = price[_window_start(origin, window):origin + 1]
                result, converged = _quiet_fit(UnobservedComponents(y, level=params['level']),
                                               method='lbfgs', maxiter=500)
                forecast = result.get_forecast(len(HORIZONS))
                mean = np.asarray(forecast.predicted_mean)
                sd = np.sqrt(np.asarray(forecast.var_pred_mean))
                extra = dict(converged=converged)
            elif kind == 'garch':
                mean, sd, quant, converged = _garch_origin(price, origin, params, task.get('seed', SEED))
                extra = dict(converged=converged)
            else:
                raise ValueError(f'Unknown univariate kind {kind}')
        except Exception as exc:  # failures are reported as missing forecasts
            status, mean = f'error: {type(exc).__name__}: {exc}', None
        for h in HORIZONS:
            if origin + h >= len(index):
                continue
            row = dict(base, target_date=index[origin + h], horizon=h, status=status, **extra)
            if mean is None or not np.isfinite(mean[h - 1]):
                row['prediction'] = np.nan
                rows.append(row)
                continue
            row['prediction'] = float(mean[h - 1])
            if quant is not None:
                for level, (lo, hi) in LEVEL_QUANTILES.items():
                    row[f'lower_{int(level * 100)}'] = float(quant[lo][h - 1])
                    row[f'upper_{int(level * 100)}'] = float(quant[hi][h - 1])
                row['sd'] = float(sd[h - 1])
            elif sd is not None:
                row['sd'] = float(sd[h - 1])
                _normal_bands(row, row['prediction'], row['sd'])
            rows.append(row)
    return pd.DataFrame(rows)


def _garch_origin(price, origin, params, seed):
    """RW mean; variance from GARCH/GJR on 100*log returns, simulated cumulative paths."""
    from arch.univariate import GARCH, Normal, StudentsT, ZeroMean
    returns = 100 * np.diff(np.log(price[:origin + 1]))
    returns = returns[np.isfinite(returns)]
    # arch draws simulation shocks from the distribution's own generator, so it is seeded here.
    generator = np.random.default_rng(seed + origin)
    distribution = Normal(seed=generator) if params['dist'] == 'normal' else StudentsT(seed=generator)
    model = ZeroMean(returns, volatility=GARCH(p=1, o=1 if params['vol'] == 'GJR' else 0, q=1),
                     distribution=distribution, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = model.fit(disp='off', options={'maxiter': 1000})
        forecast = result.forecast(horizon=len(HORIZONS), method='simulation', simulations=GARCH_SIMULATIONS,
                                   reindex=False)
    paths = np.asarray(forecast.simulations.values)[-1]
    cumulative = np.cumsum(paths, axis=1)
    ratio = np.exp(cumulative / 100)
    quant = {q: price[origin] * np.quantile(ratio, q, axis=0) for q in (0.025, 0.05, 0.95, 0.975)}
    mean = np.repeat(float(price[origin]), len(HORIZONS))
    return mean, cumulative.std(axis=0), quant, int(result.convergence_flag) == 0


def univariate_inner_keep(frame_index, rows):
    """Keep inner pairs whose target precedes the first validation origin of that h."""
    first = {h: frame_index[first_validation_origin(frame_index, h)] for h in HORIZONS}
    return rows.loc[rows['target_date'] < rows['horizon'].map(first)].copy()


def tune_univariate(spec, price, index, windows, n_jobs=1, criterion='rmse'):
    """One configuration for all h (pooled over folds and horizons), prequential within folds."""
    tasks = []
    for cid, candidate in enumerate(spec['candidates']):
        for fold, origins in enumerate(inner_origins(index, 1, windows)):
            tasks.append(dict(model_id=spec['model_id'], kind=spec['kind'], price=price, index=index,
                              origins=origins, params=candidate, candidate_id=cid, fold=fold,
                              order_checkpoints=[int(origins[0])]))
    results = parallel_map(run_univariate, tasks, n_jobs)
    inner = pd.concat([r.assign(fold=t['fold']) for r, t in zip(results, tasks)], ignore_index=True)
    inner = univariate_inner_keep(index, inner)
    inner['actual'] = inner['target_date'].map(pd.Series(price, index=index))
    inner['error'] = inner['prediction'] - inner['actual']
    expected = set(inner[KEYS].drop_duplicates().itertuples(index=False, name=None))
    table = []
    for cid, group in inner.groupby('candidate_id'):
        err = group['error'].to_numpy()
        row = dict(model_id=spec['model_id'], horizon='all', candidate_id=cid,
                   params=json.dumps(spec['candidates'][cid], sort_keys=True), n=int(np.isfinite(err).sum()),
                   rmse=float(np.sqrt(np.nanmean(err ** 2))), status=_candidate_status(group, expected),
                   n_failed=int((~group['status'].eq('ok')).sum()))
        if 'converged' in group:
            row['n_not_converged'] = int((group.drop_duplicates('origin_date')['converged'] == False).sum())  # noqa: E712
        if 'lower_90' in group:
            row['interval_score_90'] = float(np.nanmean(interval_score(group.lower_90, group.upper_90, group.actual, 0.90)))
            row['interval_score_95'] = float(np.nanmean(interval_score(group.lower_95, group.upper_95, group.actual, 0.95)))
            row['coverage_95'] = float(np.nanmean((group.actual >= group.lower_95) & (group.actual <= group.upper_95)))
        table.append(row)
    table = pd.DataFrame(table)
    if criterion == 'rmse':
        table['score'], table['criterion'] = table['rmse'], 'pooled inner price RMSE'
    else:
        table['score'] = (table['interval_score_90'] + table['interval_score_95']) / 2
        table['criterion'] = 'mean inner interval score (90% and 95%)'
    winner = select_best(table)
    table['selected'] = table['candidate_id'].eq(winner)
    return table, winner, inner


def forecast_univariate(spec, price, index, candidate_id, origins, checkpoints=(), n_jobs=1, chunks=8):
    origins = np.asarray(origins)
    parts = [p for p in np.array_split(origins, max(1, min(chunks, len(origins)))) if len(p)]
    tasks = [dict(model_id=spec['model_id'], kind=spec['kind'], price=price, index=index, origins=part,
                  params=spec['candidates'][candidate_id], candidate_id=candidate_id,
                  order_checkpoints=[int(c) for c in checkpoints]) for part in parts]
    return pd.concat(parallel_map(run_univariate, tasks, n_jobs), ignore_index=True)


def univariate_extended_origins(index):
    """Origins producing at least one target in [CALIB_TARGET_START, VAL_END]."""
    first = int(extended_origins(index, max(HORIZONS))[0])
    last = int(validation_origins(index, 1)[-1])
    return np.arange(first, last + 1)


def keep_extended(rows):
    keep = (rows['target_date'] >= CALIB_TARGET_START) & (rows['target_date'] <= VAL_END)
    return rows.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------- conformal intervals

def conformal_intervals(predictions, actual, window, levels=LEVELS, min_cal=MIN_CAL):
    """Absolute-error split conformal per h on a rolling window of matured errors.

    At origin t only errors of the same model and horizon with ``target_date <= t`` are used,
    i.e. outcomes already observed at the forecast origin.
    """
    out = []
    for h, group in predictions.groupby('horizon', sort=True):
        group = group.sort_values('target_date').reset_index(drop=True)
        targets = group['target_date'].to_numpy()
        errors = np.abs(group['target_date'].map(actual).to_numpy(float) - group['prediction'].to_numpy(float))
        result = {f'{side}_{int(level * 100)}': np.full(len(group), np.nan)
                  for level in levels for side in ('lower', 'upper')}
        cal_n = np.zeros(len(group), int)
        cal_end = np.full(len(group), np.datetime64('NaT'), dtype='datetime64[ns]')
        for j, origin in enumerate(group['origin_date'].to_numpy()):
            k = np.searchsorted(targets, origin, side='right')
            start = max(0, k - int(window))
            scores = errors[start:k]
            mask = np.isfinite(scores)
            scores = np.sort(scores[mask])
            cal_n[j] = len(scores)
            if k > start and mask.any():
                cal_end[j] = targets[start:k][mask].max()
            if len(scores) < min_cal or not np.isfinite(group['prediction'].iloc[j]):
                continue
            for level in levels:
                rank = math.ceil((len(scores) + 1) * level)
                if rank > len(scores):
                    continue
                q = scores[rank - 1]
                result[f'lower_{int(level * 100)}'][j] = group['prediction'].iloc[j] - q
                result[f'upper_{int(level * 100)}'][j] = group['prediction'].iloc[j] + q
        frame = group[KEYS + ['model_id', 'prediction']].copy()
        for name, values in result.items():
            frame[name] = values
        frame['cal_n'], frame['calibration_end_date'], frame['conformal_window'] = cal_n, cal_end, window
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def tune_conformal_window(predictions, actual, windows, block=CONFORMAL_TUNE_BLOCK):
    """Choose the calibration window on pre-validation targets only (mean 90%/95% interval score)."""
    table = []
    for cid, window in enumerate(windows):
        bands = conformal_intervals(predictions, actual, window)
        sub = bands.loc[(bands.target_date >= block[0]) & (bands.target_date <= block[1])].copy()
        sub['actual'] = sub['target_date'].map(actual)
        scores = [np.nanmean(interval_score(sub[f'lower_{int(l * 100)}'], sub[f'upper_{int(l * 100)}'],
                                            sub['actual'], l)) for l in LEVELS]
        cover = [np.nanmean((sub.actual >= sub[f'lower_{int(l * 100)}']) & (sub.actual <= sub[f'upper_{int(l * 100)}']))
                 for l in LEVELS]
        complete = sub[['lower_90', 'lower_95']].notna().all().all() and len(sub) > 0
        table.append(dict(model_id=predictions['model_id'].iloc[0], candidate_id=cid, window=window, n=len(sub),
                          interval_score_90=scores[0], interval_score_95=scores[1], coverage_90=cover[0],
                          coverage_95=cover[1], score=float(np.mean(scores)),
                          status='ok' if complete else 'incomplete'))
    table = pd.DataFrame(table)
    winner = select_best(table)
    table['selected'] = table['candidate_id'].eq(winner)
    return table, int(table.loc[table.selected, 'window'].iloc[0])


# ------------------------------------------------------------------------- metrics

def interval_score(lower, upper, actual, level):
    lower, upper, actual = (np.asarray(v, float) for v in (lower, upper, actual))
    alpha = 1 - level
    return ((upper - lower) + 2 / alpha * (lower - actual) * (actual < lower)
            + 2 / alpha * (actual - upper) * (actual > upper))


def common_keys(scored, models):
    sets = [set(scored.loc[scored.model_id.eq(m) & np.isfinite(scored.prediction), KEYS]
                .itertuples(index=False, name=None)) for m in models]
    return set.intersection(*sets)


def point_metrics(scored, models, reference='RW'):
    """RMSE, MAE, bias (prediction - actual), RMSE/RW and n on keys common to all models."""
    keys = common_keys(scored, models)
    key_frame = pd.DataFrame(sorted(keys), columns=KEYS)
    data = scored.merge(key_frame, on=KEYS)
    rows = []
    for h, group in data.groupby('horizon'):
        rw = group.loc[group.model_id.eq(reference)]
        rw_rmse = float(np.sqrt(np.mean(rw.error ** 2)))
        for model in models:
            err = group.loc[group.model_id.eq(model), 'error'].to_numpy()
            rmse = float(np.sqrt(np.mean(err ** 2)))
            rows.append(dict(model_id=model, horizon=h, n=len(err), rmse=rmse, mae=float(np.mean(np.abs(err))),
                             bias=float(np.mean(err)), rmse_ratio_vs_rw=rmse / rw_rmse))
    return pd.DataFrame(rows)


def interval_metrics(intervals):
    """Coverage, mean width and Winkler interval score per model/source/h/level."""
    rows = []
    for (model, source, h), group in intervals.groupby(['model_id', 'interval_source', 'horizon']):
        for level in LEVELS:
            lo, hi = group[f'lower_{int(level * 100)}'], group[f'upper_{int(level * 100)}']
            ok = lo.notna() & hi.notna()
            y = group['actual'][ok]
            rows.append(dict(model_id=model, interval_source=source, horizon=h, level=level, n=int(ok.sum()),
                             coverage=float(np.mean((y >= lo[ok]) & (y <= hi[ok]))),
                             mean_width=float(np.mean(hi[ok] - lo[ok])),
                             interval_score=float(np.mean(interval_score(lo[ok], hi[ok], y, level)))))
    return pd.DataFrame(rows)


def dm_tests(scored, pairs, family, hac_base_lag=HAC_BASE_LAG):
    """Diebold-Mariano on squared price errors, Newey-West HAC as in the project; Holm per family."""
    rows = []
    for left, right in pairs:
        for h in HORIZONS:
            a = scored.loc[scored.model_id.eq(left) & scored.horizon.eq(h), KEYS + ['prediction', 'actual']]
            b = scored.loc[scored.model_id.eq(right) & scored.horizon.eq(h), KEYS + ['prediction']]
            paired = a.merge(b.rename(columns={'prediction': 'rw_prediction'}), on=KEYS).dropna()
            paired = paired.sort_values('origin_date')
            stats = ridge.loss_hac(paired, h, {'hac_base_lag': hac_base_lag})
            dm = stats['mean_squared_loss_difference'] / stats['hac_se'] if stats['hac_se'] else np.nan
            rows.append(dict(family=family, model=left, versus=right, horizon=h, n=len(paired), dm_stat=dm, **stats))
    table = pd.DataFrame(rows)
    table['holm_p'] = ridge.holm(table['p_two_sided'].to_numpy())
    return table


# ------------------------------------------------------------ reproduction and manifest

def markdown_table(path, header_prefix):
    """First markdown table in ``path`` whose header cells start with ``header_prefix``."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if line.strip().startswith('|') and cells[:len(header_prefix)] == list(header_prefix):
            body = []
            for row in lines[i + 2:]:
                if not row.strip().startswith('|'):
                    break
                body.append([c.strip() for c in row.strip().strip('|').split('|')])
            return pd.DataFrame(body, columns=cells)
    raise ValueError(f'Table {header_prefix} not found in {path}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info():
    def run(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(sha=run('rev-parse', 'HEAD'), branch=run('rev-parse', '--abbrev-ref', 'HEAD'),
                status_porcelain=run('status', '--porcelain').splitlines())


def package_versions(names=('numpy', 'pandas', 'scipy', 'statsmodels', 'arch', 'lightgbm', 'catboost',
                            'xgboost', 'scikit-learn', 'matplotlib', 'pyarrow', 'nbconvert')):
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
