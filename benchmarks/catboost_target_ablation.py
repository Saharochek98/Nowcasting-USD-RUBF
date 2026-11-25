"""H13 changes the target only; use the historical Core v2 fitting code."""
import hashlib
import json

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, ridge, ridge_v2

MODELS = {'M0_DELTA': 'delta', 'M1_LOGRETURN': 'log_return'}
KEYS = ridge.KEYS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def targets(frame, h, transform):
    require(transform in MODELS.values(), 'Unknown target transform')
    require(isinstance(h, (int, np.integer)) and 0 < h < len(frame), 'Invalid horizon')
    prices = frame.close.to_numpy(float)
    # The invalid-value policy is fixed before any evaluation: abort, never drop.
    require(np.isfinite(prices).all() and (prices > 0).all(), 'All CLOSE prices must be finite and positive')
    result = ridge.supervised(frame, h, transform)
    require(np.isfinite(result[:-h]).all(), 'Nonfinite transformed target; abort experiment')
    require(np.isnan(result[-h:]).all(), 'Only terminal unavailable labels may be NaN')
    return result


def restore_price(anchor, correction, transform):
    require(transform in MODELS.values(), 'Unknown target transform')
    anchor, correction = np.asarray(anchor, float), np.asarray(correction, float)
    require(np.isfinite(anchor).all() and (anchor > 0).all(), 'Origin price must be finite and positive')
    require(np.isfinite(correction).all(), 'Nonfinite predicted target')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        prediction = anchor + correction if transform == 'delta' else anchor * np.exp(correction)
    require(np.isfinite(prediction).all(), 'Nonfinite reconstructed prediction; abort experiment')
    if transform == 'log_return':
        require((prediction > 0).all(), 'Nonpositive reconstructed log-return price; abort experiment')
    return prediction


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def training_plan(frame, h, features, transform):
    require(frame.index.is_unique and frame.index.is_monotonic_increasing, 'Invalid frame dates')
    require(frame.index.max() < pd.Timestamp('2026-01-01'), '2026 data forbidden')
    require(list(frame.columns) == ridge_v2.CORE, 'Exactly 10 original Core v2 features required')
    y = targets(frame, h, transform)
    reference_y = targets(frame, h, 'delta')
    origins = cb.origins_for_validation(frame, h, features)
    plan = []
    for origin in origins:
        train = cb.training(frame, y, h, origin, features)
        reference = cb.training(frame, reference_y, h, origin, features)
        require(np.array_equal(train, reference), 'M0/M1 eligible training indices differ')
        require((train + h <= origin).all(), 'Unmatured training label')
        plan.append((int(origin), train))
    return y, plan


def forecast(frame, h, parameters, proposal, features, model_id, progress=None):
    transform = MODELS[model_id]
    y, plan = training_plan(frame, h, features, transform)
    x = frame[ridge_v2.CORE].to_numpy(float)
    predictions, windows = [], []
    for count, (origin, train) in enumerate(plan, 1):
        model, median = cb.fit_model(x[train], y[train], parameters, proposal)
        correction = float(cb.predict(model, median, x[[origin]])[0])
        prediction = float(restore_price(frame.close.iloc[origin], correction, transform))
        record = dict(model_id=model_id, horizon=h, origin_date=frame.index[origin],
                      target_date=frame.index[origin+h], prediction_target=correction,
                      prediction=prediction, split='validation', target_transform=transform)
        predictions.append(record)
        windows.append(dict(model_id=model_id, horizon=h, origin_date=frame.index[origin],
                            train_n=len(train), train_origin_start=frame.index[train[0]],
                            train_origin_end=frame.index[train[-1]], train_label_end=frame.index[train[-1]+h],
                            train_indices_sha256=index_hash(train),
                            **{name+'_median': float(median[k]) for k, name in enumerate(ridge_v2.CORE)}))
        if progress and (count % 50 == 0 or count == len(plan)):
            progress(f'{model_id} h={h}: {count}/{len(plan)} origins')
    return pd.DataFrame(predictions), pd.DataFrame(windows)


def exact_pairs(left, right, columns):
    require(not left.duplicated(KEYS).any() and not right.duplicated(KEYS).any(), 'Duplicate evaluation keys')
    joined = left.merge(right[KEYS + columns], on=KEYS, how='outer', suffixes=('', '_reference'),
                        indicator=True, validate='one_to_one')
    require(joined._merge.eq('both').all(), 'Evaluation keys differ')
    require(len(joined) == 1778 and joined.groupby('horizon').size().eq(254).all(),
            'Expected 254 identical pairs for each of seven horizons')
    return joined.drop(columns='_merge').sort_values(['horizon', 'origin_date']).reset_index(drop=True)


def check_m0(predictions, saved, atol=1e-8, rtol=1e-10):
    joined = exact_pairs(predictions, saved, ['prediction', 'prediction_target'])
    require(joined.target_date.between('2025-01-01', '2025-12-31').all(), 'Non-2025 evaluation target')
    rows = []
    for h, group in joined.groupby('horizon'):
        for column in ['prediction', 'prediction_target']:
            require(np.isclose(group[column], group[column+'_reference'], atol=atol, rtol=rtol).all(),
                    f'M0 reproduction failed: h={h}, {column}; do not run M1')
        rows.append(dict(horizon=h, n=len(group), passed=True,
                         max_abs_prediction_difference=float((group.prediction-group.prediction_reference).abs().max()),
                         max_abs_target_difference=float((group.prediction_target-group.prediction_target_reference).abs().max())))
    return pd.DataFrame(rows)


def check_windows(m0, m1):
    columns = ['horizon', 'origin_date', 'train_n', 'train_origin_start', 'train_origin_end',
               'train_label_end', 'train_indices_sha256'] + [c+'_median' for c in ridge_v2.CORE]
    a = m0[columns].sort_values(['horizon', 'origin_date']).reset_index(drop=True)
    b = m1[columns].sort_values(['horizon', 'origin_date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    return dict(windows=len(a), identical_indices=True, identical_train_medians=True)


def metric_values(prediction, actual, rw):
    error = np.asarray(prediction, float) - np.asarray(actual, float)
    rw_error = np.asarray(rw, float) - np.asarray(actual, float)
    require(len(error) > 0 and np.isfinite(error).all() and np.isfinite(rw_error).all(), 'Invalid metric data')
    rmse, rw_rmse = np.sqrt(np.mean(error**2)), np.sqrt(np.mean(rw_error**2))
    return dict(n=len(error), rmse=float(rmse), mae=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)), rmse_ratio_vs_rw=float(rmse/rw_rmse) if rw_rmse else np.nan)


def summarize(paired, cfg):
    metrics, behavior, confusion, hac = [], [], [], []
    for h, g in paired.groupby('horizon', sort=True):
        g = g.sort_values('origin_date')
        actual_delta = g.actual - g.RW
        actual_sign = np.sign(actual_delta)
        for model, column in [('M0_DELTA', 'prediction'), ('M1_LOGRETURN', 'prediction_reference'), ('RW', 'RW')]:
            metrics.append(dict(model_id=model, horizon=h, **metric_values(g[column], g.actual, g.RW)))
            delta = g[column] - g.RW
            sign = np.sign(delta)
            correct, nonzero = sign == actual_sign, actual_sign != 0
            behavior.append(dict(model_id=model, horizon=h, n=len(g), predicted_delta_mean=delta.mean(),
                                 predicted_delta_std=delta.std(ddof=1), actual_delta_mean=actual_delta.mean(),
                                 actual_delta_std=actual_delta.std(ddof=1), bias=(g[column]-g.actual).mean(),
                                 std_ratio=delta.std(ddof=1)/actual_delta.std(ddof=1),
                                 correlation=delta.corr(actual_delta) if delta.std(ddof=1) > 0 else np.nan,
                                 direction_accuracy=correct.mean(), direction_accuracy_nonzero=correct[nonzero].mean(),
                                 n_actual_zero=int((~nonzero).sum()), n_predicted_zero=int((sign == 0).sum()),
                                 n_correct_zero=int(((sign == 0) & (~nonzero)).sum())))
            for actual in [-1, 0, 1]:
                for predicted in [-1, 0, 1]:
                    confusion.append(dict(model_id=model, horizon=h, actual_sign=actual, predicted_sign=predicted,
                                          n=int(((actual_sign == actual) & (sign == predicted)).sum())))
        losses = pd.DataFrame(dict(prediction=g.prediction_reference, actual=g.actual, rw_prediction=g.prediction))
        hac.append(dict(horizon=h, n=len(g), model='M1_LOGRETURN', versus='M0_DELTA',
                        **ridge.loss_hac(losses, h, cfg)))
    hac = pd.DataFrame(hac)
    require(hac.horizon.tolist() == list(range(1, 8)), 'Primary HAC family must contain exactly seven comparisons')
    hac['holm_p'] = ridge.holm(hac.p_two_sided.to_numpy())
    hac['reject_holm_005'] = hac.holm_p < .05
    return {'metrics': pd.DataFrame(metrics), 'behavior': pd.DataFrame(behavior),
            'direction_confusion': pd.DataFrame(confusion), 'hac_holm': hac}
