"""Expanding prequential RW with drift; compatible with core.generate_forecasts."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import core, ridge

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT/'configs/rw_drift_experiment.json').read_text())
    canonical = core.load_config(ROOT/cfg['benchmark_config'])
    if (cfg['train_start'] != canonical['splits']['train'][0]
            or [cfg['validation_start'], cfg['validation_end']] != canonical['splits']['validation']
            or cfg['horizons'] != canonical['horizons'] or cfg['min_ar_pairs'] != canonical['min_ar_pairs']
            or cfg['validation_end'] >= '2026-01-01'):
        raise ValueError('Drift training/evaluation constraints differ from benchmark')
    return cfg, canonical


def load_inputs(cfg):
    data = ridge.read_through(ROOT/cfg['price_path'], cfg['validation_end'])
    data = data.loc[cfg['train_start']:].copy()
    data['price'] = pd.to_numeric(data.price, errors='raise')
    present = data.price.dropna()
    if (present <= 0).any() or not np.isfinite(present).all():
        raise ValueError('Invalid input price')
    with (ROOT/cfg['calendar_path']).open(encoding='utf-8-sig') as stream:
        calendar = pd.DatetimeIndex([r['date'] for r in csv.DictReader(stream)
                                     if cfg['train_start'] <= r['date'] <= cfg['validation_end']])
    core.validate_calendar(calendar)
    if not data.index.isin(calendar).all():
        raise ValueError('Price outside supplied calendar')
    return data, calendar


def random_walk_with_drift(history, horizons, min_ar_pairs=100):
    """Only history ending at the origin is accepted; no horizon labels required."""
    if min_ar_pairs < 2 or not horizons or min(horizons) < 1:
        raise ValueError('Invalid minimum training size or horizons')
    changes = history.diff()  # NaNs preserve missing adjacent-session pairs.
    available = changes[np.isfinite(changes)]
    lagged = pd.concat([changes.shift(1), changes], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    meta = dict(n_pairs=len(lagged), n_changes=len(available), mu=np.nan)
    if len(lagged) < min_ar_pairs:
        return None, dict(meta, reason='insufficient_history')
    if not np.isfinite(history.iloc[-1]) or not np.isfinite(changes.iloc[-1]):
        return None, dict(meta, reason='missing_last_change')
    mu = float(available.mean())
    predictions = float(history.iloc[-1]) + np.arange(1, max(horizons)+1)*mu
    meta.update(mu=mu, training_start_date=history.index[0], last_change_date=available.index[-1])
    if not np.isfinite(predictions).all():
        return None, dict(meta, reason='nonfinite_forecast')
    return predictions, meta


def generate(data, calendar, cfg, canonical):
    if len(calendar) and (calendar.min() < pd.Timestamp(cfg['train_start'])
                          or calendar.max() > pd.Timestamp(cfg['validation_end'])):
        raise ValueError('History extends beyond allowed dates')
    predictions = core.generate_forecasts(data, calendar, canonical,
        models={'RW_DRIFT': lambda history, hs: random_walk_with_drift(history, hs, cfg['min_ar_pairs'])})
    predictions = predictions[predictions['split'].eq('validation')].copy()
    predictions['model_version'] = cfg['version']
    predictions['result_status'] = 'exploratory_after_ridge_v2'
    predictions['train_start_date'] = cfg['train_start']
    if not predictions.prediction_status.eq('available').all():
        raise ValueError('Drift unavailable on required validation dates')
    return predictions.reset_index(drop=True)
