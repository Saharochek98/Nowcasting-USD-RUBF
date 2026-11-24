"""Versioned frozen-v2 loader and explicit prequential/frozen training policies."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import ridge

ROOT = ridge.ROOT
CORE = ridge.PRICE + ['cbr_usdrub_effective', 'swaprate_prev_session', 'swaprate_current',
                     'key_rate_pct_effective', 'spread_close_minus_cbr']
SCENARIOS = {'price_only': ridge.PRICE, 'core_v2': CORE,
             'core_v2_no_spread': CORE[:-1],
             'B1_v2': CORE + ['ruonia_pct_last_reported', 'brent_eia_spot_usd_bbl_lag10d'],
             'B2_v2': CORE + ['ruonia_pct_last_reported', 'brent_eia_spot_usd_bbl_lag10d',
                              'usdcnh_investing_price_prev_date']}


def config(path=None):
    cfg = json.loads(Path(path or ROOT/'configs/ridge_experiments_v2.json').read_text())
    if (cfg['validation_end'] >= '2026-01-01' or cfg['train_target_end'] >= cfg['validation_start']
            or cfg['horizons'] != list(range(1, 8)) or cfg['scenarios'] != SCENARIOS
            or cfg['pit_status'] != 'ASSUMED_AVAILABLE'
            or cfg['alpha_grid'] != [.01, .1, 1, 10, 100, 1000, 10000, 100000]):
        raise ValueError('Unsafe v2 configuration')
    if any(end >= cfg['validation_start'] for _, end in cfg['inner_origin_windows']):
        raise ValueError('Inner tuning must precede validation')
    return cfg


def load_inputs(cfg):
    # Reuse v1's strict provenance/date checks; never modify B3 or old Core.
    checked, calendar, _ = ridge.load_inputs(cfg, include_current=True)
    source = ridge.read_through(ROOT/cfg['v2_path'], cfg['validation_end'])
    cols = list(dict.fromkeys(c for values in SCENARIOS.values() for c in values))
    frame = source[cols].apply(pd.to_numeric, errors='raise').reindex(calendar)
    for col in checked.columns:
        np.testing.assert_allclose(frame[col], checked[col], equal_nan=True)
    return frame, calendar, SCENARIOS


def policy_config(cfg, regime):
    if regime not in ['frozen', 'prequential']:
        raise ValueError('Unknown training policy')
    result = dict(cfg)
    result['train_target_end'] = cfg['validation_end'] if regime == 'prequential' else cfg['train_target_end']
    return result


def walk_forward(frame, cols, model, h, transform, cfg, regime):
    output = ridge.walk_forward(frame, cols, model, h, transform, policy_config(cfg, regime), regime)
    pred, coef, tuning, origins, fits = output
    pred['data_version'] = 'frozen_v2_core_v2'
    pred['source'] = cfg['v2_path']
    pred['pit_status'] = 'conditional EOD / pseudo-PIT' if model == 'price_only' else 'ASSUMED_AVAILABLE'
    pred['funding_availability_assumption'] = 'final swaprate_current[t] available after close t'
    pred['result_status'] = 'research_after_validation_v1'
    return pred, coef, tuning, origins, fits
