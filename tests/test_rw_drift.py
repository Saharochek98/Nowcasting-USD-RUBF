"""RW drift maturity, calendar holes, common keys and deterministic forecasts."""
import unittest

import numpy as np
import pandas as pd

from benchmarks import ridge, rw_drift
from scripts.run_rw_drift_experiment import assemble, summaries
from tests.test_ridge import fixture


class RWDriftTests(unittest.TestCase):
    def test_formula_and_initial_ar_pairs(self):
        hist = pd.Series([10.,12.,14.,16.], index=pd.bdate_range('2024-01-01', periods=4))
        pred, meta = rw_drift.random_walk_with_drift(hist, [1,2,3], 2)
        np.testing.assert_array_equal(pred, [18.,20.,22.])
        self.assertEqual((meta['mu'],meta['n_changes'],meta['n_pairs']), (2.,3,2))
        pred, meta = rw_drift.random_walk_with_drift(hist.iloc[:3], [1], 2)
        self.assertIsNone(pred)
        self.assertEqual(meta['reason'], 'insufficient_history')

    def test_missing_sessions_are_not_compressed(self):
        hist = pd.Series([10.,np.nan,14.,15.,17.,20.], index=pd.bdate_range('2024-01-01', periods=6))
        pred, meta = rw_drift.random_walk_with_drift(hist, [1,2], 2)
        self.assertEqual(meta['n_changes'], 3)
        self.assertEqual(meta['mu'], 2.)
        np.testing.assert_array_equal(pred, [22.,24.])

    def test_latest_known_change_included_and_future_not_required(self):
        index = pd.bdate_range('2024-01-01', periods=5)
        hist = pd.Series([10.,11.,13.,16.,20.], index=index)
        _, meta = rw_drift.random_walk_with_drift(hist, [7], 2)
        self.assertEqual(meta['mu'], 2.5)
        self.assertEqual(meta['last_change_date'], index[-1])

    def test_zero_drift_not_missing(self):
        hist = pd.Series([10.,11.,10.,11.,10.], index=pd.bdate_range('2024-01-01', periods=5))
        pred, meta = rw_drift.random_walk_with_drift(hist, [1,7], 2)
        self.assertEqual(meta['mu'], 0.)
        np.testing.assert_array_equal(pred, np.full(7,10.))

    def test_missing_last_change_fails_without_fallback(self):
        hist = pd.Series([10.,11.,12.,13.,np.nan,15.], index=pd.bdate_range('2024-01-01', periods=6))
        pred, meta = rw_drift.random_walk_with_drift(hist, [1], 2)
        self.assertIsNone(pred)
        self.assertEqual(meta['reason'], 'missing_last_change')

    def test_direct_horizon_intercept_is_not_h_times_daily_mean(self):
        close = np.array([10.,11.,15.,16.,17.,24.,26.])
        daily = np.diff(close).mean()
        direct_intercept = (close[2:]-close[:-2]).mean()
        self.assertNotAlmostEqual(2*daily, direct_intercept)

    def test_walk_forward_future_invariance_reproducibility_and_calendar_horizon(self):
        cfg, canonical = rw_drift.config()
        data = fixture()[['close']].rename(columns={'close':'price'})
        data['source'] = 'synthetic'
        a = rw_drift.generate(data, data.index, cfg, canonical)
        b = rw_drift.generate(data, data.index, cfg, canonical)
        pd.testing.assert_frame_equal(a,b)
        changed = data.copy()
        changed.loc['2025-07-01':,'price'] *= 3
        c = rw_drift.generate(changed, data.index, cfg, canonical)
        pd.testing.assert_frame_equal(a[a.origin_date < '2025-07-01'],c[c.origin_date < '2025-07-01'])
        self.assertTrue((a.train_end_date == a.origin_date).all())
        for row in a.itertuples():
            self.assertEqual(data.index.get_loc(row.target_date)-data.index.get_loc(row.origin_date), row.horizon)
        extended = pd.concat([data,pd.DataFrame({'price':[90.],'source':['sealed']},index=pd.to_datetime(['2026-01-05']))])
        with self.assertRaises(ValueError):
            rw_drift.generate(extended,extended.index,cfg,canonical)

    def test_common_keys_and_holm_family(self):
        cfg, canonical = rw_drift.config()
        data = fixture()[['close']].rename(columns={'close':'price'})
        data['source'] = 'synthetic'
        pred = rw_drift.generate(data,data.index,cfg,canonical)
        baseline, saved = [], []
        for model in ['B0','B1','price_only','core_v2']:
            item = pred[ridge.KEYS].copy()
            item['model_id'] = model
            item['prediction'] = item.origin_date.map(data.price)
            if model in ['B0','B1']:
                item['actual'] = item.target_date.map(data.price)
                baseline.append(item)
            else:
                saved.append(item)
        baseline,saved = pd.concat(baseline),pd.concat(saved)
        common = assemble(pred,baseline,saved,data)
        metrics,hac = summaries(common,cfg)
        self.assertEqual((len(metrics),len(hac)), (35,28))
        self.assertTrue(hac.holm_p.between(0,1).all())
        with self.assertRaises(ValueError):
            assemble(pred,baseline.iloc[1:],saved,data)


if __name__ == '__main__':
    unittest.main()
