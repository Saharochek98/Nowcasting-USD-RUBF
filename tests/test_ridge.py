"""Leakage and numerical tests for direct Ridge; synthetic data only."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import ridge


def fixture():
    calendar = pd.bdate_range('2022-04-26', '2025-12-31')
    n = len(calendar)
    x = np.arange(n)
    close = 80 + .01*x + np.sin(x/15)
    frame = pd.DataFrame({'open': close-.1, 'high': close+.3, 'low': close-.4,
                          'close': close, 'volume': 100+x%23,
                          'swaprate_prev_session': np.where(x < 154, np.nan, np.sin(x/13)/100)}, index=calendar)
    return frame


class RidgeTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.cfg = ridge.config()
        self.cfg['alpha_grid'] = [.1, 10.]
        self.cfg['permutation_repeats'] = 2

    def test_closed_form_ridge_and_unpenalized_intercept(self):
        x = np.array([[1., 0.], [2., 1.], [3., 2.], [4., 0.]])
        y = np.array([10., 13., 15., 17.])
        fit = ridge.fit_ridge(x, y, 3.)
        z = ridge.standardized(fit, x)
        np.testing.assert_allclose((z.T@z+3*np.eye(2))@fit['beta'], z.T@(y-y.mean()))
        self.assertAlmostEqual(ridge.predict_change(fit, x).mean(), y.mean())

    def test_missing_training_only_and_real_zero_preserved(self):
        fit = ridge.fit_ridge([[0., 1.], [np.nan, 2.], [2., 3.]], [1., 2., 4.], 1.)
        self.assertEqual(fit['median'][0], 1.)
        self.assertNotEqual(ridge.standardized(fit, [[0., 1.]])[0, 0], ridge.standardized(fit, [[np.nan, 1.]])[0, 0])
        before = copy.deepcopy(fit)
        ridge.predict_change(fit, [[99999., 99999.]])
        np.testing.assert_array_equal(fit['median'], before['median'])
        with self.assertRaisesRegex(ValueError, 'All-missing'):
            ridge.fit_ridge([[np.nan], [np.nan]], [1., 2.], 1.)

    def test_horizon_uses_supplied_calendar_not_elapsed_days(self):
        f = self.frame.iloc[:4].copy()
        f.index = pd.to_datetime(['2024-01-05', '2024-01-08', '2024-01-13', '2024-01-15'])
        y = ridge.supervised(f, 2, 'delta')
        self.assertAlmostEqual(y[0], f.close.iloc[2]-f.close.iloc[0])
        self.assertTrue(np.isnan(y[-1]))

    def test_labels_must_mature_and_validation_never_enters_fit(self):
        y = ridge.supervised(self.frame, 7, 'delta')
        origin = 400
        train = ridge.training_indices(self.frame, y, 7, origin, self.cfg)
        self.assertLessEqual(train[-1]+7, origin)
        train = ridge.training_indices(self.frame, y, 7, len(self.frame)-1, self.cfg)
        self.assertLessEqual(self.frame.index[train[-1]+7], pd.Timestamp('2024-12-31'))

    def test_missing_target_does_not_compress_horizon(self):
        f = self.frame.copy()
        f.iloc[100, f.columns.get_loc('close')] = np.nan
        y = ridge.supervised(f, 1, 'delta')
        self.assertTrue(np.isnan(y[99]))
        train = ridge.training_indices(f, y, 1, 110, self.cfg)
        self.assertNotIn(99, train)
        self.assertNotIn(100, train)

    def test_s2_covers_only_original_observed_funding_and_same_start(self):
        y = ridge.supervised(self.frame, 1, 'delta')
        train = ridge.training_indices(self.frame, y, 1, 600, self.cfg, 'S2')
        self.assertTrue((self.frame.index[train] >= self.cfg['s2_start']).all())
        self.assertTrue(self.frame.swaprate_prev_session.iloc[train].notna().all())

    def test_alpha_unchanged_by_validation_outcomes(self):
        changed = self.frame.copy()
        changed.loc['2025-01-01':, 'close'] *= 100
        first_outer = self.frame.index.get_indexer([pd.Timestamp('2024-12-23')])[0]
        a, folds = ridge.choose_alpha(self.frame, ridge.PRICE, 7, 'delta', self.cfg, first_outer)
        b, _ = ridge.choose_alpha(changed, ridge.PRICE, 7, 'delta', self.cfg, first_outer)
        self.assertEqual(a, b)
        self.assertTrue(all(row['fold_target_end'] < self.frame.index[first_outer] for row in folds))

    def test_future_features_and_labels_do_not_change_past_forecasts(self):
        a = ridge.walk_forward(self.frame, ridge.PRICE, 'R1', 3, 'delta', self.cfg)[0]
        changed = self.frame.copy()
        changed.loc['2025-07-01':, ridge.PRICE] *= 3
        b = ridge.walk_forward(changed, ridge.PRICE, 'R1', 3, 'delta', self.cfg)[0]
        pd.testing.assert_frame_equal(a[a.origin_date < '2025-07-01'], b[b.origin_date < '2025-07-01'])
        self.assertTrue((a.train_end_date <= pd.Timestamp('2024-12-31')).all())

    def test_reproducible_walk_forward_and_permutation_does_not_refit(self):
        a, coef, _, origins, fits = ridge.walk_forward(self.frame, ridge.PRICE, 'R1', 1, 'delta', self.cfg)
        b = ridge.walk_forward(self.frame, ridge.PRICE, 'R1', 1, 'delta', self.cfg)[0]
        pd.testing.assert_frame_equal(a, b)
        saved = a.copy(deep=True)
        with patch.object(ridge, 'fit_ridge', side_effect=AssertionError('must not refit')):
            p = ridge.permutation(self.frame, ridge.PRICE, a, origins, fits, self.cfg)
            q = ridge.permutation(self.frame, ridge.PRICE, a, origins, fits, self.cfg)
        self.assertEqual(p, q)
        pd.testing.assert_frame_equal(a, saved)
        self.assertGreater(len(coef), 0)

    def test_log_target_reconstruction_in_price_units(self):
        f = self.frame.iloc[:10]
        y = ridge.supervised(f, 3, 'log_return')
        np.testing.assert_allclose(ridge.price_from_change(f.close.iloc[:-3], y[:-3], 'log_return'), f.close.iloc[3:])

    def test_common_pair_comparison_rejects_missing_baseline_key(self):
        pred = ridge.walk_forward(self.frame, ridge.PRICE, 'R1', 1, 'delta', self.cfg)[0]
        bases = []
        for m in ['B0', 'B1']:
            b = pred[ridge.KEYS].copy()
            b['model_id'] = m
            b['prediction'] = b.origin_date.map(self.frame.close)
            b['actual'] = b.target_date.map(self.frame.close)
            bases.append(b)
        base = pd.concat(bases, ignore_index=True)
        _, metrics, _ = ridge.common_comparison(pred, base, self.frame)
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(m['n'] == len(pred) for m in metrics))
        with self.assertRaises(ValueError):
            ridge.common_comparison(pred, base.iloc[1:], self.frame)

    def test_holm_and_r5_isolated_from_b3(self):
        np.testing.assert_allclose(ridge.holm([.01, .03, .2]), [.03, .06, .2])
        cfg = ridge.config()
        frame, _, cols = ridge.load_inputs(cfg, include_current=True)
        self.assertNotIn('swaprate_current', cols['R2'])
        self.assertIn('swaprate_current', cols['R5'])
        self.assertEqual(str(frame.index.max().date()), '2025-12-30')
        self.assertEqual(cfg['r5_status'], 'PIT_UNVERIFIED')
        with (ridge.ROOT / cfg['scenario_config']).open() as f:
            self.assertNotIn('swaprate_current', json.load(f)['scenarios']['core']['features'])

    def test_r5_rejects_wrong_trade_date_or_source(self):
        cfg = ridge.config()
        original = ridge.read_through
        def wrong_date(path, end):
            frame = original(path, end)
            if Path(path) == ridge.ROOT / cfg['v2_path']:
                frame.iloc[0, frame.columns.get_loc('begin')] = '2022-04-27 00:00:00'
            return frame
        with patch.object(ridge, 'read_through', side_effect=wrong_date):
            with self.assertRaisesRegex(ValueError, 'TRADEDATE'):
                ridge.load_inputs(cfg, include_current=True)
        def wrong_source(path, end):
            frame = original(path, end)
            if Path(path) == ridge.ROOT / cfg['metadata_path']:
                frame.iloc[0, frame.columns.get_loc('source_sha256')] = 'wrong'
            return frame
        with patch.object(ridge, 'read_through', side_effect=wrong_source):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                ridge.load_inputs(cfg, include_current=True)

    def test_sealed_rows_filtered_before_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.csv'
            path.write_text('date,close\n2025-12-30,80\n2026-01-05,DO_NOT_PARSE\n', encoding='utf-8')
            frame = ridge.read_through(path, '2025-12-31').apply(pd.to_numeric, errors='raise')
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.close.iloc[0], 80)


if __name__ == '__main__':
    unittest.main()
