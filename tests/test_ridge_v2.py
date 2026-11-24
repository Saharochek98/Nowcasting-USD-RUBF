"""Matured prequential labels, train-only tuning, versioning and shrinkage."""
import copy
import unittest

import numpy as np
import pandas as pd

from benchmarks import ridge, ridge_v2
from tests.test_ridge import fixture


class RidgeV2Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = ridge_v2.config()
        self.frame = fixture()
        self.cfg['alpha_grid'] = [1., 100000.]

    def test_versioned_core_and_source(self):
        cfg = ridge_v2.config()
        frame, calendar, cols = ridge_v2.load_inputs(cfg)
        self.assertEqual([len(cols[k]) for k in cols], [5, 10, 9, 12, 13])
        self.assertIn('swaprate_current', cols['core_v2'])
        self.assertNotIn('spread_close_minus_cbr', cols['core_v2_no_spread'])
        self.assertEqual(str(calendar[-1].date()), '2025-12-30')
        self.assertEqual(frame.swaprate_prev_session.isna().sum(), 154)
        self.assertEqual(frame.swaprate_current.isna().sum(), 153)
        self.assertTrue(frame.swaprate_current.eq(0).any())
        old = ridge.config()
        self.assertEqual(old['pit_status'], 'conditional EOD / pseudo-PIT')

    def test_prequential_maturity_and_frozen_cap(self):
        frame = self.frame
        y = ridge.supervised(frame, 7, 'delta')
        origin = frame.index.get_indexer([pd.Timestamp('2025-07-01')])[0]
        pre = ridge.training_indices(frame, y, 7, origin, ridge_v2.policy_config(self.cfg, 'prequential'))
        frozen = ridge.training_indices(frame, y, 7, origin, ridge_v2.policy_config(self.cfg, 'frozen'))
        self.assertEqual(pre[-1]+7, origin)
        self.assertGreater(len(pre), len(frozen))
        self.assertLessEqual(frame.index[frozen[-1]+7], pd.Timestamp('2024-12-31'))

    def test_prequential_future_invariance_and_train_only_scaler(self):
        a, coefs, _, _, _ = ridge_v2.walk_forward(self.frame, ridge.PRICE, 'price_only', 7, 'delta', self.cfg, 'prequential')
        changed = self.frame.copy()
        changed.loc['2025-07-01':, ridge.PRICE] *= 10
        b = ridge_v2.walk_forward(changed, ridge.PRICE, 'price_only', 7, 'delta', self.cfg, 'prequential')[0]
        pd.testing.assert_frame_equal(a[a.origin_date < '2025-07-01'], b[b.origin_date < '2025-07-01'])
        first = a.iloc[0]
        y = ridge.supervised(self.frame, 7, 'delta')
        origin = self.frame.index.get_loc(first.origin_date)
        train = ridge.training_indices(self.frame, y, 7, origin, ridge_v2.policy_config(self.cfg, 'prequential'))
        expected = self.frame[ridge.PRICE].iloc[train].mean()
        actual = coefs[coefs.window_id.eq(first.window_id)].set_index('feature').scaler_mean
        np.testing.assert_allclose(actual[ridge.PRICE], expected[ridge.PRICE])
        self.assertTrue((a.train_end_date <= a.origin_date).all())
        self.assertGreater(a.train_end_date.max(), pd.Timestamp('2024-12-31'))

    def test_alpha_same_policies_and_insensitive_to_validation(self):
        first = self.frame.index.get_loc(pd.Timestamp('2024-12-23'))
        changed = self.frame.copy()
        changed.loc['2025-01-01':, 'close'] *= 100
        results = []
        for regime in ['frozen', 'prequential']:
            for frame in [self.frame, changed]:
                result, folds = ridge.choose_alpha(frame, ridge.PRICE, 7, 'delta', ridge_v2.policy_config(self.cfg, regime), first)
                results.append(result)
                self.assertTrue(all(r['fold_target_end'] < self.frame.index[first] for r in folds))
        self.assertEqual(len(set(results)), 1)

    def test_strong_regularization_shrinks_slopes_but_retains_intercept(self):
        x = np.arange(100.)[:, None]
        y = 3 + .1*x[:, 0]
        a, b = [ridge.fit_ridge(x, y, alpha) for alpha in [1, 1e12]]
        self.assertLess(np.linalg.norm(b['beta']), np.linalg.norm(a['beta'])*1e-8)
        self.assertEqual(b['intercept'], y.mean())
        self.assertGreater(abs(ridge.predict_change(b, x).mean()), 1)

    def test_v2_reproducible_and_all_comparison_keys_identical(self):
        a = ridge_v2.walk_forward(self.frame, ridge.PRICE, 'price_only', 1, 'log_return', self.cfg, 'prequential')[0]
        b = ridge_v2.walk_forward(self.frame, ridge.PRICE, 'price_only', 1, 'log_return', self.cfg, 'prequential')[0]
        pd.testing.assert_frame_equal(a, b)
        frozen = ridge_v2.walk_forward(self.frame, ridge.PRICE, 'price_only', 1, 'delta', self.cfg, 'frozen')[0]
        pd.testing.assert_frame_equal(a[ridge.KEYS], frozen[ridge.KEYS])
        self.assertTrue(a.result_status.eq('research_after_validation_v1').all())

    def test_invalid_policies_rejected(self):
        with self.assertRaises(ValueError):
            ridge_v2.policy_config(self.cfg, 'unknown')

    def test_prespecified_comparison_families_and_transform_pairing(self):
        from scripts.run_ridge_experiments_v2 import comparisons
        rows = []
        for regime in ['frozen', 'prequential']:
            for transform in ['delta', 'log_return']:
                for model in ridge_v2.SCENARIOS:
                    for h in range(1, 8):
                        for i in range(4):
                            rows.append(dict(scope=regime, transform=transform, model_id=model,
                                horizon=h, origin_date=f'2025-01-{i+1:02}', target_date=f'2025-02-{i+1:02}',
                                actual=80.+i, prediction=80.+i+.1*(i+1)+.02*(transform == 'log_return'),
                                rw_prediction=80., ar_prediction=80.+.5*i))
        result = comparisons(pd.DataFrame(rows), ridge_v2.config())
        self.assertEqual(len(result), 532)
        self.assertEqual(result.primary.sum(), 98)
        self.assertTrue(result.n.eq(4).all())
        transformed = result[result.kind.eq('target_transform')]
        self.assertTrue(transformed.mean_squared_loss_difference.gt(0).all())
        self.assertTrue(result.holm_p.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
