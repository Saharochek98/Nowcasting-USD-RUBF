"""H11-REL-01 leakage, warm-up, calendar and matched-design tests (validation 2025 only)."""
import importlib.util
import json
import unittest

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, h11_relative as H, ridge, ridge_v2

HAS_CATBOOST = importlib.util.find_spec('catboost') is not None


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h11, _, cls.proposal, cls.features = H.load_config()
        cls.frame, cls.calendar = H.load_panel(cls.features)

    def test_config_is_preregistered_and_unchanged(self):
        self.assertEqual(self.h11['status'], 'PREREGISTERED_NOT_EXECUTED')
        self.assertEqual(self.h11['funding_status'], 'ASSUMED_AVAILABLE')
        self.assertFalse(self.h11['final_test_access'])
        self.assertFalse(self.h11['new_hyperparameter_search'])
        self.assertEqual(list(self.h11['feature_building']['added_features']), H.ADDED)
        self.assertEqual(H.ARMS['R0_MATCHED'], ridge_v2.CORE)
        self.assertEqual(H.ARMS['R1'], ridge_v2.CORE+H.ADDED)

    def test_panel_stops_before_final_test(self):
        self.assertLess(self.frame.index.max(), pd.Timestamp('2026-01-01'))
        self.assertTrue(self.frame.index.equals(self.calendar))

    def test_formulas_match_manual_values(self):
        f, c = self.frame, self.frame.close.to_numpy()
        for t in [19, 100, 500, len(f)-1]:
            self.assertAlmostEqual(f.logret_1.iloc[t], np.log(c[t]/c[t-1]), places=14)
            self.assertAlmostEqual(f.logret_3.iloc[t], np.log(c[t]/c[t-3]), places=14)
            self.assertAlmostEqual(f.logret_7.iloc[t], np.log(c[t]/c[t-7]), places=14)
            self.assertAlmostEqual(f.range_rel.iloc[t], (f.high.iloc[t]-f.low.iloc[t])/f.open.iloc[t], places=14)
            v = f.volume.to_numpy()
            self.assertAlmostEqual(f.volume_ratio_20.iloc[t], v[t]/v[t-19:t+1].mean(), places=12)
            sma = c[t-19:t+1].mean()
            self.assertAlmostEqual(f.distance_sma20.iloc[t], (c[t]-sma)/sma, places=12)

    def test_warmup_is_nan_then_complete(self):
        f = self.frame
        for name, first in [('logret_1', 1), ('logret_3', 3), ('logret_7', 7), ('range_rel', 0),
                            ('volume_ratio_20', 19), ('distance_sma20', 19)]:
            self.assertTrue(f[name].iloc[:first].isna().all(), name)
            self.assertTrue(f[name].iloc[first:].notna().all(), name)
        self.assertEqual(H.WARMUP_SESSIONS, 19)

    def test_features_are_backward_only(self):
        cut = 400
        changed = self.frame[['open', 'high', 'low', 'close', 'volume']].astype(float)
        changed.iloc[cut+1:] *= 1.7
        changed.iloc[cut+1:, changed.columns.get_loc('volume')] += 12345
        a = H.relative_features(self.frame).iloc[:cut+1]
        b = H.relative_features(changed).iloc[:cut+1]
        pd.testing.assert_frame_equal(a, b)

    def test_current_bar_is_used_but_nothing_later(self):
        cut = 400
        changed = self.frame.copy()
        changed.iloc[cut, changed.columns.get_loc('close')] *= 1.01
        a, b = H.relative_features(self.frame), H.relative_features(changed)
        self.assertNotEqual(a.logret_1.iloc[cut], b.logret_1.iloc[cut])
        pd.testing.assert_frame_equal(a.iloc[:cut], b.iloc[:cut])

    def test_non_positive_denominators_are_missing(self):
        toy = pd.DataFrame(dict(open=[0.]+[1.]*24, high=[2.]*25, low=[1.]*25,
                                close=[1.]*25, volume=[0.]*25), index=pd.RangeIndex(25))
        out = H.relative_features(toy)
        self.assertTrue(np.isnan(out.range_rel.iloc[0]))
        self.assertTrue(out.volume_ratio_20.isna().all())
        self.assertFalse(np.isinf(out.to_numpy()).any())

    def test_eligible_rows_are_matured_and_skip_warmup(self):
        for h in range(1, 8):
            y = ridge.supervised(self.frame, h, 'delta')
            for origin in cb.origins_for_validation(self.frame, h, self.features)[[0, 100, -1]]:
                rows = H.eligible(self.frame, y, h, origin, self.features)
                base = ridge.training_indices(self.frame, y, h, int(origin), self.features)
                self.assertTrue(np.array_equal(rows, base[base >= 19]))
                self.assertEqual(rows.min(), 19)
                self.assertTrue((rows+h <= origin).all())
                self.assertTrue((self.frame.index[rows+h] <= pd.Timestamp('2025-12-31')).all())
                self.assertTrue(self.frame.iloc[rows][H.ADDED].notna().all().all())

    def test_horizon_is_trading_sessions(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ridge.ROOT/'data/processed/usdrubf_trading_days.csv').date))
        calendar = calendar[calendar <= '2025-12-31']
        self.assertTrue(self.frame.index.equals(calendar))
        common = pd.read_csv(ridge.ROOT/'results/rw_drift_experiment/common_pairs.csv', parse_dates=['origin_date', 'target_date'])
        for h in range(1, 8):
            origins = cb.origins_for_validation(self.frame, h, self.features)
            self.assertEqual(len(origins), 254)
            keys = set(zip(self.frame.index[origins], self.frame.index[origins+h]))
            sub = common[common.horizon == h]
            self.assertEqual(keys, set(zip(sub.origin_date, sub.target_date)))


@unittest.skipUnless(HAS_CATBOOST, 'Install requirements-catboost.txt')
class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h11, _, cls.proposal, cls.features = H.load_config()
        cls.frame, _ = H.load_panel(cls.features)

    def forecast(self, frame, arm, h, origins, warmup=H.WARMUP_SESSIONS):
        pred, _ = H.forecast_arm(frame, arm, h, H.parameters(self.h11, h), self.proposal, self.features,
                                 origins=origins, warmup=warmup)
        return pred

    def test_without_warmup_r0_equals_existing_catboost(self):
        h = 2
        origins = cb.origins_for_validation(self.frame, h, self.features)[[0, 120]]
        ours = self.forecast(self.frame, 'R0_MATCHED', h, origins, warmup=0)
        old, *_ = cb.outer_forecasts(self.frame[ridge_v2.CORE], h, H.parameters(self.h11, h),
                                     self.proposal, self.features, origins=origins)
        np.testing.assert_array_equal(ours.prediction.to_numpy(), old.prediction.to_numpy())

    def test_future_prices_do_not_change_past_forecasts(self):
        for arm in H.ARMS:
            for h in (1, 7):
                origin = int(cb.origins_for_validation(self.frame, h, self.features)[60])
                base = self.forecast(self.frame, arm, h, [origin])
                raw = self.frame[ridge_v2.CORE].astype(float)
                raw.iloc[origin+1:, :5] *= 1.3  # OHLCV after the origin bar
                raw.iloc[origin+1:, raw.columns.get_loc('swaprate_current')] = 99.
                changed = pd.concat([raw, H.relative_features(raw)], axis=1)
                after = self.forecast(changed, arm, h, [origin])
                self.assertEqual(base.prediction.iloc[0], after.prediction.iloc[0], (arm, h))

    def test_arms_share_training_windows(self):
        h = 4
        origins = cb.origins_for_validation(self.frame, h, self.features)[[0, 253]]
        r0, r1 = self.forecast(self.frame, 'R0_MATCHED', h, origins), self.forecast(self.frame, 'R1', h, origins)
        cols = ['origin_date', 'target_date', 'train_n', 'train_origin_start', 'train_origin_end', 'train_label_end']
        pd.testing.assert_frame_equal(r0[cols], r1[cols])
        self.assertTrue((r0.train_label_end <= r0.origin_date).all())
        self.assertEqual(r0.train_origin_start.iloc[0], self.frame.index[19])

    def test_imputation_uses_training_rows_only(self):
        h = 1
        origin = int(cb.origins_for_validation(self.frame, h, self.features)[10])
        y = ridge.supervised(self.frame, h, 'delta')
        train = H.eligible(self.frame, y, h, origin, self.features)
        x = self.frame[H.ARMS['R1']].to_numpy(float)
        _, median = cb.fit_model(x[train], y[train], H.parameters(self.h11, h), self.proposal)
        np.testing.assert_array_equal(median, np.nanmedian(x[train], axis=0))
        zeros = self.frame.swaprate_prev_session.iloc[train].eq(0).sum()
        self.assertGreater(zeros, 0)  # observed funding zeros stay observations, not missing


if __name__ == '__main__':
    unittest.main()
