"""Leakage and correctness tests for benchmarks/new_models.py (notebook comparison v1)."""
import importlib.util
import math
import unittest

import numpy as np
import pandas as pd

from benchmarks import new_models as nm
from benchmarks import ridge, ridge_v2

HAS = {name: importlib.util.find_spec(name) is not None
       for name in ('statsmodels', 'arch', 'lightgbm', 'catboost')}


def synthetic_panel(n=320, seed=7):
    """Random-walk close with OHLCV and funding columns shaped like Core v2."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2022-04-26', periods=n)
    close = 75 + np.cumsum(rng.normal(0, 0.6, n))
    frame = pd.DataFrame(index=index)
    frame['open'] = close + rng.normal(0, 0.2, n)
    frame['close'] = close
    frame['high'] = np.maximum(frame.open, close) + rng.uniform(0, 0.5, n)
    frame['low'] = np.minimum(frame.open, close) - rng.uniform(0, 0.5, n)
    frame['volume'] = rng.integers(1000, 5000, n).astype(float)
    frame['cbr_usdrub_effective'] = close - 0.3
    frame['swaprate_current'] = rng.normal(0, 0.01, n)
    frame.loc[frame.index[:40], 'swaprate_current'] = np.nan
    frame.loc[frame.index[60:70], 'swaprate_current'] = 0.0  # observed zeros
    frame['swaprate_prev_session'] = frame['swaprate_current'].shift(1)
    frame['key_rate_pct_effective'] = 16.0
    frame['spread_close_minus_cbr'] = 0.3
    return frame[ridge_v2.CORE]


def perturb_future(frame, origin, scale=5.0):
    """Change every predictor and the close strictly after the origin position."""
    changed = frame.copy()
    future = changed.index[origin + 1:]
    numeric = changed.columns
    changed.loc[future, numeric] = changed.loc[future, numeric] * 1.3 + scale
    return changed


class MaturityTests(unittest.TestCase):
    def test_eligible_rows_only_matured_labels_for_every_h(self):
        y = np.ones(300)
        for h in nm.HORIZONS:
            for origin in (120, 200, 250):
                rows = nm.eligible_rows(300, h, origin, y)
                self.assertTrue((rows + h <= origin).all())
                self.assertEqual(rows.max(), origin - h)  # the newest matured label is used
                self.assertEqual(len(rows), origin - h + 1)
                window = nm.eligible_rows(300, h, origin, y, window=50)
                self.assertEqual(list(window), list(rows[-50:]))

    def test_direct_target_label_is_future_close(self):
        close = np.arange(10, dtype=float)
        for h in nm.HORIZONS:
            delta = nm.direct_target(close, h, 'delta')
            level = nm.direct_target(close, h, 'level')
            self.assertTrue(np.isnan(delta[-h:]).all())
            np.testing.assert_allclose(delta[:-h], h)
            np.testing.assert_allclose(level[:-h], close[h:])

    def test_eligible_rows_equal_project_prequential_indices(self):
        frame, _, features = nm.load_panel()
        for h in nm.HORIZONS:
            y = ridge.supervised(frame, h, 'delta')
            for origin in (nm.validation_origins(frame.index, h)[0], nm.validation_origins(frame.index, h)[-1], 500):
                expected = ridge.training_indices(frame, y, h, int(origin), features)
                np.testing.assert_array_equal(nm.eligible_rows(len(frame), h, int(origin), y), expected)

    def test_keys_and_inner_folds_precede_validation(self):
        frame, _, features = nm.load_panel()
        self.assertLess(frame.index.max(), nm.FINAL_TEST_START)
        for h in nm.HORIZONS:
            origins = nm.validation_origins(frame.index, h)
            self.assertEqual(len(origins), 254)
            first = nm.first_validation_origin(frame.index, h)
            for fold in nm.inner_origins(frame.index, h, features['inner_origin_windows']):
                self.assertTrue((fold + h < first).all())
                self.assertTrue((frame.index[fold + h] < nm.VAL_START).all())


class DirectInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_panel()
        self.origin = 260

    def run_model(self, frame, kind, params, target='delta', model_cfg=None, h=3):
        task = dict(model_id='m', kind=kind, params=params, target=target, X=frame.to_numpy(float),
                    close=frame.close.to_numpy(float), index=frame.index, h=h, origins=[self.origin],
                    model_cfg=model_cfg)
        return nm.run_direct(task).iloc[0]

    def assert_invariant(self, kind, params, model_cfg=None):
        for target in ('delta', 'level'):
            for h in (1, 7):
                before = self.run_model(self.frame, kind, params, target, model_cfg, h)
                after = self.run_model(perturb_future(self.frame, self.origin), kind, params, target, model_cfg, h)
                self.assertLessEqual(before.train_label_end, before.origin_date)
                for col in [c for c in before.index if c == 'prediction' or c.startswith(('q0', 'lower', 'upper'))]:
                    self.assertEqual(before[col], after[col], f'{kind} {target} h={h} {col} saw the future')

    def test_ridge_future_invariance(self):
        self.assert_invariant('ridge', {'alpha': 10.0})

    @unittest.skipUnless(HAS['catboost'], 'catboost not installed')
    def test_catboost_future_invariance(self):
        _, proposal, _ = nm.load_panel()
        self.assert_invariant('catboost', {'depth': 2, 'l2_leaf_reg': 30}, proposal)

    @unittest.skipUnless(HAS['lightgbm'], 'lightgbm not installed')
    def test_lgbm_quantile_future_invariance_and_monotone(self):
        params = dict(max_depth=2, learning_rate=0.05, n_estimators=30)
        self.assert_invariant('lgbm_quantile', params)
        row = self.run_model(self.frame, 'lgbm_quantile', params)
        values = [row[f'q{q}'] for q in nm.QUANTILES]
        self.assertEqual(values, sorted(values))
        self.assertEqual(row.prediction, row['q0.5'])
        self.assertEqual((row.lower_90, row.upper_90), (row['q0.05'], row['q0.95']))

    def test_training_imputation_uses_only_eligible_rows(self):
        seen = {}

        def spy(x, y, alpha):
            seen['x'] = np.asarray(x)
            return original(x, y, alpha)

        original = ridge.fit_ridge
        ridge.fit_ridge = spy
        try:
            self.run_model(self.frame, 'ridge', {'alpha': 1.0}, h=5)
        finally:
            ridge.fit_ridge = original
        self.assertEqual(len(seen['x']), self.origin - 5 + 1)
        # Funding zeros stay zeros (not treated as missing) in the training matrix.
        col = ridge_v2.CORE.index('swaprate_current')
        self.assertTrue((seen['x'][60:70, col] == 0).all())


class RelativeFeatureTests(unittest.TestCase):
    def test_relative_features_use_past_rows_only(self):
        frame = synthetic_panel()
        origin = 200
        before = nm.relative_features(frame).iloc[:origin + 1]
        after = nm.relative_features(perturb_future(frame, origin)).iloc[:origin + 1]
        pd.testing.assert_frame_equal(before, after)

    def test_relative_feature_formulas(self):
        frame = synthetic_panel()
        rel = nm.relative_features(frame)
        t = 150
        c = frame.close
        self.assertAlmostEqual(rel.ret_7.iloc[t], math.log(c.iloc[t] / c.iloc[t - 7]))
        sma = c.iloc[t - 19:t + 1].mean()
        self.assertAlmostEqual(rel.sma20_gap.iloc[t], (c.iloc[t] - sma) / sma)
        self.assertTrue(rel.sma20_gap.iloc[:19].isna().all())
        self.assertTrue(rel.rsi14.dropna().between(0, 100).all())
        expected = frame.swaprate_current - frame.swaprate_prev_session
        pd.testing.assert_series_equal(rel.d_funding, expected, check_names=False)


@unittest.skipUnless(HAS['statsmodels'] and HAS['arch'], 'statsmodels/arch not installed')
class UnivariateInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_panel(n=360)
        self.origin = 330

    def forecast(self, frame, kind, params, checkpoints=()):
        task = dict(model_id=kind, kind=kind, price=frame.close.to_numpy(float), index=frame.index,
                    origins=[self.origin], params=params, order_checkpoints=list(checkpoints))
        return nm.run_univariate(task).set_index('horizon')

    def test_univariate_models_ignore_future_prices(self):
        cases = [('ar1', {'window': 250}), ('drift', {'window': 60}), ('arima', {'window': None}),
                 ('uc', {'level': 'lltrend', 'window': 250}), ('garch', {'vol': 'GJR', 'dist': 't'})]
        for kind, params in cases:
            checkpoints = [300] if kind == 'arima' else ()
            before = self.forecast(self.frame, kind, params, checkpoints)
            after = self.forecast(perturb_future(self.frame, self.origin), kind, params, checkpoints)
            self.assertTrue(before.status.eq('ok').all(), kind)
            self.assertEqual(list(before.index), list(nm.HORIZONS))
            cols = [c for c in ('prediction', 'lower_90', 'upper_90', 'lower_95', 'upper_95') if c in before]
            pd.testing.assert_frame_equal(before[cols], after[cols], obj=kind)

    def test_garch_point_forecast_is_random_walk_and_bands_widen(self):
        out = self.forecast(self.frame, 'garch', {'vol': 'GARCH', 'dist': 'normal'})
        self.assertTrue(np.allclose(out.prediction, self.frame.close.iloc[self.origin]))
        width = (out.upper_95 - out.lower_95).to_numpy()
        self.assertTrue((np.diff(width) > 0).all())

    def test_arma00_variance_is_random_walk_variance(self):
        dx = np.diff(self.frame.close.to_numpy()[:300])
        result, _ = nm.arma_fit(dx, (0, 0))
        mean, sd = nm.arma_price_forecast(result, (0, 0), 100.0)
        sigma = math.sqrt(result.params[-1])
        np.testing.assert_allclose(sd, sigma * np.sqrt(np.arange(1, 8)), rtol=1e-10)
        np.testing.assert_allclose(np.diff(mean), result.params[0], rtol=1e-10)

    def test_bic_order_selection_uses_only_supplied_window(self):
        price = self.frame.close.to_numpy()
        order_a, _ = nm.select_arma_order(np.diff(price[:300]))
        changed = perturb_future(self.frame, 300).close.to_numpy()
        order_b, _ = nm.select_arma_order(np.diff(changed[:300]))
        self.assertEqual(order_a, order_b)


class ConformalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.index = pd.bdate_range('2023-01-02', periods=220)
        self.actual = pd.Series(80 + np.cumsum(rng.normal(0, 0.5, 220)), index=self.index)
        self.h = 3
        origins = np.arange(0, 220 - self.h)
        self.pred = pd.DataFrame(dict(model_id='m', origin_date=self.index[origins],
                                      target_date=self.index[origins + self.h], horizon=self.h,
                                      prediction=self.actual.iloc[origins].to_numpy()))

    def test_only_matured_errors_enter_calibration(self):
        bands = nm.conformal_intervals(self.pred, self.actual, window=60)
        self.assertTrue((bands.calibration_end_date.dropna() <= bands.origin_date[bands.calibration_end_date.notna()]).all())
        j = 150
        origin = self.pred.origin_date.iloc[j]
        future_actual = self.actual.copy()
        future_actual.loc[future_actual.index > origin] += 1000.0  # unrealised outcomes change wildly
        changed = nm.conformal_intervals(self.pred, future_actual, window=60)
        cols = ['lower_90', 'upper_90', 'lower_95', 'upper_95', 'cal_n']
        before = bands.loc[bands.origin_date.le(origin), cols].reset_index(drop=True)
        after = changed.loc[changed.origin_date.le(origin), cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_window_rank_and_count(self):
        bands = nm.conformal_intervals(self.pred, self.actual, window=60).set_index('origin_date')
        origin = self.pred.origin_date.iloc[150]
        matured = self.pred.loc[self.pred.target_date <= origin].tail(60)
        errors = np.sort(np.abs(matured.target_date.map(self.actual) - matured.prediction).to_numpy())
        self.assertEqual(bands.loc[origin, 'cal_n'], 60)
        q90 = errors[math.ceil(61 * 0.90) - 1]
        q95 = errors[math.ceil(61 * 0.95) - 1]
        anchor = self.pred.set_index('origin_date').loc[origin, 'prediction']
        self.assertAlmostEqual(bands.loc[origin, 'upper_90'] - anchor, q90)
        self.assertAlmostEqual(anchor - bands.loc[origin, 'lower_95'], q95)

    def test_too_few_matured_errors_gives_no_interval(self):
        bands = nm.conformal_intervals(self.pred, self.actual, window=60)
        early = bands.loc[bands.cal_n < nm.MIN_CAL]
        self.assertFalse(early.empty)
        self.assertTrue(early.lower_95.isna().all())


class ScoringTests(unittest.TestCase):
    def test_interval_score(self):
        self.assertAlmostEqual(float(nm.interval_score([1], [3], [2], 0.9)[0]), 2.0)
        self.assertAlmostEqual(float(nm.interval_score([1], [3], [4], 0.9)[0]), 2.0 + 2 / 0.1 * 1)
        self.assertAlmostEqual(float(nm.interval_score([1], [3], [0.5], 0.95)[0]), 2.0 + 2 / 0.05 * 0.5)

    def test_select_best_prefers_simpler_on_ties(self):
        table = pd.DataFrame(dict(candidate_id=[0, 1, 2], score=[1.0, 1.0, 0.9], status=['ok', 'ok', 'incomplete']))
        self.assertEqual(nm.select_best(table), 0)

    def test_ridge_alpha_procedure_matches_project_for_delta(self):
        frame, _, features = nm.load_panel()
        grid = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        cfg = dict(features, alpha_grid=grid)
        for h in (1, 7):
            first = nm.first_validation_origin(frame.index, h)
            expected, _ = ridge.choose_alpha(frame, ridge_v2.CORE, h, 'delta', cfg, first)
            alpha, _, _ = nm.ridge_choose_alpha(frame, ridge_v2.CORE, h, 'delta', grid, features['inner_origin_windows'])
            self.assertEqual(alpha, expected)


if __name__ == '__main__':
    unittest.main()
