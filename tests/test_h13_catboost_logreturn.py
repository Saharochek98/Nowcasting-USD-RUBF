"""H13 mathematical, scheduling, seeded-fit and saved-experiment checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, catboost_target_ablation as ab, ridge, ridge_v2
from scripts import run_h13_catboost_logreturn as runner

ROOT = runner.ROOT
OUTPUT = ROOT/'results/h13_catboost_logreturn'


def fixture():
    dates = pd.bdate_range('2024-06-03', '2025-03-31')
    t = np.arange(len(dates))
    close = 80 + .02*t + np.sin(t/7)
    frame = pd.DataFrame({c: close + i/10 for i, c in enumerate(ridge_v2.CORE)}, index=dates)
    frame['close'] = close
    frame['volume'] = 100 + t % 11
    frame.loc[dates[:10], 'swaprate_prev_session'] = np.nan
    frame.loc[dates[10], 'swaprate_current'] = 0.
    return frame


class TargetMathTests(unittest.TestCase):
    def test_target_inverse_and_terminal_labels(self):
        frame = fixture()
        for h in range(1, 8):
            for transform in ab.MODELS.values():
                y = ab.targets(frame, h, transform)
                expected = frame.close.to_numpy()[h:]-frame.close.to_numpy()[:-h]
                if transform == 'log_return':
                    expected = np.log(frame.close.to_numpy()[h:]/frame.close.to_numpy()[:-h])
                np.testing.assert_array_equal(y[:-h], expected)
                np.testing.assert_allclose(ab.restore_price(frame.close.to_numpy()[:-h], y[:-h], transform),
                                           frame.close.to_numpy()[h:], atol=1e-12, rtol=1e-14)
                self.assertTrue(np.isnan(y[-h:]).all())

    def test_positive_finite_prices_required(self):
        for value in [0, -1, np.nan, np.inf]:
            for transform in ab.MODELS.values():
                with self.subTest(value=value, transform=transform):
                    frame = fixture()
                    frame.iloc[0, frame.columns.get_loc('close')] = value
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        ab.targets(frame, 1, transform)

    def test_invalid_predictions_abort_without_fallback(self):
        for value in [np.nan, np.inf, 1000, -1000]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ab.restore_price(80, value, 'log_return')
        with self.assertRaises(ValueError):
            ab.restore_price(0, .1, 'log_return')
        with self.assertRaises(ValueError):
            ab.restore_price(80, 1, 'unknown')
        self.assertAlmostEqual(float(ab.restore_price(100, np.log(1.1), 'log_return')), 110)
        self.assertEqual(float(ab.restore_price(100, 10, 'delta')), 110)

    def test_same_matured_indices_all_h_no_warmup(self):
        _, _, cfg = cb.config()
        frame = fixture()
        for h in range(1, 8):
            _, delta = ab.training_plan(frame, h, cfg, 'delta')
            _, log = ab.training_plan(frame, h, cfg, 'log_return')
            self.assertEqual(len(delta), len(log))
            for (origin0, train0), (origin1, train1) in zip(delta, log):
                self.assertEqual(origin0, origin1)
                np.testing.assert_array_equal(train0, train1)
                self.assertEqual(train0[0], 0)
                self.assertEqual(train0[-1]+h, origin0)
                self.assertEqual(len(train0), origin0-h+1)
                self.assertTrue((frame.index[train0+h] <= frame.index[origin0]).all())

    def test_2026_and_wrong_features_rejected(self):
        _, _, cfg = cb.config()
        frame = fixture()
        frame.index = frame.index + pd.DateOffset(years=1)
        with self.assertRaisesRegex(ValueError, '2026'):
            ab.training_plan(frame, 1, cfg, 'delta')
        with self.assertRaisesRegex(ValueError, '10 original'):
            ab.training_plan(fixture().iloc[:, :6], 1, cfg, 'delta')

    def test_hand_computed_metrics(self):
        result = ab.metric_values([2, 0], [1, 2], [1, 1])
        self.assertEqual(result['n'], 2)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertEqual(result['mae'], 1.5)
        self.assertEqual(result['bias'], -.5)
        self.assertAlmostEqual(result['rmse_ratio_vs_rw'], np.sqrt(5))

    def test_hac_matches_independent_bartlett_quadratic_form(self):
        t = np.arange(64)
        e0 = np.sin(t/3)+.1
        e1 = np.cos(t/5)-.2
        losses = pd.DataFrame(dict(prediction=e1, actual=np.zeros(64), rw_prediction=e0))
        d = e1**2-e0**2
        centered = d-d.mean()
        lag = 20
        weights = np.maximum(1-np.abs(t[:, None]-t[None, :])/(lag+1), 0)
        se = np.sqrt(centered @ weights @ centered/(64*63))
        from scipy.stats import norm
        for h in range(1, 8):
            result = ridge.loss_hac(losses, h, {'hac_base_lag': 20})
            self.assertEqual(result['hac_lag'], 20)
            self.assertAlmostEqual(result['mean_squared_loss_difference'], d.mean(), places=14)
            self.assertAlmostEqual(result['hac_se'], se, places=14)
            self.assertAlmostEqual(result['p_two_sided'], 2*norm.sf(abs(d.mean()/se)), places=14)
            self.assertAlmostEqual(result['ci_low'], d.mean()-1.95996398454*se, places=14)

    def test_holm_numeric_and_primary_family(self):
        np.testing.assert_allclose(ridge.holm([.04, .01, .2, .03]), [.09, .04, .2, .09])
        cfg = json.loads(runner.CONFIG.read_text())
        self.assertEqual(cfg['primary_family']['size'], 7)
        self.assertFalse(cfg['jensen_correction'])
        self.assertFalse(cfg['final_test_access'])
        self.assertEqual(cfg['fixed_parameters'], cb.config()[1]['fixed_parameters'])

    def test_window_audit_detects_index_changes(self):
        windows = pd.DataFrame([dict(model_id='M0_DELTA', horizon=1, origin_date=pd.Timestamp('2025-01-03'),
                                     train_n=100, train_origin_start=pd.Timestamp('2024-01-01'),
                                     train_origin_end=pd.Timestamp('2024-12-30'), train_label_end=pd.Timestamp('2025-01-03'),
                                     train_indices_sha256=ab.index_hash(np.arange(100)),
                                     **{c+'_median': 1. for c in ridge_v2.CORE})])
        self.assertTrue(ab.check_windows(windows, windows)['identical_indices'])
        changed = windows.copy()
        changed.loc[0, 'train_indices_sha256'] = ab.index_hash(np.arange(1, 101))
        with self.assertRaises(AssertionError):
            ab.check_windows(windows, changed)


@unittest.skipUnless(importlib.util.find_spec('catboost'), 'Install requirements-catboost.txt')
class SeededForecastTests(unittest.TestCase):
    def test_repeatability_future_invariance_and_train_only_imputer(self):
        _, proposal, cfg = cb.config()
        frame = fixture()
        origin = frame.index.get_loc(pd.Timestamp('2025-01-10'))
        changed = frame.copy()
        changed.loc[changed.index > frame.index[origin]] *= 2
        parameters = {'depth': 2, 'l2_leaf_reg': 30}
        windows = []
        with patch.object(cb, 'origins_for_validation', return_value=np.array([origin])):
            for model_id in ab.MODELS:
                a, wa = ab.forecast(frame, 7, parameters, proposal, cfg, model_id)
                b, wb = ab.forecast(changed, 7, parameters, proposal, cfg, model_id)
                c, wc = ab.forecast(frame, 7, parameters, proposal, cfg, model_id)
                pd.testing.assert_frame_equal(a, b, check_exact=True)
                pd.testing.assert_frame_equal(a, c, check_exact=True)
                pd.testing.assert_frame_equal(wa, wb, check_exact=True)
                pd.testing.assert_frame_equal(wa, wc, check_exact=True)
                y = ab.targets(frame, 7, ab.MODELS[model_id])
                train = cb.training(frame, y, 7, origin, cfg)
                np.testing.assert_array_equal(wa[[c+'_median' for c in ridge_v2.CORE]].to_numpy()[0],
                                              np.nanmedian(frame.iloc[train].to_numpy(), axis=0))
                windows.append(wa)
                if model_id == 'M0_DELTA':
                    native = cb.outer_forecasts(frame, 7, parameters, proposal, cfg, origins=np.array([origin]))[0]
                    np.testing.assert_array_equal(a.prediction, native.prediction)
                    np.testing.assert_array_equal(a.prediction_target, native.prediction_target)
        ab.check_windows(*windows)


@unittest.skipUnless((OUTPUT/'manifest.json').exists(), 'Run H13 to validate saved experiment')
class SavedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OUTPUT/'manifest.json').read_text(encoding='utf-8'))
        cls.pred = runner.read_predictions(OUTPUT/'predictions_oos.csv')
        cls.paired = runner.read_predictions(OUTPUT/'paired_predictions.csv')
        cls.original = runner.read_predictions(ROOT/'results/catboost_core_v2/scored_common_predictions.csv')

    def test_all_1778_original_forecasts_and_stop_gate(self):
        m0 = self.pred[self.pred.model_id.eq('M0_DELTA')].copy()
        control = ab.check_m0(m0, self.original)
        self.assertEqual(control.n.sum(), 1778)
        self.assertTrue(control.passed.all())
        m0.loc[m0.index[0], 'prediction'] += .001
        with self.assertRaisesRegex(ValueError, 'do not run M1'):
            ab.check_m0(m0, self.original)

    @unittest.skipUnless(importlib.util.find_spec('catboost'), 'Install requirements-catboost.txt')
    def test_real_m1_repeat_first_middle_last_all_horizons(self):
        _, proposal, features = cb.config()
        frame, _ = cb.load_inputs(features)
        for h in range(1, 8):
            saved = self.pred[self.pred.model_id.eq('M1_LOGRETURN') & self.pred.horizon.eq(h)].iloc[[0, 127, 253]]
            origins = frame.index.get_indexer(saved.origin_date)
            parameters = self.manifest['selected_parameters'][str(h)]
            with patch.object(cb, 'origins_for_validation', return_value=origins):
                repeated, _ = ab.forecast(frame, h, parameters, proposal, features, 'M1_LOGRETURN')
            np.testing.assert_array_equal(repeated.prediction.to_numpy(), saved.prediction.to_numpy())
            np.testing.assert_array_equal(repeated.prediction_target.to_numpy(), saved.prediction_target.to_numpy())

    def test_identical_keys_and_no_2026(self):
        m0 = self.pred[self.pred.model_id.eq('M0_DELTA')]
        m1 = self.pred[self.pred.model_id.eq('M1_LOGRETURN')]
        ab.exact_pairs(m0, m1, ['prediction'])
        self.assertFalse(self.pred.duplicated(['model_id']+ab.KEYS).any())
        self.assertTrue(self.pred.target_date.dt.year.eq(2025).all())
        self.assertTrue((self.pred.origin_date < pd.Timestamp('2026-01-01')).all())
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            ab.exact_pairs(m0, m1.iloc[1:], ['prediction'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ab.exact_pairs(pd.concat([m0, m0.iloc[:1]]), m1, ['prediction'])

    def test_saved_windows_original_and_both_models(self):
        windows = pd.read_csv(OUTPUT/'training_windows.csv', float_precision='round_trip',
                              parse_dates=['origin_date', 'train_origin_start', 'train_origin_end', 'train_label_end'])
        m0, m1 = [windows[windows.model_id.eq(model)].reset_index(drop=True) for model in ab.MODELS]
        self.assertEqual(ab.check_windows(m0, m1)['windows'], 1778)
        runner.check_original_windows(m0, ROOT/'results/catboost_core_v2')
        self.assertTrue((windows.train_label_end <= windows.origin_date).all())
        _, _, features = cb.config()
        frame, _ = cb.load_inputs(features)
        for model in ab.MODELS:
            for h in range(1, 8):
                _, plan = ab.training_plan(frame, h, features, ab.MODELS[model])
                saved = windows[windows.model_id.eq(model) & windows.horizon.eq(h)]
                self.assertEqual(saved.train_indices_sha256.tolist(), [ab.index_hash(train) for _, train in plan])

    def test_metrics_hac_directions_and_zeros(self):
        cfg = json.loads(runner.CONFIG.read_text())
        tables = ab.summarize(self.paired, cfg)
        for name, table in tables.items():
            saved = pd.read_csv(OUTPUT/f'{name}.csv', float_precision='round_trip')
            pd.testing.assert_frame_equal(table, saved, check_exact=False, rtol=1e-13, atol=1e-14)
        controls = pd.read_csv(ROOT/'results/catboost_core_v2/metrics.csv', float_precision='round_trip')
        for model, original in [('M0_DELTA', 'CatBoost_Core_v2'), ('RW', 'RW')]:
            a = tables['metrics'][tables['metrics'].model_id.eq(model)].sort_values('horizon')
            b = controls[controls.model_id.eq(original)].sort_values('horizon')
            np.testing.assert_allclose(a[['rmse','mae','bias']], b[['rmse','mae','bias']], atol=1e-8, rtol=1e-10)
        self.assertEqual(tables['hac_holm'].shape[0], 7)
        self.assertTrue(tables['direction_confusion'].groupby(['model_id','horizon']).n.sum().eq(254).all())
        rw = tables['behavior'][tables['behavior'].model_id.eq('RW')]
        self.assertTrue(rw.n_predicted_zero.eq(254).all())
        self.assertEqual(rw.n_actual_zero.tolist(), [3, 2, 2, 0, 3, 1, 1])
        self.assertTrue(rw.n_correct_zero.eq(rw.n_actual_zero).all())

    def test_manifest_inputs_outputs_and_historical_integrity(self):
        self.assertTrue(self.manifest['no_2026_numeric_data'])
        self.assertTrue(self.manifest['historical_files_unchanged'])
        self.assertTrue(self.manifest['m0_all_1778_verified'])
        self.assertEqual(self.manifest['features'], ridge_v2.CORE)
        for name, value in self.manifest['input_sha256'].items():
            self.assertEqual(runner.digest(ROOT/name), value, name)
        for name, value in self.manifest['output_sha256'].items():
            self.assertEqual(runner.digest(OUTPUT/name), value, name)
            if Path(name).suffix in {'.csv', '.json', '.md'}:
                self.assertNotIn(b'\r\n', (OUTPUT/name).read_bytes(), name)
        self.assertEqual(runner.protected_hashes(), self.manifest['protected_sha256'])


if __name__ == '__main__':
    unittest.main()
