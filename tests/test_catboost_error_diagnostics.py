"""Saved-data controls and adversarial checks; never fit forecasting models."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts import analyze_catboost_errors as diag


class ErrorDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((diag.ROOT / 'configs/benchmark_config.yaml').read_text())
        cls.pred = diag.read(diag.ROOT / 'results/catboost_core_v2/scored_common_predictions.csv')
        cls.history = diag.read(diag.ROOT / 'data/processed/USDRUBF_baseline_2026-11-18.csv')
        cls.calendar = diag.read(diag.ROOT / cls.config['calendar_path'])
        cls.saved = diag.read(diag.ROOT / 'results/catboost_core_v2/metrics.csv')
        cls.frame, cls.prices = diag.validate_pairs(cls.pred, cls.history, cls.calendar, cls.config)

    def validate(self, pred=None, history=None, calendar=None):
        return diag.validate_pairs(self.pred if pred is None else pred,
                                   self.history if history is None else history,
                                   self.calendar if calendar is None else calendar, self.config)

    def test_all_horizons_and_controls(self):
        result = diag.check_metrics(self.frame, self.saved)
        self.assertEqual(result.n.tolist(), [254] * 7)
        self.assertEqual(len(self.frame), 1778)
        self.assertLess(self.prices.index.max(), pd.Timestamp('2026-01-01'))

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.validate(pd.concat([self.pred, self.pred.iloc[[0]]]))

    def test_missing_key_rejected(self):
        with self.assertRaisesRegex(ValueError, '254'):
            self.validate(self.pred.iloc[1:])

    def test_invalid_trading_horizon_rejected(self):
        p = self.pred.copy()
        p.loc[0, 'origin_date'] = '2024-12-28'
        with self.assertRaisesRegex(ValueError, 'trading-session horizon'):
            self.validate(p)

    def test_price_and_prediction_equalities(self):
        for column, message in [('actual', 'target CLOSE'), ('RW', 'origin CLOSE'),
                                ('prediction', r'RW \+ prediction_target')]:
            with self.subTest(column=column):
                p = self.pred.copy()
                p.loc[0, column] += .01
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(p)

    def test_missing_dates_prices_and_calendar(self):
        for column in ['origin_date', 'target_date', 'prediction_target']:
            with self.subTest(column=column):
                p = self.pred.copy()
                p.loc[0, column] = np.nan
                with self.assertRaises(ValueError):
                    self.validate(p)
        history = self.history.copy()
        history.loc[history.date.eq(self.pred.iloc[0].target_date), 'price'] = np.nan
        with self.assertRaisesRegex(ValueError, 'CLOSE prices'):
            self.validate(history=history)
        with self.assertRaises(ValueError):
            self.validate(calendar=self.calendar[self.calendar.date.ne(self.pred.iloc[0].target_date)])

    def test_metric_mismatch_rejected(self):
        for column in ['rmse', 'mae', 'bias', 'n']:
            with self.subTest(column=column):
                saved = self.saved.copy()
                saved.loc[0, column] += 1
                with self.assertRaisesRegex(ValueError, 'Control'):
                    diag.check_metrics(self.frame, saved)

    def test_separate_rw_rows_exact_join(self):
        rw = self.pred.copy()
        rw['model_id'] = 'RW'
        rw['prediction'] = rw.RW
        combined = pd.concat([self.pred, rw], ignore_index=True).drop(columns='RW').sample(frac=1, random_state=42)
        frame, _ = self.validate(combined)
        pd.testing.assert_frame_equal(frame[self.frame.columns.drop('RW')], self.frame.drop(columns='RW'))
        with self.assertRaisesRegex(ValueError, 'key sets differ'):
            self.validate(combined.drop(combined[combined.model_id.eq('RW')].index[:1]))

    def test_volatility_is_exact_past_twenty_returns(self):
        vol, _, _ = diag.volatility(self.prices, self.config['splits']['train'])
        origin = pd.Timestamp('2025-06-02')
        pos = self.prices.index.get_loc(origin)
        expected = np.log(self.prices.iloc[pos - 20:pos + 1]).diff().dropna().std(ddof=1)
        self.assertAlmostEqual(vol.loc[origin], expected, places=14)

    def test_future_invariance_and_train_only_thresholds(self):
        before, bounds, _ = diag.volatility(self.prices, self.config['splits']['train'])
        changed = self.prices.copy()
        origin = pd.Timestamp('2025-06-02')
        changed.loc[changed.index > origin] *= np.linspace(2, 10, (changed.index > origin).sum())
        after, bounds_after, _ = diag.volatility(changed, self.config['splits']['train'])
        pd.testing.assert_series_equal(before.loc[:origin], after.loc[:origin])
        np.testing.assert_array_equal(bounds, bounds_after)
        changed.loc[changed.index >= '2025-01-01'] *= 3
        _, final_bounds, _ = diag.volatility(changed, self.config['splits']['train'])
        np.testing.assert_array_equal(bounds, final_bounds)

    def test_2026_prices_have_no_effect(self):
        history = self.history.copy()
        history.loc[history.date >= '2026-01-01', 'price'] = -999
        frame, prices = self.validate(history=history)
        pd.testing.assert_frame_equal(frame, self.frame)
        pd.testing.assert_series_equal(prices, self.prices)

    def test_exact_zero_direction(self):
        self.assertEqual(diag.direction([-1, -1e-14, 0, -0., 1e-14, 1]).tolist(),
                         ['down', 'down', 'zero', 'zero', 'up', 'up'])

    def test_csv_repeatability_and_confusion_counts(self):
        first, _, _ = diag.analyze(self.frame, self.prices, self.config)
        second, _, _ = diag.analyze(self.frame, self.prices, self.config)
        self.assertEqual(first['direction_confusion'].groupby('horizon').n.sum().tolist(), [254] * 7)
        self.assertTrue(first['top_errors'].groupby('horizon').size().eq(10).all())
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a', Path(folder) / 'b'
            diag.write_tables(first, a)
            diag.write_tables(second, b)
            for path in a.glob('*.csv'):
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes(), path.name)


if __name__ == '__main__':
    unittest.main()
