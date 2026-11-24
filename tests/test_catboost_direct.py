"""Leakage scheduling plus real seeded CatBoost numerical checks."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks import catboost_direct as cb, ridge, ridge_v2
from scripts.run_catboost_experiment import common_pairs, forecast
from tests.test_ridge import fixture


def panel():
    frame = fixture()
    frame['swaprate_current'] = frame.swaprate_prev_session.shift(-1)
    frame['cbr_usdrub_effective'] = frame.close-.2
    frame['spread_close_minus_cbr'] = .2
    frame['key_rate_pct_effective'] = 10+np.arange(len(frame))%4
    return frame[ridge_v2.CORE]


class TinyModel:
    def __init__(self,y): self.intercept = np.mean(y)
    def predict(self,x,thread_count=1): return self.intercept+np.asarray(x)[:,0]*.00001


def tiny_fit(x,y,parameters,proposal):
    return TinyModel(y),np.nanmedian(x,axis=0)


@unittest.skipUnless(importlib.util.find_spec('catboost'),'Install requirements-catboost.txt')
class CatBoostDirectTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.proposal,self.features = cb.config()
        self.frame = panel()
        self.parameters = cb.candidates(self.proposal)[0]

    def test_fixed_grid_loader_and_funding(self):
        frame,calendar = cb.load_inputs(self.features)
        self.assertEqual(list(frame),ridge_v2.CORE)
        self.assertEqual(len(frame.columns),10)
        self.assertEqual(str(calendar[-1].date()),'2025-12-30')
        self.assertEqual(frame.swaprate_prev_session.isna().sum(),154)
        self.assertEqual(frame.swaprate_current.isna().sum(),153)
        self.assertTrue(frame.swaprate_current.eq(0).any())
        self.assertEqual(len(cb.candidates(self.proposal)),4)

    def test_train_only_median_and_observed_zero(self):
        x = np.array([[0.,1.],[np.nan,2.],[2.,3.],[4.,4.]])
        model,median = cb.fit_model(x,[1.,2.,3.,4.],self.parameters,self.proposal)
        np.testing.assert_array_equal(median,[2.,2.5])
        before = median.copy()
        cb.predict(model,median,[[np.nan,999.]])
        np.testing.assert_array_equal(before,median)
        self.assertEqual(np.where(np.isnan(x),median,x)[0,0],0.)
        with self.assertRaises(ValueError):
            cb.fit_model([[np.nan],[np.nan]],[1.,2.],self.parameters,self.proposal)

    def test_matured_labels_for_all_horizons(self):
        origin = self.frame.index.get_loc(pd.Timestamp('2025-03-03'))
        for h in range(1,8):
            y = ridge.supervised(self.frame,h,'delta')
            train = cb.training(self.frame,y,h,origin,self.features)
            self.assertEqual(train[-1]+h,origin)
            self.assertGreater(self.frame.index[train[-1]+h],pd.Timestamp('2024-12-31'))
            self.assertTrue(np.isfinite(y[train]).all())

    def test_real_outer_future_invariance_reconstruction_and_reproducibility(self):
        origins = self.frame.index.get_indexer(pd.to_datetime(['2025-01-10','2025-03-03','2025-06-20']))
        a = cb.outer_forecasts(self.frame,7,self.parameters,self.proposal,self.features,origins=origins)
        changed = self.frame.copy()
        changed.loc['2025-07-01':] *= 100
        b = cb.outer_forecasts(changed,7,self.parameters,self.proposal,self.features,origins=origins)
        for i in range(3): pd.testing.assert_frame_equal(a[i],b[i],check_exact=True)
        pred,windows = a[:2]
        np.testing.assert_allclose(pred.prediction,self.frame.close.iloc[origins]+pred.prediction_target.to_numpy())
        self.assertTrue((pred.train_end_date <= pred.origin_date).all())
        y = ridge.supervised(self.frame,7,'delta')
        train = cb.training(self.frame,y,7,origins[0],self.features)
        self.assertAlmostEqual(windows.iloc[0].volume_median,self.frame.volume.iloc[train].median())
        for j,row in enumerate(pred.itertuples()):
            self.assertEqual(self.frame.index.get_loc(row.target_date)-origins[j],7)
        self.assertTrue(pred.pit_status.eq('ASSUMED_AVAILABLE').all())

    def test_inner_selection_precedes_outer_and_ignores_validation(self):
        proposal = copy.deepcopy(self.proposal)
        proposal['inner_origin_windows'] = [['2023-07-03','2023-07-05']]
        changed = self.frame.copy()
        changed.loc['2025-01-01':] *= 100
        ledgers = []
        with patch.object(cb,'fit_model',side_effect=tiny_fit):
            for frame in [self.frame,changed]:
                parts = [cb.inner_evaluate(frame,7,p,i,proposal,self.features)
                         for i,p in enumerate(cb.candidates(proposal))]
                ledgers.append(pd.concat(parts,ignore_index=True))
        pd.testing.assert_frame_equal(ledgers[0],ledgers[1],check_exact=True)
        inner = ledgers[0]
        self.assertTrue((inner.train_label_end <= inner.origin_date).all())
        first_outer = self.frame.index[cb.origins_for_validation(self.frame,7,self.features)[0]]
        self.assertTrue((inner.target_date < first_outer).all())
        self.assertGreater(inner[inner.candidate_id.eq(0)].train_n.iloc[-1],inner.train_n.iloc[0])
        a,_ = cb.choose_parameters(inner,proposal)
        b,_ = cb.choose_parameters(ledgers[1],proposal)
        self.assertEqual(a,b)

    def test_deterministic_grid_tie_and_missing_candidate_keys(self):
        rows = []
        for i in range(4):
            for day in [1,2]:
                rows.append(dict(candidate_id=i,origin_date=day,target_date=day+7,horizon=7,error=1.))
        inner = pd.DataFrame(rows)
        selected,_ = cb.choose_parameters(inner,self.proposal)
        self.assertEqual(selected,dict(depth=2,l2_leaf_reg=10))
        with self.assertRaises(ValueError): cb.choose_parameters(inner.iloc[1:],self.proposal)

    def test_fit_has_no_eval_set_or_early_stopping(self):
        with patch('catboost.CatBoostRegressor') as constructor:
            cb.fit_model([[1.],[2.],[3.]],[1.,2.,3.],self.parameters,self.proposal)
            self.assertEqual(constructor.call_args.kwargs['iterations'],100)
            self.assertEqual(constructor.call_args.kwargs['random_seed'],240112)
            self.assertEqual(constructor.return_value.fit.call_args.kwargs,{'use_best_model':False})

    def test_model_roundtrip_and_seeded_importance(self):
        from catboost import CatBoostRegressor
        x = np.arange(80.).reshape(40,2)
        y = np.sin(x[:,0]/5)
        a,median = cb.fit_model(x,y,self.parameters,self.proposal)
        b,_ = cb.fit_model(x,y,self.parameters,self.proposal)
        np.testing.assert_array_equal(cb.predict(a,median,x),cb.predict(b,median,x))
        np.testing.assert_array_equal(a.get_feature_importance(thread_count=1),b.get_feature_importance(thread_count=1))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'model.cbm')
            a.save_model(path)
            restored = CatBoostRegressor().load_model(path)
            np.testing.assert_array_equal(cb.predict(a,median,x),cb.predict(restored,median,x))

    def test_comparison_requires_exact_keys_and_actuals(self):
        origin = self.frame.index.get_loc(pd.Timestamp('2025-03-03'))
        pred = pd.DataFrame(dict(origin_date=[self.frame.index[origin]],target_date=[self.frame.index[origin+1]],
                                 horizon=[1],prediction=[80.]))
        base = pred[ridge.KEYS].copy()
        base['actual'] = self.frame.close.iloc[origin+1]
        for name in ['RW','AR1','RW_DRIFT','Ridge_Core_v2']: base[name] = 80.
        self.assertEqual(len(common_pairs(pred,base,self.frame)),1)
        with self.assertRaises(ValueError): common_pairs(pred,base.iloc[:0],self.frame)
        base['actual'] += 1
        with self.assertRaises(AssertionError): common_pairs(pred,base,self.frame)

    def test_all_selections_required_before_forecasting(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.run_catboost_experiment.check_design'),patch.object(cb,'load_inputs',side_effect=AssertionError('must not train')):
                with self.assertRaises(FileNotFoundError):
                    forecast(Path(folder),1,self.cfg,self.proposal,self.features)


if __name__ == '__main__': unittest.main()
