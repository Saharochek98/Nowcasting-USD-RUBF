"""Future invariance, maturity, funding, deterministic hist models and common keys."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from benchmarks import xgboost_direct as xgb, ridge, ridge_v2
from scripts.run_xgboost_experiment import common_pairs, forecast
from tests.test_catboost_direct import panel,tiny_fit


@unittest.skipUnless(importlib.util.find_spec('xgboost') and importlib.util.find_spec('sklearn'),
                     'Install requirements-xgboost.txt')
class XGBoostDirectTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.features = xgb.config()
        self.parameters = xgb.candidates(self.cfg)[0]
        self.frame = panel()

    def test_fixed_design_source_and_funding(self):
        frame,calendar = xgb.load_inputs(self.features)
        self.assertEqual(list(frame),ridge_v2.CORE)
        self.assertEqual(str(calendar[-1].date()),'2025-12-30')
        self.assertEqual(frame.swaprate_prev_session.isna().sum(),154)
        self.assertEqual(frame.swaprate_current.isna().sum(),153)
        self.assertTrue(frame.swaprate_current.eq(0).any())
        self.assertEqual(len(xgb.candidates(self.cfg)),4)
        self.assertEqual(self.cfg['hac_family']['size'],21)

    def test_train_only_imputation_and_real_zero(self):
        x = np.array([[0.,1.],[np.nan,2.],[2.,3.],[4.,4.]])
        model,median = xgb.fit_model(x,[1.,2.,3.,4.],self.parameters,self.cfg)
        np.testing.assert_array_equal(median,[2.,2.5])
        before = median.copy()
        xgb.predict(model,median,[[np.nan,999.]])
        np.testing.assert_array_equal(median,before)
        self.assertEqual(np.where(np.isnan(x),median,x)[0,0],0.)
        with self.assertRaises(ValueError):
            xgb.fit_model([[np.nan],[np.nan]],[1.,2.],self.parameters,self.cfg)

    def test_matured_prequential_labels_all_horizons(self):
        origin = self.frame.index.get_loc(pd.Timestamp('2025-03-03'))
        for h in range(1,8):
            y = ridge.supervised(self.frame,h,'delta')
            train = xgb.training(self.frame,y,h,origin,self.features)
            self.assertEqual(train[-1]+h,origin)
            self.assertGreater(self.frame.index[train[-1]+h],pd.Timestamp('2024-12-31'))

    def test_real_forecasts_ignore_future_and_reconstruct_prices(self):
        origins = self.frame.index.get_indexer(pd.to_datetime(['2025-01-10','2025-03-03','2025-06-20']))
        a = xgb.outer_forecasts(self.frame,7,self.parameters,self.cfg,self.features,origins=origins)
        changed = self.frame.copy()
        changed.loc['2025-07-01':] *= 100
        b = xgb.outer_forecasts(changed,7,self.parameters,self.cfg,self.features,origins=origins)
        for i in range(3): pd.testing.assert_frame_equal(a[i],b[i],check_exact=True)
        pred,windows = a[:2]
        np.testing.assert_array_equal(pred.prediction,self.frame.close.iloc[origins].to_numpy()+pred.prediction_target.to_numpy())
        self.assertTrue((pred.train_end_date <= pred.origin_date).all())
        for i,row in enumerate(pred.itertuples()):
            self.assertEqual(self.frame.index.get_loc(row.target_date)-origins[i],7)
        train = xgb.training(self.frame,ridge.supervised(self.frame,7,'delta'),7,origins[0],self.features)
        self.assertEqual(windows.iloc[0].volume_median,self.frame.volume.iloc[train].median())

    def test_inner_tuning_has_no_validation_or_future_access(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['inner_origin_windows'] = [['2023-07-03','2023-07-05']]
        changed = self.frame.copy()
        changed.loc['2025-01-01':] *= 100
        results = []
        with patch.object(xgb,'fit_model',side_effect=tiny_fit):
            for frame in [self.frame,changed]:
                results.append(pd.concat([xgb.inner_evaluate(frame,7,p,i,cfg,self.features)
                    for i,p in enumerate(xgb.candidates(cfg))],ignore_index=True))
        pd.testing.assert_frame_equal(results[0],results[1],check_exact=True)
        self.assertTrue((results[0].train_label_end <= results[0].origin_date).all())
        first_outer = self.frame.index[xgb.origins_for_validation(self.frame,7,self.features)[0]]
        self.assertTrue((results[0].target_date < first_outer).all())
        self.assertEqual(xgb.choose_parameters(results[0],cfg)[0],xgb.choose_parameters(results[1],cfg)[0])

    def test_fixed_grid_tie_and_exact_inner_dates(self):
        rows = [dict(candidate_id=i,origin_date=d,target_date=d+7,horizon=7,error=1.)
                for i in range(4) for d in [1,2]]
        selected,_ = xgb.choose_parameters(pd.DataFrame(rows),self.cfg)
        self.assertEqual(selected,dict(max_depth=2,reg_lambda=10))
        with self.assertRaises(ValueError): xgb.choose_parameters(pd.DataFrame(rows).iloc[1:],self.cfg)

    def test_no_eval_set_early_stopping_or_parallel_fit(self):
        with patch('xgboost.XGBRegressor') as constructor:
            xgb.fit_model([[1.],[2.],[3.]],[1.,2.,3.],self.parameters,self.cfg)
            self.assertEqual(constructor.call_args.kwargs['n_jobs'],1)
            self.assertEqual(constructor.call_args.kwargs['min_child_weight'],5)
            self.assertNotIn('early_stopping_rounds',constructor.call_args.kwargs)
            self.assertEqual(constructor.return_value.fit.call_args.kwargs,dict(verbose=False))

    def test_seed_gain_unused_feature_and_model_roundtrip(self):
        from xgboost import XGBRegressor
        x = np.column_stack([np.arange(60.),np.arange(60.)%7,np.zeros(60)])
        y = np.sin(x[:,0]/5)
        a,median = xgb.fit_model(x,y,self.parameters,self.cfg)
        b,_ = xgb.fit_model(x,y,self.parameters,self.cfg)
        np.testing.assert_array_equal(xgb.predict(a,median,x),xgb.predict(b,median,x))
        raw,share = xgb.gain(a,3)
        np.testing.assert_array_equal(raw,xgb.gain(b,3)[0])
        self.assertEqual(raw[2],0.)
        self.assertAlmostEqual(share.sum(),1.)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'model.ubj')
            a.save_model(path)
            # sklearn's load_model mutates the estimator in place.
            restored = XGBRegressor(n_jobs=1)
            restored.load_model(path)
            np.testing.assert_array_equal(xgb.predict(a,median,x),xgb.predict(restored,median,x))

    def test_common_dates_actuals_and_no_silent_drop(self):
        origin = self.frame.index.get_loc(pd.Timestamp('2025-03-03'))
        pred = pd.DataFrame(dict(origin_date=[self.frame.index[origin]],target_date=[self.frame.index[origin+1]],
                                 horizon=[1],prediction=[80.]))
        base = pred[ridge.KEYS].copy()
        base['actual'] = self.frame.close.iloc[origin+1]
        for c in ['RW','AR1','RW_DRIFT','Ridge_Core_v2','CatBoost_Core_v2']: base[c]=80.
        self.assertEqual(len(common_pairs(pred,base,self.frame)),1)
        with self.assertRaises(ValueError): common_pairs(pred,base.iloc[:0],self.frame)
        base['actual'] += 1
        with self.assertRaises(AssertionError): common_pairs(pred,base,self.frame)

    def test_all_seven_selections_sealed_before_outer(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.run_xgboost_experiment.check_design'),patch.object(xgb,'load_inputs',side_effect=AssertionError('no fit')):
                with self.assertRaises(FileNotFoundError): forecast(Path(folder),1,self.cfg,self.features)


if __name__ == '__main__': unittest.main()
