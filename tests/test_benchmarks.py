import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from benchmarks.core import (ar1, generate_forecasts, load_config, load_data,
                             load_calendar, make_splits, metrics_by_horizon, score_forecasts)


class BenchmarksTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/benchmark_config.yaml")
        self.config["min_ar_pairs"] = 3
        self.calendar = pd.date_range("2024-12-20", periods=25)
        self.data = pd.DataFrame({"price": 100 + np.cumsum(np.sin(np.arange(25))),
                                  "source": "synthetic"}, index=self.calendar)

    def test_rw_tail_and_target_split(self):
        predictions = generate_forecasts(self.data, self.calendar, self.config)
        rw = predictions[predictions.model_id.eq("B0")]
        np.testing.assert_allclose(rw.prediction, rw.origin_date.map(self.data.price))
        self.assertEqual(len(rw), 25 * 7)
        self.assertEqual(rw.target_date.isna().sum(), 28)
        crossing = rw[(rw.origin_date < "2025-01-01") & (rw.target_date >= "2025-01-01")]
        self.assertTrue(len(crossing) > 0)
        self.assertTrue(crossing.split.eq("validation").all())
        scored = score_forecasts(predictions, self.data)
        self.assertTrue(scored.loc[scored.target_date.isna(), "scored_status"].eq("unscored").all())

    def test_no_future_leakage(self):
        original = generate_forecasts(self.data, self.calendar, self.config)
        changed = self.data.copy()
        changed.loc[self.calendar[15]:, "price"] *= 10
        altered = generate_forecasts(changed, self.calendar, self.config)
        mask = original.origin_date < self.calendar[15]
        pd.testing.assert_frame_equal(original.loc[mask], altered.loc[mask])

    def test_ar_ols_and_recursive_levels(self):
        changes = [0.7]
        for _ in range(15):
            changes.append(0.2 + 0.6 * changes[-1])
        prices = pd.Series(np.r_[100, 100 + np.cumsum(changes)])
        predictions, meta = ar1(prices, range(1, 8), 3)
        self.assertAlmostEqual(meta["c"], 0.2)
        self.assertAlmostEqual(meta["phi"], 0.6)
        expected, current, delta = [], prices.iloc[-1], changes[-1]
        for _ in range(7):
            delta = 0.2 + 0.6 * delta
            current += delta
            expected.append(current)
        np.testing.assert_allclose(predictions, expected)

    def test_missing_session_preserved_and_ar_pairs(self):
        data = self.data.drop(self.calendar[10])
        predictions = generate_forecasts(data, self.calendar, self.config)
        row = predictions[(predictions.origin_date == self.calendar[9])
                          & (predictions.horizon == 1) & (predictions.model_id == "B0")].iloc[0]
        self.assertEqual(row.target_date, self.calendar[10])
        scored = score_forecasts(predictions, data, ["train", "validation"])
        self.assertTrue(scored.loc[scored.target_date.eq(self.calendar[10]), "actual"].isna().all())
        self.assertFalse(predictions.origin_date.eq(self.calendar[10]).any())
        _, meta = ar1(data.price.reindex(self.calendar), range(1, 8), 3)
        self.assertEqual(meta["n_pairs"], 20)  # 23 potential pairs minus three crossing the hole
        _, meta = ar1(data.price.reindex(self.calendar).iloc[:12], range(1, 8), 3)
        self.assertEqual(meta["reason"], "missing_last_change")

    def test_insufficient_and_singular_history(self):
        values, meta = ar1(self.data.price.iloc[:4], [1], 100)
        self.assertIsNone(values)
        self.assertEqual(meta["reason"], "insufficient_history")
        values, meta = ar1(pd.Series(range(20), dtype=float), [1], 3)
        self.assertIsNone(values)
        self.assertEqual(meta["reason"], "singular_design")

    def test_final_test_sealed_and_metrics_common_pairs(self):
        self.data.index = self.calendar = pd.date_range("2025-12-20", periods=25)
        predictions = generate_forecasts(self.data, self.calendar, self.config)
        scored = score_forecasts(predictions, self.data)
        self.assertTrue(scored.loc[scored.split.eq("final_test"), "actual"].isna().all())
        metrics = metrics_by_horizon(scored)
        for _, group in metrics.groupby(["split", "horizon"]):
            self.assertEqual(group.n.nunique(), 1)
        self.assertEqual(set(metrics.split), {"validation"})
        self.assertEqual(set(metrics.model_id), {"B0", "B1"})

    def test_split_overlap_rejected(self):
        splits = make_splits(self.data, self.config)
        self.assertEqual(sum(map(len, splits.values())), len(self.data))
        self.config["splits"]["validation"][0] = "2024-12-31"
        with self.assertRaises(ValueError):
            make_splits(self.data, self.config)

    def test_loader_rejects_invalid_and_retains_missing(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            for dates, prices in [(["2025-01-01"] * 2, [10, 11]),
                                  (["2025-01-02", "2025-01-01"], [10, 11]),
                                  (["2025-01-01"], [0]), (["2025-01-01"], ["bad"])]:
                pd.DataFrame({"date": dates, "price": prices}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_data(path)
            pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "price": [10, np.nan]}).to_csv(path, index=False)
            self.assertTrue(pd.isna(load_data(path).price.iloc[1]))


if __name__ == "__main__":
    unittest.main()


def test_final_test_cli_requires_explicit_flag(tmp_path):
    result = subprocess.run([sys.executable, "-m", "benchmarks.run",
                             "--evaluation-split", "final_test", "--output", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires explicit --include-final-test" in result.stderr
    assert not list(tmp_path.iterdir())


def test_independent_calendar_keeps_saturday_and_unknown_future_prices(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/benchmark_config.yaml")
    dates = pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-06", "2025-01-07"])
    pd.DataFrame({"date": dates}).to_csv(tmp_path / "sessions.csv", index=False)
    config.update(calendar_mode="independent", calendar_path="sessions.csv",
                  calendar_source="synthetic calendar for test only")
    data = pd.DataFrame({"price": [100., 102.], "source": "synthetic"}, index=dates[:2])
    calendar = load_calendar(data, config, tmp_path)
    assert calendar.equals(dates)
    predictions = generate_forecasts(data, calendar, config)
    scored = score_forecasts(predictions, data)
    known_future = scored.target_date.isin(dates[2:])
    assert known_future.any()
    assert scored.loc[known_future, "scored_status"].eq("unscored").all()
    assert scored.loc[known_future, "actual"].isna().all()
    config["calendar_source"] = None
    with pytest.raises(ValueError, match="provenance"):
        load_calendar(data, config, tmp_path)


def test_real_data_split_sizes_and_default_scoring_contract():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/benchmark_config.yaml")
    data = load_data(root / config["data_path"])
    splits = make_splits(data, config)
    assert {name: len(frame) for name, frame in splits.items()} == {
        "train": 683, "validation": 254, "final_test": 223}
    assert config["min_ar_pairs"] == 100
