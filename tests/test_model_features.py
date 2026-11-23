"""Offline B3 projection/scenario tests. No model training or backtest scoring."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_features_b3', ROOT / 'scripts/build_model_features.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ModelFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = builder.load_config()
        cls.source = builder.read_csv(ROOT / cls.config['source_data'], builder.SOURCE_COLUMNS)
        cls.calendar = builder.read_csv(ROOT / cls.config['calendar'])
        cls.raw = builder.read_csv(ROOT / cls.config['raw_candles'])
        cls.features = builder.read_csv(ROOT / cls.config['features_file'])
        cls.metadata = builder.read_csv(ROOT / cls.config['metadata_file'])

    def test_calendar_row_count_order_and_same_date_candles(self):
        self.assertEqual(len(self.features), 1160)
        dates = [r['date'] for r in self.features]
        self.assertEqual(len(set(dates)), 1160)
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(dates, [r['date'] for r in self.source])
        self.assertEqual(dates, [r['date'] for r in self.calendar if r['is_trading_day_expected'] == '1'])
        self.assertEqual(dates, [r['begin'][:10] for r in self.raw])
        builder.validate_source(self.source, self.calendar, self.raw, 1160)

    def test_exact_source_values_and_metadata_alignment(self):
        for number, (source, feature, meta) in enumerate(zip(self.source, self.features, self.metadata), 1):
            self.assertEqual(feature, {k: source[k] for k in ('date',) + builder.STORE_FEATURES})
            self.assertEqual(meta['date'], source['date'])
            self.assertEqual(meta['source_row_number'], str(number))
            self.assertEqual(meta['source_sha256'], self.config['source_sha256'])
            self.assertEqual(meta['pit_status'], builder.PIT_STATUS)
            for col in builder.METADATA_INPUT:
                self.assertEqual(meta[col], source[col])
        self.assertEqual(tuple(self.metadata[0]), builder.METADATA_COLUMNS)

    def test_scenarios_and_default_core_are_explicit(self):
        for name, expected in builder.SCENARIO_FEATURES.items():
            rows = builder.load_scenario(scenario=name)
            self.assertEqual(tuple(rows[0]), ('date',) + expected)
            self.assertEqual(len(rows), 1160)
            self.assertEqual(self.config['scenarios'][name]['sensitivity_only'], name != 'core')
        self.assertEqual(len(builder.CORE), 9)
        self.assertEqual(len(builder.SCENARIO_FEATURES['core_ruonia_brent']), 11)
        self.assertEqual(len(builder.STORE_FEATURES), 12)
        default = builder.load_scenario()
        self.assertEqual(tuple(default[0]), ('date',) + builder.CORE)
        self.assertTrue(set(builder.SENSITIVITY).isdisjoint(default[0]))
        with self.assertRaises(ValueError):
            builder.load_scenario(scenario='automatic_all')

    def test_no_forbidden_columns_or_future_targets(self):
        self.assertEqual(tuple(self.features[0]), ('date',) + builder.STORE_FEATURES)
        self.assertTrue(set(self.config['excluded_factors']).isdisjoint(self.features[0]))
        for forbidden in ['swaprate_current', 'value', 'begin', 'end',
                          'swaprate_available_assumed', 'close_t+1', 'target_h7']:
            config = copy.deepcopy(self.config)
            config['scenarios']['core']['features'].append(forbidden)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ValueError):
                    builder.load_config(path)

    def test_future_target_column_in_feature_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            row = {**self.features[0], 'close_t+1': '9999'}
            path.write_bytes(builder.csv_bytes([row], tuple(row)))
            with self.assertRaises(ValueError):
                builder.load_scenario(path)

    def test_missing_funding_not_filled_and_rows_not_dropped(self):
        for source, feature, meta in zip(self.source, self.features, self.metadata):
            self.assertEqual(feature['swaprate_prev_session'], source['swaprate_prev_session'])
            self.assertEqual(meta['missing_features'],
                             'swaprate_prev_session' if source['swaprate_prev_session'] == '' else '')
            self.assertEqual(meta['swaprate_available_assumed'],
                             str(bool(feature['swaprate_prev_session'])).lower())
        self.assertEqual(sum(r['swaprate_prev_session'] == '' for r in self.features), 154)
        self.assertEqual(sum(all(r[col] != '' for col in builder.CORE) for r in self.features), 1006)
        # Synthetic observed zero remains an observed zero, not a missing value.
        row = self.source[200].copy()
        row['swaprate_prev_session'] = '0.00000'
        features, metadata = builder.project_tables([row], self.config, 'synthetic')
        self.assertEqual(features[0]['swaprate_prev_session'], '0.00000')
        self.assertEqual(metadata[0]['missing_features'], '')

    def test_source_dates_funding_and_calendar_errors_rejected(self):
        duplicate = copy.deepcopy(self.source)
        duplicate[1]['date'] = duplicate[0]['date']
        with self.assertRaisesRegex(ValueError, 'unique'):
            builder.validate_source(duplicate, self.calendar, self.raw, 1160)
        for col, invalid in [('swaprate_source_trade_date', '2020-01-01'),
                             ('swaprate_prev_session', '99999'),
                             ('brent_eia_reference_date', self.source[200]['date']),
                             ('ruonia_source_dateupdate', '2099-01-01 14:00:00'),
                             ('usdcnh_investing_reference_date', '2099-01-01')]:
            rows = copy.deepcopy(self.source)
            rows[200][col] = invalid
            with self.assertRaises(ValueError):
                builder.validate_source(rows, self.calendar, self.raw, 1160)
        calendar = copy.deepcopy(self.calendar)
        calendar[0]['is_trading_day_expected'] = '0'
        with self.assertRaisesRegex(ValueError, 'calendar'):
            builder.validate_source(self.source, calendar, self.raw, 1160)

    def test_no_global_preprocessing_and_prefix_independent_of_future(self):
        self.assertEqual(self.config['transformations'], [])
        self.assertIs(self.config['train_only_preprocessing_required'], True)
        # Projection of an early prefix is identical with/without distant rows.
        full, _ = builder.project_tables(self.source, self.config, self.config['source_sha256'])
        prefix, _ = builder.project_tables(self.source[:100], self.config, self.config['source_sha256'])
        self.assertEqual(prefix, full[:100])
        altered = copy.deepcopy(self.source)
        for row in altered[100:]:
            for col in builder.STORE_FEATURES:
                row[col] = '99999999'
        changed, _ = builder.project_tables(altered, self.config, self.config['source_sha256'])
        self.assertEqual(prefix, changed[:100])
        self.assertEqual(full[0]['close'], self.source[0]['close'])

    def test_unsafe_preprocessing_config_rejected(self):
        for key, value in [('train_only_preprocessing_required', False),
                           ('transformations', ['global_standard_scaler']),
                           ('future_targets_in_features', True), ('default_scenario', 'core_ruonia_brent')]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ValueError):
                    builder.load_config(path)

    def test_rebuild_deterministic_matches_checked_in_outputs(self):
        a, b = builder.build_artifacts(), builder.build_artifacts()
        self.assertEqual(a, b)
        self.assertEqual(a[0], (ROOT / self.config['features_file']).read_bytes())
        self.assertEqual(a[1], (ROOT / self.config['metadata_file']).read_bytes())
        self.assertEqual(a[2], (ROOT / 'reports/features_daily_v1_coverage.md').read_bytes())

    def test_cli_compare_is_read_only_and_detects_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            report = output / 'coverage.md'
            args = ['--output-dir', str(output), '--report', str(report)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(builder.main(args), 0)
                paths = [output / 'features_daily_v1.csv', output / 'features_daily_v1_metadata.csv', report]
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
                self.assertEqual(builder.main(args + ['--compare']), 0)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})
                paths[0].write_bytes(b'corrupted\n')
                with self.assertRaisesRegex(ValueError, 'differs'):
                    builder.main(args + ['--compare'])

    def test_inputs_and_protocol_never_modified(self):
        paths = [ROOT / self.config[k] for k in ('source_data', 'calendar', 'raw_candles',
                                                'availability_catalog', 'lag_policy_document')]
        paths += [ROOT / 'docs/backtest_protocol.md', ROOT / 'docs/research_charter.md']
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        builder.build_artifacts()
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                builder.main(['--report', str(ROOT / 'docs/backtest_protocol.md')])


if __name__ == '__main__':
    unittest.main()
