"""B2 audit: snapshot constraints and the real builder's as-of algorithm.

No network, model fits, scoring, or writes to project datasets. Synthetic source
loaders bypass frozen-export size checks; build(), asof(), calendar validation,
and every join expression are production code, not a test-only reimplementation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('feature_builder_b2', ROOT / 'scripts/build_external_features.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = records(ROOT / 'data/processed/model_dataset_usdrubf_all_features_v2.csv')
        cls.catalog = records(ROOT / 'docs/feature_availability_catalog.csv')
        cls.by_name = {r['feature_name']: r for r in cls.catalog}

    def test_catalog_covers_exactly_all_columns_without_duplicates(self):
        self.assertEqual(len(self.panel[0]), 30)
        self.assertEqual(len(self.catalog), len(self.by_name))
        self.assertEqual(list(self.by_name), list(self.panel[0]))
        required = {'feature_name', 'economic_group', 'economic_interpretation',
                    'source_name', 'source_url', 'source_file', 'unit',
                    'observation_date_field', 'effective_date_field', 'published_at',
                    'available_at', 'availability_time_zone', 'current_join_rule',
                    'assumed_lag', 'availability_status', 'data_revision_risk',
                    'model_usage_proposal', 'evidence_reference', 'open_question'}
        for r in self.catalog:
            self.assertTrue(required <= r.keys())
            self.assertNotIn(None, r)
            self.assertTrue(all(r[k] for k in required))
            self.assertNotIn('???', str(r))
            self.assertIn(r['availability_status'], {'verified', 'assumed', 'unknown'})
            self.assertIn(r['column_role'], {'economic', 'metadata', 'constant_raw'})

    def test_metadata_is_not_publication_proof_or_economic_input(self):
        metadata = [r for r in self.catalog if r['column_role'] == 'metadata']
        self.assertEqual(len(metadata), 16)
        for r in metadata:
            self.assertEqual(r['usage_group'], 'metadata')
            self.assertIn('metadata_only', r['model_usage_proposal'])
            self.assertNotEqual(r['availability_status'], 'verified')
        # Exact publication times must not be synthesized from dates in the CSV.
        for r in self.catalog:
            self.assertFalse(r['published_at'].startswith(('2022-', '2023-', '2024-', '2025-', '2026-')))
        self.assertIn('unknown', self.by_name['end']['available_at'])

    def test_current_funding_excluded_even_when_populated(self):
        r = self.by_name['swaprate_current']
        self.assertEqual(r['usage_group'], 'C')
        self.assertEqual(r['availability_status'], 'unknown')
        self.assertIn('excluded', r['model_usage_proposal'])
        self.assertTrue(any(row['swaprate_current'] for row in self.panel))
        flag = self.by_name['swaprate_available_assumed']
        self.assertEqual(flag['column_role'], 'metadata')
        self.assertIn('presence only', flag['economic_interpretation'])

    def test_spread_units_and_inherited_availability(self):
        r = self.by_name['spread_close_minus_cbr']
        self.assertEqual(r['depends_on'].split('|'), ['close', 'cbr_usdrub_effective'])
        self.assertIn('max', r['model_usage_proposal'])
        for dep in r['depends_on'].split('|'):
            self.assertEqual(r['unit'], self.by_name[dep]['unit'])
            self.assertEqual(r['availability_status'], self.by_name[dep]['availability_status'])
        for row in self.panel:
            self.assertEqual(Decimal(row['spread_close_minus_cbr']),
                             Decimal(row['close']) - Decimal(row['cbr_usdrub_effective']))

    def test_actual_dates_and_lags_in_frozen_panel(self):
        for i, row in enumerate(self.panel):
            d = date.fromisoformat(row['date'])
            for ref, age, minimum in [
                ('cbr_effective_date', 'cbr_rate_age_calendar_days', 0),
                ('ruonia_reference_date', 'ruonia_age_calendar_days', 1),
                ('brent_eia_reference_date', 'brent_eia_age_calendar_days', 10),
                ('usdcnh_investing_reference_date', 'usdcnh_investing_age_calendar_days', 1),
            ]:
                delta = (d - date.fromisoformat(row[ref])).days
                self.assertGreaterEqual(delta, minimum)
                self.assertEqual(int(row[age]), delta)
            self.assertLessEqual(date.fromisoformat(row['key_rate_reference_date']), d)
            self.assertLessEqual(datetime.fromisoformat(row['ruonia_source_dateupdate']).date(), d)
            self.assertEqual(row['swaprate_source_trade_date'], self.panel[i-1]['date'] if i else '')
            self.assertEqual(row['swaprate_prev_session'], self.panel[i-1]['swaprate_current'] if i else '')
            self.assertEqual(row['swaprate_available_assumed'], str(bool(row['swaprate_prev_session'])).lower())

    def test_production_calendar_uses_existing_schema(self):
        days = [date.fromisoformat(row['date']) for row in self.panel]
        builder.check_calendar(ROOT / 'data/processed/trading_calendar.csv', days)

    def test_audit_did_not_change_input_bytes(self):
        fingerprints = json.loads((ROOT / 'results/feature_pit_audit/input_fingerprints.json').read_text())
        hashes = fingerprints['lf_normalized_sha256']
        # Git checkout may translate LF/CRLF; preserve content across platforms.
        def content_digest(path):
            return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for path, expected in hashes.items():
            if path not in {'docs/feature_availability_catalog.csv', 'scripts/build_external_features.py'}:
                self.assertEqual(content_digest(ROOT / path), expected, path)
        self.assertEqual(content_digest(ROOT / 'results/feature_pit_audit/catalog_b1_snapshot.csv'),
                         hashes['docs/feature_availability_catalog.csv'])


def synthetic_sources():
    days = [date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5)]
    candles = [{'date': str(d), 'begin': f'{d} 00:00:00', 'end': f'{d} 23:59:59',
                'open': '100', 'close': str(100+i), 'high': '103', 'low': '99',
                'value': '0', 'volume': '10'} for i, d in enumerate(days)]
    return {
        'days': days, 'candles': candles,
        'cbr': {'2024-02-29': '90.0000', '2024-03-02': '91.0000', '2024-03-06': '92.0000'},
        'funding': {'2024-03-01': '0.1', '2024-03-04': '', '2024-03-05': '-0.2', '2024-03-06': '0.3'},
        'key': [{'date': '2024-02-28', 'key_rate_pct': '16', 'source_page': '1'},
                {'date': '2024-03-04', 'key_rate_pct': '17', 'source_page': '2'},
                {'date': '2024-03-06', 'key_rate_pct': '18', 'source_page': '2'}],
        'ruonia': [{'reference_date': '2024-02-28', 'ruonia_pct': '15', 'source_dateupdate_timestamp': '2024-02-29 14:00:00'},
                   {'reference_date': '2024-02-29', 'ruonia_pct': '16', 'source_dateupdate_timestamp': '2024-03-02 14:00:00'},
                   {'reference_date': '2024-03-01', 'ruonia_pct': '17', 'source_dateupdate_timestamp': '2024-03-04 23:59:00'},
                   {'reference_date': '2024-03-04', 'ruonia_pct': '18', 'source_dateupdate_timestamp': '2024-03-05 14:00:00'},
                   {'reference_date': '2024-03-05', 'ruonia_pct': '19', 'source_dateupdate_timestamp': '2024-03-06 14:00:00'}],
        'oil': [{'observation_date': str(date(2024, 2, 19)+timedelta(days=i)),
                 'brent_eia_usd_bbl': str(80+i)} for i in range(18)],
        'cn': [{'reference_date': str(date(2024, 2, 29)+timedelta(days=i)),
                'price': str(7+Decimal(i)/100)} for i in range(7)],
    }


def build_fixture(sources):
    def dated(key, column):
        rows = sources[key]
        return [date.fromisoformat(r[column]) for r in rows], rows
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'calendar.csv'
        path.write_text('date,is_trading_day_expected\n' +
                        ''.join(f'{d},1\n' for d in sources['days']), encoding='utf-8')
        # All join logic remains in the real build() function. No builder changes.
        with patch.object(builder, 'load_candles', return_value=(sources['days'], sources['candles'])), \
             patch.object(builder, 'load_cbr', return_value=sources['cbr']), \
             patch.object(builder, 'load_funding', return_value=sources['funding']), \
             patch.object(builder, 'load_key_rates', return_value=dated('key', 'date')), \
             patch.object(builder, 'load_ruonia', return_value=dated('ruonia', 'reference_date')), \
             patch.object(builder, 'load_brent', return_value=dated('oil', 'observation_date')), \
             patch.object(builder, 'load_investing', return_value=dated('cn', 'reference_date')):
            return builder.build(Path(folder), path)


class AsOfAlgorithmTests(unittest.TestCase):
    def test_real_asof_boundaries_empty_and_leap_day(self):
        days = [date(2024, 2, 29), date(2024, 3, 4)]
        values = ['leap', 'monday']
        self.assertIsNone(builder.asof(days, values, date(2024, 2, 28)))
        self.assertEqual(builder.asof(days, values, days[0]), 'leap')
        self.assertEqual(builder.asof(days, values, date(2024, 3, 3)), 'leap')
        self.assertEqual(builder.asof(days, values, days[1]), 'monday')

    def test_real_build_join_rules_and_missing_funding(self):
        a, b, c = build_fixture(synthetic_sources())
        self.assertEqual(a['cbr_effective_date'], '2024-02-29')
        self.assertEqual(b['cbr_effective_date'], '2024-03-02')
        self.assertEqual(b['swaprate_source_trade_date'], '2024-03-01')  # Friday -> Monday
        self.assertEqual(b['swaprate_prev_session'], '0.1')
        self.assertEqual(c['swaprate_prev_session'], '')
        self.assertEqual(c['swaprate_available_assumed'], 'false')
        self.assertEqual(a['ruonia_reference_date'], '2024-02-28')  # later DateUpdate excluded
        self.assertEqual(b['ruonia_reference_date'], '2024-03-01')
        # Documents the existing date-only limitation; does NOT certify early-day availability.
        self.assertEqual(b['ruonia_source_dateupdate'], '2024-03-04 23:59:00')
        self.assertEqual(a['brent_eia_reference_date'], '2024-02-20')  # exactly ten calendar days
        self.assertEqual(a['usdcnh_investing_reference_date'], '2024-02-29')
        self.assertEqual(b['key_rate_pct_effective'], '17')

    def test_future_source_value_mutations_leave_earlier_outputs_unchanged(self):
        source = synthetic_sources()
        baseline = build_fixture(source)
        for origin in source['days']:
            altered = copy.deepcopy(source)
            for d in altered['cbr']:
                if date.fromisoformat(d) > origin:
                    altered['cbr'][d] = '9999.0000'
            for d in altered['funding']:
                if date.fromisoformat(d) > origin:
                    altered['funding'][d] = '9999'
            for group, ref, cutoff, value in [
                ('key', 'date', origin, 'key_rate_pct'),
                ('oil', 'observation_date', origin-timedelta(days=10), 'brent_eia_usd_bbl'),
                ('cn', 'reference_date', origin-timedelta(days=1), 'price'),
            ]:
                for r in altered[group]:
                    if date.fromisoformat(r[ref]) > cutoff:
                        r[value] = '9999'
            for r in altered['ruonia']:
                if (date.fromisoformat(r['reference_date']) >= origin or
                        datetime.fromisoformat(r['source_dateupdate_timestamp']).date() > origin):
                    r['ruonia_pct'] = '9999'
            for r in altered['candles']:
                if date.fromisoformat(r['date']) > origin:
                    r['close'] = '9999'
            result = build_fixture(altered)
            expected = [r for r in baseline if date.fromisoformat(r['date']) <= origin]
            actual = [r for r in result if date.fromisoformat(r['date']) <= origin]
            self.assertEqual(expected, actual)

    def test_appending_future_source_records_leaves_prefix_unchanged(self):
        source = synthetic_sources()
        baseline = build_fixture(source)
        extra = copy.deepcopy(source)
        extra['cbr']['2024-03-20'] = '9999.0000'
        extra['funding']['2024-03-20'] = '9999'
        for group, ref, field in [('key', 'date', 'key_rate_pct'),
                                   ('oil', 'observation_date', 'brent_eia_usd_bbl'),
                                   ('cn', 'reference_date', 'price'),
                                   ('ruonia', 'reference_date', 'ruonia_pct')]:
            r = copy.deepcopy(extra[group][-1])
            r[ref], r[field] = '2024-03-20', '9999'
            if group == 'ruonia':
                r['source_dateupdate_timestamp'] = '2024-03-21 14:00:00'
            extra[group].append(r)
        self.assertEqual(build_fixture(extra), baseline)


if __name__ == '__main__':
    unittest.main()
