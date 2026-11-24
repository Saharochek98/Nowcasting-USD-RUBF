"""Synthetic and frozen-panel B4 checks, without fitting or network access."""
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_funding_missingness as audit


def fixture():
    dates = ['2022-12-02', '2022-12-05', '2022-12-06', '2022-12-07',
             '2022-12-08', '2022-12-09', '2022-12-12']
    current = ['', '', '0.1', '0', '0.2', '-0.1', '0.00000']
    rows = []
    for i, day in enumerate(dates):
        previous = current[i-1] if i else ''
        rows.append({'date': day, 'swaprate_current': current[i],
                     'swaprate_prev_session': previous,
                     'swaprate_source_trade_date': dates[i-1] if i else '',
                     'swaprate_available_assumed': str(previous != '').lower()})
    calendar = [{'date': d, 'is_trading_day_expected': '1'} for d in dates]
    return rows, calendar


class FundingAuditTests(unittest.TestCase):
    def test_contiguous_missing_uses_trading_steps_across_weekend(self):
        rows, calendar = fixture()
        summary, _, _, segments, _ = audit.analyze(rows, calendar, audit.split_bounds())
        self.assertTrue(summary['all_trading_session_checks_passed'])
        self.assertEqual([(r['column'], r['trading_steps']) for r in segments],
                         [('swaprate_current', 2), ('swaprate_prev_session', 3)])
        self.assertEqual(segments[0]['start_date'], '2022-12-02')
        self.assertEqual(segments[0]['end_date'], '2022-12-05')

    def test_numeric_zero_and_blank_are_distinct(self):
        self.assertIsNone(audit.funding(''))
        for observed in ['0', '0.00000', '-0.000', '0e-5']:
            self.assertEqual(audit.funding(observed), 0)
            self.assertIsNotNone(audit.funding(observed))
        rows, _ = fixture()
        stats = audit.statistics(rows, 'swaprate_current')
        self.assertEqual((stats['missing'], stats['observed_zero'], stats['observed_nonzero']), (2, 2, 3))
        for invalid in ['NaN', 'Infinity', 'null', ' ', '0 ', 'bad']:
            with self.assertRaises(ValueError):
                audit.funding(invalid)

    def test_first_row_is_distinguished_from_missing_predecessor_value(self):
        rows, calendar = fixture()
        summary = audit.analyze(rows, calendar, audit.split_bounds())[0]
        self.assertEqual(summary['previous_missing_reasons'],
                         {'first_row_no_predecessor': 1, 'previous_source_missing': 2})
        self.assertEqual(rows[1]['swaprate_source_trade_date'], '2022-12-02')
        self.assertEqual(rows[1]['swaprate_available_assumed'], 'false')

    def test_artificial_gap_after_coverage_detected(self):
        rows, calendar = fixture()
        rows[4]['swaprate_current'] = ''
        rows[5]['swaprate_prev_session'] = ''
        rows[5]['swaprate_available_assumed'] = 'false'
        summary, _, _, segments, _ = audit.analyze(rows, calendar, audit.split_bounds())
        self.assertTrue(summary['all_trading_session_checks_passed'])
        for col in audit.FIELDS:
            self.assertEqual(summary['funding'][col]['missing_after_first_observed'], 1)
            self.assertEqual(sum(s['column'] == col for s in segments), 2)
        self.assertEqual(summary['funding']['swaprate_prev_session']['continuous_covered_suffix_start'], '2022-12-12')

    def test_lag_violation_detected(self):
        rows, calendar = fixture()
        rows[4]['swaprate_prev_session'] = '99'
        summary = audit.analyze(rows, calendar, audit.split_bounds())[0]
        self.assertFalse(summary['all_trading_session_checks_passed'])
        self.assertEqual(summary['checks']['lag_matches'], 1)

    def test_source_date_and_presence_flag_violation_detected(self):
        rows, calendar = fixture()
        rows[1]['swaprate_source_trade_date'] = '2022-12-04'  # Sunday is not predecessor
        rows[4]['swaprate_available_assumed'] = 'false'  # previous is observed zero
        checks = audit.analyze(rows, calendar, audit.split_bounds())[0]['checks']
        self.assertEqual(checks['source_date_matches'], 1)
        self.assertEqual(checks['presence_flag_matches'], 1)

    def test_equal_numeric_values_with_different_format_are_not_lag_errors(self):
        rows, calendar = fixture()
        rows[4]['swaprate_prev_session'] = '-0.000'
        checks = audit.analyze(rows, calendar, audit.split_bounds())[0]['checks']
        self.assertEqual(checks['lag_matches'], 0)
        self.assertEqual(checks['representation_differences'], 1)

    def test_no_sorting_deduplication_or_calendar_compression(self):
        rows, calendar = fixture()
        original = copy.deepcopy(rows)
        audit.analyze(rows, calendar, audit.split_bounds())
        self.assertEqual(rows, original)
        for invalid in [rows[::-1], [rows[0], rows[0]] + rows[1:], rows[:2] + rows[3:]]:
            with self.assertRaises(ValueError):
                audit.analyze(invalid, calendar, audit.split_bounds())

    def test_prefix_diagnostics_and_completed_periods_independent_of_future(self):
        rows, calendar = fixture()
        changed = copy.deepcopy(rows)
        changed[5]['swaprate_current'] = '999'
        changed[6]['swaprate_prev_session'] = '999'
        self.assertEqual(audit.classify_rows(rows)[:5], audit.classify_rows(changed)[:5])
        self.assertEqual(audit.analyze(rows[:5], calendar, audit.split_bounds()),
                         audit.analyze(changed[:5], calendar, audit.split_bounds()))
        # Aggregate totals legitimately depend on the selected sample; historical
        # per-row/prefix results must not use unobserved suffix values.
        prefix = audit.periods(rows[:5], 7)
        self.assertEqual(prefix, audit.periods(changed[:5], 7))

    def test_all_missing_and_all_zero_edge_cases(self):
        rows, calendar = fixture()
        for row in rows:
            row['swaprate_current'] = row['swaprate_prev_session'] = ''
            row['swaprate_available_assumed'] = 'false'
        summary = audit.analyze(rows, calendar, audit.split_bounds())[0]
        self.assertIsNone(summary['funding']['swaprate_current']['continuous_covered_suffix_start'])
        self.assertIsNone(summary['funding']['swaprate_current']['missing_after_first_observed'])
        for i, row in enumerate(rows):
            row['swaprate_current'] = '0'
            row['swaprate_prev_session'] = '0' if i else ''
            row['swaprate_available_assumed'] = 'true' if i else 'false'
        summary = audit.analyze(rows, calendar, audit.split_bounds())[0]
        self.assertEqual(summary['funding']['swaprate_current']['observed_zero'], 7)
        self.assertEqual(summary['funding']['swaprate_prev_session']['missing'], 1)
        self.assertTrue(summary['all_trading_session_checks_passed'])

    def test_real_panel_counts_lags_zeros_and_splits(self):
        rows = audit.read_csv(audit.SOURCE, audit.SOURCE_COLUMNS)
        summary, years, months, segments, zeros = audit.analyze(rows, audit.read_csv(audit.CALENDAR), audit.split_bounds())
        current, previous = [summary['funding'][col] for col in audit.FIELDS]
        self.assertEqual(summary['n_rows'], 1160)
        self.assertEqual((current['missing'], previous['missing']), (153, 154))
        self.assertEqual((current['observed_zero'], previous['observed_zero']), (198, 198))
        self.assertEqual(current['first_observed_date'], '2022-12-05')
        self.assertEqual(previous['first_observed_date'], '2022-12-06')
        self.assertEqual(previous['last_missing_date'], '2022-12-05')
        self.assertEqual(summary['previous_missing_reasons'],
                         {'first_row_no_predecessor': 1, 'previous_source_missing': 153})
        self.assertTrue(summary['all_trading_session_checks_passed'])
        self.assertEqual(len(years), 10)
        self.assertEqual(len(months), 112)
        self.assertEqual([s['trading_steps'] for s in segments], [153, 154])
        self.assertEqual([current['missing_after_first_observed'], previous['missing_after_first_observed']], [0, 0])
        splits = {s['split']: s for s in summary['split_coverage'] if s['column'] == 'swaprate_prev_session'}
        self.assertEqual([(s['n_rows'], s['missing']) for s in splits.values()], [(683, 154), (254, 0), (223, 0)])
        self.assertEqual(sum(z['observed_zero'] for z in zeros if z['column'] == 'swaprate_prev_session'), 198)

    def test_reproducibility_compare_and_input_immutability(self):
        protected = list((ROOT / 'data').rglob('*.csv')) + [ROOT / 'docs/feature_availability_catalog.csv',
                    ROOT / 'configs/feature_scenarios_v1.json', ROOT / 'docs/backtest_protocol.md']
        fingerprints = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        first, second = audit.build_artifacts(), audit.build_artifacts()
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(['--output-dir', folder]), 0)
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(folder).iterdir()}
                self.assertEqual(audit.main(['--output-dir', folder, '--compare']), 0)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(folder).iterdir()})
                (Path(folder) / 'audit_summary.json').write_text('{}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit.main(['--output-dir', folder, '--compare'])
        self.assertEqual(fingerprints, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
        for name, content in first.items():
            self.assertTrue(audit.artifact_matches(name, content, (audit.OUTPUT / name).read_bytes()), name)

    def test_only_live_protocol_hash_may_change(self):
        content = audit.build_artifacts()['audit_summary.json']
        changed = json.loads(content)
        changed['input_sha256_lf_normalized']['docs/backtest_protocol.md'] = '0' * 64
        self.assertTrue(audit.artifact_matches('audit_summary.json', content, json.dumps(changed).encode()))
        changed['b3_funding_linkage_passed'] = False
        self.assertFalse(audit.artifact_matches('audit_summary.json', content, json.dumps(changed).encode()))
        changed = json.loads(content)
        changed['input_sha256_lf_normalized']['data/processed/features_daily_v1.csv'] = '0' * 64
        self.assertFalse(audit.artifact_matches('audit_summary.json', content, json.dumps(changed).encode()))


if __name__ == '__main__':
    unittest.main()
