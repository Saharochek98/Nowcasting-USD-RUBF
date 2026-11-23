"""Offline B4 funding audit. Never fills, sorts, drops or models observations.

Run from any directory; --compare compares the five generated artifacts without
writing. Research evidence is recorded separately from empirical panel facts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_model_features import SOURCE_COLUMNS, read_csv, ordered_dates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/model_dataset_usdrubf_all_features_v2.csv'
CALENDAR = ROOT / 'data/processed/trading_calendar.csv'
SPLITS = ROOT / 'configs/benchmark_config.yaml'
OUTPUT = ROOT / 'results/funding_missingness_audit'
FIELDS = ('swaprate_current', 'swaprate_prev_session')
REQUIRED = ('date', *FIELDS, 'swaprate_source_trade_date', 'swaprate_available_assumed')
SOURCE_SHA256 = 'da5b582e622ce5d3efb6fb619c0b9258dc15fc7d18b6d76860ce5fb021107103'
REGIMES = (
    ('pre_stabilizing_component', '0001-01-01', '2022-12-04'),
    ('stabilizing_component_spot', '2022-12-05', '2024-06-23'),
    ('cbr_vwap', '2024-06-24', '2024-12-19'),
    ('cbr_holiday_clarification', '2024-12-20', '2026-03-22'),
    ('unified_session', '2026-03-23', '9999-12-31'),
)
EVIDENCE = [
    {'url': 'https://www.moex.com/n47096', 'published_date': '2022-04-20',
     'effective_date': '2022-04-26', 'fact': 'Trading launch; not first SWAPRATE release.'},
    {'url': 'https://www.moex.com/n52418', 'published_date': '2022-10-20',
     'effective_date': '2022-12-05',
     'fact': 'Evening VM replaces swap points with a stabilizing price-gap component. Economic regime change verified; exact historical database publication start unknown.'},
    {'url': 'https://www.moex.com/a8141', 'published_date': None,
     'effective_date': '2024-06-24',
     'fact': 'Current undated page distinguishes old spot-gap and newer CBR/VWAP formula; zero is possible in the tolerance band, and pre-release displayed zero is also documented. Individual historical zeros not certified.'},
    {'url': 'https://www.moex.com/n75925', 'published_date': '2024-12-18',
     'effective_date': '2024-12-20', 'fact': 'Holiday funding uses last published CBR rate.'},
    {'url': 'https://www.moex.com/n98363?nt=101', 'published_date': '2026-03-13',
     'effective_date': '2026-03-23', 'fact': 'Unified session changes trade-date attribution and clearing; not proof of candle/history equivalence.'},
]


def funding(value):
    """Only the exact empty CSV field is missing. Observed zero stays Decimal(0)."""
    if value == '':
        return None
    if not isinstance(value, str) or value.strip() != value:
        raise ValueError(f'Noncanonical funding field: {value!r}')
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f'Invalid funding: {value!r}') from exc
    if not result.is_finite():
        raise ValueError('NaN/Inf text is not an empty CSV field')
    return result


def split_bounds(path=SPLITS):
    """Read the three explicit quoted date pairs in the existing config.

    Restricted parser for this repository's splits section, not a general YAML
    parser. Fail on unsupported structure instead of guessing new boundaries.
    """
    text = Path(path).read_text(encoding='utf-8-sig')
    section = re.search(r'^splits:\s*\n((?:[ \t]+[^\n]*\n?)+)', text, re.M)
    if not section:
        raise ValueError('Missing splits section')
    entries = re.findall(r"^  (train|validation|final_test): \['(\d{4}-\d{2}-\d{2})', '(\d{4}-\d{2}-\d{2})'\]\s*$",
                         section.group(1), re.M)
    if len(entries) != 3 or [x[0] for x in entries] != ['train', 'validation', 'final_test']:
        raise ValueError('Unsupported split configuration')
    result = {name: [start, end] for name, start, end in entries}
    previous_end = None
    for start, end in result.values():
        if date.fromisoformat(start) > date.fromisoformat(end) or (previous_end and start <= previous_end):
            raise ValueError('Invalid/overlapping splits')
        previous_end = end
    return result


def split_for(day, bounds):
    matches = [name for name, (start, end) in bounds.items() if start <= day <= end]
    if len(matches) != 1:
        raise ValueError(f'Date not in exactly one split: {day}')
    return matches[0]


def validate_rows(rows, calendar):
    if not rows or any(not set(REQUIRED) <= r.keys() for r in rows):
        raise ValueError('Missing required funding schema/rows')
    days = ordered_dates([r['date'] for r in rows])
    ordered_dates([r['date'] for r in calendar])
    if any(r['is_trading_day_expected'] not in {'0', '1'} for r in calendar):
        raise ValueError('Invalid calendar flag')
    expected = [r['date'] for r in calendar if r['is_trading_day_expected'] == '1'
                and str(days[0]) <= r['date'] <= str(days[-1])]
    if [r['date'] for r in rows] != expected:
        raise ValueError('Rows disagree with trading calendar; no sorting/reindexing allowed')
    for row in rows:
        for col in FIELDS:
            funding(row[col])


def classify_rows(rows):
    """Causal row diagnostics: row t uses only t and its calendar predecessor."""
    diagnostics = []
    for i, row in enumerate(rows):
        prev = funding(row[FIELDS[1]])
        expected = funding(rows[i-1][FIELDS[0]]) if i else None
        expected_date = rows[i-1]['date'] if i else ''
        reason = ''
        if prev is None:
            reason = ('first_row_no_predecessor' if i == 0 else
                      'previous_source_missing' if expected is None else 'unexpected_missing')
        diagnostics.append({
            'date': row['date'], 'previous_missing_reason': reason,
            'lag_matches': prev == expected,
            'source_date_matches': row['swaprate_source_trade_date'] == expected_date,
            'presence_flag_matches': row['swaprate_available_assumed'] == str(prev is not None).lower(),
            'representation_differs': i > 0 and prev == expected and row[FIELDS[1]] != rows[i-1][FIELDS[0]],
        })
    return diagnostics


def statistics(rows, column):
    values = [(r['date'], funding(r[column])) for r in rows]
    observed = [(d, v) for d, v in values if v is not None]
    zeros = [d for d, v in observed if v == 0]
    missing = [d for d, v in values if v is None]
    return {'column': column, 'n_rows': len(rows), 'observed': len(observed),
            'missing': len(missing), 'observed_zero': len(zeros),
            'observed_nonzero': len(observed) - len(zeros),
            'missing_fraction': f'{len(missing)/len(rows):.8f}' if rows else '',
            'zero_fraction_observed': f'{len(zeros)/len(observed):.8f}' if observed else '',
            'first_observed_date': observed[0][0] if observed else '',
            'last_observed_date': observed[-1][0] if observed else '',
            'first_zero_date': zeros[0] if zeros else '', 'last_zero_date': zeros[-1] if zeros else '',
            'first_missing_date': missing[0] if missing else '',
            'last_missing_date': missing[-1] if missing else ''}


def periods(rows, width):
    groups = {}
    for row in rows:
        groups.setdefault(row['date'][:width], []).append(row)
    return [{'period': key, **statistics(group, col)}
            for key, group in groups.items() for col in FIELDS]


def missing_segments(rows, diagnostics):
    result = []
    for column in FIELDS:
        start = None
        for i in range(len(rows) + 1):
            missing = i < len(rows) and funding(rows[i][column]) is None
            if missing and start is None:
                start = i
            if not missing and start is not None:
                reasons = Counter(d['previous_missing_reason'] for d in diagnostics[start:i]) if column == FIELDS[1] else {}
                result.append({'column': column, 'start_date': rows[start]['date'],
                               'end_date': rows[i-1]['date'], 'trading_steps': i-start,
                               'start_row_1based': start+1, 'end_row_1based': i,
                               'touches_start': str(start == 0).lower(),
                               'touches_end': str(i == len(rows)).lower(),
                               'first_row_no_predecessor': reasons.get('first_row_no_predecessor', 0),
                               'previous_source_missing': reasons.get('previous_source_missing', 0),
                               'unexpected_missing': reasons.get('unexpected_missing', 0)})
                start = None
    return result


def coverage(rows, col):
    missing = [i for i, row in enumerate(rows) if funding(row[col]) is None]
    observed = [i for i, row in enumerate(rows) if funding(row[col]) is not None]
    first = observed[0] if observed else None
    suffix = missing[-1]+1 if missing else 0
    return {**statistics(rows, col),
            'missing_after_first_observed': sum(i > first for i in missing) if first is not None else None,
            'continuous_covered_suffix_start': rows[suffix]['date'] if suffix < len(rows) else None,
            'continuous_covered_suffix_rows': len(rows)-suffix}


def zero_summary(rows):
    result = []
    for col in FIELDS:
        for name, start, end in REGIMES:
            # Group by the source value's trade label, not publication time.
            group = [r for r in rows if start <= (r['date'] if col == FIELDS[0]
                                                else r['swaprate_source_trade_date']) <= end]
            result.append({'regime': name, 'regime_start': start, 'regime_end': end,
                           'grouping_basis': 'source_trade_date', **statistics(group, col),
                           'zero_strings': json.dumps(dict(Counter(r[col] for r in group
                                                                  if funding(r[col]) == 0)), sort_keys=True)})
        if col == FIELDS[1]:
            group = [r for r in rows if r['swaprate_source_trade_date'] == '']
            result.append({'regime': 'no_predecessor', 'regime_start': '', 'regime_end': '',
                           'grouping_basis': 'source_trade_date', **statistics(group, col), 'zero_strings': '{}'})
    return result


def analyze(rows, calendar, bounds):
    validate_rows(rows, calendar)
    diagnostics = classify_rows(rows)
    checks = {name: sum(not d[name] for d in diagnostics) for name in
              ('lag_matches', 'source_date_matches', 'presence_flag_matches')}
    checks['representation_differences'] = sum(bool(d['representation_differs']) for d in diagnostics)
    summary = {
        'schema_version': 'funding_missingness_audit_v1', 'pit_status': 'conditional EOD / pseudo-PIT',
        'n_rows': len(rows), 'first_date': rows[0]['date'], 'last_date': rows[-1]['date'],
        'checks': checks, 'all_trading_session_checks_passed': all(checks[k] == 0 for k in checks if k != 'representation_differences'),
        'funding': {col: coverage(rows, col) for col in FIELDS},
        'previous_missing_reasons': dict(Counter(d['previous_missing_reason'] for d in diagnostics if d['previous_missing_reason'])),
        'split_bounds': bounds,
        'split_coverage': [{'split': name, **statistics([r for r in rows if split_for(r['date'], bounds) == name], col)}
                           for name in bounds for col in FIELDS],
        'zero_validity': 'Observed numeric zero in frozen CSV, not independently certified final funding; never converted to missing.',
        'raw_external_export': {'filename': 'contractresults-USDRUBF-26042022-20112026.csv',
                                'inspected_in_b4': False,
                                'limitation': 'Absent from repository; original export content and historical first publication not verified.'},
        'official_evidence_checked_on': '2026-11-23', 'official_evidence': EVIDENCE,
        'publication_history_status': 'unknown', 'models_fitted': False, 'forecast_metrics_computed': False,
        'archive_checks': [
            {'url': 'https://www.moex.com/ru/derivatives/contractresults.aspx?code=USDRUBF',
             'result': 'Page confirms first circulation date 2022-04-26; historical export values not retrieved.'},
            {'url': 'https://iss.moex.com/iss/history/engines/futures/markets/forts/boards/RFUD/securities/USDRUBF.json?from=2022-11-28&till=2022-12-07&iss.only=history&history.columns=TRADEDATE,SECID,BOARDID,SWAPRATE',
             'result': 'Not accessible through web tool on 2026-11-23; no historical values verified.'},
            {'url': 'https://iss.moex.com/iss/history/engines/futures/markets/forts/columns.json',
             'result': 'Not accessible through web tool; historical field migration unknown.'},
        ],
        'segments_are_trading_steps': True,
        'limitations': ['Coverage is not historical available_at verification.',
                        'Descriptive split counts use origin labels, not horizon-specific scored target pairs.',
                        'Observed source-date regime grouping does not identify publication times.',
                        'No early SWAPRATE archive/schema migration history retrieved; do not assert economic absence before 2022-12-05.'],
    }
    return summary, periods(rows, 4), periods(rows, 7), missing_segments(rows, diagnostics), zero_summary(rows)


def csv_content(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    # BOM supports spreadsheet readers; .gitattributes pins canonical LF bytes.
    return buffer.getvalue().encode('utf-8-sig')


def build_artifacts():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Frozen v2 SHA-256 mismatch')
    rows = read_csv(SOURCE, SOURCE_COLUMNS)
    if len(rows) != 1160:
        raise ValueError('Expected exactly 1160 source observations')
    calendar = read_csv(CALENDAR)
    expected = [r['date'] for r in calendar if r['is_trading_day_expected'] == '1']
    if [r['date'] for r in rows] != expected:
        raise ValueError('Full panel/calendar mismatch')
    summary, yearly, monthly, segments, zeros = analyze(rows, calendar, split_bounds())
    if not summary['all_trading_session_checks_passed']:
        raise ValueError('Funding lag/source-date/presence-flag checks failed')
    # B3 linkage is audited without altering/rebuilding B3.
    store = read_csv(ROOT / 'data/processed/features_daily_v1.csv')
    metadata = read_csv(ROOT / 'data/processed/features_daily_v1_metadata.csv')
    if ([r['date'] for r in store] != [r['date'] for r in rows]
            or [r['date'] for r in metadata] != [r['date'] for r in rows]):
        raise ValueError('B3 linkage mismatch')
    for row, model, meta in zip(rows, store, metadata):
        if (row['swaprate_prev_session'] != model['swaprate_prev_session']
                or any(row[k] != meta[k] for k in ('swaprate_source_trade_date', 'swaprate_available_assumed'))):
            raise ValueError('B3 funding value/metadata differs from source')
    protected = [SOURCE, CALENDAR, SPLITS, ROOT / 'data/processed/features_daily_v1.csv',
                 ROOT / 'data/processed/features_daily_v1_metadata.csv',
                 ROOT / 'data/raw/USDRUBF_daily_2026-11-18.csv',
                 ROOT / 'docs/feature_availability_catalog.csv',
                 ROOT / 'configs/feature_scenarios_v1.json', ROOT / 'docs/backtest_protocol.md']
    summary['input_sha256_lf_normalized'] = {p.relative_to(ROOT).as_posix():
                                            hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                            for p in protected}
    summary['source_sha256_bytes'] = SOURCE_SHA256
    summary['b3_funding_linkage_passed'] = True
    artifacts = {name: csv_content(content) for name, content in (
        ('yearly_summary.csv', yearly), ('monthly_summary.csv', monthly),
        ('missing_segments.csv', segments), ('zero_funding_summary.csv', zeros))}
    summary['artifact_sha256_bytes'] = {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}
    artifacts['audit_summary.json'] = (json.dumps(summary, indent=2, sort_keys=True) + '\n').encode('utf-8')
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--compare', action='store_true', help='Read-only byte comparison')
    args = parser.parse_args(argv)
    artifacts = build_artifacts()
    for name in artifacts:
        path = (args.output_dir / name).resolve()
        if not args.compare and path.is_relative_to(ROOT) and path.exists() and path.parent != OUTPUT:
            parser.error(f'Refusing to overwrite existing repository artifact: {path}')
    for name, content in artifacts.items():
        path = args.output_dir / name
        if args.compare:
            if not path.exists() or path.read_bytes() != content:
                raise ValueError(f'Artifact differs or missing: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(('MATCH' if args.compare else 'WROTE'), name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
