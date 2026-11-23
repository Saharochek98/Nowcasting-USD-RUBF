"""Build the frozen v1 feature store and audit metadata without fitting models.

Run from any directory. Standard library only. --compare validates existing
outputs byte-for-byte without writing them. Scenario loading defaults to core.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/feature_scenarios_v1.json'
PIT_STATUS = 'conditional EOD / pseudo-PIT'
CORE = ('open', 'close', 'high', 'low', 'volume', 'cbr_usdrub_effective',
        'swaprate_prev_session', 'key_rate_pct_effective', 'spread_close_minus_cbr')
SENSITIVITY = ('ruonia_pct_last_reported', 'brent_eia_spot_usd_bbl_lag10d',
               'usdcnh_investing_price_prev_date')
STORE_FEATURES = CORE + SENSITIVITY
SCENARIO_FEATURES = {'core': CORE, 'core_ruonia_brent': CORE + SENSITIVITY[:2],
                     'core_ruonia_brent_usdcnh': STORE_FEATURES}
METADATA_INPUT = (
    'begin', 'end', 'cbr_effective_date', 'cbr_rate_age_calendar_days',
    'swaprate_source_trade_date', 'swaprate_available_assumed',
    'key_rate_reference_date', 'key_rate_pdf_source_page',
    'ruonia_reference_date', 'ruonia_source_dateupdate', 'ruonia_age_calendar_days',
    'brent_eia_reference_date', 'brent_eia_age_calendar_days',
    'usdcnh_investing_reference_date', 'usdcnh_investing_age_calendar_days',
)
SOURCE_COLUMNS = ('date', 'open', 'close', 'high', 'low', 'value', 'volume',
                  'begin', 'end', 'cbr_usdrub_effective', 'cbr_effective_date',
                  'cbr_rate_age_calendar_days', 'swaprate_current',
                  'swaprate_prev_session', 'swaprate_source_trade_date',
                  'swaprate_available_assumed', 'spread_close_minus_cbr',
                  'key_rate_pct_effective', 'key_rate_reference_date',
                  'key_rate_pdf_source_page', 'ruonia_pct_last_reported',
                  'ruonia_reference_date', 'ruonia_source_dateupdate',
                  'ruonia_age_calendar_days', 'brent_eia_spot_usd_bbl_lag10d',
                  'brent_eia_reference_date', 'brent_eia_age_calendar_days',
                  'usdcnh_investing_price_prev_date', 'usdcnh_investing_reference_date',
                  'usdcnh_investing_age_calendar_days')
METADATA_COLUMNS = ('date',) + METADATA_INPUT + (
    'source_dataset', 'source_sha256', 'source_row_number', 'schema_version',
    'lag_policy_version', 'pit_status', 'price_type', 'missing_features')


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (config['schema_version'] != 'features_daily_v1'
            or config['lag_policy_version'] != 'b2-approved-b3-v1'
            or config['default_scenario'] != 'core'
            or config['pit_status'] != PIT_STATUS
            or config['train_only_preprocessing_required'] is not True
            or config['future_targets_in_features'] is not False
            or config['transformations'] != []
            or config['date_column'] != 'date'
            or config['observed_target_level'] != 'close'):
        raise ValueError('Unsupported schema/policy or unsafe preprocessing configuration')
    if set(config['scenarios']) != set(SCENARIO_FEATURES):
        raise ValueError('Unexpected scenarios')
    excluded = set(config['excluded_factors'])
    if not {'swaprate_current', 'value', *METADATA_INPUT} <= excluded:
        raise ValueError('Required exclusions missing')
    for name, expected in SCENARIO_FEATURES.items():
        scenario = config['scenarios'][name]
        if tuple(scenario['features']) != expected or excluded.intersection(expected):
            raise ValueError(f'Unauthorized predictors in {name}; explicit allowlist required')
        if (scenario['pit_status'] != PIT_STATUS
                or scenario['lag_policy_version'] != config['lag_policy_version']
                or scenario['sensitivity_only'] is not (name != 'core')
                or (name != 'core' and not scenario['additional_pit_assumptions'])):
            raise ValueError(f'Incorrect PIT/sensitivity policy: {name}')
    return config


def read_csv(path, columns=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or (columns is not None and tuple(reader.fieldnames) != tuple(columns))):
            raise ValueError(f'Unexpected or duplicate CSV columns: {path}')
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f'Empty or malformed CSV: {path}')
    return rows


def ordered_dates(values):
    parsed = [date.fromisoformat(value) for value in values]
    if any(str(d) != v for d, v in zip(parsed, values)) or parsed != sorted(set(parsed)):
        raise ValueError('Dates must be unique, strictly increasing ISO dates')
    return parsed


def validate_source(rows, calendar, raw, expected_rows):
    days = ordered_dates([r['date'] for r in rows])
    if len(rows) != expected_rows:
        raise ValueError('Unexpected source row count')
    ordered_dates([r['date'] for r in calendar])
    if any(r['is_trading_day_expected'] not in {'0', '1'} for r in calendar):
        raise ValueError('Invalid calendar flag')
    if [r['date'] for r in rows] != [r['date'] for r in calendar if r['is_trading_day_expected'] == '1']:
        raise ValueError('Source dates disagree with USDRUBF calendar')
    raw_columns = ('open', 'close', 'high', 'low', 'value', 'volume', 'begin', 'end')
    if len(raw) != len(rows):
        raise ValueError('Raw candle count differs')
    for i, (row, original, day) in enumerate(zip(rows, raw, days)):
        if any(row[k] != original[k] for k in raw_columns):
            raise ValueError(f'Raw candle changed at {day}')
        if any(datetime.fromisoformat(row[k]).date() != day for k in ('begin', 'end')):
            raise ValueError(f'Candle label mismatch at {day}')
        for col in STORE_FEATURES:
            if row[col] == '' and col == 'swaprate_prev_session':
                continue
            if row[col] == '' or not Decimal(row[col]).is_finite():
                raise ValueError(f'Invalid economic value: {day} {col}')
        previous_date = str(days[i-1]) if i else ''
        previous_value = rows[i-1]['swaprate_current'] if i else ''
        if (row['swaprate_source_trade_date'] != previous_date
                or row['swaprate_prev_session'] != previous_value
                or row['swaprate_available_assumed'] != str(bool(previous_value)).lower()):
            raise ValueError(f'Previous-session funding/flag mismatch at {day}')
        if Decimal(row['spread_close_minus_cbr']) != Decimal(row['close']) - Decimal(row['cbr_usdrub_effective']):
            raise ValueError(f'Spread mismatch at {day}')
        for ref, age, minimum in (
            ('cbr_effective_date', 'cbr_rate_age_calendar_days', 0),
            ('ruonia_reference_date', 'ruonia_age_calendar_days', 1),
            ('brent_eia_reference_date', 'brent_eia_age_calendar_days', 10),
            ('usdcnh_investing_reference_date', 'usdcnh_investing_age_calendar_days', 1),
        ):
            delta = (day - date.fromisoformat(row[ref])).days
            if delta < minimum or int(row[age]) != delta:
                raise ValueError(f'Invalid source date/age at {day}: {ref}')
        if (date.fromisoformat(row['key_rate_reference_date']) > day
                or datetime.fromisoformat(row['ruonia_source_dateupdate']).date() > day):
            raise ValueError(f'Future reference/update date at {day}')


def validate_catalog(catalog):
    names = [r['feature_name'] for r in catalog]
    if len(names) != len(set(names)) or set(names) != set(SOURCE_COLUMNS):
        raise ValueError('Availability catalog does not match source schema')
    by_name = {r['feature_name']: r for r in catalog}
    for name in STORE_FEATURES:
        row = by_name[name]
        if row['column_role'] != 'economic' or row['usage_group'] != ('A' if name in CORE else 'B'):
            raise ValueError(f'Availability policy disagrees with feature: {name}')


def project_tables(rows, config, source_sha256):
    """Pure row projection. No shifts, targets, imputation, fitting or rescaling."""
    features, metadata = [], []
    for number, row in enumerate(rows, 1):
        features.append({key: row[key] for key in ('date',) + STORE_FEATURES})
        metadata.append({
            'date': row['date'], **{key: row[key] for key in METADATA_INPUT},
            'source_dataset': config['source_data'], 'source_sha256': source_sha256,
            'source_row_number': str(number), 'schema_version': config['schema_version'],
            'lag_policy_version': config['lag_policy_version'], 'pit_status': PIT_STATUS,
            'price_type': 'CANDLE_CLOSE',
            'missing_features': '|'.join(key for key in STORE_FEATURES if row[key] == ''),
        })
    return features, metadata


def csv_bytes(rows, columns):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def coverage_report(features, config, hashes):
    lines = ['# Покрытие features_daily_v1', '',
             'Сгенерировано scripts/build_model_features.py без обучения и оценки моделей.',
             '', 'Режим: **conditional EOD / pseudo-PIT**. Сценарии согласованы A в задании B3.',
             'Один store содержит объединение 12 значений; default core использует ровно 9.',
             '', f"Период: {features[0]['date']} — {features[-1]['date']}; строк: {len(features)}.",
             'Даты и пропуски сохранены; число признаков ниже включает известный close_t, исключает date.', '']
    for name, scenario in config['scenarios'].items():
        cols = scenario['features']
        missing = {key: sum(r[key] == '' for r in features) for key in cols}
        constants = [key for key in cols if len({r[key] for r in features if r[key] != ''}) == 1]
        complete = sum(all(r[key] != '' for key in cols) for r in features)
        lines += [f"## {scenario['name']}", '',
                  f"Строк: {len(features)}; признаков: {len(cols)}; полных строк: {complete}.",
                  f"Минимальная дата: {features[0]['date']}; максимальная: {features[-1]['date']}.",
                  'Постоянные экономические столбцы (без пропусков): ' + (', '.join(constants) or 'нет') + '.',
                  'Сценарий чувствительности: ' + ('да' if scenario['sensitivity_only'] else 'нет') + '.',
                  '', '| Признак | Пропуски | Доля |', '|---|---:|---:|']
        lines += [f'| `{key}` | {count} | {count/len(features):.6%} |' for key, count in missing.items()]
        lines += ['', 'Ограничения: funding не заполнен; строки не удалены. Импутация/scaler/PCA только внутри train/walk-forward.']
        if scenario['additional_pit_assumptions']:
            lines += [''] + ['- ' + assumption for assumption in scenario['additional_pit_assumptions']]
        lines += ['']
    lines += ['## Контрольные SHA-256', '', '| Артефакт | SHA-256 |', '|---|---|']
    lines += [f'| {name} | `{digest}` |' for name, digest in hashes.items()]
    lines += ['', 'Исторические available_at не восстановлены. Проверки дат не доказывают строгий PIT.',
              'Календарь остаётся предварительным; правила публикаций и ревизий наследуются из B1/B2.',
              'Метрики прогнозов и выбор сценария по test не выполнялись.', '']
    return '\n'.join(lines).encode('utf-8')


def load_scenario(features_path=None, *, scenario='core', config_path=DEFAULT_CONFIG):
    """Return date plus explicit predictors as strings; empty funding stays ''."""
    config = load_config(config_path)
    if scenario not in config['scenarios']:
        raise ValueError(f'Unknown scenario: {scenario}')
    path = Path(features_path) if features_path else ROOT / config['features_file']
    rows = read_csv(path, ('date',) + STORE_FEATURES)
    ordered_dates([r['date'] for r in rows])
    cols = ('date',) + tuple(config['scenarios'][scenario]['features'])
    return [{key: row[key] for key in cols} for row in rows]


def build_artifacts(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    source = ROOT / config['source_data']
    source_hash = sha256(source.read_bytes())
    if source_hash != config['source_sha256']:
        raise ValueError('Source SHA-256 differs from frozen v2; do not silently replace snapshot')
    rows = read_csv(source, SOURCE_COLUMNS)
    calendar = read_csv(ROOT / config['calendar'])
    raw = read_csv(ROOT / config['raw_candles'])
    validate_source(rows, calendar, raw, config['expected_rows'])
    validate_catalog(read_csv(ROOT / config['availability_catalog']))
    features, metadata = project_tables(rows, config, source_hash)
    feature_bytes = csv_bytes(features, ('date',) + STORE_FEATURES)
    metadata_bytes = csv_bytes(metadata, METADATA_COLUMNS)
    hashes = {'source v2': source_hash, 'features_daily_v1.csv': sha256(feature_bytes),
              'features_daily_v1_metadata.csv': sha256(metadata_bytes),
              'scenario config (LF normalized)': sha256(Path(config_path).read_bytes().replace(b'\r\n', b'\n'))}
    return feature_bytes, metadata_bytes, coverage_report(features, config, hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/features_daily_v1_coverage.md')
    parser.add_argument('--compare', action='store_true', help='Read-only exact comparison of all three outputs')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    destinations = [args.output_dir / 'features_daily_v1.csv',
                    args.output_dir / 'features_daily_v1_metadata.csv', args.report]
    inputs = {(ROOT / config[key]).resolve() for key in
              ('source_data', 'calendar', 'raw_candles', 'availability_catalog', 'lag_policy_document')}
    inputs.add(args.config.resolve())
    resolved = [p.resolve() for p in destinations]
    if len(set(resolved)) != len(resolved) or inputs.intersection(resolved):
        parser.error('Outputs must be distinct and cannot overwrite inputs')
    if args.report.suffix.lower() != '.md':
        parser.error('--report must be a Markdown file')
    # Protect existing repo files except the three documented build outputs.
    allowed = {ROOT / config['features_file'], ROOT / config['metadata_file'],
               ROOT / 'reports/features_daily_v1_coverage.md'}
    for path in resolved:
        if path.is_relative_to(ROOT) and path.exists() and path not in allowed:
            parser.error(f'Refusing to replace existing repository artifact: {path}')
    artifacts = build_artifacts(args.config)
    for path, content in zip(destinations, artifacts):
        if args.compare:
            if not path.exists() or path.read_bytes() != content:
                raise ValueError(f'Output differs or is missing: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(('MATCH' if args.compare else 'WROTE'), path.name, sha256(content))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
