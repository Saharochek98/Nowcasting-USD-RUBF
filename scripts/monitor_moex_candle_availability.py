"""Manually observe MOEX daily candles; observations never establish finality.

Standard library only. Defaults to ONE request; --duration enables a finite run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    MOSCOW = ZoneInfo('Europe/Moscow')
except ZoneInfoNotFoundError:
    # Windows may lack IANA data. Correct for live observations since 2014.
    MOSCOW = timezone(timedelta(hours=3), 'Europe/Moscow')

ENDPOINT = ('https://iss.moex.com/iss/engines/futures/markets/forts/'
            'boards/rfud/securities/USDRUBF/candles.json')
FIELDS = ['run_id', 'requested_at', 'received_at', 'timezone', 'url',
          'http_status', 'error', 'server_date', 'retry_after', 'response_sha256',
          'event', 'first_seen_at', 'previous_seen_at', 'begin', 'end',
          'open', 'close', 'high', 'low', 'value', 'volume', 'candle_json',
          'metadata_json']


def now():
    return datetime.now(MOSCOW).isoformat(timespec='milliseconds')


def parse_candles(payload):
    block = payload['candles']
    columns, data = block['columns'], block['data']
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(x, str) for x in columns)
            or len(set(columns)) != len(columns) or not isinstance(data, list)):
        raise ValueError('Invalid candles columns/data')
    if not {'begin', 'end', 'close', 'volume'} <= set(columns):
        raise ValueError('Missing required candle columns')
    records, seen = [], set()
    for values in data:
        if not isinstance(values, list) or len(values) != len(columns):
            raise ValueError('Candle row length differs from columns')
        row = dict(zip(columns, values))
        for key in ('begin', 'end'):
            datetime.fromisoformat(row[key])
        if row['begin'] in seen:
            raise ValueError('Duplicate candle begin')
        seen.add(row['begin'])
        records.append(row)
    return sorted(records, key=lambda row: row['begin'])


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def observe(url, state, run_id, timeout=20, opener=urlopen):
    """Return log rows, retaining state across empty responses and failures."""
    base = dict.fromkeys(FIELDS, '')
    base.update(run_id=run_id, requested_at=now(), timezone='Europe/Moscow', url=url)
    raw = b''
    try:
        request = Request(url, headers={'User-Agent': 'USDRUBF-availability-audit/1.0'})
        with opener(request, timeout=timeout) as response:
            base['http_status'] = response.status
            base['server_date'] = response.headers.get('Date', '')
            base['retry_after'] = response.headers.get('Retry-After', '')
            raw = response.read()
        base['received_at'] = now()
        base['response_sha256'] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        candles = parse_candles(payload)
        base['metadata_json'] = encode(payload['candles'].get('metadata', {}))
        if not candles:
            return [{**base, 'event': 'empty'}]
        # Validate all records before mutating state.
        encoded = [encode(candle) for candle in candles]
        output = []
        for candle, serialized in zip(candles, encoded):
            key = candle['begin']
            previous = state.get(key)
            event = ('first_seen' if previous is None else
                     'changed' if previous['candle_json'] != serialized else 'unchanged')
            first = previous['first_seen_at'] if previous else base['received_at']
            row = {**base, **{k: candle.get(k, '') for k in
                              ('begin', 'end', 'open', 'close', 'high', 'low',
                               'value', 'volume')},
                   'event': event, 'first_seen_at': first,
                   'previous_seen_at': previous['received_at'] if previous else '',
                   'candle_json': serialized}
            state[key] = row
            output.append(row)
        return output
    except HTTPError as exc:
        base.update(http_status=exc.code, retry_after=exc.headers.get('Retry-After', '')
                    if exc.headers else '', error=f'HTTPError: {exc.reason}')
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        base['error'] = f'{type(exc).__name__}: {exc}'
    base.update(received_at=now(), event='error')
    if raw:
        base['response_sha256'] = hashlib.sha256(raw).hexdigest()
    return [base]


def restore(path):
    state = {}
    if path.exists() and path.stat().st_size:
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError('Existing output has incompatible CSV header')
            for row in reader:
                if row['event'] in {'first_seen', 'changed', 'unchanged'}:
                    state[row['begin']] = row
    return state


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Expected a finite positive number')
    return value


def instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError('Use ISO timestamp with offset, e.g. +03:00')
    return result.astimezone(MOSCOW)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='One request (default)')
    mode.add_argument('--duration', type=positive, help='Finite observation window, seconds')
    parser.add_argument('--interval', type=positive, default=60, help='Seconds; minimum 15')
    parser.add_argument('--timeout', type=positive, default=20)
    parser.add_argument('--start-at', type=instant, help='Optional start timestamp with offset')
    parser.add_argument('--date', type=date.fromisoformat, help='Observe one candle label date')
    parser.add_argument('--output', type=Path,
                        default=Path('results/moex_availability_monitor/observations.csv'))
    args = parser.parse_args(argv)
    if args.interval < 15:
        parser.error('--interval must be at least 15 seconds')
    # Never append observations to original datasets, even if explicitly passed.
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo / 'data') or output.suffix.lower() != '.csv':
        parser.error('--output must be a CSV outside data/')
    state = restore(output)
    while args.start_at and datetime.now(MOSCOW) < args.start_at:
        time.sleep(min(30, (args.start_at - datetime.now(MOSCOW)).total_seconds()))
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    deadline = time.monotonic() + (args.duration or 0)
    with output.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        while True:
            # Rolling window includes previous/current labels across midnight.
            today = datetime.now(MOSCOW).date()
            params = {'interval': 24, 'from': str(args.date or today - timedelta(days=60)),
                      'till': str(args.date or today), 'iss.reverse': 'true'}
            rows = observe(ENDPOINT + '?' + urlencode(params), state, run_id, args.timeout)
            writer.writerows(rows)
            handle.flush()
            print(rows[-1]['received_at'], rows[-1]['event'],
                  rows[-1]['begin'], rows[-1]['error'], flush=True)
            if not args.duration:
                return 1 if rows[-1]['event'] == 'error' else 0
            # No immediate retries; honor numeric or HTTP-date Retry-After.
            delay = args.interval
            retry = rows[-1]['retry_after']
            if retry:
                from email.utils import parsedate_to_datetime
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry) -
                                            datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, OverflowError):
                        pass
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                return 0
            time.sleep(delay)
            if time.monotonic() >= deadline:
                return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
