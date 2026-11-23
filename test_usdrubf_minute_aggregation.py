#!/usr/bin/env python3
"""Compare MOEX ISS daily USDRUBF candles against 1-minute candle aggregations.

Run from the repository root:
  python test_usdrubf_minute_aggregation.py --daily data/raw/USDRUBF_daily_2026-11-18.csv --all-unusual
  python test_usdrubf_minute_aggregation.py --daily data/raw/USDRUBF_daily_2026-11-18.csv --dates 2024-03-19 2025-03-19 2022-09-30

Requests require internet. Output never changes the checked-in datasets.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = 'https://iss.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/USDRUBF/candles.json'


def fetch_minutes(day, timeout=25, pause=0.15):
    """Fetch 1m candles for one calendar date with ISS pagination."""
    records = []
    start = 0
    while True:
        query = urlencode({'interval': 1, 'from': day.isoformat(), 'till': day.isoformat(), 'start': start, 'iss.meta': 'off'})
        req = Request(ENDPOINT + '?' + query, headers={'User-Agent': 'USDRUBF-research-audit/1.0'})
        with urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status} on {day} at offset {start}')
            payload = json.load(response)
        section = payload.get('candles', {})
        names = section.get('columns', [])
        chunk = section.get('data', [])
        if not names:
            raise ValueError(f'No candles schema for {day}; API response may be an error: {str(payload)[:200]}')
        required = {'begin', 'end', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(names):
            raise ValueError(f'Missing candle fields: {required - set(names)}')
        records.extend(dict(zip(names, row)) for row in chunk)
        if not chunk:
            break
        start += len(chunk)
        if len(chunk) < 100:
            # Typical ISS page size; still perform one extra request for certainty.
            pass
        if start > 10000:
            raise RuntimeError(f'Pagination exceeded expected size on {day}')
        time.sleep(pause)
    # Defense against unexpected duplicate pagination or overlapping API data.
    unique = {r['begin']: r for r in records}
    if len(unique) != len(records):
        raise ValueError(f'Duplicate minute candle beginnings in {day}')
    return sorted(unique.values(), key=lambda r: r['begin'])


def aggregate(minutes, a, b):
    chosen = [r for r in minutes if a <= datetime.fromisoformat(r['begin']) < b]
    if not chosen:
        return None
    return {
        'open': Decimal(str(chosen[0]['open'])),
        'high': max(Decimal(str(r['high'])) for r in chosen),
        'low': min(Decimal(str(r['low'])) for r in chosen),
        'close': Decimal(str(chosen[-1]['close'])),
        'volume': sum((Decimal(str(r['volume'])) for r in chosen), Decimal('0')),
        'minute_rows': len(chosen),
        'first_minute': chosen[0]['begin'],
        'last_minute': chosen[-1]['begin'],
    }


def candidate_windows(day):
    prev = day - timedelta(days=1)
    next_day = day + timedelta(days=1)
    return {
        'calendar_00_to_24': (datetime.combine(day, datetime.min.time()), datetime.combine(next_day, datetime.min.time())),
        'previous_evening_19_05_to_18_50': (datetime.fromisoformat(prev.isoformat()+' 19:05:00'), datetime.fromisoformat(day.isoformat()+' 18:50:00')),
        'previous_evening_19_05_to_19_30': (datetime.fromisoformat(prev.isoformat()+' 19:05:00'), datetime.fromisoformat(day.isoformat()+' 19:30:00')),
        'same_day_00_to_19_30': (datetime.fromisoformat(day.isoformat()+' 00:00:00'), datetime.fromisoformat(day.isoformat()+' 19:30:00')),
    }


def load_daily(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = csv.DictReader(handle)
        return {r['begin'][:10]: r for r in rows}


def compare(row, candidate):
    if candidate is None:
        return {'match_ohlcv': False, 'match_volume': False, 'match_close': False, 'volume_diff': None, 'close_diff': None}
    keys = ['open', 'high', 'low', 'close', 'volume']
    matches = {k: Decimal(str(row[k])) == candidate[k] for k in keys}
    return {
        'match_ohlcv': all(matches.values()),
        'match_volume': matches['volume'],
        'match_close': matches['close'],
        'volume_diff': str(candidate['volume']-Decimal(str(row['volume']))),
        'close_diff': str(candidate['close']-Decimal(str(row['close']))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--daily', type=Path, required=True)
    parser.add_argument('--dates', nargs='*', default=['2024-03-19', '2025-03-19', '2022-09-30'])
    parser.add_argument('--all-unusual', action='store_true', help='All candles with end 19:30:21/22')
    parser.add_argument('--output', type=Path, default=Path('results/moex_minute_audit/comparison.csv'))
    parser.add_argument('--pause', type=float, default=0.25, help='Pause between API requests in seconds')
    args = parser.parse_args()
    daily = load_daily(args.daily)
    dates = sorted(k for k,v in daily.items() if v['end'][11:] in ('19:30:21','19:30:22')) if args.all_unusual else args.dates
    if not dates:
        raise ValueError('No target dates selected')
    cache = {}
    output = []
    for day_str in dates:
        if day_str not in daily:
            raise ValueError(f'Daily candle missing: {day_str}')
        day = date.fromisoformat(day_str)
        try:
            for d in (day-timedelta(days=1), day):
                if d not in cache:
                    cache[d] = fetch_minutes(d, pause=args.pause)
                    print(f'{d}: downloaded {len(cache[d])} minute records', flush=True)
            combined = sorted(cache[day-timedelta(days=1)] + cache[day], key=lambda r:r['begin'])
            print(f'\n{day} daily={daily[day_str]["close"]} volume={daily[day_str]["volume"]} end={daily[day_str]["end"]}')
            for name, (a,b) in candidate_windows(day).items():
                candidate = aggregate(combined, a, b)
                result = compare(daily[day_str], candidate)
                record={'date':day_str, 'daily_end':daily[day_str]['end'], 'window':name, 'status':'ok', **result}
                if candidate:
                    record.update({k:str(candidate[k]) for k in ('open','high','low','close','volume','minute_rows','first_minute','last_minute')})
                output.append(record)
                print(name, 'match OHLCV:', result['match_ohlcv'], 'match volume:',result['match_volume'], 'match close:',result['match_close'], 'volume delta:',result['volume_diff'])
        except Exception as exc:
            print(f'ERROR on {day}: {type(exc).__name__}: {exc}', file=sys.stderr)
            output.append({'date':day_str, 'daily_end':daily[day_str]['end'], 'window':'all', 'status':f'ERROR: {type(exc).__name__}: {exc}'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns=['date','daily_end','window','status','match_ohlcv','match_volume','match_close','volume_diff','close_diff','open','high','low','close','volume','minute_rows','first_minute','last_minute']
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(output)
    print(f'\nSaved {args.output} | rows={len(output)}')
    if any(r['status'].startswith('ERROR') for r in output):
        print('WARNING: Some dates could not be verified. Do not infer a positive result.', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
