"""Regression tests for the checked-in preliminary USDRUBF factor panel v2.

 Run from repository root: python -m unittest discover -s tests -p 'test_external_features.py' -v
 Optional source-to-output check: set USDRUBF_SOURCE_DIR to the frozen raw-export directory.
 """
from __future__ import annotations
import csv
import hashlib
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_external_features as builder

DATASET=ROOT/'data/processed/model_dataset_usdrubf_all_features_v2.csv'
CANDLES=ROOT/'data/raw/USDRUBF_daily_2026-11-18.csv'
CALENDAR=ROOT/'data/processed/trading_calendar.csv'


def check_integrity(rows,raw,calendar):
    """Independent checks of CSV snapshot, source candles and provisional calendar."""
    if len(rows)!=1160 or len(raw)!=1160 or len(rows)!=len(raw):
        raise AssertionError('Wrong number of USDRUBF observations')
    if list(rows[0])!=builder.COLUMNS:
        raise AssertionError('Expected exactly 30 v2 columns in documented order')
    dates=[date.fromisoformat(x['date']) for x in rows]
    if dates!=sorted(set(dates)) or dates[0]!=date(2022,4,26) or dates[-1]!=date(2026,11,18):
        raise AssertionError('Trading dates not unique / sorted / within frozen range')
    expected=[date.fromisoformat(r['date']) for r in calendar
        if r['forts_market_trade_date_expected']=='1']
    if expected != dates:
        raise AssertionError('Trading calendar disagrees with factor-panel dates')

    for i,(row,rawrow,d) in enumerate(zip(rows,raw,dates)):
        if row['begin'][:10]!=row['date']:
            raise AssertionError(f'Bad candle begin date at {d}')
        if any(row[k]!=rawrow[k] for k in builder.BASE_COLS[1:]):
            raise AssertionError(f'OHLCV was changed relative to MOEX raw candles at {d}')
        o,c,h,l=(Decimal(row[k]) for k in ('open','close','high','low'))
        if min(o,c,l,h)<=0 or not (l<=o<=h and l<=c<=h):
            raise AssertionError(f'OHLC constraints violated on {d}')
        if Decimal(row['volume'])<0:
            raise AssertionError(f'Negative trading volume at {d}')
        if not row['cbr_usdrub_effective'] or not row['key_rate_pct_effective']:
            raise AssertionError(f'Missing CBR price or key rate on {d}')
        if Decimal(row['spread_close_minus_cbr']) != c-Decimal(row['cbr_usdrub_effective']):
            raise AssertionError(f'Wrong spread at {d}')
        cbr=date.fromisoformat(row['cbr_effective_date'])
        if cbr>d or int(row['cbr_rate_age_calendar_days'])!=(d-cbr).days:
            raise AssertionError(f'CBR effective-date/age error at {d}')
        key=date.fromisoformat(row['key_rate_reference_date'])
        if key>d:raise AssertionError(f'Future key rate on {d}')

        if row['swaprate_source_trade_date'] != (dates[i-1].isoformat() if i else ''):
            raise AssertionError(f'Wrong previous funding trade date at {d}')
        if i and row['swaprate_prev_session']!=rows[i-1]['swaprate_current']:
            raise AssertionError(f'Wrong previous funding value at {d}')
        if row['swaprate_available_assumed'] not in ('true','false'):
            raise AssertionError(f'Invalid funding availability flag on {d}')
        if (row['swaprate_available_assumed']=='true') != bool(row['swaprate_prev_session']):
            raise AssertionError(f'Funding flag not consistent with prev-session value on {d}')
        if row['swaprate_current'] and not Decimal(row['swaprate_current']).is_finite():
            raise AssertionError(f'Funding is not a finite number on {d}')

        ruo=date.fromisoformat(row['ruonia_reference_date'])
        updated=datetime.fromisoformat(row['ruonia_source_dateupdate']).date()
        if not (ruo<d and updated<=d) or int(row['ruonia_age_calendar_days'])!=(d-ruo).days:
            raise AssertionError(f'RUONIA join constraint violated at {d}')
        brent=date.fromisoformat(row['brent_eia_reference_date'])
        if brent>d-timedelta(days=10) or int(row['brent_eia_age_calendar_days'])!=(d-brent).days:
            raise AssertionError(f'EIA 10-day lag constraint violated at {d}')
        cn=date.fromisoformat(row['usdcnh_investing_reference_date'])
        if cn>=d or int(row['usdcnh_investing_age_calendar_days'])!=(d-cn).days:
            raise AssertionError(f'Investing USD/CNH date constraint violated at {d}')
        if Decimal(row['usdcnh_investing_price_prev_date'])<=0:
            raise AssertionError(f'Nonpositive USD/CNH at {d}')

    if sum(not r['swaprate_current'] for r in rows)!=153:
        raise AssertionError('Expected 153 missing current funding observations')
    if sum(not r['swaprate_prev_session'] for r in rows)!=154:
        raise AssertionError('Expected 154 missing lagged funding observations')
    return True


class SnapshotIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields, cls.rows=read_csv_with_fields(DATASET)
        cls.raw=builder.load_csv(CANDLES)
        cls.calendar=builder.load_csv(CALENDAR)

    def test_full_snapshot_integrity(self):
        self.assertTrue(check_integrity(self.rows,self.raw,self.calendar))
        self.assertEqual(hashlib.sha256(DATASET.read_bytes()).hexdigest(),builder.SNAPSHOT_SHA256)

    def test_no_duplicate_and_calendar_alignment(self):
        self.assertEqual(len({r['date'] for r in self.rows}),1160)
        self.assertEqual(sum(x['forts_market_trade_date_expected']=='1' for x in self.calendar),1160)

    def test_no_forward_looking_join(self):
        for row in self.rows:
            d=date.fromisoformat(row['date'])
            self.assertLess(date.fromisoformat(row['ruonia_reference_date']),d)
            self.assertLess(date.fromisoformat(row['usdcnh_investing_reference_date']),d)
            self.assertLessEqual(date.fromisoformat(row['brent_eia_reference_date']), d-timedelta(days=10))
            self.assertLessEqual(date.fromisoformat(row['cbr_effective_date']),d)

    def test_reject_corrupted_close(self):
        example=self.rows[0].copy()
        example['close']='99.9999'
        rows=[example]+self.rows[1:]
        with self.assertRaisesRegex(AssertionError,'OHLCV was changed'):
            check_integrity(rows,self.raw,self.calendar)

    def test_reject_corrupted_funding_lag(self):
        rows=[x.copy() for x in self.rows]
        rows[200]['swaprate_prev_session']='12345'
        with self.assertRaisesRegex(AssertionError,'Wrong previous funding value'):
            check_integrity(rows,self.raw,self.calendar)

    def test_reject_future_external_reference_date(self):
        rows=[x.copy() for x in self.rows]
        rows[250]['brent_eia_reference_date']=rows[250]['date']
        with self.assertRaisesRegex(AssertionError,'EIA 10-day lag'):
            check_integrity(rows,self.raw,self.calendar)

    def test_reject_calendar_mismatch(self):
        calendar=[x.copy() for x in self.calendar]
        calendar[0]['forts_market_trade_date_expected']='0'
        with self.assertRaisesRegex(AssertionError,'Trading calendar disagrees'):
            check_integrity(self.rows,self.raw,calendar)

    def test_no_missing_nonfunding_features(self):
        for key in ('cbr_usdrub_effective','key_rate_pct_effective','ruonia_pct_last_reported',
            'brent_eia_spot_usd_bbl_lag10d','usdcnh_investing_price_prev_date'):
            self.assertFalse(any(not row[key] for row in self.rows),key)

    def test_optional_full_rebuild_from_external_sources(self):
        sources=os.environ.get('USDRUBF_SOURCE_DIR')
        if not sources:
            self.skipTest('Set USDRUBF_SOURCE_DIR to validate exact raw-export reproduction')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model_dataset.csv'
            builder.write(builder.build(Path(sources), CALENDAR),path)
            builder.verify_exact(path,DATASET)


def read_csv_with_fields(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        return r.fieldnames,list(r)


if __name__ == '__main__':unittest.main()