"""Rebuild the FROZEN USDRUBF external-feature dataset v2 from source exports.

Reproducibility is snapshot-based: place the exact original source exports in
--sources, not a freshly refreshed export with later revisions. Run, e.g.:

 python scripts/build_external_features.py \
    --sources /path/to/usdrubf-source-exports \
    --calendar data/processed/trading_calendar.csv \
    --output /tmp/model_dataset_usdrubf_all_features_v2.csv \
    --compare data/processed/model_dataset_usdrubf_all_features_v2.csv

Requires only Python standard library if normalized Brent + key-rate CSVs are
present. For raw CBR key-rate PDF install 'pymupdf'; for raw EIA .xls install
'xlrd>=2'. No external network requests; the program never silently updates
historical data or substitutes source values.

Caution: the historical point-in-time availability is UNVERIFIED (pseudo-PIT).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import zipfile
from bisect import bisect_right
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

CANDLE_FILE = 'USDRUBF_daily_2026-11-18.csv'
CBR_FILE = 'RC_F01_01_2020_T21_11_2026.xlsx'
MOEX_FILE = 'contractresults-USDRUBF-26042022-20112026.csv'
RUONIA_RAW = 'RC_F01_01_2020_T19_11_2026.xlsx'
RUONIA_CSV = 'normalized_ruonia_2020-2026.csv'
KEY_RATE_CSV = 'CBR_key_rate_from_pdf_2020-2026.csv'
BRENT_RAW = 'RBRTEd.xls'
BRENT_CSV = 'normalized_brent_eia_1987-2026.csv'
INVESTING_FILE = 'USD to CNH Historical Exchage Rates.csv'
BASE_COLS = ['date','open','close','high','low','value','volume','begin','end']
EXTRA_COLS = [
    'cbr_usdrub_effective','cbr_effective_date','cbr_rate_age_calendar_days',
    'swaprate_current','swaprate_prev_session','swaprate_source_trade_date',
    'swaprate_available_assumed','spread_close_minus_cbr',
    'key_rate_pct_effective','key_rate_reference_date','key_rate_pdf_source_page',
    'ruonia_pct_last_reported','ruonia_reference_date','ruonia_source_dateupdate',
    'ruonia_age_calendar_days','brent_eia_spot_usd_bbl_lag10d',
    'brent_eia_reference_date','brent_eia_age_calendar_days',
    'usdcnh_investing_price_prev_date','usdcnh_investing_reference_date',
    'usdcnh_investing_age_calendar_days',
]
COLUMNS = BASE_COLS + EXTRA_COLS
BRENT_LAG_DAYS = 10
FIRST_DAY, LAST_DAY, N_ROWS = date(2022,4,26), date(2026,11,18), 1160
SNAPSHOT_SHA256 = 'da5b582e622ce5d3efb6fb619c0b9258dc15fc7d18b6d76860ce5fb021107103'
XML_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
EXCEL_EPOCH = datetime(1899,12,30)


def load_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'CSV missing header: {path}')
        return list(reader)


def raw_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        lines = file.read().splitlines()
    if lines[:2] != ['history', '']:
        raise ValueError('MOEX contractresults expected header: history followed by a blank line')
    return list(csv.DictReader(lines[2:]))


def excel_datetime(value) -> datetime:
    return EXCEL_EPOCH + timedelta(days=float(value))


def xlsx_records(path: Path) -> list[dict[str,str]]:
    """Read these CBR exports with the stdlib, no openpyxl dependency."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read('xl/worksheets/sheet1.xml')
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for si in root.findall(f'{XML_NS}si'):
                shared.append(''.join(n.text or '' for n in si.iter(f'{XML_NS}t')))
    root = ET.fromstring(xml)
    rows=[]
    for node in root.findall(f'.//{XML_NS}sheetData/{XML_NS}row'):
        cells={}
        for c in node.findall(f'{XML_NS}c'):
            reference=c.attrib['r']
            col=re.match(r'([A-Z]+)',reference).group(1)
            v=c.find(f'{XML_NS}v')
            if c.attrib.get('t') == 'inlineStr':
                parts=c.findall(f'.//{XML_NS}t')
                value=''.join(t.text or '' for t in parts)
            elif v is None:
                value=''
            elif c.attrib.get('t') == 's':
                value=shared[int(v.text)]
            else:
                value=v.text or ''
            cells[col]=value
        rows.append(cells)
    if not rows:
        raise ValueError(f'XLSX is empty: {path}')
    headers={col:value for col,value in rows[0].items() if value}
    return [{label:cells.get(col,'') for col,label in headers.items()} for cells in rows[1:]]


def unique_sorted(records: list[dict], date_key: str, source: str):
    by_date={}
    for row in records:
        d=date.fromisoformat(row[date_key])
        if d in by_date:
            raise ValueError(f'Duplicate {d} in {source}')
        by_date[d]=row
    ds=sorted(by_date)
    return ds, [by_date[d] for d in ds]


def asof(days, records, cutoff):
    i=bisect_right(days,cutoff)-1
    return records[i] if i>=0 else None


def iso_from_excel(value: str):
    return excel_datetime(value).date().isoformat()


def load_cbr(path: Path):
    rates={}
    for row in xlsx_records(path):
        if not row.get('data'):continue
        if int(float(row['nominal']))!=1 or row.get('cdx')!='Доллар США':
            raise ValueError('Unexpected CBR currency or nominal')
        d=iso_from_excel(row['data'])
        if d in rates:raise ValueError(f'Duplicate CBR rate {d}')
        rates[d]=f"{Decimal(row['curs']):.4f}"
    if len(rates)!=1699:raise ValueError(f'Expected 1699 historical CBR USD rows; got {len(rates)}')
    return dict(sorted(rates.items()))


def load_ruonia(sources:Path):
    raw=sources/RUONIA_RAW
    if raw.exists():
        result=[]
        for row in xlsx_records(raw):
            if not row.get('DT'):continue
            d=iso_from_excel(row['DT'])
            # Excel stores a fractional day; rounding to the nearest second.
            timestamp=excel_datetime(row['DateUpdate'])
            timestamp=timestamp.replace(microsecond=0)
            result.append({'reference_date':d,
                'ruonia_pct':f"{Decimal(row['ruo']):.4f}",
                'source_dateupdate_timestamp':timestamp.isoformat(' ')})
    else:
        result=load_csv(sources/RUONIA_CSV)
    days,records=unique_sorted(result,'reference_date','RUONIA')
    if len(days)!=1689:raise ValueError(f'Expected 1689 RUONIA rows; got {len(days)}')
    return days,records


def load_key_rates(sources:Path):
    frozen=sources/KEY_RATE_CSV
    if frozen.exists():
        result=load_csv(frozen)
    else:
        pdfs=list(sources.glob('cbr_hd_base_keyrate*.pdf'))
        if len(pdfs)!=1:
            raise FileNotFoundError('Missing original key-rate PDF (cbr_hd_base_keyrate*.pdf) or '+KEY_RATE_CSV)
        try:import fitz
        except ImportError as exc:
            raise RuntimeError('Install pymupdf for PDF processing: pip install pymupdf') from exc
        result=[]
        match=re.compile(r'(?m)^(\d{2}\.\d{2}\.\d{4})\s*\n(\d{1,2},\d{2})\s*$')
        with fitz.open(pdfs[0]) as pdf:
            for page_no,page in enumerate(pdf,1):
                for date_str,rate_str in match.findall(page.get_text()):
                    d=datetime.strptime(date_str,'%d.%m.%Y').date().isoformat()
                    result.append({'date':d, 'key_rate_pct':str(float(rate_str.replace(',','.'))),
                        'source_page':str(page_no)})
    days,records=unique_sorted(result,'date','CBR key rate')
    if len(days)!=1746:raise ValueError(f'Expected 1746 key rate records; got {len(days)}')
    return days,records


def load_brent(sources:Path, *, prefer_raw:bool=False):
    raw=sources/BRENT_RAW
    normalized=sources/BRENT_CSV
    if prefer_raw or not normalized.exists():
        if not raw.exists():raise FileNotFoundError('Missing EIA Brent .xls export: '+str(raw))
        try: import xlrd
        except ImportError as exc:
            raise RuntimeError('To read the raw EIA .xls install xlrd: pip install xlrd') from exc
        book=xlrd.open_workbook(str(raw))
        sheet=book.sheet_by_name('Data 1')
        result=[]
        for i in range(sheet.nrows):
            if sheet.ncols<2:raise ValueError('Unrecognized EIA Excel layout')
            dval=sheet.cell_value(i,0)
            price=sheet.cell_value(i,1)
            if isinstance(dval,(int,float)) and isinstance(price,(int,float)) and price>0:
                d=xlrd.xldate_as_datetime(dval,book.datemode).date().isoformat()
                result.append({'observation_date':d,'brent_eia_usd_bbl':f'{price:.2f}'})
    else:
        result=load_csv(normalized)
    days,records=unique_sorted(result,'observation_date','EIA Brent')
    if len(days)!=9132:raise ValueError(f'Expected 9132 EIA Brent records; got {len(days)}')
    return days,records


def load_investing(path:Path):
    rows=[]
    for row in load_csv(path):
        d=datetime.strptime(row['Date'],'%d/%m/%Y').date().isoformat()
        price=row['Price'].strip().replace(',','')
        if not price or Decimal(price)<=0:raise ValueError(f'Bad USD/CNH at {d}')
        rows.append({'reference_date':d, 'price':price})
    days,records=unique_sorted(rows,'reference_date','Investing USD/CNH')
    if len(days)!=1275:raise ValueError(f'Expected 1275 Investing rows, got {len(days)}')
    return days,records


def load_candles(path:Path):
    raw=load_csv(path)
    if len(raw)!=N_ROWS:raise ValueError(f'Expected {N_ROWS} MOEX candles; got {len(raw)}')
    if list(raw[0])!=BASE_COLS[1:]:raise ValueError('MOEX candle columns changed')
    result=[]
    for x in raw:
        d=x['begin'][:10]
        o, h, l, c=(Decimal(x[k]) for k in ('open','high','low','close'))
        if l <= 0 or not (l <= min(o,c) and max(o,c) <= h):
            raise ValueError(f'Invalid MOEX OHLC at {d}')
        if Decimal(x['volume']) < 0:
            raise ValueError(f'Negative MOEX volume at {d}')
        result.append({'date':d,**x})
    days,_=unique_sorted(result,'date','MOEX candles')
    if [r['date'] for r in result]!=[d.isoformat() for d in days]:
        raise ValueError('Candle dates not sorted')
    if days[0]!=FIRST_DAY or days[-1]!=LAST_DAY:raise ValueError('Unexpected candle date range')
    return days,result


def load_funding(path:Path):
    result={}
    for row in raw_csv(path):
        if row['SECID']!='USDRUBF' or row['BOARDID']!='RFUD':
            raise ValueError('Unexpected MOEX history instrument/board')
        key=datetime.strptime(row['TRADEDATE'],'%d.%m.%Y').date().isoformat()
        if key in result:raise ValueError('Duplicate funding date: '+key)
        result[key]=row['SWAPRATE']
    if len(result)!=1161:raise ValueError(f'Expected 1161 MOEX history rows; got {len(result)}')
    return result


def check_calendar(path:Path, days:list[date]):
    if not path.exists():raise FileNotFoundError(f'Missing independent provisional calendar: {path}')
    lines=load_csv(path)
    actual=[date.fromisoformat(x['date']) for x in lines if x['is_trading_day_expected']=='1']
    if len(actual)!=len(set(actual)):raise ValueError('Duplicate dates in calendar')
    if actual!=days:raise ValueError('MOEX trade dates differ from expected provisional calendar')


def build(sources:Path, calendar:Path) -> list[dict]:
    days,candles=load_candles(sources/CANDLE_FILE)
    check_calendar(calendar,days)
    cbr=load_cbr(sources/CBR_FILE)
    funding=load_funding(sources/MOEX_FILE)
    kr_days,krs=load_key_rates(sources)
    ruo_days,ruos=load_ruonia(sources)
    oil_days,oils=load_brent(sources)
    cn_days,cns=load_investing(sources/INVESTING_FILE)
    cbr_days=list(date.fromisoformat(s) for s in cbr.keys())
    cbr_values=list(cbr.values())
    output=[]
    for i,(d,candle) in enumerate(zip(days,candles)):
        day=d.isoformat()
        cbr_index=bisect_right(cbr_days,d)-1
        if cbr_index<0:raise ValueError(f'No available CBR USD rate on {day}')
        cbr_day=cbr_days[cbr_index]
        rate=cbr_values[cbr_index]
        prev_day=days[i-1].isoformat() if i else ''
        swap=funding[day]
        previous=funding[prev_day] if prev_day else ''
        key=asof(kr_days,krs,d)
        oil=asof(oil_days,oils,d-timedelta(days=BRENT_LAG_DAYS))
        cn=asof(cn_days,cns,d-timedelta(days=1))
        ruo=None
        end=bisect_right(ruo_days,d-timedelta(days=1))
        for candidate in reversed(ruos[:end]):
            if datetime.fromisoformat(candidate['source_dateupdate_timestamp']).date()<=d:
                ruo=candidate;break
        if not all((key,oil,cn,ruo)):
            raise ValueError('An external factor is missing on '+day)
        output.append({**candle,
            'cbr_usdrub_effective':rate,
            'cbr_effective_date':cbr_day.isoformat(),
            'cbr_rate_age_calendar_days':str((d-cbr_day).days),
            'swaprate_current':swap,
            'swaprate_prev_session':previous,
            'swaprate_source_trade_date':prev_day,
            'swaprate_available_assumed':str(bool(previous)).lower(),
            'spread_close_minus_cbr':f"{Decimal(candle['close'])-Decimal(rate):.4f}",
            'key_rate_pct_effective':key['key_rate_pct'],
            'key_rate_reference_date':key['date'],
            'key_rate_pdf_source_page':key['source_page'],
            'ruonia_pct_last_reported':ruo['ruonia_pct'],
            'ruonia_reference_date':ruo['reference_date'],
            'ruonia_source_dateupdate':ruo['source_dateupdate_timestamp'],
            'ruonia_age_calendar_days':str((d-date.fromisoformat(ruo['reference_date'])).days),
            'brent_eia_spot_usd_bbl_lag10d':oil['brent_eia_usd_bbl'],
            'brent_eia_reference_date':oil['observation_date'],
            'brent_eia_age_calendar_days':str((d-date.fromisoformat(oil['observation_date'])).days),
            'usdcnh_investing_price_prev_date':cn['price'],
            'usdcnh_investing_reference_date':cn['reference_date'],
            'usdcnh_investing_age_calendar_days':str((d-date.fromisoformat(cn['reference_date'])).days),
        })
    return output


def write(rows:list[dict],path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def verify_exact(output:Path,comparison:Path):
    generated=output.read_bytes()
    original=comparison.read_bytes()
    if generated!=original:
        digest=hashlib.sha256(generated).hexdigest()
        # Report first differing row, not only a hash; avoid truncating data silently.
        reread=load_csv(output)
        existing=load_csv(comparison)
        for i,(a,b) in enumerate(zip(reread,existing),1):
            for key in COLUMNS:
                if a[key]!=b[key]:
                    raise ValueError(f'Nonreproducible v2 at row {i}, date {a["date"]}, '
                        f'{key}: generated={a[key]!r} vs checked-in={b[key]!r}; sha256={digest}')
        raise ValueError(f'CSV bytes differ, but parsed rows match; generated sha256={digest}')
    print('EXACT REPRODUCTION verified; SHA-256:',hashlib.sha256(generated).hexdigest())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--sources',type=Path,required=True,help='Directory with frozen original source exports')
    parser.add_argument('--calendar',type=Path,default=Path('data/processed/trading_calendar.csv'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,default=None,help='Compare byte-for-byte against committed v2 CSV')
    args=parser.parse_args(argv)
    result=build(args.sources,args.calendar)
    write(result,args.output)
    print('Created',args.output,'with',len(result),'rows and',len(COLUMNS),'columns')
    if args.compare:verify_exact(args.output,args.compare)
    elif hashlib.sha256(args.output.read_bytes()).hexdigest()!=SNAPSHOT_SHA256:
        print('WARNING: output differs from frozen v2 SHA-256; check raw file versions and source metadata',file=sys.stderr)

if __name__=='__main__':main()
