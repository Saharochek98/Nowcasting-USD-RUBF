"""Offline synthetic ISS responses; no network or model evaluation."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/monitor_moex_candle_availability.py'
spec = importlib.util.spec_from_file_location('monitor', SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def payload(close=80, volume=10, end='2026-11-20 23:59:59'):
    return {'candles': {'columns': ['volume', 'end', 'close', 'begin', 'extra'],
                        'data': [[volume, end, close, '2026-11-20 00:00:00', 123]],
                        'metadata': {'close': {'type': 'double'}}}}


class Response:
    status = 200
    headers = {'Date': 'Fri, 20 Nov 2026 21:10:00 GMT'}

    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MonitorTests(unittest.TestCase):
    def call(self, data, state):
        return monitor.observe('https://example.invalid', state, 'synthetic',
                               opener=lambda *a, **kw: Response(json.dumps(data).encode()))

    def test_first_change_and_no_finality(self):
        state = {}
        first = self.call(payload(), state)[0]
        self.assertEqual(first['event'], 'first_seen')
        self.assertTrue(first['received_at'].endswith('+03:00'))
        self.assertEqual(json.loads(first['candle_json'])['extra'], 123)
        self.assertEqual(self.call(payload(), state)[0]['event'], 'unchanged')
        change = self.call(payload(volume=11), state)[0]
        self.assertEqual(change['event'], 'changed')
        self.assertEqual(change['first_seen_at'], first['first_seen_at'])
        self.assertNotIn('final', change)
        self.assertEqual(self.call(payload(end='2026-11-20 23:59:42'), state)[0]['event'], 'changed')

    def test_empty_and_error_preserve_state(self):
        state = {}
        self.call(payload(), state)
        original = dict(state)
        empty = payload()
        empty['candles']['data'] = []
        self.assertEqual(self.call(empty, state)[0]['event'], 'empty')
        self.assertEqual(self.call({}, state)[0]['event'], 'error')
        self.assertEqual(state, original)

    def test_schema_and_short_rows(self):
        bad = payload()
        bad['candles']['data'][0].pop()
        self.assertEqual(self.call(bad, {})[0]['event'], 'error')
        bad['candles']['columns'] = ['close']
        with self.assertRaises(ValueError):
            monitor.parse_candles(bad)

    def test_timeout_http_and_invalid_json(self):
        for exc, status in [(TimeoutError('slow'), ''), (URLError('offline'), ''),
                            (HTTPError('url', 429, 'limit', {'Retry-After': '120'}, None), 429)]:
            def fail(*args, **kwargs):
                raise exc
            row = monitor.observe('https://example.invalid', {}, 'test', opener=fail)[0]
            self.assertEqual(row['event'], 'error')
            self.assertEqual(row['http_status'], status)
            self.assertTrue(row['error'])
            if status == 429:
                self.assertEqual(row['retry_after'], '120')
        row = monitor.observe('https://example.invalid', {}, 'test', opener=lambda *a, **kw: Response(b'<html>'))[0]
        self.assertEqual(row['http_status'], 200)
        self.assertEqual(row['event'], 'error')
        self.assertTrue(row['response_sha256'])

    def test_resume_and_single_request_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'observations.csv'
            real_observe = monitor.observe
            with patch.object(monitor, 'observe', side_effect=lambda u, s, r, t:
                              real_observe(u, s, r, t, opener=lambda *a, **kw:
                                           Response(json.dumps(payload()).encode()))) as call:
                self.assertEqual(monitor.main(['--output', str(path)]), 0)
                self.assertEqual(call.call_count, 1)
            state = monitor.restore(path)
            self.assertEqual(self.call(payload(), state)[0]['event'], 'unchanged')

    def test_invalid_arguments_and_dataset_protection(self):
        for args in [['--interval', '1'], ['--duration', 'nan'], ['--timeout', 'inf'],
                     ['--start-at', '2026-11-23T23:45:00'],
                     ['--output', str(SCRIPT.parents[1] / 'data/raw/do_not_write.csv')]]:
            with self.assertRaises(SystemExit) as error:
                monitor.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_retry_after_stops_before_next_request_outside_window(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'observations.csv'
            row = dict.fromkeys(monitor.FIELDS, '')
            row.update(event='error', retry_after='120')
            with patch.object(monitor, 'observe', return_value=[row]) as call:
                self.assertEqual(monitor.main(['--duration', '30', '--interval', '15',
                                               '--output', str(path)]), 0)
                self.assertEqual(call.call_count, 1)

    def test_finite_monitor_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'observations.csv'
            row = dict.fromkeys(monitor.FIELDS, '')
            row['event'] = 'empty'
            with patch.object(monitor, 'observe', return_value=[row]) as call, \
                 patch.object(monitor.time, 'monotonic', side_effect=[0, 0, 15, 15]), \
                 patch.object(monitor.time, 'sleep') as sleep:
                self.assertEqual(monitor.main(['--duration', '30', '--interval', '15',
                                               '--output', str(path)]), 0)
                self.assertEqual(call.call_count, 2)
                sleep.assert_called_once_with(15)

    def test_once_error_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            row = dict.fromkeys(monitor.FIELDS, '')
            row['event'] = 'error'
            with patch.object(monitor, 'observe', return_value=[row]) as call:
                self.assertEqual(monitor.main(['--once', '--output',
                                               str(Path(folder) / 'log.csv')]), 1)
                self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
