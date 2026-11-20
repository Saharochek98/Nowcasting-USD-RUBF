"""Consistency audit of the supplied reconstruction, not a historical MOEX audit."""
import pandas as pd

from .core import validate_calendar


KNOWN_SATURDAYS = ("2024-04-27", "2024-11-02", "2024-12-28", "2025-11-01")


def _dates(frame):
    labels = frame["date"].astype("string")
    if not labels.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).all():
        raise ValueError("Expected nonmissing YYYY-MM-DD date labels")
    dates = pd.DatetimeIndex(pd.to_datetime(labels, format="%Y-%m-%d", errors="raise"))
    validate_calendar(dates)
    return dates


def audit_calendars(full_path, sessions_path, raw_path):
    """Return diagnostics even for malformed files; never repair inputs.

    Missing expected candles are reported, but are not structural errors:
    the forecasting engine must retain those calendar dates and their NaNs.
    """
    result = {"status": "failed", "calendar_source": "reconstructed_moex_calendar_provisional",
              "verification_status": "provisional_not_independently_verified",
              "checks": {}, "counts": {}, "errors": []}
    checks, counts = result["checks"], result["counts"]
    try:
        full = pd.read_csv(full_path)
        sessions = pd.read_csv(sessions_path)
        raw = pd.read_csv(raw_path)
        dates, trading = _dates(full), _dates(sessions)
        checks["dates_unique_sorted_valid"] = True
        expected_span = pd.date_range(dates.min(), dates.max(), freq="D")
        checks["full_calendar_contiguous"] = dates.equals(expected_span)
        raw_dates = pd.DatetimeIndex(pd.to_datetime(raw["begin"], errors="raise").dt.normalize())
        validate_calendar(raw_dates)
        flags = full[["is_trading_day_expected", "has_candle"]]
        if not flags.isin([0, 1]).all().all():
            raise ValueError("Calendar flags must be nonmissing binary 0/1")
        expected = dates[full.is_trading_day_expected.eq(1)]
        observed_flags = pd.Series(dates.isin(raw_dates)).to_numpy(dtype=int)
        checks["weekday_labels_match"] = (full.weekday.to_numpy() == dates.day_name()).all().item()
        checks["evidence_present"] = bool(full.moex_evidence.fillna("").str.strip().ne("").all())
        checks["sessions_within_full_calendar"] = len(trading.difference(dates)) == 0
        checks["sessions_equal_expected_dates"] = trading.equals(expected)
        checks["has_candle_matches_raw"] = bool((full.has_candle.to_numpy() == observed_flags).all())
        checks["raw_dates_within_full_calendar"] = len(raw_dates.difference(dates)) == 0
        missing, unexpected = expected.difference(raw_dates), raw_dates.difference(expected)
        result["missing_expected_dates"] = list(missing.strftime("%Y-%m-%d"))
        result["unexpected_candle_dates"] = list(unexpected.strftime("%Y-%m-%d"))
        result["known_saturdays"] = {
            day: bool(pd.Timestamp(day) in expected and pd.Timestamp(day) in trading)
            for day in KNOWN_SATURDAYS
            if dates.min() <= pd.Timestamp(day) <= dates.max()}
        result["known_saturday_candles"] = {
            day: bool(pd.Timestamp(day) in raw_dates) for day in result["known_saturdays"]}
        checks["known_saturdays_preserved"] = all(result["known_saturdays"].values())
        counts.update(calendar_dates=len(dates), expected_trading_dates=len(expected),
                      listed_trading_dates=len(trading), raw_candles=len(raw_dates),
                      missing_expected_candles=len(missing), unexpected_candles=len(unexpected))
        result["range"] = [str(dates.min().date()), str(dates.max().date())]
        result["evidence_counts"] = {str(k): int(v) for k, v in full.moex_evidence.value_counts().items()}
        claims = {"calendar_dates": 1668, "expected_trading_dates": 1160,
                  "listed_trading_dates": 1160, "raw_candles": 1160,
                  "missing_expected_candles": 0, "unexpected_candles": 0}
        result["readme_claims_match"] = all(counts[k] == v for k, v in claims.items())
        result["readme_claims"] = claims
        result["errors"] = [name for name, passed in checks.items() if not passed]
        if unexpected.size:
            result["errors"].append("raw_candles_on_unexpected_dates")
        result["status"] = "failed" if result["errors"] else ("gaps" if len(missing) else "passed")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def calendar_report(result):
    lines = ["# USDRUBF reconstructed calendar audit", "",
             f"Consistency status: **{result['status']}**.",
             "Calendar status: **provisional**, not a fully verified independent session archive.", "",
             "Inputs: `data/processed/trading_calendar.csv`, `usdrubf_trading_days.csv`,",
             "and raw MOEX ISS candle `begin` dates. Expected dates come from the supplied",
             "production-calendar/MOEX-notice reconstruction; has_candle is checked separately.", "",
             "| Count | Actual |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in result["counts"].items()]
    lines += ["", f"README numerical claims reproduced: {result.get('readme_claims_match', False)}.", "",
              "| Consistency check | Passed |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in result["checks"].items()]
    lines += ["", f"Known working Saturdays: {result.get('known_saturdays', {})}.",
              f"Candles on known working Saturdays: {result.get('known_saturday_candles', {})}.",
              f"Missing expected dates: {result.get('missing_expected_dates', [])}.",
              f"Unexpected candle dates: {result.get('unexpected_candle_dates', [])}.",
              f"Evidence label counts: {result.get('evidence_counts', {})}.",
              f"Errors: {result['errors']}.", "",
              "Equality to the reconstruction does not prove an independent historical MOEX audit.",
              "No prices, calendars or flags were repaired. The supplied calendar ends on 2026-11-18;",
              "no later target dates are invented. Historical candle availability remains pseudo-PIT."]
    return "\n".join(lines) + "\n"
