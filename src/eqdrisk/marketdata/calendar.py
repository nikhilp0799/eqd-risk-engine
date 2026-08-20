"""Trading-day calendars and year-fraction conventions.

Used everywhere a lookback window or a curve tenor needs to count trading
days (not calendar days) or convert a date span into a year fraction.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas_market_calendars as mcal

DAYS_PER_YEAR = {
    "ACT/365F": 365.0,
    "ACT/360": 360.0,
}


@lru_cache(maxsize=8)
def _calendar(name: str):
    return mcal.get_calendar(name)


def trading_days(start: dt.date, end: dt.date, calendar: str = "NYSE") -> list[dt.date]:
    """Trading days in [start, end], inclusive."""
    schedule = _calendar(calendar).schedule(start_date=start, end_date=end)
    return [ts.date() for ts in schedule.index]


def is_trading_day(day: dt.date, calendar: str = "NYSE") -> bool:
    return len(trading_days(day, day, calendar)) == 1


def last_n_trading_days(asof: dt.date, n: int, calendar: str = "NYSE") -> list[dt.date]:
    """The `n` trading days on or before `asof`, oldest first.

    Widens the calendar-day search window until enough trading days are
    found (handles long holiday clusters without over-fetching by default).
    """
    lookback_days = int(n * 1.6) + 10
    start = asof - dt.timedelta(days=lookback_days)
    days = trading_days(start, asof, calendar)
    while len(days) < n and start > asof - dt.timedelta(days=365 * 2):
        lookback_days *= 2
        start = asof - dt.timedelta(days=lookback_days)
        days = trading_days(start, asof, calendar)
    return days[-n:]


def year_fraction(start: dt.date, end: dt.date, daycount: str = "ACT/365F") -> float:
    basis = DAYS_PER_YEAR.get(daycount)
    if basis is None:
        raise ValueError(f"unsupported daycount convention: {daycount}")
    return (end - start).days / basis
