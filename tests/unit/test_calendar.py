import datetime as dt

from eqdrisk.marketdata.calendar import (
    is_trading_day,
    last_n_trading_days,
    trading_days,
    year_fraction,
)


def test_weekend_and_holiday_excluded():
    assert not is_trading_day(dt.date(2026, 1, 1))  # New Year's Day
    assert not is_trading_day(dt.date(2026, 8, 22))  # Saturday
    assert is_trading_day(dt.date(2026, 8, 20))  # ordinary Thursday


def test_trading_days_excludes_weekends():
    days = trading_days(dt.date(2026, 8, 17), dt.date(2026, 8, 21))
    assert days == [
        dt.date(2026, 8, 17),
        dt.date(2026, 8, 18),
        dt.date(2026, 8, 19),
        dt.date(2026, 8, 20),
        dt.date(2026, 8, 21),
    ]


def test_last_n_trading_days_returns_n_days_ending_on_asof():
    days = last_n_trading_days(dt.date(2026, 8, 20), 5)
    assert len(days) == 5
    assert days[-1] == dt.date(2026, 8, 20)
    assert days == sorted(days)


def test_year_fraction_act_365f():
    assert year_fraction(dt.date(2026, 1, 1), dt.date(2027, 1, 1)) == 365 / 365.0
