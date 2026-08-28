import datetime as dt

import pandas as pd
import pytest

from eqdrisk.marketdata.quality import (
    CROSSED,
    LOW_OI,
    OK,
    STALE,
    WIDE_SPREAD,
    ZERO_BID,
    classify_quotes,
    rejection_counts,
    staleness_reference_ts,
    wide_spread_threshold,
)

FRESH = pd.Timestamp("2026-08-20 16:00:00", tz="America/New_York")


def _row(**overrides):
    base = {
        "strike": 100.0,
        "bid": 1.0,
        "ask": 1.05,
        "asof_ts": FRESH,
        "last_trade_ts": FRESH,
        "open_interest": 100,
    }
    base.update(overrides)
    return base


def test_all_ok_when_clean():
    df = pd.DataFrame([_row(), _row(strike=101.0)])
    tagged = classify_quotes(df, spot=100.0)
    assert (tagged["reason"] == OK).all()


def test_zero_bid_detected():
    df = pd.DataFrame([_row(bid=0.0)])
    tagged = classify_quotes(df, spot=100.0)
    assert tagged["reason"].iloc[0] == ZERO_BID


def test_crossed_detected():
    df = pd.DataFrame([_row(bid=2.0, ask=1.0)])
    tagged = classify_quotes(df, spot=100.0)
    assert tagged["reason"].iloc[0] == CROSSED


def test_stale_detected():
    df = pd.DataFrame([_row(last_trade_ts=FRESH - pd.Timedelta(minutes=60))])
    tagged = classify_quotes(df, spot=100.0, stale_minutes=30)
    assert tagged["reason"].iloc[0] == STALE


def test_low_oi_detected():
    df = pd.DataFrame([_row(open_interest=2)])
    tagged = classify_quotes(df, spot=100.0, low_oi_threshold=10)
    assert tagged["reason"].iloc[0] == LOW_OI


def test_wide_spread_detected_atm():
    # ATM (moneyness=0): threshold is 10%. mid=1.025, spread=0.5 -> ~49% relative spread.
    df = pd.DataFrame([_row(bid=0.8, ask=1.25)])
    tagged = classify_quotes(df, spot=100.0)
    assert tagged["reason"].iloc[0] == WIDE_SPREAD


def test_wide_spread_threshold_widens_with_moneyness():
    moneyness = pd.Series([0.0, 0.2, 0.4, 0.8])
    tau = wide_spread_threshold(moneyness)
    assert tau.iloc[0] == pytest.approx(0.10)
    assert tau.iloc[2] == pytest.approx(0.40)
    assert tau.iloc[3] == pytest.approx(0.40)  # clipped, doesn't keep widening past 0.4
    assert (tau.diff().dropna() >= 0).all()  # monotone non-decreasing


def test_priority_order_zero_bid_beats_crossed_and_stale():
    # bid=0 AND crossed AND stale simultaneously -> ZERO_BID wins (checked first).
    df = pd.DataFrame([_row(bid=0.0, ask=-1.0, last_trade_ts=FRESH - pd.Timedelta(days=5))])
    tagged = classify_quotes(df, spot=100.0)
    assert tagged["reason"].iloc[0] == ZERO_BID


def test_rejection_counts_excludes_ok():
    df = pd.DataFrame([_row(), _row(bid=0.0), _row(bid=2.0, ask=1.0)])
    tagged = classify_quotes(df, spot=100.0)
    counts = rejection_counts(tagged)
    assert counts == {ZERO_BID: 1, CROSSED: 1}


# 2026-08-20/21/24 are NYSE trading days; 2026-08-22/23 (Sat/Sun) are not.
TRADING_DAY = dt.date(2026, 8, 21)
NEXT_TRADING_DAY = dt.date(2026, 8, 24)
CANONICAL_CLOSE_TIME = dt.time(16, 0, 0)


def test_staleness_reference_mid_session_is_capture_time_itself():
    capture = pd.Timestamp("2026-08-21 13:00:00", tz="America/New_York")
    ref = staleness_reference_ts(capture, TRADING_DAY, CANONICAL_CLOSE_TIME)
    assert ref == capture


def test_staleness_reference_after_close_caps_at_canonical_close():
    capture = pd.Timestamp("2026-08-21 22:39:00", tz="America/New_York")
    ref = staleness_reference_ts(capture, TRADING_DAY, CANONICAL_CLOSE_TIME)
    assert ref == pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")


def test_staleness_reference_before_open_falls_back_to_prior_trading_day_close():
    """Real bug found live (2026-08-28): a 00:32 ET pre-market run measured
    staleness against literal 'now,' before the session had even opened, and
    rejected 100% of SPX as STALE even though the most recent trade — from the
    prior day's completely normal close — was fresh by any reasonable standard.
    """
    capture = pd.Timestamp("2026-08-24 00:32:00", tz="America/New_York")
    ref = staleness_reference_ts(capture, NEXT_TRADING_DAY, CANONICAL_CLOSE_TIME)
    assert ref == pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")


def test_staleness_reference_on_a_non_trading_day_falls_back_to_prior_trading_day_close():
    weekend_day = dt.date(2026, 8, 22)  # Saturday
    capture = pd.Timestamp("2026-08-22 10:00:00", tz="America/New_York")
    ref = staleness_reference_ts(capture, weekend_day, CANONICAL_CLOSE_TIME)
    assert ref == pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")
