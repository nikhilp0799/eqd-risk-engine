import datetime as dt

import pandas as pd
import pytest

from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.schema import AutocallPosition, BarrierPosition, VanillaPosition


def _closes(points: dict[str, float]) -> pd.Series:
    return pd.Series(list(points.values()), index=[dt.date.fromisoformat(d) for d in points])


def _barrier(**kw) -> BarrierPosition:
    base = dict(
        id="B", underlying="X", strike=100.0, barrier=80.0, expiry=dt.date(2026, 12, 18), qty=-10
    )
    return BarrierPosition(**{**base, **kw})


def _autocall(**kw) -> AutocallPosition:
    base = dict(
        id="A",
        underlying="X",
        notional=1_000_000.0,
        autocall_barrier=1.0,
        coupon_barrier=0.75,
        put_barrier=0.65,
        coupon=0.02,
        expiry=dt.date(2027, 8, 14),
        issue_date=dt.date(2026, 8, 14),
        initial_level=100.0,
    )
    return AutocallPosition(**{**base, **kw})


def test_vanilla_is_live_before_expiry_and_settles_at_the_expiry_close():
    p = VanillaPosition(
        id="V", underlying="X", cp="C", strike=240.0, expiry=dt.date(2026, 10, 16), qty=-500
    )
    closes = _closes({"2026-10-15": 342.0, "2026-10-16": 343.46, "2026-10-19": 342.94})
    assert lifecycle._vanilla(p, dt.date(2026, 10, 15), closes).status == lifecycle.LIVE
    for asof in (dt.date(2026, 10, 16), dt.date(2026, 10, 19)):
        st = lifecycle._vanilla(p, asof, closes)
        assert st.status == lifecycle.SETTLED
        assert st.settled_value == pytest.approx(-500 * 103.46)  # the P004 case, 2026-10-16


def test_barrier_knocks_in_on_a_close_at_the_barrier_and_settles_by_whether_it_did():
    closes = _closes(
        {"2026-09-01": 95.0, "2026-09-02": 80.0, "2026-09-03": 90.0, "2026-12-18": 85.0}
    )
    st = lifecycle._barrier(_barrier(), dt.date(2026, 9, 3), closes)
    assert st.status == lifecycle.KNOCKED_IN
    assert "2026-09-02" in (st.note or "")
    settled = lifecycle._barrier(_barrier(), dt.date(2026, 12, 18), closes)
    assert settled.settled_value == pytest.approx(-10 * 15.0)

    never = _closes({"2026-09-01": 95.0, "2026-12-18": 85.0})
    assert lifecycle._barrier(_barrier(), dt.date(2026, 12, 18), never).settled_value == 0.0


def test_barrier_ignores_closes_before_its_monitoring_start():
    closes = _closes({"2026-08-01": 70.0, "2026-09-01": 95.0})
    p = _barrier(monitoring_start=dt.date(2026, 8, 15))
    assert lifecycle._barrier(p, dt.date(2026, 9, 1), closes).status == lifecycle.LIVE


def test_autocall_pays_coupon_then_misses_then_redeems_with_memory():
    # Observations 2026-11-14 (a Saturday: fixed on Monday the 16th), 2027-02-14, 2027-05-14.
    closes = _closes(
        {
            "2026-11-16": 90.0,  # >= 75 coupon, < 100 autocall: coupon 20,000 paid
            "2027-02-15": 70.0,  # < 75: coupon missed, memory 1
            "2027-05-14": 101.0,  # >= 100: redeemed at par + 2 coupons (one owed)
        }
    )
    p = _autocall()
    st = lifecycle._autocall(p, dt.date(2026, 11, 20), closes)
    assert (st.status, st.coupons_paid, st.memory) == (lifecycle.LIVE, 20_000.0, 0)
    st = lifecycle._autocall(p, dt.date(2027, 3, 1), closes)
    assert (st.coupons_paid, st.memory) == (20_000.0, 1)
    st = lifecycle._autocall(p, dt.date(2027, 5, 14), closes)
    assert st.status == lifecycle.SETTLED
    assert st.settled_value == pytest.approx(20_000 + 1_000_000 + 40_000)


def test_autocall_observation_not_yet_fixed_is_not_applied():
    # 2026-11-14 is a Saturday: on Friday the 13th nothing has been observed yet.
    closes = _closes({"2026-11-13": 120.0})
    st = lifecycle._autocall(_autocall(), dt.date(2026, 11, 13), closes)
    assert st.status == lifecycle.LIVE and st.coupons_paid == 0.0


def test_autocall_maturity_below_the_put_barrier_pays_the_loss():
    p = _autocall(expiry=dt.date(2026, 11, 13), issue_date=dt.date(2026, 8, 13))
    closes = _closes({"2026-11-13": 50.0})
    st = lifecycle._autocall(p, dt.date(2026, 11, 13), closes)
    assert st.status == lifecycle.SETTLED
    assert st.settled_value == pytest.approx(1_000_000 * 0.5)


def test_rolls_enter_the_book_on_their_active_from_date():
    p = VanillaPosition(
        id="R",
        underlying="X",
        cp="C",
        strike=1.0,
        expiry=dt.date(2027, 1, 15),
        qty=1,
        active_from=dt.date(2026, 10, 16),
    )
    assert not lifecycle.is_active(p, dt.date(2026, 10, 15))
    assert lifecycle.is_active(p, dt.date(2026, 10, 16))


def test_upcoming_events_lists_expiries_and_observations_within_a_week():
    asof = dt.date(2026, 11, 10)
    assert lifecycle.upcoming_events(_autocall(), asof) == ["A autocall observation 2026-11-14"]
    v = VanillaPosition(
        id="V", underlying="X", cp="C", strike=1.0, expiry=dt.date(2026, 11, 13), qty=1
    )
    assert lifecycle.upcoming_events(v, asof) == ["V expires 2026-11-13"]
    assert lifecycle.upcoming_events(v, dt.date(2026, 11, 1)) == []
