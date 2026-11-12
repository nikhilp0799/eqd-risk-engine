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


def test_an_observation_says_what_it_decides_at_todays_level():
    """P008 on 2026-11-11: NVDA 173.73 against a 225.16 fixing, observation on
    Sunday 2026-11-15, so the fixing is the Monday close."""
    p8 = _autocall(
        id="P008",
        underlying="NVDA",
        notional=5_000_000.0,
        coupon=0.0225,
        expiry=dt.date(2028, 8, 15),
        issue_date=dt.date(2026, 8, 15),
        initial_level=225.160004,
    )
    (event,) = lifecycle.upcoming_events(p8, dt.date(2026, 11, 11), spot=173.729996)
    assert event == (
        "P008 autocall observation 2026-11-15: fixing on the 2026-11-16 close; NVDA 173.73 is "
        "77.2% of the 225.16 initial fixing; coupon barrier 75% (a 2.8% fall reaches it): "
        "112,500 coupon paid at or above it; autocall barrier 100% (a 29.6% rise reaches it): "
        "redeems the 5,000,000 notional at or above it"
    )
    # Two coupons missed: the next one pays all three; below the barrier it needs a rise.
    (owed,) = lifecycle.upcoming_events(p8, dt.date(2026, 11, 12), spot=166.19, memory=2)
    assert "a 1.6% rise reaches it): 337,500 coupon paid at or above it, including 2" in owed
    assert "put barrier" not in owed


def test_the_final_observation_names_the_put_barrier():
    final = lifecycle.observation_detail(_autocall(), dt.date(2027, 8, 14), 70.0, 0)
    assert "put barrier 65% (a 7.1% fall reaches it): below it the notional is repaid" in final


def test_event_lines_skip_what_has_settled(tmp_path):
    from eqdrisk.config import BaseConfig, Paths, Universe
    from eqdrisk.io import store
    from eqdrisk.io.schemas import UNDERLYING_REQUIRED_NOT_NULL, UNDERLYING_SCHEMA, validate
    from eqdrisk.portfolio.schema import Portfolio

    cfg = BaseConfig(
        run_date=dt.date(2026, 11, 13),
        universe=Universe(index=[], single_names=["X"]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path / "curated")),
        calendar="NYSE",
        daycount="ACT/365F",
    )
    days = [dt.date(2026, 11, 12), dt.date(2026, 11, 13)]
    closes = pd.DataFrame(
        {
            "underlying": "X",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": [1.1, 1.2],
            "volume": 1,
            "asof_date": days,
        }
    )
    table = validate(closes, UNDERLYING_SCHEMA, UNDERLYING_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "curated" / "underlyings", ["asof_date"])
    lifecycle._closes.cache_clear()

    settles_today = VanillaPosition(
        id="V", underlying="X", cp="C", strike=1.0, expiry=dt.date(2026, 11, 13), qty=1
    )
    next_week = VanillaPosition(
        id="W", underlying="X", cp="C", strike=1.0, expiry=dt.date(2026, 11, 20), qty=1
    )
    pf = Portfolio(positions=[settles_today, next_week])
    # 11-12: V is a day out; W's 11-20 is past the week.
    assert lifecycle.event_lines(cfg, pf, dt.date(2026, 11, 12)) == ["V expires 2026-11-13"]
    # 11-13: V settled at today's close, so it is gone; W is now inside the week.
    assert lifecycle.event_lines(cfg, pf, dt.date(2026, 11, 13)) == ["W expires 2026-11-20"]
