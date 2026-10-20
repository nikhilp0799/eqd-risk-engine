"""Position lifecycle from observed closing fixings (planning/position_lifecycle_plan.md).

Before this, an option past its expiry was re-priced every day at intrinsic value on
that day's forward with a full +-1 delta: P004 (short 500 AAPL 240 calls, expired
2026-10-16) kept moving with AAPL and reported a 500-share short delta that no longer
existed. Barrier knock-in and autocall observations were never applied either.

Everything here is recomputed from the stored daily closes (`underlyings`), so a
position's status on any date is reproducible from data, not from a mutable book:
- vanilla: settled in cash at the expiry close (intrinsic value);
- down-and-in put barrier: knocked in once any close since monitoring start is at or
  below the barrier (daily closes, a discrete-monitoring approximation of a
  continuously monitored barrier), then priced as the vanilla put it has become;
  settled at expiry;
- autocallable: at each past observation, the fixing (the first close on or after the
  observation date) decides redemption, coupon (paid in cash, memory reset) or a missed
  coupon (memory + 1); settled on redemption or at maturity.
Variance swaps are not settled: the book marks P006 as a fresh swap at its own fair
strike each day, so there is no traded strike to settle against (documented limit).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.portfolio.schema import (
    AutocallPosition,
    BarrierPosition,
    Position,
    VanillaPosition,
)

LIVE = "live"
KNOCKED_IN = "knocked_in"
SETTLED = "settled"

# Expiry or autocall observation within this many calendar days is flagged.
NEAR_EVENT_DAYS = 7


class LifecycleError(ValueError):
    """A lifecycle event is due but its fixing is not in the stored data."""


@dataclass
class PositionStatus:
    status: str = LIVE
    settled_value: float | None = None
    settle_date: dt.date | None = None
    # Autocallable: coupons already paid in cash, and coupons the memory still owes.
    coupons_paid: float = 0.0
    memory: int = 0
    note: str | None = None


def is_active(p: Position, asof: dt.date) -> bool:
    """In the book on `asof`: a roll enters on its `active_from` date."""
    return p.active_from is None or p.active_from <= asof


@lru_cache(maxsize=64)
def _closes(curated_root: str, underlying: str, asof: dt.date) -> pd.Series:
    root = Path(curated_root) / "underlyings"
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.Series(dtype=float)
    df = store.query(
        f"SELECT asof_date, close FROM u WHERE underlying = '{underlying}' "
        f"AND asof_date <= DATE '{asof.isoformat()}' ORDER BY asof_date",
        views={"u": str(root)},
    ).to_pandas()
    return pd.Series(
        df["close"].to_numpy(dtype=float), index=pd.to_datetime(df["asof_date"]).dt.date
    )


def _fixing_on_or_before(closes: pd.Series, day: dt.date, what: str) -> float:
    before = closes[closes.index <= day]
    if before.empty:
        raise LifecycleError(f"no close on or before {day} for {what}")
    return float(before.iloc[-1])


def _fixing_on_or_after(closes: pd.Series, day: dt.date, what: str) -> tuple[dt.date, float]:
    after = closes[closes.index >= day]
    if after.empty:
        raise LifecycleError(f"no close on or after {day} for {what}")
    return after.index[0], float(after.iloc[0])


def quarterly_observation_dates(expiry: dt.date, after: dt.date | None) -> list[dt.date]:
    """Quarterly observation dates counting back from `expiry`, oldest first,
    strictly after `after` (the issue date)."""
    dates = []
    d = pd.Timestamp(expiry)
    # No issue date: bound the walk (a note longer than 30 years is not plausible).
    floor = after or dt.date(expiry.year - 30, 1, 1)
    while d.date() > floor:
        dates.append(d.date())
        d = d - pd.DateOffset(months=3)
    return list(reversed(dates))


def _vanilla(p: VanillaPosition, asof: dt.date, closes: pd.Series) -> PositionStatus:
    if asof < p.expiry:
        return PositionStatus()
    s_t = _fixing_on_or_before(closes, p.expiry, f"{p.id} expiry")
    intrinsic = max(s_t - p.strike, 0.0) if p.cp == "C" else max(p.strike - s_t, 0.0)
    return PositionStatus(
        status=SETTLED,
        settled_value=p.qty * intrinsic,
        settle_date=p.expiry,
        note=f"settled in cash at the {p.expiry} close ({s_t:,.2f})",
    )


def _barrier(p: BarrierPosition, asof: dt.date, closes: pd.Series) -> PositionStatus:
    start = p.monitoring_start or (closes.index[0] if len(closes) else asof)
    watched = closes[(closes.index >= start) & (closes.index <= min(asof, p.expiry))]
    hits = watched[watched <= p.barrier]
    knocked = not hits.empty
    if asof >= p.expiry:
        s_t = _fixing_on_or_before(closes, p.expiry, f"{p.id} expiry")
        payoff = max(p.strike - s_t, 0.0) if knocked else 0.0
        return PositionStatus(
            status=SETTLED,
            settled_value=p.qty * payoff,
            settle_date=p.expiry,
            note=f"settled at the {p.expiry} close ({s_t:,.2f}), "
            + ("knocked in" if knocked else "never knocked in"),
        )
    if knocked:
        return PositionStatus(status=KNOCKED_IN, note=f"knocked in on {hits.index[0]}")
    return PositionStatus()


def _autocall(p: AutocallPosition, asof: dt.date, closes: pd.Series) -> PositionStatus:
    if p.initial_level is None:
        return PositionStatus()  # a note issued today: no past observations
    s0 = p.initial_level
    status = PositionStatus()
    obs_dates = quarterly_observation_dates(p.expiry, p.issue_date)
    for i, obs in enumerate(obs_dates):
        if obs > asof:
            break
        fix_date, level = _fixing_on_or_after(closes, obs, f"{p.id} observation {obs}")
        if fix_date > asof:
            break
        is_last = i == len(obs_dates) - 1
        coupon = p.notional * p.coupon * (1 + status.memory)
        if level >= p.autocall_barrier * s0:
            return PositionStatus(
                status=SETTLED,
                settled_value=status.coupons_paid + p.notional + coupon,
                settle_date=fix_date,
                coupons_paid=status.coupons_paid,
                note=f"autocalled at the {fix_date} fixing ({level:,.2f})",
            )
        if is_last:
            if level >= p.coupon_barrier * s0:
                final = p.notional + coupon
            elif level >= p.put_barrier * s0:
                final = p.notional
            else:
                final = p.notional * level / s0
            return PositionStatus(
                status=SETTLED,
                settled_value=status.coupons_paid + final,
                settle_date=fix_date,
                coupons_paid=status.coupons_paid,
                note=f"matured at the {fix_date} fixing ({level:,.2f})",
            )
        if level >= p.coupon_barrier * s0:
            status.coupons_paid += coupon
            status.memory = 0
        else:
            status.memory += 1
    return status


def position_status(cfg: BaseConfig, asof: dt.date, p: Position) -> PositionStatus:
    """Status of `p` on `asof` from the stored closes (see module docstring)."""
    if not isinstance(p, VanillaPosition | BarrierPosition | AutocallPosition):
        return PositionStatus()
    closes = _closes(str(cfg.paths.curated), p.underlying, asof)
    if isinstance(p, VanillaPosition):
        return _vanilla(p, asof, closes)
    if isinstance(p, BarrierPosition):
        return _barrier(p, asof, closes)
    return _autocall(p, asof, closes)


def upcoming_events(p: Position, asof: dt.date, days: int = NEAR_EVENT_DAYS) -> list[str]:
    """Expiry or autocall observation within `days` calendar days of `asof`."""
    horizon = asof + dt.timedelta(days=days)
    events = []
    expiry = getattr(p, "expiry", None)
    if expiry is not None and asof <= expiry <= horizon:
        events.append(f"{p.id} expires {expiry}")
    if isinstance(p, AutocallPosition):
        for obs in quarterly_observation_dates(p.expiry, p.issue_date):
            if asof < obs <= horizon and obs != p.expiry:
                events.append(f"{p.id} autocall observation {obs}")
    return events
