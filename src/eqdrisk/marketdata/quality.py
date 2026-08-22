"""Quote-level quality filters, with reason codes — nothing is silently dropped.

This covers the filters that don't need a forward price or an implied vol:
ZERO_BID, CROSSED, STALE, LOW_OI, WIDE_SPREAD. The remaining README filters
(NO_ARB_INTRINSIC, ITM_SIDE, EXTREME_K, IV_SOLVE_FAIL, THIN_SLICE) need a
forward and/or a solved IV and live in `vol/implied.py`.

Built and used ahead of the rest of Step 3 on purpose: Step 2's implied-forward
regression was found to be noisy on real data specifically because it fit
against raw, unfiltered quotes (some literally years stale). Cleaning the
input here first is the actual fix — see `forward.fit_forward`, which now
calls `classify_quotes` before running the regression.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"

ZERO_BID = "ZERO_BID"
CROSSED = "CROSSED"
STALE = "STALE"
LOW_OI = "LOW_OI"
WIDE_SPREAD = "WIDE_SPREAD"
OK = "OK"

DEFAULT_STALE_MINUTES = 30
DEFAULT_LOW_OI_THRESHOLD = 10


def wide_spread_threshold(spot_moneyness: pd.Series) -> pd.Series:
    """tau(moneyness): ~10% relative-spread tolerance ATM, widening to ~40% by
    |K/S - 1| = 0.4 and beyond. Uses spot-moneyness, not forward-moneyness (`k`)
    — a forward price isn't available yet at this stage, which is the point:
    this filtering happens before (and to clean the inputs for) the forward
    estimate, not after."""
    return 0.10 + 0.30 * (spot_moneyness.abs() / 0.4).clip(upper=1.0)


def classify_quotes(
    chain: pd.DataFrame,
    spot: float,
    stale_minutes: int = DEFAULT_STALE_MINUTES,
    low_oi_threshold: int = DEFAULT_LOW_OI_THRESHOLD,
    reference_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tag each row with exactly one reason code (`OK` if it survives everything).

    Priority order when multiple issues apply: ZERO_BID > CROSSED > STALE >
    LOW_OI > WIDE_SPREAD. `chain` must have `asof_ts` (capture time, constant
    per ingest run) and `last_trade_ts` columns per the chain schema.

    `reference_ts` is the point staleness is measured against — defaults to
    `asof_ts` (literal capture time) if not given. Callers running an
    end-of-day process after the market has closed should pass the trading
    day's canonical close instead: measuring "staleness" against wall-clock
    "now" at 10pm makes almost everything look stale purely because trading
    has stopped, not because the quote is actually unreliable.
    """
    df = chain.copy()
    reason = pd.Series(OK, index=df.index, dtype="object")

    zero_bid = df["bid"] <= 0
    reason = reason.mask(zero_bid, ZERO_BID)

    crossed = (df["bid"] > df["ask"]) & (reason == OK)
    reason = reason.mask(crossed, CROSSED)

    ref = df["asof_ts"] if reference_ts is None else reference_ts
    age_minutes = (ref - df["last_trade_ts"]).dt.total_seconds() / 60.0
    stale = (age_minutes > stale_minutes) & (reason == OK)
    reason = reason.mask(stale, STALE)

    low_oi = (df["open_interest"] < low_oi_threshold) & (reason == OK)
    reason = reason.mask(low_oi, LOW_OI)

    moneyness = df["strike"] / spot - 1.0
    mid = ((df["ask"] + df["bid"]) / 2.0).replace(0.0, np.nan)
    spread_rel = (df["ask"] - df["bid"]) / mid
    wide = (spread_rel > wide_spread_threshold(moneyness)) & (reason == OK)
    reason = reason.mask(wide, WIDE_SPREAD)

    df["reason"] = reason
    return df


def rejection_counts(tagged: pd.DataFrame) -> dict[str, int]:
    """Per-reason rejection counts, excluding `OK` survivors."""
    counts = tagged["reason"].value_counts().to_dict()
    counts.pop(OK, None)
    return {str(k): int(v) for k, v in counts.items()}


def staleness_reference_ts(
    capture_ts: pd.Timestamp, asof: dt.date, canonical_snap_time: dt.time
) -> pd.Timestamp:
    """The point staleness should be measured against: whichever is earlier of the
    actual capture time and that trading day's canonical close.

    Running mid-day, this is just `capture_ts` (live intraday freshness). Running
    after the close — including any dev/demo run at an arbitrary hour — this is the
    close itself, so "how long ago did this last trade" doesn't spuriously include
    the hours nothing has traded because the market is shut.
    """
    canonical_close = pd.Timestamp.combine(asof, canonical_snap_time).tz_localize(NY_TZ)
    return min(capture_ts, canonical_close)
