"""Implied vol extraction: Black-76 inversion plus the filter-chain reason codes
that need a forward price and/or a solved IV (`quality.py` covers the ones that don't).

Works in Black-76 forward space throughout: k = log(K/F) (log-moneyness),
w = sigma^2 * T (total implied variance) — README 3.1.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import IMPLIED_VOL_REQUIRED_NOT_NULL, IMPLIED_VOL_SCHEMA, validate
from eqdrisk.marketdata.calendar import trading_days, year_fraction
from eqdrisk.marketdata.curve import bootstrap_curve
from eqdrisk.marketdata.forward import (  # noqa: F401  (re-exported for callers/tests)
    RELIABLE_BP_THRESHOLD,
    RELIABLE_R2_THRESHOLD,
    build_forward_curve,
    is_reliable_forward,
    reliable_forwards,
)
from eqdrisk.marketdata.quality import OK, classify_quotes, staleness_reference_ts
from eqdrisk.pricing.blackscholes import call_price, put_price
from eqdrisk.pricing.blackscholes import vega as bs_vega

NO_ARB_INTRINSIC = "NO_ARB_INTRINSIC"
ITM_SIDE = "ITM_SIDE"
EXTREME_K = "EXTREME_K"
IV_SOLVE_FAIL = "IV_SOLVE_FAIL"
THIN_SLICE = "THIN_SLICE"
NO_RELIABLE_FORWARD = "NO_RELIABLE_FORWARD"

SIGMA_LO, SIGMA_HI = 1e-4, 5.0
EXTREME_K_MULTIPLE = 4.0
MIN_SLICE_QUOTES = 8  # README's own number: fewer surviving quotes -> THIN_SLICE


def invert_iv(
    price: float, forward: float, strike: float, T: float, discount_factor: float, is_call: bool
) -> float | None:
    """Solve for sigma via Brent's method. None means IV_SOLVE_FAIL (no sign change
    in [SIGMA_LO, SIGMA_HI], or the solver didn't converge)."""
    pricer = call_price if is_call else put_price

    def objective(sigma: float) -> float:
        return pricer(forward, strike, T, sigma, discount_factor) - price

    f_lo, f_hi = objective(SIGMA_LO), objective(SIGMA_HI)
    if f_lo * f_hi > 0:
        return None
    try:
        return float(brentq(objective, SIGMA_LO, SIGMA_HI, xtol=1e-13, rtol=1e-14))
    except (ValueError, RuntimeError):
        return None


def extract_slice_ivs(
    chain_expiry: pd.DataFrame,
    spot: float,
    forward: float,
    discount_factor: float,
    T: float,
    reference_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Classify every quote in one (underlying, expiry) slice and invert IV for survivors.

    Returns one row per input quote with `k`, `iv`, `total_variance`, `vega`, `weight`
    (all null for rejected rows) and a `reason` column — nothing is silently dropped.
    """
    df = classify_quotes(chain_expiry, spot, reference_ts=reference_ts).copy()
    df["k"] = np.log(df["strike"] / forward)

    # NO_ARB_INTRINSIC before ITM_SIDE, deliberately: an OTM option's forward-intrinsic
    # value (max(F-K,0) for calls, max(K-F,0) for puts) is trivially zero, so this check
    # only ever binds on ITM quotes — checking it first catches a genuinely mispriced ITM
    # quote with its own specific reason, before ITM_SIDE would otherwise discard it (for
    # an unrelated reason: we simply don't use the ITM side, mispriced or not).
    mid = (df["bid"] + df["ask"]) / 2.0
    intrinsic = np.where(
        df["cp"] == "C",
        np.maximum(forward - df["strike"], 0.0),
        np.maximum(df["strike"] - forward, 0.0),
    )
    no_arb = (mid < discount_factor * intrinsic - 1e-9) & (df["reason"] == OK)
    df["reason"] = df["reason"].mask(no_arb, NO_ARB_INTRINSIC)

    itm_side = ((df["cp"] == "C") & (df["k"] <= 0)) | ((df["cp"] == "P") & (df["k"] >= 0))
    df["reason"] = df["reason"].mask((df["reason"] == OK) & itm_side, ITM_SIDE)

    iv = pd.Series(np.nan, index=df.index)
    for i in df.index[df["reason"] == OK]:
        row = df.loc[i]
        solved = invert_iv(mid.loc[i], forward, row["strike"], T, discount_factor, row["cp"] == "C")
        if solved is None:
            df.loc[i, "reason"] = IV_SOLVE_FAIL
        else:
            iv.loc[i] = solved
    df["iv"] = iv

    extreme = (df["reason"] == OK) & (df["k"].abs() > EXTREME_K_MULTIPLE * df["iv"] * np.sqrt(T))
    df["reason"] = df["reason"].mask(extreme, EXTREME_K)

    survivor_count = (df["reason"] == OK).sum()
    if 0 < survivor_count < MIN_SLICE_QUOTES:
        df["reason"] = df["reason"].mask(df["reason"] == OK, THIN_SLICE)

    df["total_variance"] = np.where(df["reason"] == OK, df["iv"] ** 2 * T, np.nan)

    quote_vega = pd.Series(np.nan, index=df.index)
    for i in df.index[df["reason"] == OK]:
        quote_vega.loc[i] = bs_vega(
            forward, df.loc[i, "strike"], T, df.loc[i, "iv"], discount_factor
        )
    df["vega"] = quote_vega

    # Floor spread at a nominal tick size rather than dividing by zero: a locked
    # (bid==ask) market is unusual but not invalid — ZERO_BID/CROSSED already ruled out
    # bid<=0 and bid>ask, so a literal zero-width spread here is a genuine two-sided
    # quote that deserves a high (not undefined/NaN) weight.
    MIN_SPREAD = 0.01
    spread = (df["ask"] - df["bid"]).clip(lower=MIN_SPREAD)
    df["weight"] = np.where(df["reason"] == OK, quote_vega / spread, np.nan)
    return df


# Stale fill (planning/input_stability_plan.md, decision 2): an expiry that had
# usable quotes on the previous stored day but none today keeps yesterday's
# quotes for a few trading days, so the surface's long end doesn't vanish and
# reappear with thinly traded long-dated quotes.
IV_COLUMNS = [
    "asof_date",
    "underlying",
    "expiry",
    "strike",
    "cp",
    "T",
    "k",
    "iv",
    "total_variance",
    "vega",
    "weight",
    "reason",
    "carried_from",
    "forward_source",
]

# Where a slice's forward came from (planning/input_stability_plan.md, C7):
# "fit" = that expiry's own put-call parity fit; "curve" = that day's forward
# curve, for a single-name expiry with no reliable fit of its own.
FORWARD_FROM_FIT = "fit"
FORWARD_FROM_CURVE = "curve"

STALE_FILL_MAX_TRADING_DAYS = 3
STALE_FILL_WEIGHT = 0.5


def carry_forward_quotes(
    prev_ok: pd.DataFrame,
    expiries_ok_today: set[dt.date],
    asof: dt.date,
    daycount: str = "ACT/365F",
    calendar: str = "NYSE",
) -> pd.DataFrame:
    """OK quotes from the previous stored day, re-expressed for `asof`, for every
    expiry with no OK quote today. Done at the quote level (not by copying fitted
    parameters) because SPX's surface can be a joint SSVI fit, where one slice
    can't be copied on its own.

    Same log-moneyness and implied vol; T shortened to today's, so total variance
    is recomputed; weight halved once (a quote carried a second time keeps its
    already-halved weight). `carried_from` is the date the quote was actually
    observed, and a quote older than `STALE_FILL_MAX_TRADING_DAYS` trading days
    is dropped."""
    if prev_ok.empty:
        return prev_ok.iloc[0:0]
    df = prev_ok.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    if "carried_from" not in df.columns:
        df["carried_from"] = None
    if "forward_source" not in df.columns:
        df["forward_source"] = None
    already_carried = df["carried_from"].notna()
    origin = df["carried_from"].where(already_carried, df["asof_date"])
    df["carried_from"] = pd.to_datetime(origin).dt.date

    df = df[~df["expiry"].isin(expiries_ok_today) & (df["expiry"] > asof)]
    if df.empty:
        return df
    age = {
        d: len(trading_days(d + dt.timedelta(days=1), asof, calendar))
        for d in df["carried_from"].unique()
    }
    df = df[df["carried_from"].map(age) <= STALE_FILL_MAX_TRADING_DAYS]
    if df.empty:
        return df

    already_carried = df["carried_from"] != pd.to_datetime(df["asof_date"]).dt.date
    df["asof_date"] = asof
    df["T"] = [year_fraction(asof, e, daycount) for e in df["expiry"]]
    df["total_variance"] = df["iv"] ** 2 * df["T"]
    df["weight"] = df["weight"].where(already_carried, df["weight"] * STALE_FILL_WEIGHT)
    df["reason"] = OK
    return df


def observed_quotes(ivs: pd.DataFrame) -> pd.DataFrame:
    """Quotes actually observed on their `asof_date` (excludes stale-fill rows),
    for consumers that compare the model with the day's market prices."""
    if "carried_from" not in ivs.columns:
        return ivs
    return ivs[ivs["carried_from"].isna()]


@dataclass
class IVExtractionResult:
    asof: dt.date
    rejection_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_expiries: dict[str, list[dt.date]] = field(default_factory=dict)
    carried_expiries: dict[str, list[dt.date]] = field(default_factory=dict)
    curve_forward_expiries: dict[str, list[dt.date]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"IV extraction — {self.asof}"]
        for underlying, counts in self.rejection_counts.items():
            lines.append(f"  {underlying}: {counts}")
            skipped = self.skipped_expiries.get(underlying)
            if skipped:
                lines.append(f"    skipped (no reliable forward): {skipped}")
            from_curve = self.curve_forward_expiries.get(underlying)
            if from_curve:
                lines.append(f"    forward taken from the forward curve: {from_curve}")
            carried = self.carried_expiries.get(underlying)
            if carried:
                lines.append(f"    carried from a previous day (stale fill): {carried}")
        return "\n".join(lines)


def run_iv_extraction(cfg: BaseConfig, asof: dt.date) -> IVExtractionResult:
    curated_root = Path(cfg.paths.curated)
    chains_root = curated_root / "chains"
    forwards_root = curated_root / "forwards"
    universe = cfg.universe.index + cfg.universe.single_names

    iv_root = curated_root / "implied_vols"
    prev_date = store.latest_available_date(iv_root, asof - dt.timedelta(days=1))

    rates_curve = None
    curves_date = store.latest_available_date(curated_root / "curves", asof)
    if curves_date is not None:
        rates_curve = bootstrap_curve(
            store.query(
                f"SELECT * FROM curves WHERE asof_date = DATE '{curves_date.isoformat()}'",
                views={"curves": str(curated_root / "curves")},
            ).to_pandas()
        )

    result = IVExtractionResult(asof=asof)
    out_rows = []

    for underlying in universe:
        chain = store.query(
            f"SELECT * FROM chains WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"chains": str(chains_root)},
        ).to_pandas()
        if chain.empty:
            continue
        spot = float(chain["underlying_px"].iloc[0])
        reference_ts = staleness_reference_ts(
            chain["asof_ts"].iloc[0], asof, cfg.canonical_snap_time, cfg.calendar
        )

        forwards = pd.DataFrame()
        if forwards_root.exists() and any(forwards_root.rglob("*.parquet")):
            forwards = store.query(
                f"SELECT * FROM fwd WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{underlying}'",
                views={"fwd": str(forwards_root)},
            ).to_pandas()
        forward_by_expiry = {row["expiry"]: row for _, row in forwards.iterrows()}

        # Single names (American options) often have long expiries with clean
        # out-of-the-money quotes but too few strikes where BOTH legs are clean
        # for a forward fit (the in-the-money leg rarely trades): measured on all
        # history, 80% of NVDA's no-forward expiry-days beyond 0.5y had >= 8 clean
        # OTM quotes but a median of 2 usable call-put pairs (the fit needs 6).
        # Those expiries take their forward from the day's forward curve
        # (reliable fits + long-end carry) instead of being dropped.
        fallback_curve = None
        reliable = reliable_forwards(forwards)
        if (
            underlying in cfg.universe.single_names
            and rates_curve is not None
            and not reliable.empty
        ):
            fallback_curve = build_forward_curve(reliable)
        underlying_from_curve: list[dt.date] = []

        underlying_counts: dict[str, int] = {}
        underlying_skipped: list[dt.date] = []
        expiries_ok_today: set[dt.date] = set()

        for expiry, chain_expiry in chain.groupby("expiry"):
            expiry_date = pd.Timestamp(expiry).date()
            T = year_fraction(asof, expiry_date, cfg.daycount)
            fwd_row = forward_by_expiry.get(expiry_date)

            forward_source: str | None = None
            forward = discount_factor = float("nan")
            if fwd_row is not None and not reliable_forwards(fwd_row.to_frame().T).empty:
                forward_source = FORWARD_FROM_FIT
                forward = float(fwd_row["forward"])
                # A "curve_df" forward was fitted WITH the curve's discount factor,
                # so its IVs must use that one too, not the drifting parity DF.
                discount_factor = float(
                    fwd_row["discount_factor_curve"]
                    if fwd_row.get("method") == "curve_df"
                    else fwd_row["discount_factor_implied"]
                )
            elif fallback_curve is not None and rates_curve is not None and T > 0:
                forward = fallback_curve.forward(T)
                discount_factor = rates_curve.discount_factor(T)
                forward_source = FORWARD_FROM_CURVE
                underlying_from_curve.append(expiry_date)

            if forward_source is None:
                underlying_skipped.append(expiry_date)
                underlying_counts[NO_RELIABLE_FORWARD] = underlying_counts.get(
                    NO_RELIABLE_FORWARD, 0
                ) + len(chain_expiry)
                tagged = chain_expiry.copy()
                tagged["reason"] = NO_RELIABLE_FORWARD
                tagged["k"] = np.nan
                tagged["iv"] = np.nan
                tagged["total_variance"] = np.nan
                tagged["vega"] = np.nan
                tagged["weight"] = np.nan
            else:
                tagged = extract_slice_ivs(
                    chain_expiry, spot, forward, discount_factor, T, reference_ts
                )
                for reason, n in tagged["reason"].value_counts().items():
                    if reason != OK:
                        underlying_counts[reason] = underlying_counts.get(reason, 0) + int(n)
                if (tagged["reason"] == OK).any():
                    expiries_ok_today.add(expiry_date)

            tagged["asof_date"] = asof
            tagged["underlying"] = underlying
            tagged["expiry"] = expiry_date
            tagged["T"] = T
            tagged["carried_from"] = None
            tagged["forward_source"] = forward_source
            out_rows.append(tagged[IV_COLUMNS])

        if prev_date is not None:
            prev_ok = store.query(
                f"SELECT * FROM iv WHERE asof_date = DATE '{prev_date.isoformat()}' "
                f"AND underlying = '{underlying}' AND reason = 'OK'",
                views={"iv": str(iv_root)},
            ).to_pandas()
            carried = carry_forward_quotes(
                prev_ok, expiries_ok_today, asof, cfg.daycount, cfg.calendar
            )
            if not carried.empty:
                out_rows.append(carried[IV_COLUMNS])
                result.carried_expiries[underlying] = sorted(set(carried["expiry"]))

        if underlying_from_curve:
            result.curve_forward_expiries[underlying] = underlying_from_curve
        result.rejection_counts[underlying] = underlying_counts
        if underlying_skipped:
            result.skipped_expiries[underlying] = underlying_skipped

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
        table = validate(out_df, IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "implied_vols", ["asof_date", "underlying"])

    return result
