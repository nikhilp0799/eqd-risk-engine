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
from eqdrisk.marketdata.calendar import year_fraction
from eqdrisk.marketdata.quality import OK, classify_quotes
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

# Gate for "is this Step 2 forward usable at all downstream" — deliberately much more
# permissive than Step 2's own strict acceptance-bar flags (flag_r2 < 0.999, bp > 5).
# An R²=0.9999 fit that misses the institutional 5bp target by 8bp is still an
# excellent forward for computing moneyness; only genuinely broken fits (the
# hundreds-to-thousands-of-bp disasters seen pre-quality-filter) should be excluded
# here. See planning/decisions.md, 2026-08-21, for the reasoning.
RELIABLE_R2_THRESHOLD = 0.995
RELIABLE_BP_THRESHOLD = 150.0


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


def is_reliable_forward(r_squared: float, discount_factor_diff_bp: float) -> bool:
    return (
        r_squared >= RELIABLE_R2_THRESHOLD and abs(discount_factor_diff_bp) <= RELIABLE_BP_THRESHOLD
    )


def extract_slice_ivs(
    chain_expiry: pd.DataFrame, spot: float, forward: float, discount_factor: float, T: float
) -> pd.DataFrame:
    """Classify every quote in one (underlying, expiry) slice and invert IV for survivors.

    Returns one row per input quote with `k`, `iv`, `total_variance`, `vega`, `weight`
    (all null for rejected rows) and a `reason` column — nothing is silently dropped.
    """
    df = classify_quotes(chain_expiry, spot).copy()
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

    spread = df["ask"] - df["bid"]
    df["weight"] = np.where(
        (df["reason"] == OK) & (spread > 0), quote_vega / spread.replace(0, np.nan), np.nan
    )
    return df


@dataclass
class IVExtractionResult:
    asof: dt.date
    rejection_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_expiries: dict[str, list[dt.date]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"IV extraction — {self.asof}"]
        for underlying, counts in self.rejection_counts.items():
            lines.append(f"  {underlying}: {counts}")
            skipped = self.skipped_expiries.get(underlying)
            if skipped:
                lines.append(f"    skipped (no reliable forward): {skipped}")
        return "\n".join(lines)


def run_iv_extraction(cfg: BaseConfig, asof: dt.date) -> IVExtractionResult:
    curated_root = Path(cfg.paths.curated)
    chains_root = curated_root / "chains"
    forwards_root = curated_root / "forwards"
    universe = cfg.universe.index + cfg.universe.single_names

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

        forwards = pd.DataFrame()
        if forwards_root.exists() and any(forwards_root.rglob("*.parquet")):
            forwards = store.query(
                f"SELECT * FROM fwd WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{underlying}'",
                views={"fwd": str(forwards_root)},
            ).to_pandas()
        forward_by_expiry = {row["expiry"]: row for _, row in forwards.iterrows()}

        underlying_counts: dict[str, int] = {}
        underlying_skipped: list[dt.date] = []

        for expiry, chain_expiry in chain.groupby("expiry"):
            expiry_date = pd.Timestamp(expiry).date()
            T = year_fraction(asof, expiry_date, cfg.daycount)
            fwd_row = forward_by_expiry.get(expiry_date)

            if fwd_row is None or not is_reliable_forward(
                fwd_row["r_squared"], fwd_row["discount_factor_diff_bp"]
            ):
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
                    chain_expiry,
                    spot,
                    float(fwd_row["forward"]),
                    float(fwd_row["discount_factor_implied"]),
                    T,
                )
                for reason, n in tagged["reason"].value_counts().items():
                    if reason != OK:
                        underlying_counts[reason] = underlying_counts.get(reason, 0) + int(n)

            tagged["asof_date"] = asof
            tagged["underlying"] = underlying
            tagged["expiry"] = expiry_date
            tagged["T"] = T
            out_rows.append(
                tagged[
                    [
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
                    ]
                ]
            )

        result.rejection_counts[underlying] = underlying_counts
        if underlying_skipped:
            result.skipped_expiries[underlying] = underlying_skipped

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
        table = validate(out_df, IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "implied_vols", ["asof_date", "underlying"])

    return result
