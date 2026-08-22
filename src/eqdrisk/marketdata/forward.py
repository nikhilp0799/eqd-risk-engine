"""Implied forward (put-call parity regression) and implied dividend yield.

For a European option, C(K) - P(K) = P(0,T)(F - K): regressing C-P against K
recovers the discount factor (slope) and the forward (intercept/slope) in one
shot, and cross-checks against the bootstrapped curve (README 2.2).

SPX is European, so this is exact. AAPL/NVDA/JPM/XLE are American-style listed
equity options, where early-exercise value (mainly on puts, when dividends are
involved) breaks the parity identity — running this regression on them anyway
is a deliberate, documented approximation (locked decision, 2026-08-20): the
README wants single-name implied-dividend divergence analysis, and the bias
this introduces is itself worth surfacing rather than avoiding.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    DISCOUNT_CURVE_REQUIRED_NOT_NULL,
    DISCOUNT_CURVE_SCHEMA,
    FORWARD_REQUIRED_NOT_NULL,
    FORWARD_SCHEMA,
    validate,
)
from eqdrisk.marketdata import quality
from eqdrisk.marketdata.calendar import year_fraction
from eqdrisk.marketdata.curve import TENOR_YEARS, bootstrap_curve
from eqdrisk.marketdata.quality import classify_quotes, staleness_reference_ts

MIN_STRIKES = 6
MONEYNESS_BAND = 0.3  # tighter than the old 0.5 — forward extraction wants near-the-money strikes
R2_FLAG_THRESHOLD = 0.999
DISCOUNT_FACTOR_BP_FLAG_THRESHOLD = 5.0
DIVIDEND_LOOKBACK_DAYS = 370  # trailing ~12mo of announced dividends


@dataclass
class ForwardFitResult:
    underlying: str
    expiry: dt.date
    T: float
    n_strikes: int
    forward: float
    discount_factor_implied: float
    r_squared: float


def _mid(bid: pd.Series, ask: pd.Series) -> pd.Series:
    return (bid + ask) / 2.0


def _matched_pairs(clean_legs: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Merge surviving call/put legs on strike, restricted to a near-the-money band.

    The moneyness band here is on top of (not instead of) `quality.classify_quotes` —
    real forward-price extraction deliberately uses near-the-money strikes (a
    "synthetic forward" is often built from a single ATM-ish pair); wings add
    little signal for *this* purpose even when individually well-formed quotes.
    """
    near = clean_legs[(clean_legs["strike"] / spot - 1).abs() <= MONEYNESS_BAND]
    calls = near[near["cp"] == "C"][["strike", "bid", "ask"]].rename(
        columns={"bid": "bid_c", "ask": "ask_c"}
    )
    puts = near[near["cp"] == "P"][["strike", "bid", "ask"]].rename(
        columns={"bid": "bid_p", "ask": "ask_p"}
    )
    return calls.merge(puts, on="strike", how="inner")


def fit_forward(
    chain_expiry: pd.DataFrame,
    spot: float,
    underlying: str,
    expiry: dt.date,
    T: float,
    reference_ts: pd.Timestamp | None = None,
) -> ForwardFitResult | None:
    """Fit one (underlying, expiry) slice. Returns None if too few clean matched strikes
    or the fit is economically nonsensical (non-positive implied discount factor).

    Quality-filters legs first (`quality.classify_quotes` — ZERO_BID, CROSSED, STALE,
    LOW_OI, WIDE_SPREAD) before matching call/put pairs, rather than fitting against
    the raw chain and hoping the regression averages out the noise. This was found to
    matter a lot on real data: some far-wing quotes are literally years stale.
    """
    tagged = classify_quotes(chain_expiry, spot, reference_ts=reference_ts)
    pairs = _matched_pairs(tagged[tagged["reason"] == quality.OK], spot)
    if len(pairs) < MIN_STRIKES:
        return None

    y = (_mid(pairs["bid_c"], pairs["ask_c"]) - _mid(pairs["bid_p"], pairs["ask_p"])).to_numpy()
    strikes = pairs["strike"].to_numpy(dtype=float)
    spread = (pairs["ask_c"] - pairs["bid_c"]) + (pairs["ask_p"] - pairs["bid_p"])
    spread = spread.clip(lower=spread[spread > 0].min() if (spread > 0).any() else 1e-6)
    weights = (1.0 / spread).to_numpy()

    X = sm.add_constant(strikes)
    fit = sm.WLS(y, X, weights=weights).fit()
    alpha, neg_beta = fit.params
    beta = -neg_beta
    if beta <= 0:
        return None

    return ForwardFitResult(
        underlying=underlying,
        expiry=expiry,
        T=T,
        n_strikes=len(pairs),
        forward=alpha / beta,
        discount_factor_implied=beta,
        r_squared=float(fit.rsquared),
    )


def implied_dividend_yield(forward: float, discount_factor: float, spot: float, T: float) -> float:
    """q = -1/T * log(F * P(0,T) / S0) — README 2.3."""
    return float(-np.log(forward * discount_factor / spot) / T)


def announced_dividend_yield(
    dividends_root: Path, underlying: str, spot: float, asof: dt.date
) -> float | None:
    """Trailing ~12-month announced dividend yield, or None if no dividend history exists
    (e.g. SPX — an index has no per-name dividend series via yfinance)."""
    if not dividends_root.exists() or not any(dividends_root.rglob("*.parquet")):
        return None
    table = store.query(
        f"SELECT sum(amount) AS total FROM div WHERE underlying = '{underlying}' "
        f"AND ex_date > DATE '{(asof - dt.timedelta(days=DIVIDEND_LOOKBACK_DAYS)).isoformat()}' "
        f"AND ex_date <= DATE '{asof.isoformat()}'",
        views={"div": str(dividends_root)},
    )
    total = table.column("total")[0].as_py()
    if not total:
        return None
    return float(total) / spot


@dataclass
class ForwardConstructionResult:
    asof: dt.date
    fits: list[ForwardFitResult] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Forward/curve construction — {self.asof}"]
        by_underlying: dict[str, int] = {}
        for f in self.fits:
            by_underlying[f.underlying] = by_underlying.get(f.underlying, 0) + 1
        for underlying, n in by_underlying.items():
            lines.append(f"  {underlying}: {n} expiries fitted")
        if self.flagged:
            lines.append("  FLAGGED:")
            lines.extend(f"    {msg}" for msg in self.flagged)
        return "\n".join(lines)


def run_forward_construction(cfg: BaseConfig, asof: dt.date) -> ForwardConstructionResult:
    curated_root = Path(cfg.paths.curated)
    chains_root = curated_root / "chains"
    curves_root = curated_root / "curves"
    dividends_root = curated_root / "dividends"
    universe = cfg.universe.index + cfg.universe.single_names

    curves_date = store.latest_available_date(curves_root, asof)
    if curves_date is None:
        raise ValueError(f"no curated rates available on or before {asof}")
    rates = store.query(
        f"SELECT * FROM curves WHERE asof_date = DATE '{curves_date.isoformat()}'",
        views={"curves": str(curves_root)},
    ).to_pandas()
    curve = bootstrap_curve(rates)

    curve_pillar_rows = rates[rates["tenor"].isin(TENOR_YEARS)].copy()
    curve_pillar_rows["asof_date"] = asof
    curve_pillar_rows["T"] = curve_pillar_rows["tenor"].map(TENOR_YEARS)
    curve_pillar_rows["discount_factor"] = np.exp(
        -(curve_pillar_rows["rate"] / 100.0) * curve_pillar_rows["T"]
    )
    curve_table = validate(
        curve_pillar_rows[["asof_date", "tenor", "T", "rate", "discount_factor"]],
        DISCOUNT_CURVE_SCHEMA,
        DISCOUNT_CURVE_REQUIRED_NOT_NULL,
    )
    store.write_partitioned(curve_table, curated_root / "discount_curves", ["asof_date"])

    result = ForwardConstructionResult(asof=asof, fits=[])
    forward_rows = []

    for underlying in universe:
        chain = store.query(
            f"SELECT * FROM chains WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"chains": str(chains_root)},
        ).to_pandas()
        if chain.empty:
            continue
        spot = float(chain["underlying_px"].iloc[0])
        div_yield = announced_dividend_yield(dividends_root, underlying, spot, asof)
        reference_ts = staleness_reference_ts(
            chain["asof_ts"].iloc[0], asof, cfg.canonical_snap_time
        )

        for expiry, chain_expiry in chain.groupby("expiry"):
            expiry_date = pd.Timestamp(expiry).date()
            T = year_fraction(asof, expiry_date, cfg.daycount)
            if T <= 0:
                continue
            fit = fit_forward(chain_expiry, spot, underlying, expiry_date, T, reference_ts)
            if fit is None:
                continue
            result.fits.append(fit)

            df_curve = curve.discount_factor(T)
            diff_bp = (fit.discount_factor_implied - df_curve) / df_curve * 10_000
            # Deliberate choice: use the regression's OWN discount factor here, not
            # df_curve. Since forward = alpha/beta and beta = discount_factor_implied,
            # substituting q = -1/T * log(F * P(0,T) / S0) with P(0,T) = beta makes
            # F*P(0,T) = alpha — i.e. q depends only on the regression's intercept,
            # not its (noisier, per the flags above) slope. Using df_curve instead
            # would inject the option-market-vs-Treasury financing basis (the thing
            # discount_factor_diff_bp already measures) directly into the dividend
            # estimate, contaminating it with a different economic effect.
            q_impl = implied_dividend_yield(fit.forward, fit.discount_factor_implied, spot, T)
            flag_r2 = fit.r_squared < R2_FLAG_THRESHOLD
            flag_bp = abs(diff_bp) > DISCOUNT_FACTOR_BP_FLAG_THRESHOLD

            if flag_r2 or flag_bp:
                result.flagged.append(
                    f"{underlying} {expiry_date}: R²={fit.r_squared:.4f} "
                    f"(flag={flag_r2}), DF diff={diff_bp:+.1f}bp (flag={flag_bp}), "
                    f"n={fit.n_strikes}"
                )

            forward_rows.append(
                {
                    "asof_date": asof,
                    "underlying": underlying,
                    "expiry": expiry_date,
                    "T": T,
                    "n_strikes": fit.n_strikes,
                    "forward": fit.forward,
                    "discount_factor_implied": fit.discount_factor_implied,
                    "discount_factor_curve": df_curve,
                    "discount_factor_diff_bp": diff_bp,
                    "r_squared": fit.r_squared,
                    "implied_dividend_yield": q_impl,
                    "announced_dividend_yield": div_yield,
                    "dividend_yield_diff": (None if div_yield is None else q_impl - div_yield),
                    "flag_r2": flag_r2,
                    "flag_discount_factor_bp": flag_bp,
                }
            )

    if forward_rows:
        forward_table = validate(
            pd.DataFrame(forward_rows), FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL
        )
        store.write_partitioned(
            forward_table, curated_root / "forwards", ["asof_date", "underlying"]
        )

    return result
