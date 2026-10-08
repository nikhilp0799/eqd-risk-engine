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

# Reliability, ONE definition used by both the vol surface (IV extraction) and the
# pricing forward curve (planning/input_stability_plan.md, C3).
# - "parity_2p" (European index options): R^2 and the parity-implied discount
#   factor's distance from the curve; the original rule (decisions.md, 2026-08-21).
RELIABLE_R2_THRESHOLD = 0.995
RELIABLE_BP_THRESHOLD = 150.0
# - "curve_df" (American single names): how tightly the strikes agree on the
#   forward. Measured on all stored history (2026-10-07): median 4-27bp, worst
#   58bp (NVDA beyond 1y); 100bp accepts every observed fit and still rejects a
#   genuinely broken one. The old DF test rejected 77% of NVDA fits beyond 1y.
RELIABLE_DISPERSION_BP = 100.0


def is_reliable_forward(
    r_squared: float,
    discount_factor_diff_bp: float,
    method: str | None = None,
    dispersion_bp: float | None = None,
) -> bool:
    if method == "curve_df":
        return dispersion_bp is not None and dispersion_bp <= RELIABLE_DISPERSION_BP
    return (
        r_squared >= RELIABLE_R2_THRESHOLD and abs(discount_factor_diff_bp) <= RELIABLE_BP_THRESHOLD
    )


def reliable_forwards(forwards: pd.DataFrame) -> pd.DataFrame:
    """The rows of a `forwards` slice that are reliable. Rows written before the
    `reliable` column existed fall back to the original two-parameter rule."""
    if forwards.empty:
        return forwards
    if "reliable" in forwards.columns:
        stored = forwards["reliable"]
        legacy = stored.isna()
    else:
        stored = pd.Series(False, index=forwards.index)
        legacy = pd.Series(True, index=forwards.index)
    legacy_ok = forwards.apply(
        lambda r: is_reliable_forward(r["r_squared"], r["discount_factor_diff_bp"]), axis=1
    )
    keep = stored.where(~legacy, legacy_ok).astype(bool)
    return forwards[keep]


@dataclass
class ForwardCurve:
    """Continuous forward term structure F(0,T), interpolated log-linearly (same
    convention as `curve.Curve`'s discount-factor interpolation) across the day's
    calibrated per-expiry forwards, flat before the shortest pillar. Needed by
    Step 6's local-vol grid, which evaluates sigma_loc at (spot, calendar-time)
    pairs that fall off the handful of expiries options actually settle on.

    Beyond the longest pillar the log-forward continues at `long_end_carry`
    (per year) rather than staying flat: a flat forward means zero carry, so
    every change in which expiries are reliable moved long-dated forwards for no
    market reason (planning/input_stability_plan.md, C4).
    """

    pillar_T: np.ndarray
    pillar_log_forward: np.ndarray
    long_end_carry: float = 0.0

    def forward(self, T: float) -> float:
        log_f = np.interp(T, self.pillar_T, self.pillar_log_forward)
        beyond = T - self.pillar_T[-1]
        if beyond > 0:
            log_f += self.long_end_carry * beyond
        return float(np.exp(log_f))


# The long-end carry is the log-forward slope from the pillar at least this far
# before the last one (or the first pillar if none is), so two nearly coincident
# long expiries can't produce a wild slope. Measured on stored history
# (2026-10-07), last-segment carry was 2.7-5.4%/yr for NVDA/SPX/AAPL.
LONG_END_CARRY_MIN_SPAN_YEARS = 0.25


def build_forward_curve(forwards_for_underlying: pd.DataFrame) -> ForwardCurve:
    """`forwards_for_underlying` must have `T` and `forward` columns for one
    underlying's expiries on one asof date (i.e. a slice of the `forwards` table)."""
    df = forwards_for_underlying.sort_values("T")
    if df.empty:
        raise ValueError("no forward pillars to build a forward curve from")
    T = df["T"].to_numpy(dtype=float)
    log_f = np.log(df["forward"].to_numpy(dtype=float))
    carry = 0.0
    if len(T) >= 2:
        earlier = np.nonzero(T <= T[-1] - LONG_END_CARRY_MIN_SPAN_YEARS)[0]
        i = int(earlier[-1]) if len(earlier) else 0
        if T[-1] > T[i]:
            carry = float((log_f[-1] - log_f[i]) / (T[-1] - T[i]))
    return ForwardCurve(pillar_T=T, pillar_log_forward=log_f, long_end_carry=carry)


@dataclass
class ForwardFitResult:
    underlying: str
    expiry: dt.date
    T: float
    n_strikes: int
    forward: float
    discount_factor_implied: float
    r_squared: float
    # "parity_2p": discount factor and forward both fitted from put-call parity
    # (European index options). "curve_df": discount factor taken from the rates
    # curve and only the forward fitted (American single-name options, whose
    # parity-implied discount factor drifts with maturity; planning/
    # input_stability_plan.md). `discount_factor_implied`/`r_squared` always come
    # from the two-parameter fit, kept as diagnostics.
    method: str = "parity_2p"
    # Weighted spread, across strikes, of the forward each strike implies, in bp
    # of the forward: how much the strikes agree. Only for "curve_df".
    dispersion_bp: float | None = None


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
    curve_discount_factor: float | None = None,
) -> ForwardFitResult | None:
    """Fit one (underlying, expiry) slice. Returns None if too few clean matched strikes
    or the fit is economically nonsensical (non-positive implied discount factor).

    With `curve_discount_factor`, the forward is fitted with the discount factor
    fixed at the curve's: each strike's pair implies F_i = (C - P)/DF + K, and F
    is their spread-weighted mean (method "curve_df").

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

    if curve_discount_factor is not None:
        implied = y / curve_discount_factor + strikes
        forward = float(np.average(implied, weights=weights))
        dispersion = float(np.sqrt(np.average((implied - forward) ** 2, weights=weights)))
        if forward <= 0:
            return None
        return ForwardFitResult(
            underlying=underlying,
            expiry=expiry,
            T=T,
            n_strikes=len(pairs),
            forward=forward,
            discount_factor_implied=float(beta),
            r_squared=float(fit.rsquared),
            method="curve_df",
            dispersion_bp=10_000.0 * dispersion / forward,
        )

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
            chain["asof_ts"].iloc[0], asof, cfg.canonical_snap_time, cfg.calendar
        )

        # American single-name options: parity's implied discount factor drifts with
        # maturity, so fix it at the curve's and fit only the forward.
        use_curve_df = underlying in cfg.universe.single_names

        for expiry, chain_expiry in chain.groupby("expiry"):
            expiry_date = pd.Timestamp(expiry).date()
            T = year_fraction(asof, expiry_date, cfg.daycount)
            if T <= 0:
                continue
            df_curve = curve.discount_factor(T)
            fit = fit_forward(
                chain_expiry,
                spot,
                underlying,
                expiry_date,
                T,
                reference_ts,
                curve_discount_factor=df_curve if use_curve_df else None,
            )
            if fit is None:
                continue
            result.fits.append(fit)

            diff_bp = (fit.discount_factor_implied - df_curve) / df_curve * 10_000
            # Deliberate choice: use the regression's OWN discount factor here, not
            # df_curve. Since forward = alpha/beta and beta = discount_factor_implied,
            # substituting q = -1/T * log(F * P(0,T) / S0) with P(0,T) = beta makes
            # F*P(0,T) = alpha — i.e. q depends only on the regression's intercept,
            # not its (noisier, per the flags above) slope. Using df_curve instead
            # would inject the option-market-vs-Treasury financing basis (the thing
            # discount_factor_diff_bp already measures) directly into the dividend
            # estimate, contaminating it with a different economic effect.
            # For "curve_df" the forward was built WITH the curve's discount factor,
            # so the implied yield must use it too (it then absorbs borrow and the
            # financing basis, which is what a single-name carry is).
            df_for_q = df_curve if fit.method == "curve_df" else fit.discount_factor_implied
            q_impl = implied_dividend_yield(fit.forward, df_for_q, spot, T)
            flag_r2 = fit.r_squared < R2_FLAG_THRESHOLD
            flag_bp = abs(diff_bp) > DISCOUNT_FACTOR_BP_FLAG_THRESHOLD
            reliable = is_reliable_forward(
                fit.r_squared, diff_bp, method=fit.method, dispersion_bp=fit.dispersion_bp
            )

            if fit.method == "curve_df" and not reliable:
                result.flagged.append(
                    f"{underlying} {expiry_date}: strikes disagree on the forward by "
                    f"{fit.dispersion_bp:.0f}bp (limit {RELIABLE_DISPERSION_BP:.0f}bp), "
                    f"n={fit.n_strikes}"
                )
            elif fit.method != "curve_df" and (flag_r2 or flag_bp):
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
                    "method": fit.method,
                    "dispersion_bp": fit.dispersion_bp,
                    "reliable": reliable,
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
