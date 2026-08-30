"""Daily mark-to-model for the portfolio defined in `configs/portfolio.yaml`
(README Step 7). Dispatches each position to the pricer its `type` needs
(Steps 5-6 built all of them already) and aggregates Greeks by underlying,
expiry bucket, and moneyness bucket.

Every pricer here already existed before this module — Step 7 is integration,
not new pricing math. The one piece of genuinely new logic is pricing a
position whose expiry doesn't land exactly on a calibrated `vol_surface`
pillar: `vol/local_vol.py`'s T-interpolation (built for Dupire in Step 6.1)
already evaluates total variance at an arbitrary query T, so vanilla legs reuse
`local_variance_at(...).w` directly; `pricing/varswap.py`'s
`fair_variance_strike_from_w_func` (added in this step) generalises the
variance-swap replication the same way. Barrier and autocallable positions
don't need this at all — `build_local_vol_grid` already interpolates across
pillars internally for any (S, t) query.

**Deliberate, documented scope limits for this step (not a business decision,
an engineering one — flagged same as every prior step's honest simplifications):**
- Greeks for the two Monte Carlo-priced structures (barrier, autocall) are
  limited to delta/gamma/vega — theta/rho/vanna/volga would each need another
  bump-and-reval MC pass, and the README's own Step 6 risk-characteristics
  discussion for both instruments centres on exactly delta/gamma/vega anyway.
- The variance swap position (P006) has no trade-inception strike in its own
  config (only a `vega_notional`) — marked as if trading AT today's own fair
  strike (price = 0 by construction, a fresh swap has no value at its own fair
  strike) with vega reported directly as `vega_notional` (that is literally
  what a variance swap's vega notional means: P&L per one point of realised-
  vs-fair vol). Delta/gamma/theta/rho/vanna/volga are reported as 0 for this
  position, not because they're truly zero, but because a variance swap's risk
  is conventionally expressed in vega-notional terms directly, not decomposed
  through a strike the config doesn't specify.

Split into `load_market_state` (query the curated store once) and `mark_with_state`
(price every position given a market state) so Step 11's stress testing can load
today's real market ONCE and reprice under many `MarketShock`s without re-querying
the store or rebuilding local-vol grids from scratch each time. `mark_portfolio`
is the convenience wrapper that does both for the plain, unshocked daily mark.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import PORTFOLIO_MARKS_REQUIRED_NOT_NULL, PORTFOLIO_MARKS_SCHEMA, validate
from eqdrisk.marketdata.calendar import year_fraction
from eqdrisk.marketdata.curve import Curve, bootstrap_curve
from eqdrisk.marketdata.forward import ForwardCurve, build_forward_curve
from eqdrisk.portfolio.schema import (
    AutocallPosition,
    BarrierPosition,
    EquityPosition,
    Portfolio,
    VanillaPosition,
    VarSwapPosition,
)
from eqdrisk.pricing.autocallable import AutocallableSpec, autocallable_greeks
from eqdrisk.pricing.barrier_mc import down_and_in_put_greeks
from eqdrisk.pricing.blackscholes import compute_greeks
from eqdrisk.pricing.varswap import fair_variance_strike_from_w_func
from eqdrisk.stress.shock import MarketShock, shock_local_vol_grid, shocked_spot, shocked_w
from eqdrisk.vol.implied import EXTREME_K_MULTIPLE
from eqdrisk.vol.local_vol import LocalVolGrid, build_local_vol_grid

EXPIRY_BUCKETS = [
    (0.25, "0-3m"),
    (0.5, "3-6m"),
    (1.0, "6-12m"),
    (2.0, "1-2y"),
    (float("inf"), "2y+"),
]
MONEYNESS_BUCKETS = [
    (-0.2, "deep_otm_put"),
    (-0.05, "otm_put"),
    (0.05, "atm"),
    (0.2, "otm_call"),
    (float("inf"), "deep_otm_call"),
]
STRUCTURED_MONEYNESS_LABEL = "structured"

# Daily-run MC settings — a fraction of Step 6's own exhaustive validation-run
# path counts, since this runs once per position per day, not as a one-off
# convergence study. Documented tradeoff: more MC noise in the stderr column
# than Step 6's own tests carry, in exchange for a portfolio run that finishes
# in seconds, not minutes.
MC_N_PATHS = 50_000
BARRIER_N_STEPS = 64
AUTOCALL_N_STEPS_PER_PERIOD = 8
MC_SEED = 12345


@dataclass
class MCSettings:
    """Overridable MC cost knobs — `mark_with_state`'s default (`None`) reuses
    the plain daily-mark constants above unchanged. Step 11.2's hypothetical
    grid needs a much cheaper setting (42 full reprices, not one), so it builds
    its own `MCSettings` rather than paying Step 7's per-mark cost 42 times."""

    n_paths: int = MC_N_PATHS
    barrier_n_steps: int = BARRIER_N_STEPS
    autocall_n_steps_per_period: int = AUTOCALL_N_STEPS_PER_PERIOD
    seed: int = MC_SEED


def _expiry_bucket(T: float) -> str:
    for threshold, label in EXPIRY_BUCKETS:
        if T <= threshold:
            return label
    return EXPIRY_BUCKETS[-1][1]


def _moneyness_bucket(k: float | None) -> str:
    if k is None:
        return STRUCTURED_MONEYNESS_LABEL
    for threshold, label in MONEYNESS_BUCKETS:
        if k <= threshold:
            return label
    return MONEYNESS_BUCKETS[-1][1]


@dataclass
class PositionMark:
    position_id: str
    type: str
    underlying: str
    expiry: dt.date
    T: float
    price: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    k: float | None = None
    stderr: float | None = None
    note: str | None = None

    @property
    def expiry_bucket(self) -> str:
        return _expiry_bucket(self.T)

    @property
    def moneyness_bucket(self) -> str:
        return _moneyness_bucket(self.k)


@dataclass
class PortfolioMarkResult:
    asof: dt.date
    marks: list[PositionMark] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def total_value(self) -> float:
        return sum(m.price for m in self.marks)

    def aggregate(self, by: str) -> pd.DataFrame:
        """`by` in {"underlying", "expiry_bucket", "moneyness_bucket"}."""
        rows = [
            {
                "group": getattr(m, by),
                "price": m.price,
                "delta": m.delta,
                "gamma": m.gamma,
                "vega": m.vega,
                "theta": m.theta,
                "rho": m.rho,
                "vanna": m.vanna,
                "volga": m.volga,
            }
            for m in self.marks
        ]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "group",
                    "price",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "rho",
                    "vanna",
                    "volga",
                ]
            )
        return pd.DataFrame(rows).groupby("group", as_index=False).sum()

    def render(self) -> str:
        lines = [f"Portfolio mark — {self.asof}", f"  total value: {self.total_value():,.2f}"]
        for axis in ("underlying", "expiry_bucket", "moneyness_bucket"):
            lines.append(f"  by {axis}:")
            for _, row in self.aggregate(axis).iterrows():
                lines.append(
                    f"    {row['group']}: price={row['price']:,.0f} delta={row['delta']:,.1f} "
                    f"gamma={row['gamma']:,.2f} vega={row['vega']:,.1f}"
                )
        for m in self.marks:
            if m.note:
                lines.append(f"  {m.position_id}: {m.note}")
        for pos_id, reason in self.skipped.items():
            lines.append(f"  SKIPPED {pos_id}: {reason}")
        return "\n".join(lines)


@dataclass
class MarketState:
    """Everything a mark needs from the curated store, loaded ONCE — Step 11's
    stress testing loads this once per `asof` and reprices it under many
    `MarketShock`s, rather than re-querying the store or rebuilding local-vol
    grids (expensive — Step 6.1's own T-interpolation) from scratch per shock."""

    curve: Curve
    spot: dict[str, float]
    surface: dict[str, pd.DataFrame]
    forward_curve: dict[str, ForwardCurve]
    grids: dict[str, LocalVolGrid]


def load_market_state(cfg: BaseConfig, asof: dt.date, portfolio: Portfolio) -> MarketState | None:
    """Returns None if there's no curated rates data at all for this date —
    nothing downstream is priceable without a discount curve."""
    curated_root = Path(cfg.paths.curated)
    curves_date = store.latest_available_date(curated_root / "curves", asof)
    if curves_date is None:
        return None
    rates = store.query(
        f"SELECT * FROM curves WHERE asof_date = DATE '{curves_date.isoformat()}'",
        views={"curves": str(curated_root / "curves")},
    ).to_pandas()
    curve = bootstrap_curve(rates)

    underlyings_root = curated_root / "underlyings"
    vol_surface_root = curated_root / "vol_surface"
    forwards_root = curated_root / "forwards"

    underlyings = sorted({p.underlying for p in portfolio.positions})
    spot: dict[str, float] = {}
    surface: dict[str, pd.DataFrame] = {}
    forward_curve: dict[str, ForwardCurve] = {}
    for u in underlyings:
        if underlyings_root.exists() and any(underlyings_root.rglob("*.parquet")):
            spot_df = store.query(
                f"SELECT close FROM u WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{u}'",
                views={"u": str(underlyings_root)},
            ).to_pandas()
            if not spot_df.empty:
                spot[u] = float(spot_df["close"].iloc[0])

        if vol_surface_root.exists() and any(vol_surface_root.rglob("*.parquet")):
            surf = store.query(
                f"SELECT * FROM vs WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{u}'",
                views={"vs": str(vol_surface_root)},
            ).to_pandas()
            if not surf.empty:
                surface[u] = surf

        if forwards_root.exists() and any(forwards_root.rglob("*.parquet")):
            fwd = store.query(
                f"SELECT * FROM fwd WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{u}'",
                views={"fwd": str(forwards_root)},
            ).to_pandas()
            if not fwd.empty:
                forward_curve[u] = build_forward_curve(fwd)

    # One LocalVolGrid per underlying that actually needs Monte Carlo (barrier/autocall),
    # built once and reused across that underlying's MC-priced positions.
    mc_underlyings = {
        p.underlying
        for p in portfolio.positions
        if isinstance(p, (BarrierPosition, AutocallPosition))
    }
    grids: dict[str, LocalVolGrid] = {}
    for u in mc_underlyings:
        if u not in spot or u not in surface:
            continue
        s0 = spot[u]
        max_T = max(
            year_fraction(asof, p.expiry, cfg.daycount)
            for p in portfolio.positions
            if p.underlying == u and isinstance(p, (BarrierPosition, AutocallPosition))
        )
        s_grid = np.linspace(max(1.0, s0 * 0.1), s0 * 3.0, 100)
        t_grid = np.linspace(0.01, max_T, 60)
        grid = build_local_vol_grid(surface[u], forward_curve[u], s_grid, t_grid)
        if grid is not None:
            grids[u] = grid

    return MarketState(
        curve=curve, spot=spot, surface=surface, forward_curve=forward_curve, grids=grids
    )


def mark_with_state(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio: Portfolio,
    state: MarketState,
    shock: MarketShock | dict[str, MarketShock] | None = None,
    mc_settings: MCSettings | None = None,
) -> PortfolioMarkResult:
    """Price every position given an already-loaded `MarketState`, optionally
    under a `MarketShock` — either ONE shock applied to every underlying
    (11.2's hypothetical grid: the same spot/vol move everywhere) or a
    per-underlying `dict` (11.1's historical replays: SPX, AAPL, NVDA... each
    really moved by a different amount during a given episode, and using each
    name's own real historical move is the whole point of a *historical*
    replay rather than a hypothetical one). Default: a no-op, i.e. today's
    real, unshocked marks. `mc_settings` overrides the MC cost knobs for
    barrier/autocall positions (default `None` reuses the plain daily-mark
    constants) — Step 11.2's 42-cell grid needs a much cheaper setting than
    one daily mark does. Does NOT persist — only `mark_portfolio`'s own plain
    daily mark does; shocked/scenario marks are Step 11's concern, not this
    table's.
    """
    mc = mc_settings if mc_settings is not None else MCSettings()
    shock_map = shock if isinstance(shock, dict) else {}
    uniform_shock = shock if isinstance(shock, MarketShock) else MarketShock()

    def shock_for(underlying: str) -> MarketShock:
        return shock_map.get(underlying, uniform_shock)

    result = PortfolioMarkResult(asof=asof)

    for p in portfolio.positions:
        u = p.underlying
        if u not in state.spot or u not in state.surface or u not in state.forward_curve:
            result.skipped[p.id] = f"no spot/surface/forward data for {u} on {asof}"
            continue
        position_shock = shock_for(u)

        try:
            if isinstance(p, VanillaPosition):
                mark = _mark_vanilla(
                    p,
                    asof,
                    cfg,
                    state.spot[u],
                    state.surface[u],
                    state.forward_curve[u],
                    state.curve,
                    position_shock,
                )
            elif isinstance(p, EquityPosition):
                mark = PositionMark(
                    position_id=p.id,
                    type="equity",
                    underlying=u,
                    expiry=asof,
                    T=0.0,
                    price=p.qty * shocked_spot(state.spot[u], position_shock),
                    delta=p.qty,
                )
            elif isinstance(p, VarSwapPosition):
                mark = _mark_varswap(
                    p,
                    asof,
                    cfg,
                    state.surface[u],
                    state.forward_curve[u],
                    state.curve,
                    position_shock,
                )
            elif isinstance(p, BarrierPosition):
                if u not in state.grids:
                    result.skipped[p.id] = f"no local-vol grid available for {u}"
                    continue
                mark = _mark_barrier(
                    p,
                    asof,
                    cfg,
                    state.spot[u],
                    state.grids[u],
                    state.curve,
                    state.forward_curve[u],
                    position_shock,
                    mc,
                )
            elif isinstance(p, AutocallPosition):
                if u not in state.grids:
                    result.skipped[p.id] = f"no local-vol grid available for {u}"
                    continue
                mark = _mark_autocall(
                    p,
                    asof,
                    cfg,
                    state.spot[u],
                    state.grids[u],
                    state.curve,
                    state.forward_curve[u],
                    position_shock,
                    mc,
                )
            else:  # pragma: no cover - exhaustive over the discriminated union
                raise TypeError(f"unknown position type: {p!r}")
        except ValueError as exc:
            result.skipped[p.id] = str(exc)
            continue

        result.marks.append(mark)

    return result


def mark_portfolio(cfg: BaseConfig, asof: dt.date, portfolio_path: str) -> PortfolioMarkResult:
    portfolio = Portfolio.from_yaml(portfolio_path)
    state = load_market_state(cfg, asof, portfolio)
    if state is None:
        result = PortfolioMarkResult(asof=asof)
        result.skipped["_all_"] = "no curated rates available"
        return result

    result = mark_with_state(cfg, asof, portfolio, state)
    if result.marks:
        _persist(result, Path(cfg.paths.curated))
    return result


def _mark_vanilla(
    p: VanillaPosition,
    asof: dt.date,
    cfg: BaseConfig,
    spot: float,
    surface: pd.DataFrame,
    fwd_curve: ForwardCurve,
    curve: Curve,
    shock: MarketShock,
) -> PositionMark:
    T = year_fraction(asof, p.expiry, cfg.daycount)
    spot = shocked_spot(spot, shock)
    forward = fwd_curve.forward(T) * (1 + shock.spot_shock_pct)  # same carry, shocked spot
    discount_factor = curve.discount_factor(T)
    k = float(np.log(p.strike / forward))
    w = shocked_w(surface, k, T, shock)
    sigma = float(np.sqrt(max(w, 1e-12) / T))

    g = compute_greeks(p.cp == "C", forward, p.strike, T, sigma, discount_factor, spot)
    return PositionMark(
        position_id=p.id,
        type="vanilla",
        underlying=p.underlying,
        expiry=p.expiry,
        T=T,
        price=p.qty * g.price,
        delta=p.qty * g.delta_spot,
        gamma=p.qty * g.gamma_spot,
        vega=p.qty * g.vega,
        theta=p.qty * g.theta,
        rho=p.qty * g.rho,
        vanna=p.qty * g.vanna_spot,
        volga=p.qty * g.volga,
        k=k,
    )


def _mark_varswap(
    p: VarSwapPosition,
    asof: dt.date,
    cfg: BaseConfig,
    surface: pd.DataFrame,
    fwd_curve: ForwardCurve,
    curve: Curve,
    shock: MarketShock,
) -> PositionMark:
    T = year_fraction(asof, p.expiry, cfg.daycount)
    forward = fwd_curve.forward(T) * (1 + shock.spot_shock_pct)
    discount_factor = curve.discount_factor(T)

    def w_func(k: np.ndarray) -> np.ndarray:
        return np.array([shocked_w(surface, float(kk), T, shock) for kk in np.atleast_1d(k)])

    atm_iv = float(np.sqrt(max(shocked_w(surface, 0.0, T, shock), 1e-12) / T))
    k_cap = EXTREME_K_MULTIPLE * atm_iv * np.sqrt(T)
    fair_var = fair_variance_strike_from_w_func(w_func, forward, T, discount_factor, -k_cap, k_cap)
    fair_strike_pct = float(np.sqrt(max(fair_var, 0.0)) * 100)

    return PositionMark(
        position_id=p.id,
        type="varswap",
        underlying=p.underlying,
        expiry=p.expiry,
        T=T,
        price=0.0,
        vega=p.vega_notional,
        k=None,
        note=f"fair_strike={fair_strike_pct:.2f}% (marked as a fresh swap at its own fair strike)",
    )


def _implied_q(spot: float, forward: float, r: float, T: float) -> float:
    """Back out a continuous dividend yield consistent with the day's own
    (interpolated) forward curve: F(0,T) = S0*exp((r-q)T) => q = r - ln(F/S0)/T.
    Lets the barrier/autocall Monte Carlo (which works in (S0, r, q) space, not
    Black-76's (F, discount_factor) space) stay consistent with the SAME forward
    curve the vanilla legs price off, without a separate q-interpolation."""
    return r - np.log(forward / spot) / T


def _mark_barrier(
    p: BarrierPosition,
    asof: dt.date,
    cfg: BaseConfig,
    spot: float,
    grid: LocalVolGrid,
    curve: Curve,
    fwd_curve: ForwardCurve,
    shock: MarketShock,
    mc_settings: MCSettings,
) -> PositionMark:
    T = year_fraction(asof, p.expiry, cfg.daycount)
    spot = shocked_spot(spot, shock)
    forward = fwd_curve.forward(T) * (1 + shock.spot_shock_pct)
    r = curve.zero_rate(T)
    q = _implied_q(spot, forward, r, T)
    grid = shock_local_vol_grid(grid, fwd_curve, shock)
    greeks = down_and_in_put_greeks(
        spot,
        p.strike,
        p.barrier,
        T,
        grid,
        r,
        q,
        mc_settings.n_paths,
        mc_settings.barrier_n_steps,
        mc_settings.seed,
    )
    return PositionMark(
        position_id=p.id,
        type="barrier",
        underlying=p.underlying,
        expiry=p.expiry,
        T=T,
        price=p.qty * greeks.price,
        delta=p.qty * greeks.delta,
        gamma=p.qty * greeks.gamma,
        vega=p.qty * greeks.vega,
        k=float(np.log(p.strike / spot)),
    )


def _mark_autocall(
    p: AutocallPosition,
    asof: dt.date,
    cfg: BaseConfig,
    spot: float,
    grid: LocalVolGrid,
    curve: Curve,
    fwd_curve: ForwardCurve,
    shock: MarketShock,
    mc_settings: MCSettings,
) -> PositionMark:
    T = year_fraction(asof, p.expiry, cfg.daycount)
    spot = shocked_spot(spot, shock)
    forward = fwd_curve.forward(T) * (1 + shock.spot_shock_pct)
    r = curve.zero_rate(T)
    q = _implied_q(spot, forward, r, T)
    grid = shock_local_vol_grid(grid, fwd_curve, shock)

    obs_times = _quarterly_obs_times(asof, p.expiry, cfg.daycount)
    spec = AutocallableSpec(
        notional=p.notional,
        autocall_barrier=p.autocall_barrier,
        coupon_barrier=p.coupon_barrier,
        put_barrier=p.put_barrier,
        coupon_rate=p.coupon,
        obs_times=obs_times,
    )
    greeks = autocallable_greeks(
        spec,
        spot,
        grid,
        r,
        q,
        mc_settings.n_paths,
        mc_settings.autocall_n_steps_per_period,
        mc_settings.seed,
    )
    return PositionMark(
        position_id=p.id,
        type="autocall",
        underlying=p.underlying,
        expiry=p.expiry,
        T=T,
        price=greeks.price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        k=None,
    )


def _quarterly_obs_times(asof: dt.date, expiry: dt.date, daycount: str) -> np.ndarray:
    """Quarterly observation dates counting BACK from `expiry` (calendar-correct
    3-month steps) until reaching a date on or before `asof`, then converted to
    year-fractions from `asof`. `next_power_of_two` in the MC engine tolerates
    a non-power-of-two observation count fine (it rounds the total step count,
    not the observation count) — no alignment constraint here."""
    dates = []
    d = pd.Timestamp(expiry)
    while d.date() > asof:
        dates.append(d.date())
        d = d - pd.DateOffset(months=3)
    dates.reverse()
    return np.array([year_fraction(asof, d, daycount) for d in dates])


def _persist(result: PortfolioMarkResult, curated_root: Path) -> None:
    rows = [
        {
            "asof_date": result.asof,
            "position_id": m.position_id,
            "type": m.type,
            "underlying": m.underlying,
            "expiry": m.expiry,
            "T": m.T,
            "price": m.price,
            "delta": m.delta,
            "gamma": m.gamma,
            "vega": m.vega,
            "theta": m.theta,
            "rho": m.rho,
            "vanna": m.vanna,
            "volga": m.volga,
            "k": m.k,
            "expiry_bucket": m.expiry_bucket,
            "moneyness_bucket": m.moneyness_bucket,
            "stderr": m.stderr,
            "note": m.note,
        }
        for m in result.marks
    ]
    table = validate(pd.DataFrame(rows), PORTFOLIO_MARKS_SCHEMA, PORTFOLIO_MARKS_REQUIRED_NOT_NULL)
    store.write_partitioned(table, curated_root / "portfolio_marks", ["asof_date"])
