"""Daily P&L explain (README Step 12): decompose one day's ACTUAL (full-reval)
portfolio P&L into a fixed-order Greek-attributable waterfall — time -> rates/
divs -> spot -> vol, per the README's own mandated convention — reporting the
residual (actual minus explained) at each step, not just the total.

Unlike Step 11's stress testing (a HYPOTHETICAL percentage shock applied to one
real day), this needs TWO real, already-observed market states — the whole
point is explaining what the market actually did. Four intermediate states are
built by mixing fields from `load_market_state(day0)` and `load_market_state(day1)`
in the mandated order; at each step, the ACTUAL P&L (full reval, exact) is
compared against an EXPLAINED P&L (that step's Greeks, evaluated at the START
of the step, times that step's own driving-variable change).

**A real, non-obvious property of this design, not an oversight:** using
per-step, start-of-step Greeks (rather than one global Taylor expansion at the
very first state) means the sequential order itself absorbs cross-terms like
vanna (delta-vol interaction) into whichever step comes LATER — e.g. the vol
step's vega is evaluated AFTER the spot step has already moved the spot level,
so it already reflects some of what a separate vanna term would otherwise
capture. This is exactly the README's own "P&L explain is path-dependent in the
order you apply moves" point (12.2), made concrete rather than just stated.

**One deliberate, documented simplification:** this project's forward is a
jointly-fitted regression output (Step 2), not a closed-form S*exp((r-q)T) that
can be decomposed into rate/dividend/spot contributions after the fact. So the
"rates/divs" step moves ONLY the discount curve (a real, measurable rho
effect); the forward curve (which embeds that day's own carry) moves together
with spot in the next step. Consequence: `dividend_rho` never has a driving
variable to apply against in this decomposition and is not used — any real
carry P&L shows up inside the spot bucket instead.

**The vanillas' spot move is the forward's, not the close's** (2026-11-14,
planning/spx_forward_basis_plan.md): a vanilla is priced off the forward, and
SPX options trade to 16:15 while the cash index closes at 16:00, so the
option-implied spot runs -45bp to +24bp from the close day to day. Measured on
the close, P002 (deep in-the-money SPX call) carried that basis as residual,
correlation 0.997 over 36 day pairs. The basis is printed per underlying instead.

**MC exotics' carry and smile shape are explained by revaluation** (2026-11-25,
planning/p008_alert_days_plan.md): two more states split the spot step into pure
spot (the forward curve moved with spot alone) plus the change in the forward's
carry, and the vol step into the ATM level by maturity plus the smile's shape.
The Greeks answer for pure spot and level; carry and shape, which no Greek
covers, are full-revaluation P&L, counted as explained and stored apart
(`reval_pnl`, `carry_pnl`, `shape_pnl`) so no report mistakes them for Greek
predictions. On P008's five alert days nearly all the residual was these two.

**Positions with an incomplete Greek set contribute their own honest gap, not a
fabricated explanation:** MC-priced positions (barrier, autocall) started with
delta/gamma/vega only (Step 7's scope limit); vanna/volga, theta and rho were
added as the explain found each gap, rho last (2026-11-13: until then every
curve move was all residual in the rates step). Where a step's Greek doesn't
exist for a position type, `explained` is 0 for that position in that step
(not "explained = actual", which would silently hide the gap), so the residual
is that step's entire actual P&L. This is precisely the README's own 12.4
point made concrete: "if your residual spikes, your Greeks are incomplete."
Variance swap positions are excluded from the Greek-based calculation entirely
— Step 7 marks them at exactly 0 by construction (a fresh swap at today's own
fair strike), so there is no real P&L to explain for them in the first place.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    PNL_EXPLAIN_BY_POSITION_REQUIRED_NOT_NULL,
    PNL_EXPLAIN_BY_POSITION_SCHEMA,
    PNL_EXPLAIN_REQUIRED_NOT_NULL,
    PNL_EXPLAIN_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import last_n_trading_days
from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.mark import (
    VEGA_BAND_EDGES,
    MarketState,
    MCSettings,
    PortfolioMarkResult,
    PositionMark,
    build_grids_for_state,
    load_market_state,
    mark_with_state,
)
from eqdrisk.portfolio.quote_noise import load_quote_noise
from eqdrisk.portfolio.schema import AutocallPosition, Portfolio
from eqdrisk.vol.local_vol import local_variance_at

STEP_NAMES: tuple[str, ...] = ("time", "rates_divs", "spot", "vol")

# The daily mark's own settings (same paths and seeds, so common random numbers
# hold across the five states) plus band vegas for the autocallable.
EXPLAIN_VOL_STATE_SETTINGS = MCSettings(vega_bands=VEGA_BAND_EDGES)

# README 12.3's own suggested number ("e.g. 5bp of NAV") — applied to both the
# total residual and, per position, the same convention (no evidence anywhere in
# this project that a different per-position number was intended).
RESIDUAL_ALERT_THRESHOLD_BP = 5.0

# Carry and smile-shape revaluation lines for MC exotics (2026-11-25,
# planning/p008_alert_days_plan.md). Two more states split the spot step into
# pure spot + the change in the forward's carry, and the vol step into the ATM
# level by maturity + the smile's shape. No Greek covers carry or shape, so
# each is explained by full revaluation and stored as `reval_pnl`; the Greeks
# answer for pure spot and level. On P008's five alert days the residual was
# almost all carry and shape (RMS $11,903 -> $1,138 without them), and the two
# offset on other days, so they are covered together. Price-only, on the daily
# mark's paths and seeds, so common random numbers hold with the other states.
REVAL_STATE_SETTINGS = MCSettings(greeks=False, exotic_theta=False)
MC_EXOTIC_TYPES = ("barrier", "autocall")


def _atm_iv(surface: pd.DataFrame, T: float) -> float | None:
    lv = local_variance_at(surface, 0.0, T)
    if lv is None:
        return None
    return float((max(lv.w, 1e-12) / T) ** 0.5)


def _vol_move(
    m: PositionMark, surface_prev: pd.DataFrame, surface_cur: pd.DataFrame
) -> float | None:
    """The day's ATM vol move where the position's vol exposure sits (its
    `vol_times`), measured only at maturities BOTH days' surfaces cover.

    Beyond its last pillar a surface is extrapolated, not observed, so when the
    long end appears or disappears between two days (thinly traded long-dated
    quotes passing the quality filters one day and not the next) an ATM vol read
    beyond it moves for no market reason: measured on real NVDA data under the
    old flat-total-variance extrapolation, a fake +12 vol-point move for the
    1.88y autocallable on 2026-09-03 and 09-17. Flat-implied-vol extrapolation
    (C4) shrinks this but two days can still extrapolate from different last
    pillars, so the move is measured only where both surfaces have data."""
    t_last = min(float(surface_prev["T"].max()), float(surface_cur["T"].max()))
    moves = []
    for t in m.vol_times or [m.T]:
        t_eff = min(t, t_last)
        a_prev, a_cur = _atm_iv(surface_prev, t_eff), _atm_iv(surface_cur, t_eff)
        if a_prev is None or a_cur is None:
            return None
        moves.append(a_cur - a_prev)
    return float(np.mean(moves)) if moves else None


def _forward_atm_iv(surface: pd.DataFrame, t_lo: float, t_hi: float) -> float | None:
    """ATM forward vol over [t_lo, t_hi) from ATM total variance; the implied vol
    at `t_hi` when the forward variance is not positive (calendar arbitrage)."""
    a_hi = _atm_iv(surface, t_hi)
    if a_hi is None:
        return None
    if t_lo <= 0.0:
        return a_hi
    a_lo = _atm_iv(surface, t_lo)
    if a_lo is None:
        return None
    fwd_var = (a_hi**2 * t_hi - a_lo**2 * t_lo) / (t_hi - t_lo)
    return float(fwd_var**0.5) if fwd_var > 0.0 else a_hi


def _band_vol_moves(
    m: PositionMark, surface_prev: pd.DataFrame, surface_cur: pd.DataFrame
) -> list[float] | None:
    """The day's ATM forward vol move over each VEGA_BAND_EDGES band, where both
    surfaces have data and the position is alive; a band entirely past that takes
    the ATM move at its end, as `_vol_move` does (its vega is then near zero)."""
    t_end = min(m.T, float(surface_prev["T"].max()), float(surface_cur["T"].max()))
    edges = (0.0, *VEGA_BAND_EDGES, np.inf)
    moves = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        t_hi = min(hi, t_end)
        t_lo = min(lo, t_hi)
        if t_hi - t_lo < 1e-6:
            a_prev, a_cur = _atm_iv(surface_prev, t_end), _atm_iv(surface_cur, t_end)
        else:
            a_prev = _forward_atm_iv(surface_prev, t_lo, t_hi)
            a_cur = _forward_atm_iv(surface_cur, t_lo, t_hi)
        if a_prev is None or a_cur is None:
            return None
        moves.append(a_cur - a_prev)
    return moves


def _banded_vol_explained(
    m: PositionMark, surface_prev: pd.DataFrame, surface_cur: pd.DataFrame
) -> float | None:
    """Each band's vega times its band's move, plus volga at the vega-weighted
    move (planning/p008_vol_weighting_plan.md). None without band vegas."""
    if not m.bucket_vega:
        return None
    moves = _band_vol_moves(m, surface_prev, surface_cur)
    if moves is None:
        return None
    first = sum(v * d for v, d in zip(m.bucket_vega, moves, strict=True))
    total = sum(m.bucket_vega)
    d_mean = first / total if total else float(np.mean(moves))
    return first + 0.5 * m.volga * d_mean**2


# Change in the surface's longest calibrated maturity (years) treated as a
# coverage change worth flagging for a position whose exposure extends beyond it.
COVERAGE_CHANGE_FLAG_YEARS = 0.1


def _coverage_note(
    m: PositionMark, surface_prev: pd.DataFrame, surface_cur: pd.DataFrame
) -> str | None:
    t_prev, t_cur = float(surface_prev["T"].max()), float(surface_cur["T"].max())
    if max(m.vol_times or [m.T]) <= min(t_prev, t_cur):
        return None
    if abs(t_cur - t_prev) <= COVERAGE_CHANGE_FLAG_YEARS:
        return None
    return (
        f"surface coverage changed for {m.underlying} (longest maturity {t_prev:.2f}y -> "
        f"{t_cur:.2f}y); residual may reflect data coverage, not market moves"
    )


def _carry_held_forward(fc: ForwardCurve, spot_prev: float, spot_cur: float) -> ForwardCurve:
    """`fc` moved with spot alone: every forward scaled by spot_cur / spot_prev,
    so the carry (the forward against spot) is the previous day's."""
    return ForwardCurve(
        pillar_T=fc.pillar_T,
        pillar_log_forward=fc.pillar_log_forward + np.log(spot_cur / spot_prev),
        long_end_carry=fc.long_end_carry,
    )


def _ssvi_as_raw_svi(row: pd.Series) -> pd.Series:
    """An SSVI slice as the identical raw-SVI slice (Gatheral and Jacquier 2014),
    so its level can move by its `a` alone. SSVI slices are never wing-guarded,
    so the quoted range is dropped and the guard stays off."""
    row = row.copy()
    if row["model"] != "SSVI":
        return row
    theta, rho, eta = float(row["theta"]), float(row["rho"]), float(row["eta"])
    phi = eta / np.sqrt(theta)
    row["model"] = "SVI"
    row["a"] = theta / 2 * (1 - rho**2)
    row["b"] = theta * phi / 2
    row["m"] = -rho / phi
    row["sigma"] = np.sqrt(1 - rho**2) / phi
    row["k_min"] = np.nan
    row["k_max"] = np.nan
    return row


def _relevelled_surface(
    surface_prev: pd.DataFrame, surface_cur: pd.DataFrame
) -> pd.DataFrame | None:
    """`surface_prev`'s smiles, each shifted in total variance by the change in
    the T-smoothed ATM total variance at its maturity: the current day's ATM
    level by maturity (what the band vegas measure) with the previous day's
    shape. None when either surface has too few pillars to read ATM."""
    rows = []
    for _, row in surface_prev.iterrows():
        T = float(row["T"])
        lv_prev, lv_cur = (
            local_variance_at(surface_prev, 0.0, T),
            local_variance_at(surface_cur, 0.0, T),
        )
        if lv_prev is None or lv_cur is None:
            return None
        raw = _ssvi_as_raw_svi(row)
        raw["a"] = float(raw["a"]) + (lv_cur.w - lv_prev.w)
        rows.append(raw)
    return pd.DataFrame(rows).reset_index(drop=True)


def _marks_by_id(result: PortfolioMarkResult) -> dict[str, PositionMark]:
    return {m.position_id: m for m in result.marks}


def _forward_spot_move(m: PositionMark, state_prev: MarketState, state_cur: MarketState) -> float:
    """The spot move a vanilla's price sees: its forward's relative move at its
    own maturity, in spot units, so the sticky-moneyness delta (a bump of spot
    and forward together) applies unchanged. The close's move without a forward."""
    u = m.underlying
    d_close = state_cur.spot[u] - state_prev.spot[u]
    if u not in state_prev.forward_curve or u not in state_cur.forward_curve:
        return d_close
    f_prev = state_prev.forward_curve[u].forward(m.T)
    f_cur = state_cur.forward_curve[u].forward(m.T)
    return state_prev.spot[u] * (f_cur / f_prev - 1.0)


def _spot_taylor(m: PositionMark, d_spot: float) -> float:
    """Sticky-moneyness delta/gamma where available: this explain's spot step
    rebuilds the vol surface around the new spot from the same log-moneyness
    smile, so the Greeks must move vol the same way. With sticky-strike Greeks
    P008's spot-step residual averaged 19.3bp over 23 day pairs; with
    sticky-moneyness Greeks, 7.1bp (planning/revaluation_ladder_plan.md)."""
    delta = m.delta_sm if m.delta_sm is not None else m.delta
    gamma = m.gamma_sm if m.gamma_sm is not None else m.gamma
    return delta * d_spot + 0.5 * gamma * d_spot**2


def _basis_bp(state: MarketState, u: str) -> float:
    """The option-implied spot (the forward at the front pillar) against the close."""
    return 10_000.0 * (state.forward_curve[u].forward(0.0) / state.spot[u] - 1.0)


def _explained_for_step(
    step: str,
    m_prev: PositionMark,
    days_elapsed: int,
    state_prev: MarketState,
    state_cur: MarketState,
) -> float:
    if m_prev.type == "varswap":
        return 0.0  # always marked to 0 by construction (see module docstring) — nothing to explain

    if step == "time":
        # Exotic theta (one-day revaluation) since 2026-10-07; before, the time
        # step explained nothing for MC exotics.
        explains_time = m_prev.type in ("vanilla", "barrier", "autocall")
        return m_prev.theta * days_elapsed if explains_time else 0.0

    if step == "rates_divs":
        # MC exotics have a rho since 2026-11-13 (planning/same_day_rates_plan.md).
        if m_prev.type not in ("vanilla", "barrier", "autocall"):
            return 0.0
        r_prev = state_prev.curve.zero_rate(m_prev.T)
        r_cur = state_cur.curve.zero_rate(m_prev.T)
        return m_prev.rho * (r_cur - r_prev)

    if step == "spot":
        u = m_prev.underlying
        if u not in state_prev.spot or u not in state_cur.spot:
            return 0.0
        if m_prev.type == "vanilla":
            return _spot_taylor(m_prev, _forward_spot_move(m_prev, state_prev, state_cur))
        return _spot_taylor(m_prev, state_cur.spot[u] - state_prev.spot[u])

    if step == "vol":
        if m_prev.vega == 0.0 and m_prev.volga == 0.0:
            return 0.0  # no vol exposure at all (e.g. equity, T=0) -- nothing to look up
        u = m_prev.underlying
        if u not in state_prev.surface or u not in state_cur.surface:
            return 0.0
        banded = _banded_vol_explained(m_prev, state_prev.surface[u], state_cur.surface[u])
        if banded is not None:
            return banded
        d_sigma = _vol_move(m_prev, state_prev.surface[u], state_cur.surface[u])
        if d_sigma is None:
            return 0.0
        return m_prev.vega * d_sigma + 0.5 * m_prev.volga * d_sigma**2

    raise ValueError(f"unknown step: {step}")  # pragma: no cover - exhaustive over STEP_NAMES


@dataclass
class StepResult:
    step: str
    actual_pnl: float
    explained_pnl: float
    # Monte Carlo standard error of `residual` from the RQMC sub-runs of the
    # MC-priced positions; 0.0 when every position is closed-form.
    residual_se: float = 0.0
    # The part of `explained_pnl` measured by revaluation, not Greeks: the MC
    # exotics' carry (spot step) and smile shape (vol step).
    reval_pnl: float = 0.0

    @property
    def residual(self) -> float:
        return self.actual_pnl - self.explained_pnl


@dataclass
class PnLExplainResult:
    day0: dt.date
    day1: dt.date
    steps: list[StepResult] = field(default_factory=list)
    by_position_residual: dict[str, float] = field(default_factory=dict)
    by_position_residual_se: dict[str, float] = field(default_factory=dict)
    total_residual_se: float = 0.0
    # Quote noise (portfolio/quote_noise.py): sqrt(day0^2 + day1^2) of each MC
    # exotic's stored mark SD from bid-ask, only when both days were measured.
    by_position_quote_noise: dict[str, float] = field(default_factory=dict)
    total_quote_noise: float = 0.0
    # MC exotics' revaluation lines: the carry change inside the spot step and
    # the smile-shape change inside the vol step, both counted as explained.
    by_position_carry: dict[str, float] = field(default_factory=dict)
    by_position_shape: dict[str, float] = field(default_factory=dict)
    nav: float = 0.0
    breaches: list[str] = field(default_factory=list)
    # Residuals whose noise band (2 x MC SE and quote noise combined) is itself
    # wider than the alert threshold: neither an alert nor a clean bill of health.
    inconclusive: list[str] = field(default_factory=list)
    # Per-position caveats (e.g. surface coverage changed between the two days).
    notes: dict[str, str] = field(default_factory=dict)
    # Per underlying held in vanillas: the option-implied spot against the close
    # on both days, and what reading the spot step through the forward added.
    basis_lines: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def total_actual(self) -> float:
        return sum(s.actual_pnl for s in self.steps)

    def total_explained(self) -> float:
        return sum(s.explained_pnl for s in self.steps)

    def total_residual(self) -> float:
        return sum(s.residual for s in self.steps)

    def _bp(self, dollars: float) -> float:
        """`dollars` expressed in basis points of `nav` — 0.0 if NAV is zero
        (nothing to normalize against, not a divide-by-zero crash)."""
        return 0.0 if self.nav == 0.0 else 10_000.0 * dollars / self.nav

    def total_residual_bp(self) -> float:
        return self._bp(self.total_residual())

    def by_position_residual_bp(self) -> dict[str, float]:
        return {pid: self._bp(r) for pid, r in self.by_position_residual.items()}

    def render(self) -> str:
        lines = [f"P&L explain — {self.day0} -> {self.day1}  (NAV={self.nav:,.2f})"]
        for s in self.steps:
            lines.append(
                f"  {s.step:>10}: actual={s.actual_pnl:+,.2f}  explained={s.explained_pnl:+,.2f}"
                f"  residual={s.residual:+,.2f} +- {2 * s.residual_se:,.2f} (2 SE)"
                + (f"  [revaluation {s.reval_pnl:+,.2f}]" if s.reval_pnl else "")
            )
        lines.append(
            f"  {'TOTAL':>10}: actual={self.total_actual():+,.2f}  "
            f"explained={self.total_explained():+,.2f}  residual={self.total_residual():+,.2f}"
            f" +- {2 * self.total_residual_se:,.2f}"
            f"  ({self.total_residual_bp():+.1f} +- {self._bp(2 * self.total_residual_se):.1f}bp"
            f" of NAV, 2 SE)" + _quote_noise_text(self.total_quote_noise, self._bp)
        )
        for b in self.basis_lines:
            lines.append(f"  {b}")
        if self.by_position_residual:
            lines.append("  residual by position:")
            for pid, r in sorted(self.by_position_residual.items(), key=lambda kv: -abs(kv[1])):
                bp = self._bp(r)
                se = self.by_position_residual_se.get(pid, 0.0)
                lines.append(
                    f"    {pid}: {r:+,.2f} +- {2 * se:,.2f}  "
                    f"({bp:+.1f} +- {self._bp(2 * se):.1f}bp of NAV)"
                    + _quote_noise_text(self.by_position_quote_noise.get(pid, 0.0), self._bp)
                    + _reval_text(self.by_position_carry.get(pid), self.by_position_shape.get(pid))
                )
        for b in self.breaches:
            lines.append(f"  ALERT: {b}")
        for i in self.inconclusive:
            lines.append(f"  INCONCLUSIVE: {i}")
        for pid, note in sorted(self.notes.items()):
            lines.append(f"  NOTE {pid}: {note}")
        if not self.breaches and not self.inconclusive:
            lines.append(f"  no residual breaches (threshold {RESIDUAL_ALERT_THRESHOLD_BP:.0f}bp)")
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def _reval_text(carry: float | None, shape: float | None) -> str:
    if carry is None and shape is None:
        return ""
    parts = [
        f"{name} {v:+,.2f}"
        for name, v in (("carry", carry), ("smile shape", shape))
        if v is not None
    ]
    return "; explained by revaluation: " + ", ".join(parts)


def _quote_noise_text(quote_noise: float, to_bp: Callable[[float], float]) -> str:
    if quote_noise <= 0.0:
        return ""
    return f"; quote noise +- {2 * quote_noise:,.2f} ({to_bp(2 * quote_noise):.1f}bp, 2 SD)"


def _combine_quote_noise(
    marks: dict[str, PositionMark], day0_sd: dict[str, float], day1_sd: dict[str, float]
) -> dict[str, float]:
    """Per MC exotic, sqrt(day0^2 + day1^2) of its stored quote-noise SD. The two
    days' quotes are independent draws of bid-ask noise, and the residual carries
    both marks. A position missing either day gets none (not half a band)."""
    return {
        pid: float(np.hypot(day0_sd[pid], day1_sd[pid]))
        for pid, m in marks.items()
        if m.type in ("barrier", "autocall") and pid in day0_sd and pid in day1_sd
    }


def _mixed_state(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio: Portfolio,
    curve_from: MarketState,
    spot_from: MarketState,
    surface_from: MarketState,
    forward_from: MarketState,
) -> MarketState:
    grids = build_grids_for_state(
        cfg, asof, portfolio, spot_from.spot, surface_from.surface, forward_from.forward_curve
    )
    return MarketState(
        curve=curve_from.curve,
        spot=spot_from.spot,
        surface=surface_from.surface,
        forward_curve=forward_from.forward_curve,
        grids=grids,
    )


def _revaluation_states(
    cfg: BaseConfig,
    day1: dt.date,
    portfolio: Portfolio,
    state_day0: MarketState,
    state_day1: MarketState,
    state_3: MarketState,
) -> dict[str, tuple[MarketState, set[str]]]:
    """The mid states of the spot and vol steps, each with the underlyings it
    actually splits (an underlying it cannot split keeps the plain step).

    spot: day1 curve and spot, day0 surface, day0 forward curve moved with spot
    alone; the spot step's Greeks answer for state 2 -> this, carry for the rest.
    vol: state 3 with each underlying's day0 surface re-levelled to day1's ATM
    term structure; the band vegas answer for state 3 -> this, shape for the rest.
    """
    spot_fwd = dict(state_day0.forward_curve)
    spot_split = set()
    for u, fc in state_day0.forward_curve.items():
        if u in state_day0.spot and u in state_day1.spot:
            spot_fwd[u] = _carry_held_forward(fc, state_day0.spot[u], state_day1.spot[u])
            spot_split.add(u)
    level_surface = dict(state_3.surface)
    vol_split = set()
    for u, surf_prev in state_3.surface.items():
        if u not in state_day1.surface:
            continue
        relevelled = _relevelled_surface(surf_prev, state_day1.surface[u])
        if relevelled is not None:
            level_surface[u] = relevelled
            vol_split.add(u)

    def build(spot, surface, forward) -> MarketState:
        grids = build_grids_for_state(cfg, day1, portfolio, spot, surface, forward)
        return MarketState(
            curve=state_day1.curve, spot=spot, surface=surface, forward_curve=forward, grids=grids
        )

    return {
        "spot": (build(state_day1.spot, state_day0.surface, spot_fwd), spot_split),
        "vol": (build(state_3.spot, level_surface, state_3.forward_curve), vol_split),
    }


def missing_market_data(cfg: BaseConfig, asof: dt.date, portfolio: Portfolio) -> list[str]:
    """What `asof` lacks of its OWN market data for this book: spot for every
    underlying, plus forwards and a calibrated vol surface for every underlying
    that holds an option-like position. No fallback to an earlier date: rates
    may legitimately lag a day (`load_market_state` handles that), but a day
    without its own option data is not a day the book can be revalued on.

    Added 2026-10-04: on days the option-chain ingest failed, the explain used
    to run on a half-loaded market and persist junk rows (all-zero pairs, and
    one -$676K residual against a NAV of 0)."""
    curated = Path(cfg.paths.curated)
    need_spot = {p.underlying for p in portfolio.positions}
    need_options = {p.underlying for p in portfolio.positions if p.type != "equity"}

    have_spot: set[str] = set()
    spot_root = curated / "underlyings"
    if (spot_root / f"asof_date={asof.isoformat()}").exists():
        have_spot = set(
            store.query(
                f"SELECT DISTINCT underlying FROM t WHERE asof_date = DATE '{asof.isoformat()}'",
                views={"t": str(spot_root)},
            )
            .column("underlying")
            .to_pylist()
        )

    missing = [f"spot {u}" for u in sorted(need_spot - have_spot)]
    for table, label in (("forwards", "forwards"), ("vol_surface", "vol surface")):
        for u in sorted(need_options):
            partition = curated / table / f"asof_date={asof.isoformat()}" / f"underlying={u}"
            if not (partition.exists() and any(partition.glob("*.parquet"))):
                missing.append(f"{label} {u}")
    return missing


def _standard_error(per_batch: np.ndarray) -> float:
    """MC standard error of the mean of `per_batch` (randomized QMC sub-runs)."""
    if len(per_batch) < 2:
        return 0.0
    return float(np.std(per_batch, ddof=1) / np.sqrt(len(per_batch)))


def _batch_residuals(
    step: str,
    m_prev: PositionMark,
    m_cur: PositionMark,
    days_elapsed: int,
    state_prev: MarketState,
    state_cur: MarketState,
) -> np.ndarray | None:
    """One residual per RQMC sub-run, or `None` for a closed-form position. The
    explained P&L is the same formula as `_explained_for_step`, applied to each
    sub-run's own price and Greeks."""
    if not m_prev.batches or not m_cur.batches:
        return None
    out = []
    for i, (b_prev, b_cur) in enumerate(zip(m_prev.batches, m_cur.batches, strict=True)):
        sub_prev = dataclasses.replace(
            m_prev,
            price=b_prev["price"],
            delta=b_prev["delta"],
            gamma=b_prev["gamma"],
            delta_sm=b_prev.get("delta_sm"),
            gamma_sm=b_prev.get("gamma_sm"),
            vega=b_prev["vega"],
            vanna=b_prev["vanna"],
            volga=b_prev["volga"],
            rho=b_prev.get("rho", m_prev.rho),
            batches=None,
            bucket_vega=m_prev.bucket_vega_batches[i] if m_prev.bucket_vega_batches else None,
            bucket_vega_batches=None,
        )
        explained = _explained_for_step(step, sub_prev, days_elapsed, state_prev, state_cur)
        out.append((b_cur["price"] - b_prev["price"]) - explained)
    return np.array(out)


# How far back the daily run looks for a usable day0 across failed-ingest days.
MAX_DAY0_LOOKBACK_TRADING_DAYS = 10


def latest_complete_day_before(
    cfg: BaseConfig, day1: dt.date, portfolio: Portfolio
) -> dt.date | None:
    """The most recent trading day before `day1` with complete market data of
    its own, so a failed ingest day is explained ACROSS (e.g. 2026-09-14 ->
    09-16 around a lost 09-15) instead of costing two explains. `None` if none
    within `MAX_DAY0_LOOKBACK_TRADING_DAYS`."""
    candidates = last_n_trading_days(day1, MAX_DAY0_LOOKBACK_TRADING_DAYS + 1, cfg.calendar)
    for day in reversed([d for d in candidates if d < day1]):
        if not missing_market_data(cfg, day, portfolio):
            return day
    return None


def run_pnl_explain(
    cfg: BaseConfig, day0: dt.date, day1: dt.date, portfolio_path: str
) -> PnLExplainResult:
    """Persists only when both days have complete market data of their own and
    every position prices in all five states; otherwise reports why in
    `skipped["_all_"]` and writes nothing."""
    portfolio = Portfolio.from_yaml(portfolio_path)
    result = PnLExplainResult(day0=day0, day1=day1)
    gaps = {d: missing_market_data(cfg, d, portfolio) for d in (day0, day1)}
    if any(gaps.values()):
        result.skipped["_all_"] = "incomplete market data: " + "; ".join(
            f"{d} missing {', '.join(m)}" for d, m in gaps.items() if m
        )
        return result

    state_day0 = load_market_state(cfg, day0, portfolio)
    state_day1 = load_market_state(cfg, day1, portfolio)
    if state_day0 is None or state_day1 is None:
        result.skipped["_all_"] = "missing curated rates for day0 and/or day1"
        return result

    days_elapsed = (day1 - day0).days

    state_0 = state_day0
    state_1 = _mixed_state(cfg, day1, portfolio, state_day0, state_day0, state_day0, state_day0)
    state_2 = _mixed_state(cfg, day1, portfolio, state_day1, state_day0, state_day0, state_day0)
    state_3 = _mixed_state(cfg, day1, portfolio, state_day1, state_day1, state_day0, state_day1)
    state_4 = state_day1

    asofs = [day0, day1, day1, day1, day1]
    states = [state_0, state_1, state_2, state_3, state_4]
    # The vol step's Greeks are state 3's; only there are band vegas priced.
    settings = [None, None, None, EXPLAIN_VOL_STATE_SETTINGS, None]
    marks = [
        _marks_by_id(mark_with_state(cfg, asofs[i], portfolio, states[i], mc_settings=settings[i]))
        for i in range(5)
    ]

    # Revaluation lines (REVAL_STATE_SETTINGS): the spot step's carry and the vol
    # step's smile shape, for the underlyings where the split could be built.
    mid = _revaluation_states(cfg, day1, portfolio, state_day0, state_day1, states[3])
    mid_marks = {
        step: _marks_by_id(
            mark_with_state(cfg, day1, portfolio, st, mc_settings=REVAL_STATE_SETTINGS)
        )
        for step, (st, _) in mid.items()
    }

    # Every step must cover the SAME positions, or the step totals (and the
    # NAV-relative residual) mix different books. A position that fails to
    # price in any of the five states is left out of every step and reported.
    # The book is what was held on day0: a roll that starts by day1 is a new
    # trade with no day0 mark, so it is neither required nor explained.
    priced_all = set.intersection(*(set(m) for m in marks))
    unpriced = {p.id for p in portfolio.positions if lifecycle.is_active(p, day0)} - priced_all
    if unpriced:
        result.skipped["_all_"] = (
            f"positions not priced in every state: {', '.join(sorted(unpriced))}"
        )
        return result

    # RQMC sub-runs (see `portfolio.mark.rqmc_greeks`): each MC-priced position
    # carries one set of values per sub-run, on the same sub-run seeds in all
    # five states. Each sub-run gives its own residual; the reported residual is
    # their mean (exactly the residual of the averaged values) and their spread
    # gives its standard error. Closed-form positions contribute the same
    # residual to every sub-run.
    n_batches = max((len(m.batches) for m in marks[0].values() if m.batches), default=1)
    fixings = _fixing_events(cfg, portfolio, day0, day1, priced_all)
    position_residuals: dict[str, float] = {}
    position_batches: dict[str, np.ndarray] = {}

    for step_idx, step in enumerate(STEP_NAMES):
        marks_prev, marks_cur = marks[step_idx], marks[step_idx + 1]
        actual_total = 0.0
        explained_total = 0.0
        reval_total = 0.0
        step_batches = np.zeros(n_batches)
        state_mid, split_underlyings = mid.get(step, (None, set()))

        for pid in priced_all:
            m_prev, m_cur = marks_prev[pid], marks_cur[pid]
            actual = m_cur.price - m_prev.price
            settling = m_cur.status == lifecycle.SETTLED and m_prev.status != lifecycle.SETTLED
            fixing = step == "time" and pid in fixings
            if settling:
                # Expiry, autocall or maturity between the two days: the move to
                # the settled value is settlement P&L, not something the Greeks
                # should have predicted (planning/position_lifecycle_plan.md).
                explained = actual
                result.notes[pid] = f"settled: {m_cur.note}; {actual:+,.2f} booked as settlement"
            elif fixing:
                # A coupon fixing between the two days: every state after day0
                # carries day1's lifecycle status, so the outcome lands in the time
                # step, where theta cannot predict it (planning/fixing_event_explain_plan.md).
                explained = actual
                result.notes[pid] = (
                    f"{fixings[pid]}; {actual:+,.2f} booked as fixing in the time step"
                )
            split = (
                state_mid is not None
                and m_prev.type in MC_EXOTIC_TYPES
                and m_prev.underlying in split_underlyings
                and pid in mid_marks[step]
            )
            if not (settling or fixing) and split:
                # Greeks for the move up to the mid state (pure spot, or the ATM
                # level), revaluation for the rest (carry, or smile shape).
                m_mid = mid_marks[step][pid]
                assert state_mid is not None
                reval = m_cur.price - m_mid.price
                explained = (
                    _explained_for_step(step, m_prev, days_elapsed, states[step_idx], state_mid)
                    + reval
                )
                reval_total += reval
                lines = result.by_position_carry if step == "spot" else result.by_position_shape
                lines[pid] = reval
            elif not (settling or fixing):
                explained = _explained_for_step(
                    step, m_prev, days_elapsed, states[step_idx], states[step_idx + 1]
                )
            actual_total += actual
            explained_total += explained
            position_residuals[pid] = position_residuals.get(pid, 0.0) + (actual - explained)

            if settling or fixing:
                batches: np.ndarray | None = np.zeros(n_batches)
            elif split:
                # The revaluation part is exact per sub-run too, so each sub-run's
                # residual is its Greeks' miss on the move to the mid state.
                assert state_mid is not None
                batches = _batch_residuals(
                    step, m_prev, mid_marks[step][pid], days_elapsed, states[step_idx], state_mid
                )
            else:
                batches = _batch_residuals(
                    step, m_prev, m_cur, days_elapsed, states[step_idx], states[step_idx + 1]
                )
            per_batch = batches if batches is not None else np.full(n_batches, actual - explained)
            step_batches += per_batch
            position_batches[pid] = position_batches.get(pid, np.zeros(n_batches)) + per_batch

        result.steps.append(
            StepResult(
                step=step,
                actual_pnl=actual_total,
                explained_pnl=explained_total,
                residual_se=_standard_error(step_batches),
                reval_pnl=reval_total,
            )
        )

    for pid in priced_all:
        m = marks[3][pid]
        u = m.underlying
        if (
            m.type in ("vanilla", "barrier", "autocall")
            and m.status != lifecycle.SETTLED
            and u in states[3].surface
            and u in states[4].surface
        ):
            note = _coverage_note(m, states[3].surface[u], states[4].surface[u])
            if note:
                result.notes[pid] = "; ".join(n for n in (result.notes.get(pid), note) if n)

    result.basis_lines = _basis_lines(marks[2], marks[3], states, priced_all)
    result.by_position_residual_se = {
        pid: _standard_error(v) for pid, v in position_batches.items()
    }
    result.total_residual_se = _standard_error(
        np.sum(list(position_batches.values()), axis=0) if position_batches else np.zeros(1)
    )
    curated_root = Path(cfg.paths.curated)
    result.by_position_quote_noise = _combine_quote_noise(
        {pid: marks[0][pid] for pid in priced_all},
        load_quote_noise(curated_root, day0),
        load_quote_noise(curated_root, day1),
    )
    result.total_quote_noise = float(
        np.sqrt(sum(q**2 for q in result.by_position_quote_noise.values()))
    )
    result.by_position_residual = position_residuals
    result.nav = sum(m.price for m in marks[-1].values())
    result.breaches = _detect_breaches(result)
    _persist(result, Path(cfg.paths.curated))
    return result


def _fixing_events(
    cfg: BaseConfig, portfolio: Portfolio, day0: dt.date, day1: dt.date, priced: set[str]
) -> dict[str, str]:
    """Autocalls whose coupon state changed between the two days without settling:
    a coupon paid or missed at a fixing in between, described for the note."""
    out = {}
    for p in portfolio.positions:
        if p.id not in priced or not isinstance(p, AutocallPosition):
            continue
        try:
            s0 = lifecycle.position_status(cfg, day0, p)
            s1 = lifecycle.position_status(cfg, day1, p)
        except lifecycle.LifecycleError:
            continue
        if s1.status == lifecycle.SETTLED or (s0.coupons_paid, s0.memory) == (
            s1.coupons_paid,
            s1.memory,
        ):
            continue
        paid = s1.coupons_paid - s0.coupons_paid
        what = f"coupon of {paid:,.0f} paid" if paid > 0 else "coupon missed"
        out[p.id] = f"{what} at a fixing after {day0} (coupon memory {s0.memory} -> {s1.memory})"
    return out


def _basis_lines(
    marks_prev: dict[str, PositionMark],
    marks_cur: dict[str, PositionMark],
    states: list[MarketState],
    priced: set[str],
) -> list[str]:
    """`states` are the five explain states; the spot step runs states[2] -> [3]."""
    added: dict[str, float] = {}
    for pid in priced:
        m = marks_prev[pid]
        if m.type != "vanilla" or lifecycle.SETTLED in (m.status, marks_cur[pid].status):
            continue
        u = m.underlying
        via_close = _spot_taylor(m, states[3].spot[u] - states[2].spot[u])
        via_forward = _spot_taylor(m, _forward_spot_move(m, states[2], states[3]))
        added[u] = added.get(u, 0.0) + via_forward - via_close
    return [
        f"basis {u}: front forward vs close {_basis_bp(states[0], u):+.1f}bp -> "
        f"{_basis_bp(states[4], u):+.1f}bp; vanillas' spot step through the forward "
        f"instead of the close: {added[u]:+,.2f}"
        for u in sorted(added)
        if all(u in states[i].forward_curve for i in (0, 4))
    ]


def _classify(
    dollars: float, se: float, nav: float, label: str, quote_noise: float = 0.0
) -> tuple[str | None, str | None]:
    """(alert, inconclusive) message for one residual, at most one non-None.

    Breach beyond noise (planning/pnl_explain_noise_plan.md, user decision
    2026-10-05): an alert only when the residual exceeds the threshold even
    after allowing 2 standard deviations of noise. If the residual clears that
    bar it is an alert however noisy it is. Otherwise, if the 2-SD band alone is
    wider than the threshold, the residual cannot be told apart from noise
    either way: inconclusive, not an alert and not a clean pass.

    Noise is the Monte Carlo SE and, for MC exotics, the quote noise of both
    days' marks (planning/p008_noise_band_plan.md, 2026-11-22), independent and
    added in quadrature."""
    noise = float(np.hypot(se, quote_noise))
    threshold = RESIDUAL_ALERT_THRESHOLD_BP * nav / 10_000.0 if nav else 0.0
    bp = 10_000.0 * dollars / nav if nav else 0.0
    band_bp = 10_000.0 * 2 * noise / nav if nav else 0.0
    what = "2 SD, MC and quote noise" if quote_noise > 0.0 else "2 SE"
    shown = f"{label} residual {bp:+.1f}bp of NAV (+- {band_bp:.1f}bp, {what})"
    if abs(dollars) - 2 * noise > threshold:
        return f"{shown} exceeds the {RESIDUAL_ALERT_THRESHOLD_BP:.0f}bp threshold", None
    if 2 * noise > threshold:
        source = "Monte Carlo and quote noise are" if quote_noise > 0.0 else "Monte Carlo noise is"
        return (
            None,
            f"{shown}: {source} wider than the {RESIDUAL_ALERT_THRESHOLD_BP:.0f}bp threshold",
        )
    return None, None


def _detect_breaches(result: PnLExplainResult) -> list[str]:
    """README 12.3: 'set a threshold (e.g. 5bp of NAV) and generate an alert when
    breached' — applied to both the total residual and, per position, the same
    convention, beyond Monte Carlo noise (see `_classify`). Returns the alert
    messages and sets `result.inconclusive` for noise-dominated residuals."""
    breaches, inconclusive = [], []
    checks = [
        ("total", result.total_residual(), result.total_residual_se, result.total_quote_noise)
    ] + [
        (
            f"position {pid}",
            r,
            result.by_position_residual_se.get(pid, 0.0),
            result.by_position_quote_noise.get(pid, 0.0),
        )
        for pid, r in sorted(result.by_position_residual.items())
    ]
    for label, dollars, se, quote_noise in checks:
        alert, unclear = _classify(dollars, se, result.nav, label, quote_noise)
        if alert:
            breaches.append(alert)
        if unclear:
            inconclusive.append(unclear)
    result.inconclusive = inconclusive
    return breaches


def _persist(result: PnLExplainResult, curated_root: Path) -> None:
    if not result.steps:
        return
    rows = [
        {
            "asof_date": result.day1,
            "day0": result.day0,
            "step": s.step,
            "actual_pnl": s.actual_pnl,
            "explained_pnl": s.explained_pnl,
            "residual": s.residual,
            "residual_se": s.residual_se,
            "total_residual_se": result.total_residual_se,
            "nav": result.nav,
            "total_quote_noise": (
                result.total_quote_noise if result.by_position_quote_noise else None
            ),
            "reval_pnl": s.reval_pnl,
        }
        for s in result.steps
    ]
    table = validate(pd.DataFrame(rows), PNL_EXPLAIN_SCHEMA, PNL_EXPLAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, curated_root / "pnl_explain", ["asof_date"])

    if result.by_position_residual:
        by_position_rows = [
            {
                "asof_date": result.day1,
                "day0": result.day0,
                "position_id": pid,
                "residual": r,
                "residual_se": result.by_position_residual_se.get(pid, 0.0),
                "quote_noise": result.by_position_quote_noise.get(pid),
                "carry_pnl": result.by_position_carry.get(pid),
                "shape_pnl": result.by_position_shape.get(pid),
                "note": result.notes.get(pid),
            }
            for pid, r in result.by_position_residual.items()
        ]
        by_position_table = validate(
            pd.DataFrame(by_position_rows),
            PNL_EXPLAIN_BY_POSITION_SCHEMA,
            PNL_EXPLAIN_BY_POSITION_REQUIRED_NOT_NULL,
        )
        store.write_partitioned(
            by_position_table, curated_root / "pnl_explain_by_position", ["asof_date"]
        )
