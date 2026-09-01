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

**Positions with an incomplete Greek set contribute their own honest gap, not a
fabricated explanation:** MC-priced positions (barrier, autocall) only have
delta/gamma/vega (Step 7's own documented scope limit — no theta/rho/vanna/
volga). Where a step's Greek doesn't exist for a position type, `explained` is
0 for that position in that step (not "explained = actual", which would
silently hide the gap) — so their time/rates-divs residual is, honestly, their
entire actual P&L in those buckets. This is precisely the README's own 12.4
point made concrete: "if your residual spikes, your Greeks are incomplete."
Variance swap positions are excluded from the Greek-based calculation entirely
— Step 7 marks them at exactly 0 by construction (a fresh swap at today's own
fair strike), so there is no real P&L to explain for them in the first place.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import PNL_EXPLAIN_REQUIRED_NOT_NULL, PNL_EXPLAIN_SCHEMA, validate
from eqdrisk.portfolio.mark import (
    MarketState,
    PortfolioMarkResult,
    PositionMark,
    build_grids_for_state,
    load_market_state,
    mark_with_state,
)
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.vol.local_vol import local_variance_at

STEP_NAMES: tuple[str, ...] = ("time", "rates_divs", "spot", "vol")


def _atm_iv(surface: pd.DataFrame, T: float) -> float | None:
    lv = local_variance_at(surface, 0.0, T)
    if lv is None:
        return None
    return float((max(lv.w, 1e-12) / T) ** 0.5)


def _marks_by_id(result: PortfolioMarkResult) -> dict[str, PositionMark]:
    return {m.position_id: m for m in result.marks}


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
        return m_prev.theta * days_elapsed if m_prev.type == "vanilla" else 0.0

    if step == "rates_divs":
        if m_prev.type != "vanilla":
            return 0.0
        r_prev = state_prev.curve.zero_rate(m_prev.T)
        r_cur = state_cur.curve.zero_rate(m_prev.T)
        return m_prev.rho * (r_cur - r_prev)

    if step == "spot":
        u = m_prev.underlying
        if u not in state_prev.spot or u not in state_cur.spot:
            return 0.0
        d_spot = state_cur.spot[u] - state_prev.spot[u]
        return m_prev.delta * d_spot + 0.5 * m_prev.gamma * d_spot**2

    if step == "vol":
        if m_prev.vega == 0.0 and m_prev.volga == 0.0:
            return 0.0  # no vol exposure at all (e.g. equity, T=0) -- nothing to look up
        u = m_prev.underlying
        if u not in state_prev.surface or u not in state_cur.surface:
            return 0.0
        atm_prev = _atm_iv(state_prev.surface[u], m_prev.T)
        atm_cur = _atm_iv(state_cur.surface[u], m_prev.T)
        if atm_prev is None or atm_cur is None:
            return 0.0
        d_sigma = atm_cur - atm_prev
        return m_prev.vega * d_sigma + 0.5 * m_prev.volga * d_sigma**2

    raise ValueError(f"unknown step: {step}")  # pragma: no cover - exhaustive over STEP_NAMES


@dataclass
class StepResult:
    step: str
    actual_pnl: float
    explained_pnl: float

    @property
    def residual(self) -> float:
        return self.actual_pnl - self.explained_pnl


@dataclass
class PnLExplainResult:
    day0: dt.date
    day1: dt.date
    steps: list[StepResult] = field(default_factory=list)
    by_position_residual: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def total_actual(self) -> float:
        return sum(s.actual_pnl for s in self.steps)

    def total_explained(self) -> float:
        return sum(s.explained_pnl for s in self.steps)

    def total_residual(self) -> float:
        return sum(s.residual for s in self.steps)

    def render(self) -> str:
        lines = [f"P&L explain — {self.day0} -> {self.day1}"]
        for s in self.steps:
            lines.append(
                f"  {s.step:>10}: actual={s.actual_pnl:+,.2f}  explained={s.explained_pnl:+,.2f}"
                f"  residual={s.residual:+,.2f}"
            )
        lines.append(
            f"  {'TOTAL':>10}: actual={self.total_actual():+,.2f}  "
            f"explained={self.total_explained():+,.2f}  residual={self.total_residual():+,.2f}"
        )
        if self.by_position_residual:
            lines.append("  residual by position:")
            for pid, r in sorted(self.by_position_residual.items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"    {pid}: {r:+,.2f}")
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


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


def run_pnl_explain(
    cfg: BaseConfig, day0: dt.date, day1: dt.date, portfolio_path: str
) -> PnLExplainResult:
    portfolio = Portfolio.from_yaml(portfolio_path)
    state_day0 = load_market_state(cfg, day0, portfolio)
    state_day1 = load_market_state(cfg, day1, portfolio)
    result = PnLExplainResult(day0=day0, day1=day1)
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
    marks = [_marks_by_id(mark_with_state(cfg, asofs[i], portfolio, states[i])) for i in range(5)]

    position_residuals: dict[str, float] = {}

    for step_idx, step in enumerate(STEP_NAMES):
        marks_prev, marks_cur = marks[step_idx], marks[step_idx + 1]
        actual_total = 0.0
        explained_total = 0.0

        priced_both = set(marks_prev) & set(marks_cur)
        for pid in priced_both:
            m_prev, m_cur = marks_prev[pid], marks_cur[pid]
            actual = m_cur.price - m_prev.price
            explained = _explained_for_step(
                step, m_prev, days_elapsed, states[step_idx], states[step_idx + 1]
            )
            actual_total += actual
            explained_total += explained
            position_residuals[pid] = position_residuals.get(pid, 0.0) + (actual - explained)

        skipped_here = {p.id for p in portfolio.positions} - priced_both
        for pid in skipped_here:
            result.skipped[f"{pid}@{step}"] = "position not priced in both states for this step"

        result.steps.append(
            StepResult(step=step, actual_pnl=actual_total, explained_pnl=explained_total)
        )

    result.by_position_residual = position_residuals
    _persist(result, Path(cfg.paths.curated))
    return result


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
        }
        for s in result.steps
    ]
    table = validate(pd.DataFrame(rows), PNL_EXPLAIN_SCHEMA, PNL_EXPLAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, curated_root / "pnl_explain", ["asof_date"])
