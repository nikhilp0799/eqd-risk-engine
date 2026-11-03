"""Reverse stress (README 11.4): the most plausible 10-day market move that
loses a given amount.

Factors: the 10-day log moves of each name's spot and its vol index (SPX, AAPL,
NVDA, VIX, VXAPL, VXN), plausibility measured by the squared Mahalanobis
distance under their covariance over all 10-day periods since 2010-06-01
(unconditional: how unusual the move is, whatever the regime). A factor vector
becomes a market shock exactly as a VaR scenario does: spot by its relative
move, each vol index's relative move onto the surface through the Step 8.3 betas.

Solve min x' S^-1 x s.t. book P&L(x) <= -target with SLSQP, the P&L by full
revaluation (common random numbers, so it is smooth enough for finite-difference
gradients), from several starting directions: the linearised answer (S times the
P&L gradient at zero, the exact solution if P&L were linear), the historical
sell-off direction and its opposite, the rally. Each start is first scaled along
its direction until it reaches the target. Vol-index levels are kept inside
their range since 2010, spot moves inside +-60%.

Reported per target: the shock, its Mahalanobis distance d, the one-sided normal
probability of a move at least d standard deviations along that direction,
Phi(-d), and the shock in principal-component units of the covariance.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    REVERSE_STRESS_DETAIL_REQUIRED_NOT_NULL,
    REVERSE_STRESS_DETAIL_SCHEMA,
    REVERSE_STRESS_REQUIRED_NOT_NULL,
    REVERSE_STRESS_SCHEMA,
    validate,
)
from eqdrisk.portfolio.mark import PositionMark
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.risk.proxy import VOL_INDEX_FOR
from eqdrisk.risk.var import (
    RevalContext,
    VolProxy,
    active_underlyings,
    full_pnl,
    load_factor_history,
    prepare_revaluation,
    scenario_values,
    single_threaded_children,
)
from eqdrisk.stress.conditional import (
    HISTORY_START,
    HORIZON_DAYS,
    INDEX_UNDERLYING,
    fit_crash_response,
    log_moves,
)
from eqdrisk.stress.shock import MarketShock

TARGET_LOSSES: tuple[float, ...] = (500_000.0, 1_000_000.0, 2_000_000.0)
MAX_SPOT_MOVE = 0.60
# Finite-difference step, in log-move units: at 0.005 the VXN derivative moved
# 2x between steps on real data (Monte Carlo noise in the NVDA note); 0.01 is steadier.
FD_STEP = 0.01
MAX_ITER = 40
# Converged = the distance stopped improving: SLSQP's own test rarely passes,
# because Monte Carlo noise (~$1K) keeps the loss constraint from settling. On the
# 2026-10-29 book, 150 iterations gave d = 3.2035 against 3.2076 after 40.
PLATEAU_ITERATIONS = 10
PLATEAU_TOLERANCE = 0.005
N_REPORTED_PCS = 3


@dataclass
class ReverseProblem:
    """Everything a worker needs to evaluate the book under a factor vector."""

    ctx: RevalContext
    base: np.ndarray
    marks: list[PositionMark]
    proxies: dict[str, VolProxy]
    underlyings: list[str]
    factors: list[str]  # spot names first, then their vol indices
    cov_inv: np.ndarray
    bounds: list[tuple[float, float]]

    def shocks(self, x: np.ndarray) -> dict[str, MarketShock]:
        move = dict(zip(self.factors, np.expm1(x), strict=True))
        return {
            u: MarketShock(
                spot_shock_pct=float(move[u]),
                node_vol=self.proxies[u].node_vol(float(move[VOL_INDEX_FOR[u]])),
            )
            for u in self.underlyings
        }

    def position_pnl(self, x: np.ndarray) -> np.ndarray:
        values = scenario_values(self.ctx, self.shocks(x))
        return full_pnl(self.base, values[None], self.marks)[0]

    def pnl(self, x: np.ndarray) -> float:
        return float(self.position_pnl(x).sum())

    def distance2(self, x: np.ndarray) -> float:
        return float(x @ self.cov_inv @ x)

    def pnl_gradient(self, x: np.ndarray, at: float | None = None) -> np.ndarray:
        at = self.pnl(x) if at is None else at
        grad = np.empty(len(x))
        for i in range(len(x)):
            step = np.zeros(len(x))
            step[i] = FD_STEP
            grad[i] = (self.pnl(x + step) - at) / FD_STEP
        return grad

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo, hi = np.array(self.bounds).T
        return np.clip(x, lo, hi)


@dataclass
class Solution:
    target: float
    start: str
    x: np.ndarray
    pnl: float
    distance: float
    converged: bool


def _scale_to_target(
    problem: ReverseProblem, direction: np.ndarray, target: float
) -> np.ndarray | None:
    """The shortest multiple of `direction` (inside the bounds) that loses `target`."""
    hi = 1.0
    while problem.pnl(problem.clip(hi * direction)) > -target:
        hi *= 2.0
        if hi > 64:
            return None
    lo = 0.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if problem.pnl(problem.clip(mid * direction)) <= -target:
            hi = mid
        else:
            lo = mid
    return problem.clip(hi * direction)


def solve(
    problem: ReverseProblem, target: float, start: str, direction: np.ndarray
) -> Solution | None:
    x0 = _scale_to_target(problem, direction, target)
    if x0 is None:
        return None
    cache: dict[bytes, float] = {}

    def pnl(x: np.ndarray) -> float:
        key = np.round(x, 12).tobytes()
        if key not in cache:
            cache[key] = problem.pnl(x)
        return cache[key]

    def constraint(x: np.ndarray) -> float:
        return (-pnl(x) - target) / target

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        return -problem.pnl_gradient(x, pnl(x)) / target

    history: list[float] = []
    res = minimize(
        problem.distance2,
        x0,
        jac=lambda x: 2.0 * problem.cov_inv @ x,
        method="SLSQP",
        bounds=problem.bounds,
        constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jac}],
        options={"maxiter": MAX_ITER, "ftol": 1e-6},
        callback=lambda xk: history.append(float(np.sqrt(problem.distance2(xk)))),
    )
    # The iterate can sit a little short of the target (noise in the constraint):
    # move it out along its own direction until it loses the target exactly.
    end = res.x if np.any(res.x) else x0
    direction = end / np.sqrt(problem.distance2(end))
    x = _scale_to_target(problem, direction, target)
    if x is None or problem.distance2(x) > problem.distance2(x0):
        x = x0
    recent = history[-PLATEAU_ITERATIONS:]
    plateau = len(history) >= PLATEAU_ITERATIONS and (
        max(recent) - min(recent) <= PLATEAU_TOLERANCE * min(recent)
    )
    return Solution(
        target=target,
        start=start,
        x=x,
        pnl=pnl(x),
        distance=float(np.sqrt(problem.distance2(x))),
        converged=bool(res.success) or plateau,
    )


_WORKER: dict[str, ReverseProblem] = {}


def _init_worker(problem: ReverseProblem) -> None:
    _WORKER["problem"] = problem


def _solve_job(job: tuple[float, str, np.ndarray]) -> Solution | None:
    target, start, direction = job
    return solve(_WORKER["problem"], target, start, direction)


def factor_covariance(levels: pd.DataFrame, factors: list[str]) -> np.ndarray:
    """Zero-mean covariance of 10-day log moves over every period."""
    moves = log_moves(levels, factors, HORIZON_DAYS).to_numpy()
    return moves.T @ moves / len(moves)


def factor_bounds(
    levels: pd.DataFrame, factors: list[str], underlyings: Sequence[str], asof: dt.date
) -> list[tuple[float, float]]:
    bounds = []
    for f in factors:
        if f in underlyings:
            bounds.append((float(np.log1p(-MAX_SPOT_MOVE)), float(np.log1p(MAX_SPOT_MOVE))))
        else:
            history = levels[f].dropna()
            today = float(history.loc[asof])
            bounds.append(
                (float(np.log(history.min() / today)), float(np.log(history.max() / today)))
            )
    return bounds


def start_directions(
    problem: ReverseProblem, levels: pd.DataFrame, gradient: np.ndarray
) -> dict[str, np.ndarray]:
    """Unit-distance starting directions (Mahalanobis length 1)."""
    cov = np.linalg.inv(problem.cov_inv)
    others = [f for f in problem.factors if f != INDEX_UNDERLYING]
    crash = fit_crash_response(levels, others)
    sell_off = np.array(
        [-1.0 if f == INDEX_UNDERLYING else -crash[f].beta for f in problem.factors]
    )
    raw = {"linear": -cov @ gradient, "sell_off": sell_off, "rally": -sell_off}
    return {k: v / np.sqrt(v @ problem.cov_inv @ v) for k, v in raw.items() if np.any(v)}


@dataclass
class ReverseStressResult:
    asof: dt.date
    factors: list[str] = field(default_factory=list)
    solutions: list[Solution] = field(default_factory=list)  # best per target
    n_starts: dict[float, int] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Reverse stress — {self.asof} (10-day moves)"]
        for s in self.solutions:
            moves = ", ".join(
                f"{f} {np.expm1(v):+.1%}" for f, v in zip(self.factors, s.x, strict=True)
            )
            lines.append(
                f"  loss ${s.target:,.0f}: d = {s.distance:.2f}, "
                f"probability {norm.sf(s.distance):.2%} (start: {s.start}"
                f"{'' if s.converged else ', not converged'}), P&L ${s.pnl:,.0f}"
            )
            lines.append(f"    {moves}")
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def run_reverse_stress(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio_path: str,
    targets: Sequence[float] = TARGET_LOSSES,
    levels: pd.DataFrame | None = None,
    max_workers: int | None = None,
    persist: bool = True,
) -> ReverseStressResult:
    result = ReverseStressResult(asof=asof)
    portfolio = Portfolio.from_yaml(portfolio_path)
    if levels is None:
        levels = load_factor_history(active_underlyings(portfolio, asof), HISTORY_START, asof)
    levels = levels.loc[levels.index <= asof]
    prep = prepare_revaluation(cfg, asof, portfolio, levels)
    if isinstance(prep, str):
        result.skipped["_all_"] = prep
        return result
    factors = prep.underlyings + [VOL_INDEX_FOR[u] for u in prep.underlyings]
    if INDEX_UNDERLYING not in factors:
        result.skipped["_all_"] = f"the book has no {INDEX_UNDERLYING} position"
        return result
    cov = factor_covariance(levels, factors)
    problem = ReverseProblem(
        ctx=prep.ctx,
        base=prep.base,
        marks=prep.marks,
        proxies=prep.proxies,
        underlyings=prep.underlyings,
        factors=factors,
        cov_inv=np.linalg.inv(cov),
        bounds=factor_bounds(levels, factors, prep.underlyings, asof),
    )
    result.factors = factors
    gradient = problem.pnl_gradient(np.zeros(len(factors)))
    directions = start_directions(problem, levels, gradient)
    jobs = [(t, name, d) for t in targets for name, d in directions.items()]

    workers = min(len(jobs), max_workers if max_workers is not None else os.cpu_count() or 1)
    if workers <= 1:
        _init_worker(problem)
        found = [_solve_job(j) for j in jobs]
    else:
        with (
            single_threaded_children(),
            ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(problem,)
            ) as executor,
        ):
            found = list(executor.map(_solve_job, jobs))

    for target in targets:
        candidates = [s for s in found if s is not None and s.target == target]
        result.n_starts[target] = len(candidates)
        if not candidates:
            result.skipped[f"${target:,.0f}"] = "no start direction reaches this loss"
            continue
        result.solutions.append(min(candidates, key=lambda s: s.distance))

    _tables(result, problem, cov)
    if persist and result.solutions:
        _persist(result, Path(cfg.paths.curated))
    return result


def _tables(result: ReverseStressResult, problem: ReverseProblem, cov: np.ndarray) -> None:
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    eigvec = eigvec * np.sign(eigvec.sum(axis=0))  # loadings sum positive
    summary, detail = [], []
    for s in result.solutions:
        summary.append(
            {
                "target_loss": s.target,
                "pnl": s.pnl,
                "mahalanobis": s.distance,
                "probability": float(norm.sf(s.distance)),
                "start": s.start,
                "converged": s.converged,
                "n_starts": result.n_starts[s.target],
            }
        )

        def add(kind: str, name: str, value: float, target: float = s.target) -> None:
            detail.append({"target_loss": target, "kind": kind, "name": name, "value": value})

        for f, v in zip(problem.factors, s.x, strict=True):
            add("factor_move", f, float(np.expm1(v)))
        for i in range(N_REPORTED_PCS):
            add("pc_sigma", f"PC{i + 1}", float(eigvec[:, i] @ s.x / np.sqrt(eigval[i])))
        for m, v in zip(problem.marks, problem.position_pnl(s.x), strict=True):
            add("position_pnl", m.position_id, float(v))
    for i in range(N_REPORTED_PCS):
        for f, loading in zip(problem.factors, eigvec[:, i], strict=True):
            detail.append(
                {
                    "target_loss": 0.0,
                    "kind": "pc_loading",
                    "name": f"PC{i + 1} {f}",
                    "value": loading,
                }
            )
        detail.append(
            {
                "target_loss": 0.0,
                "kind": "pc_share",
                "name": f"PC{i + 1}",
                "value": float(eigval[i] / eigval.sum()),
            }
        )
    result.summary = pd.DataFrame(summary)
    result.detail = pd.DataFrame(detail)


def _persist(result: ReverseStressResult, curated_root: Path) -> None:
    for df, schema, required, name in (
        (result.summary, REVERSE_STRESS_SCHEMA, REVERSE_STRESS_REQUIRED_NOT_NULL, "reverse_stress"),
        (
            result.detail,
            REVERSE_STRESS_DETAIL_SCHEMA,
            REVERSE_STRESS_DETAIL_REQUIRED_NOT_NULL,
            "reverse_stress_detail",
        ),
    ):
        table = validate(df.assign(asof_date=result.asof), schema, required)
        store.write_partitioned(table, curated_root / name, ["asof_date"])
