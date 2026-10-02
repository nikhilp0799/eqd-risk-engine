"""Real, out-of-sample comparison between a trained hedge policy and the
existing Black-Scholes delta-hedge baseline, on a HELD-OUT path set the policy
was never trained on (`planning/deep_hedging_plan.md`). Reports honest
descriptive statistics either way, including if the learned policy does not
beat the baseline under some configuration — that is itself a real finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from eqdrisk.ml.baseline import (
    autocallable_static_delta_hedge_pnl,
    barrier_static_delta_hedge_pnl,
    bs_delta_hedge,
    static_hedge_turnover,
)
from eqdrisk.ml.hedge_model import HedgeNet, rollout_autocallable_hedged_pnl, rollout_hedged_pnl
from eqdrisk.ml.market import HedgingMarketInputs
from eqdrisk.ml.payoffs import down_and_in_put_payoff, vanilla_payoff
from eqdrisk.ml.simulate import SimulatedPaths, simulate_training_paths
from eqdrisk.pricing.autocallable import AutocallableSpec, autocallable_greeks
from eqdrisk.pricing.barrier_mc import down_and_in_put_greeks

CVAR_ALPHA = 0.95  # expected shortfall in the worst (1 - CVAR_ALPHA) tail of outcomes


@dataclass
class HedgeStats:
    mean: float
    std: float
    cvar: float  # mean of the worst (1 - CVAR_ALPHA) fraction of hedged P&L outcomes
    # Mean shares traded per path (Phase 6). Same counting convention for the
    # learned hedge and its benchmark within each instrument; see
    # `baseline.static_hedge_turnover`.
    turnover: float


def _stats(hedged_pnl: np.ndarray, turnover: np.ndarray) -> HedgeStats:
    sorted_pnl = np.sort(hedged_pnl)
    n_tail = max(1, int(len(sorted_pnl) * (1 - CVAR_ALPHA)))
    return HedgeStats(
        mean=float(np.mean(hedged_pnl)),
        std=float(np.std(hedged_pnl, ddof=1)),
        cvar=float(np.mean(sorted_pnl[:n_tail])),
        turnover=float(np.mean(turnover)),
    )


@dataclass
class ComparisonResult:
    learned: HedgeStats
    baseline: HedgeStats
    n_eval_paths: int


def autocall_benchmark_delta(
    inputs: HedgingMarketInputs,
    spec: AutocallableSpec,
    greeks_n_paths: int = 2_000,
    # 16, not 4: at 4 steps per quarter the first quarter's simulated vol was
    # measured 11% below converged (2026-10-01).
    greeks_n_steps_per_period: int = 16,
    greeks_seed: int = 4242,
) -> float:
    """The static benchmark's hedge ratio: the note's real bump-and-reval MC
    delta at inception. Depends only on the market, not on any trained net."""
    return autocallable_greeks(
        spec,
        inputs.spot,
        inputs.grid,
        inputs.r,
        inputs.q,
        greeks_n_paths,
        greeks_n_steps_per_period,
        greeks_seed,
    ).delta


def barrier_benchmark_delta(
    inputs: HedgingMarketInputs,
    strike: float,
    barrier: float,
    greeks_n_paths: int = 2_000,
    greeks_n_steps: int = 64,
    greeks_seed: int = 4242,
) -> float:
    """The static benchmark's hedge ratio for the down-and-in put, with its
    own Brownian-bridge correction. Depends only on the market."""
    return down_and_in_put_greeks(
        inputs.spot,
        strike,
        barrier,
        inputs.T,
        inputs.grid,
        inputs.r,
        inputs.q,
        greeks_n_paths,
        greeks_n_steps,
        greeks_seed,
    ).delta


def compare_vanilla_on_paths(
    inputs: HedgingMarketInputs,
    net: HedgeNet,
    strike: float,
    is_call: bool,
    sim: SimulatedPaths,
    cost_bps: float,
) -> ComparisonResult:
    """Learned policy vs the BS-delta benchmark on ANY path set on the policy's
    time grid (simulated, stressed, or historical, Phase 7). The benchmark
    always uses the calibrated ATM vol from `inputs`: both hedgers are as
    calibrated on the as-of date, whatever the paths then do."""
    with torch.no_grad():
        learned_pnl, learned_turnover, _ = rollout_hedged_pnl(
            sim.paths,
            sim.t_grid,
            net,
            strike,
            inputs.T,
            cost_bps,
            lambda p: vanilla_payoff(p, strike, is_call),
            return_trading=True,
        )

    baseline_pnl, baseline_turnover = bs_delta_hedge(
        sim.paths.numpy(),
        sim.t_grid,
        strike,
        is_call,
        inputs.T,
        inputs.r,
        inputs.q,
        inputs.atm_implied_vol(),
        cost_bps,
    )

    return ComparisonResult(
        learned=_stats(learned_pnl.numpy(), learned_turnover.numpy()),
        baseline=_stats(baseline_pnl, baseline_turnover),
        n_eval_paths=sim.paths.shape[0],
    )


def compare_autocall_on_paths(
    inputs: HedgingMarketInputs,
    spec: AutocallableSpec,
    net: HedgeNet,
    sim: SimulatedPaths,
    cost_bps: float,
    static_delta: float,
) -> ComparisonResult:
    """Same as `compare_vanilla_on_paths`, for the autocallable; `sim` is on the
    observation grid (t = 0 then one column per `spec.obs_times`)."""
    obs_levels = sim.paths[:, 1:]
    with torch.no_grad():
        learned_pnl, learned_turnover, _ = rollout_autocallable_hedged_pnl(
            obs_levels,
            spec.obs_times,
            net,
            spec,
            inputs.spot,
            inputs.T,
            cost_bps,
            return_trading=True,
        )

    baseline_pnl = autocallable_static_delta_hedge_pnl(
        obs_levels.numpy(), spec, inputs.spot, static_delta, cost_bps
    )
    baseline_turnover = static_hedge_turnover(len(baseline_pnl), static_delta, count_exit=True)

    return ComparisonResult(
        learned=_stats(learned_pnl.numpy(), learned_turnover.numpy()),
        baseline=_stats(baseline_pnl, baseline_turnover),
        n_eval_paths=obs_levels.shape[0],
    )


def compare_barrier_on_paths(
    inputs: HedgingMarketInputs,
    net: HedgeNet,
    strike: float,
    barrier: float,
    sim: SimulatedPaths,
    cost_bps: float,
    static_delta: float,
) -> ComparisonResult:
    """Same as `compare_vanilla_on_paths`, for the down-and-in put."""
    with torch.no_grad():
        learned_pnl, learned_turnover, _ = rollout_hedged_pnl(
            sim.paths,
            sim.t_grid,
            net,
            strike,
            inputs.T,
            cost_bps,
            lambda p: down_and_in_put_payoff(p, strike, barrier),
            return_trading=True,
        )

    baseline_pnl = barrier_static_delta_hedge_pnl(
        sim.paths.numpy(), strike, barrier, static_delta, cost_bps
    )
    baseline_turnover = static_hedge_turnover(len(baseline_pnl), static_delta, count_exit=False)

    return ComparisonResult(
        learned=_stats(learned_pnl.numpy(), learned_turnover.numpy()),
        baseline=_stats(baseline_pnl, baseline_turnover),
        n_eval_paths=sim.paths.shape[0],
    )


def evaluate_vanilla_hedge(
    inputs: HedgingMarketInputs,
    net: HedgeNet,
    strike: float,
    is_call: bool,
    n_paths_eval: int,
    n_steps: int,
    cost_bps: float,
    seed_eval: int,
) -> ComparisonResult:
    """`seed_eval` MUST differ from training's `seed_train` — a held-out set,
    never seen during training, is the only honest basis for this comparison."""
    sim = simulate_training_paths(inputs, n_paths_eval, n_steps, seed_eval)
    return compare_vanilla_on_paths(inputs, net, strike, is_call, sim, cost_bps)


def evaluate_autocall_hedge(
    inputs: HedgingMarketInputs,
    spec: AutocallableSpec,
    net: HedgeNet,
    n_paths_eval: int,
    cost_bps: float,
    seed_eval: int,
    greeks_n_paths: int = 2_000,
    greeks_n_steps_per_period: int = 16,
    greeks_seed: int = 4242,
    static_delta: float | None = None,
    substeps: int = 1,
) -> ComparisonResult:
    """Phase 3. The baseline's static delta is a REAL bump-and-reval MC Greek
    (`pricing/autocallable.py::autocallable_greeks`), computed once — see
    `baseline.autocallable_static_delta_hedge_pnl` for why it is static, not
    dynamically re-hedged. `seed_eval` MUST differ from training's
    `seed_train`, same held-out discipline as `evaluate_vanilla_hedge`.
    `static_delta`, if given, skips re-computing that Greek (it depends only on
    the market, not on the trained network)."""
    sim = simulate_training_paths(inputs, n_paths_eval, len(spec.obs_times), seed_eval, substeps)
    if static_delta is None:
        static_delta = autocall_benchmark_delta(
            inputs, spec, greeks_n_paths, greeks_n_steps_per_period, greeks_seed
        )
    return compare_autocall_on_paths(inputs, spec, net, sim, cost_bps, static_delta)


def evaluate_barrier_hedge(
    inputs: HedgingMarketInputs,
    net: HedgeNet,
    strike: float,
    barrier: float,
    n_paths_eval: int,
    n_steps: int,
    cost_bps: float,
    seed_eval: int,
    greeks_n_paths: int = 2_000,
    greeks_n_steps: int = 64,
    greeks_seed: int = 4242,
    static_delta: float | None = None,
) -> ComparisonResult:
    """Phase 5. The baseline's static delta is a REAL bump-and-reval MC Greek
    (`pricing/barrier_mc.py::down_and_in_put_greeks`, WITH its own Brownian-
    bridge continuity correction — unaffected by deep hedging's own simpler
    discrete-monitoring path generator), computed once — see
    `baseline.barrier_static_delta_hedge_pnl` for why it is static. `seed_eval`
    MUST differ from training's `seed_train`, same held-out discipline as
    every other `evaluate_*` function here. `static_delta`, if given, skips
    re-computing that Greek."""
    sim = simulate_training_paths(inputs, n_paths_eval, n_steps, seed_eval)
    if static_delta is None:
        static_delta = barrier_benchmark_delta(
            inputs, strike, barrier, greeks_n_paths, greeks_n_steps, greeks_seed
        )
    return compare_barrier_on_paths(inputs, net, strike, barrier, sim, cost_bps, static_delta)
