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

from eqdrisk.ml.baseline import autocallable_static_delta_hedge_pnl, bs_delta_hedge_pnl
from eqdrisk.ml.hedge_model import HedgeNet, rollout_autocallable_hedged_pnl, rollout_hedged_pnl
from eqdrisk.ml.market import HedgingMarketInputs
from eqdrisk.ml.payoffs import vanilla_payoff
from eqdrisk.ml.simulate import simulate_training_paths
from eqdrisk.pricing.autocallable import AutocallableSpec, autocallable_greeks

CVAR_ALPHA = 0.95  # expected shortfall in the worst (1 - CVAR_ALPHA) tail of outcomes


@dataclass
class HedgeStats:
    mean: float
    std: float
    cvar: float  # mean of the worst (1 - CVAR_ALPHA) fraction of hedged P&L outcomes


def _stats(hedged_pnl: np.ndarray) -> HedgeStats:
    sorted_pnl = np.sort(hedged_pnl)
    n_tail = max(1, int(len(sorted_pnl) * (1 - CVAR_ALPHA)))
    return HedgeStats(
        mean=float(np.mean(hedged_pnl)),
        std=float(np.std(hedged_pnl, ddof=1)),
        cvar=float(np.mean(sorted_pnl[:n_tail])),
    )


@dataclass
class ComparisonResult:
    learned: HedgeStats
    baseline: HedgeStats
    n_eval_paths: int


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

    with torch.no_grad():
        learned_pnl = rollout_hedged_pnl(
            sim.paths,
            sim.t_grid,
            net,
            strike,
            inputs.T,
            cost_bps,
            lambda p: vanilla_payoff(p, strike, is_call),
        ).numpy()

    baseline_pnl = bs_delta_hedge_pnl(
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
        learned=_stats(learned_pnl),
        baseline=_stats(baseline_pnl),
        n_eval_paths=sim.paths.shape[0],
    )


def evaluate_autocall_hedge(
    inputs: HedgingMarketInputs,
    spec: AutocallableSpec,
    net: HedgeNet,
    n_paths_eval: int,
    cost_bps: float,
    seed_eval: int,
    greeks_n_paths: int = 2_000,
    greeks_n_steps_per_period: int = 4,
    greeks_seed: int = 4242,
) -> ComparisonResult:
    """Phase 3. The baseline's static delta is a REAL bump-and-reval MC Greek
    (`pricing/autocallable.py::autocallable_greeks`), computed once — see
    `baseline.autocallable_static_delta_hedge_pnl` for why it is static, not
    dynamically re-hedged. `seed_eval` MUST differ from training's
    `seed_train`, same held-out discipline as `evaluate_vanilla_hedge`."""
    n_steps = len(spec.obs_times)
    sim = simulate_training_paths(inputs, n_paths_eval, n_steps, seed_eval)
    obs_levels = sim.paths[:, 1:]

    with torch.no_grad():
        learned_pnl = rollout_autocallable_hedged_pnl(
            obs_levels, spec.obs_times, net, spec, inputs.spot, inputs.T, cost_bps
        )
    assert isinstance(learned_pnl, torch.Tensor)

    greeks = autocallable_greeks(
        spec,
        inputs.spot,
        inputs.grid,
        inputs.r,
        inputs.q,
        greeks_n_paths,
        greeks_n_steps_per_period,
        greeks_seed,
    )
    baseline_pnl = autocallable_static_delta_hedge_pnl(
        obs_levels.numpy(), spec, inputs.spot, greeks.delta, cost_bps
    )

    return ComparisonResult(
        learned=_stats(learned_pnl.numpy()),
        baseline=_stats(baseline_pnl),
        n_eval_paths=obs_levels.shape[0],
    )
