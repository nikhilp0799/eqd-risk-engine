"""Training loop for the vanilla + variance-loss vertical slice (Phase 1,
`planning/deep_hedging_plan.md`). Full-batch gradient descent over one FIXED
set of Sobol-generated training paths — Sobol's own low-discrepancy coverage
makes resampling per epoch unnecessary at this path count, and full-batch
keeps the rollout's `n_steps`-deep backprop-through-time simple to reason
about. A held-out, differently-seeded path set (never trained on) is used for
evaluation — same discipline as every other validation in this project.

Phase 6: optional early stopping on a THIRD, validation path set (distinct from
both training and the held-out evaluation set), because a fixed 300-epoch
budget was measured to stop well short of convergence (loss still falling
5-8% over the last 50 epochs on real data).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch

from eqdrisk.ml.hedge_model import (
    HedgeNet,
    rollout_autocallable_hedged_pnl,
    rollout_hedged_pnl,
)
from eqdrisk.ml.losses import (
    DEFAULT_COST_LAMBDA,
    DEFAULT_CVAR_ALPHA,
    cost_adjusted_loss,
    cvar_loss,
    variance_loss,
)
from eqdrisk.ml.market import HedgingMarketInputs
from eqdrisk.ml.payoffs import down_and_in_put_payoff, vanilla_payoff
from eqdrisk.ml.simulate import SimulatedPaths, simulate_training_paths
from eqdrisk.pricing.autocallable import AutocallableSpec

LossType = Literal["variance", "cvar", "cost"]


@dataclass
class TrainConfig:
    n_paths_train: int = 8_000
    n_steps: int = 32
    # With early stopping on (`n_paths_val > 0`) this is the hard cap, not the budget.
    epochs: int = 100
    lr: float = 1e-3
    cost_bps: float = 5.0
    seed_train: int = 1
    hidden: int = 32
    # Early stopping: off unless a validation set is requested.
    n_paths_val: int = 0
    seed_val: int = 500
    eval_every: int = 10
    patience: int = 100  # epochs without a qualifying improvement before stopping
    min_rel_improvement: float = 1e-3


@dataclass
class TrainResult:
    net: HedgeNet
    loss_history: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_val_loss: float | None = None
    stopped_early: bool = False


def train_vanilla_hedge(
    inputs: HedgingMarketInputs,
    strike: float,
    is_call: bool,
    cfg: TrainConfig,
) -> TrainResult:
    sim = simulate_training_paths(inputs, cfg.n_paths_train, cfg.n_steps, cfg.seed_train)
    net = HedgeNet(hidden=cfg.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    def payoff_fn(paths: torch.Tensor) -> torch.Tensor:
        return vanilla_payoff(paths, strike, is_call)

    loss_history: list[float] = []
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        hedged_pnl = rollout_hedged_pnl(
            sim.paths, sim.t_grid, net, strike, inputs.T, cfg.cost_bps, payoff_fn
        )
        loss = variance_loss(hedged_pnl)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return TrainResult(net=net, loss_history=loss_history, epochs_run=cfg.epochs)


def _loss_from_hedged_pnl(
    hedged_pnl: torch.Tensor,
    trading_cost: torch.Tensor,
    loss_type: LossType,
    cvar_alpha: float,
    cost_lambda: float,
) -> torch.Tensor:
    if loss_type == "cost":
        return cost_adjusted_loss(hedged_pnl, trading_cost, lambda_cost=cost_lambda)
    if loss_type == "cvar":
        return cvar_loss(hedged_pnl, alpha=cvar_alpha)
    return variance_loss(hedged_pnl)


# Maps a path set to per-path (hedged_pnl, turnover, dollar trading cost) for the
# current network.
Rollout = Callable[[SimulatedPaths], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _fit(
    net: HedgeNet,
    rollout: Rollout,
    train_paths: SimulatedPaths,
    val_paths: SimulatedPaths | None,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: TrainConfig,
) -> TrainResult:
    """Full-batch Adam on `train_paths`. With `val_paths`, stops once the
    validation objective has gone `cfg.patience` epochs without improving by
    more than `cfg.min_rel_improvement` (relative), and returns the weights from
    the best validation check rather than the last epoch."""
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_history: list[float] = []
    best_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0
    stopped_early = False

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        pnl, _, cost = rollout(train_paths)
        loss = objective(pnl, cost)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if val_paths is None or epoch % cfg.eval_every:
            continue
        with torch.no_grad():
            val_pnl, _, val_cost = rollout(val_paths)
            val = float(objective(val_pnl, val_cost).item())
        if best_val is None or val < best_val - cfg.min_rel_improvement * abs(best_val):
            best_val, best_state, epochs_since_best = val, copy.deepcopy(net.state_dict()), 0
        else:
            epochs_since_best += cfg.eval_every
            if epochs_since_best >= cfg.patience:
                stopped_early = True
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return TrainResult(
        net=net,
        loss_history=loss_history,
        epochs_run=len(loss_history),
        best_val_loss=best_val,
        stopped_early=stopped_early,
    )


def _val_paths(inputs: HedgingMarketInputs, cfg: TrainConfig) -> SimulatedPaths | None:
    if cfg.n_paths_val <= 0:
        return None
    return simulate_training_paths(inputs, cfg.n_paths_val, cfg.n_steps, cfg.seed_val)


def _train_hedge_with_rollout(
    inputs: HedgingMarketInputs,
    strike: float,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    loss_type: LossType,
    cfg: TrainConfig,
    cvar_alpha: float,
    cost_lambda: float,
) -> TrainResult:
    """Shared body for any single-maturity, no-early-exit payoff hedged with
    `hedge_model.rollout_hedged_pnl` — used by both the vanilla (Phase 2) and
    down-and-in-put (Phase 5) instruments, which need no rollout logic beyond
    what `rollout_hedged_pnl` already provides (unlike the autocallable, whose
    early-redemption logic genuinely needed its own rollout function)."""
    net = HedgeNet(hidden=cfg.hidden)

    def rollout(sim: SimulatedPaths) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return rollout_hedged_pnl(
            sim.paths,
            sim.t_grid,
            net,
            strike,
            inputs.T,
            cfg.cost_bps,
            payoff_fn,
            return_trading=True,
        )

    def objective(pnl: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        return _loss_from_hedged_pnl(pnl, cost, loss_type, cvar_alpha, cost_lambda)

    train_paths = simulate_training_paths(inputs, cfg.n_paths_train, cfg.n_steps, cfg.seed_train)
    return _fit(net, rollout, train_paths, _val_paths(inputs, cfg), objective, cfg)


def train_vanilla_hedge_general(
    inputs: HedgingMarketInputs,
    strike: float,
    is_call: bool,
    loss_type: LossType,
    cfg: TrainConfig,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    cost_lambda: float = DEFAULT_COST_LAMBDA,
) -> TrainResult:
    """Phase 2: the same vanilla rollout as `train_vanilla_hedge`, generalized
    to all three loss objectives the project owner chose."""

    def payoff_fn(paths: torch.Tensor) -> torch.Tensor:
        return vanilla_payoff(paths, strike, is_call)

    return _train_hedge_with_rollout(
        inputs, strike, payoff_fn, loss_type, cfg, cvar_alpha, cost_lambda
    )


def train_barrier_hedge(
    inputs: HedgingMarketInputs,
    strike: float,
    barrier: float,
    loss_type: LossType,
    cfg: TrainConfig,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    cost_lambda: float = DEFAULT_COST_LAMBDA,
) -> TrainResult:
    """Phase 5: the down-and-in put, reusing the same no-early-exit rollout as
    the vanilla instrument — see `planning/deep_hedging_plan.md`'s Phase 5
    section for why no new rollout logic was needed here."""

    def payoff_fn(paths: torch.Tensor) -> torch.Tensor:
        return down_and_in_put_payoff(paths, strike, barrier)

    return _train_hedge_with_rollout(
        inputs, strike, payoff_fn, loss_type, cfg, cvar_alpha, cost_lambda
    )


def train_autocall_hedge(
    inputs: HedgingMarketInputs,
    spec: AutocallableSpec,
    loss_type: LossType,
    cfg: TrainConfig,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    cost_lambda: float = DEFAULT_COST_LAMBDA,
) -> TrainResult:
    """Phase 3. `cfg.n_steps` MUST equal `len(spec.obs_times)` (one rebalance
    per quarterly observation, not daily — a documented simplification, see
    `planning/deep_hedging_plan.md`) and both must be a power of two, the same
    exactness requirement `pricing/autocallable.py::price_autocallable`
    already documents for its own observation-date alignment."""
    net = HedgeNet(hidden=cfg.hidden)

    def rollout(sim: SimulatedPaths) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_levels = sim.paths[:, 1:]  # drop t=0 -> one column per spec.obs_times entry
        return rollout_autocallable_hedged_pnl(
            obs_levels,
            spec.obs_times,
            net,
            spec,
            inputs.spot,
            inputs.T,
            cfg.cost_bps,
            return_trading=True,
        )

    def objective(pnl: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        return _loss_from_hedged_pnl(pnl, cost, loss_type, cvar_alpha, cost_lambda)

    train_paths = simulate_training_paths(inputs, cfg.n_paths_train, cfg.n_steps, cfg.seed_train)
    return _fit(net, rollout, train_paths, _val_paths(inputs, cfg), objective, cfg)
