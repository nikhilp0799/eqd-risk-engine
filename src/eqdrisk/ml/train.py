"""Training loop for the vanilla + variance-loss vertical slice (Phase 1,
`planning/deep_hedging_plan.md`). Full-batch gradient descent over one FIXED
set of Sobol-generated training paths — Sobol's own low-discrepancy coverage
makes resampling per epoch unnecessary at this path count, and full-batch
keeps the rollout's `n_steps`-deep backprop-through-time simple to reason
about. A held-out, differently-seeded path set (never trained on) is used for
evaluation — same discipline as every other validation in this project.
"""

from __future__ import annotations

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
from eqdrisk.ml.payoffs import vanilla_payoff
from eqdrisk.ml.simulate import simulate_training_paths
from eqdrisk.pricing.autocallable import AutocallableSpec

LossType = Literal["variance", "cvar", "cost"]


@dataclass
class TrainConfig:
    n_paths_train: int = 8_000
    n_steps: int = 32
    epochs: int = 100
    lr: float = 1e-3
    cost_bps: float = 5.0
    seed_train: int = 1
    hidden: int = 32


@dataclass
class TrainResult:
    net: HedgeNet
    loss_history: list[float] = field(default_factory=list)


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

    return TrainResult(net=net, loss_history=loss_history)


def _loss_from_hedged_pnl(
    hedged_pnl: torch.Tensor,
    turnover: torch.Tensor,
    loss_type: LossType,
    cvar_alpha: float,
    cost_lambda: float,
) -> torch.Tensor:
    if loss_type == "cost":
        return cost_adjusted_loss(hedged_pnl, turnover, lambda_turnover=cost_lambda)
    if loss_type == "cvar":
        return cvar_loss(hedged_pnl, alpha=cvar_alpha)
    return variance_loss(hedged_pnl)


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
    sim = simulate_training_paths(inputs, cfg.n_paths_train, cfg.n_steps, cfg.seed_train)
    net = HedgeNet(hidden=cfg.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    def payoff_fn(paths: torch.Tensor) -> torch.Tensor:
        return vanilla_payoff(paths, strike, is_call)

    loss_history: list[float] = []
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        hedged_pnl, turnover = rollout_hedged_pnl(
            sim.paths,
            sim.t_grid,
            net,
            strike,
            inputs.T,
            cfg.cost_bps,
            payoff_fn,
            return_turnover=True,
        )
        loss = _loss_from_hedged_pnl(hedged_pnl, turnover, loss_type, cvar_alpha, cost_lambda)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return TrainResult(net=net, loss_history=loss_history)


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
    sim = simulate_training_paths(inputs, cfg.n_paths_train, cfg.n_steps, cfg.seed_train)
    obs_levels = sim.paths[:, 1:]  # drop t=0 -> one column per spec.obs_times entry
    net = HedgeNet(hidden=cfg.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    loss_history: list[float] = []
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        hedged_pnl, turnover = rollout_autocallable_hedged_pnl(
            obs_levels,
            spec.obs_times,
            net,
            spec,
            inputs.spot,
            inputs.T,
            cfg.cost_bps,
            return_turnover=True,
        )
        loss = _loss_from_hedged_pnl(hedged_pnl, turnover, loss_type, cvar_alpha, cost_lambda)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return TrainResult(net=net, loss_history=loss_history)
