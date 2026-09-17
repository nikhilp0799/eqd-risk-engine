"""The learned hedging policy and its differentiable rollouts (vanilla, Phase 1;
autocallable, Phase 3 — `planning/deep_hedging_plan.md`). One small network is
applied at EVERY rebalancing step (shared weights across time — the standard
deep-hedging architecture), taking real, cheaply computable state (log-
moneyness, time-to-maturity) and outputting the new hedge ratio. See the
plan's architecture note for why the price PATH itself does not need
gradients — only this network's decisions and the P&L arithmetic below do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import torch
from torch import nn

from eqdrisk.ml.payoffs import autocallable_payoff_and_alive_schedule
from eqdrisk.pricing.autocallable import AutocallableSpec


class HedgeNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).double()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@overload
def rollout_hedged_pnl(
    paths: torch.Tensor,
    t_grid: np.ndarray,
    net: HedgeNet,
    strike: float,
    T: float,
    cost_bps: float,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    return_turnover: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def rollout_hedged_pnl(
    paths: torch.Tensor,
    t_grid: np.ndarray,
    net: HedgeNet,
    strike: float,
    T: float,
    cost_bps: float,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    return_turnover: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rollout_hedged_pnl(
    paths: torch.Tensor,
    t_grid: np.ndarray,
    net: HedgeNet,
    strike: float,
    T: float,
    cost_bps: float,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    return_turnover: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One hedged-P&L value per path: accumulated trading P&L (financed by
    holding `net`'s chosen number of shares between each rebalance, net of bps
    transaction costs on every trade), minus the option payoff owed at expiry
    (this is the short-option-writer's hedged position — sold the option,
    hedges with the underlying). `paths`/`t_grid` come from
    `simulate.simulate_training_paths` and carry no gradient of their own;
    `net`'s parameters are the only thing being learned here. If
    `return_turnover`, also returns total per-path traded share volume — the
    cost-adjusted training objective (`losses.cost_adjusted_loss`) penalizes
    this directly, on top of the bps cost already deducted above."""
    n_paths, n_steps_plus1 = paths.shape
    n_steps = n_steps_plus1 - 1
    holdings = torch.zeros(n_paths, dtype=paths.dtype)
    trading_pnl = torch.zeros(n_paths, dtype=paths.dtype)
    total_turnover = torch.zeros(n_paths, dtype=paths.dtype)
    cost_rate = cost_bps / 10_000.0

    for i in range(n_steps):
        s_t = paths[:, i]
        tau = max(T - float(t_grid[i]), 0.0)
        log_moneyness = torch.log(s_t / strike)
        tau_feature = torch.full_like(s_t, tau / T if T > 0 else 0.0)
        features = torch.stack([log_moneyness, tau_feature], dim=1)
        new_holdings = net(features)
        trade = new_holdings - holdings
        total_turnover = total_turnover + torch.abs(trade)
        trading_pnl = trading_pnl - trade * s_t - cost_rate * torch.abs(trade) * s_t
        holdings = new_holdings

    s_T = paths[:, -1]
    trading_pnl = trading_pnl + holdings * s_T  # liquidate remaining shares at expiry
    payoff = payoff_fn(paths)
    hedged_pnl = trading_pnl - payoff
    if return_turnover:
        return hedged_pnl, total_turnover
    return hedged_pnl


@overload
def rollout_autocallable_hedged_pnl(
    obs_levels: torch.Tensor,
    obs_times: np.ndarray,
    net: HedgeNet,
    spec: AutocallableSpec,
    initial_level: float,
    T: float,
    cost_bps: float,
    return_turnover: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def rollout_autocallable_hedged_pnl(
    obs_levels: torch.Tensor,
    obs_times: np.ndarray,
    net: HedgeNet,
    spec: AutocallableSpec,
    initial_level: float,
    T: float,
    cost_bps: float,
    return_turnover: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rollout_autocallable_hedged_pnl(
    obs_levels: torch.Tensor,
    obs_times: np.ndarray,
    net: HedgeNet,
    spec: AutocallableSpec,
    initial_level: float,
    T: float,
    cost_bps: float,
    return_turnover: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Same accounting convention as `rollout_hedged_pnl`, in DOLLAR-notional
    terms rather than shares (natural for a note quoted in notional): the
    network's raw output is interpreted as a fraction of `spec.notional` held
    as stock exposure, rebalanced only at each quarterly `obs_times` (not
    daily — a documented simplification, see `planning/deep_hedging_plan.md`).
    Hedging STOPS the moment a path autocalls/redeems (`alive_schedule` from
    `payoffs.autocallable_payoff_and_alive_schedule`) — holding a hedge against
    a note that no longer exists would be a real error, not a simplification."""
    payoff, alive_schedule = autocallable_payoff_and_alive_schedule(obs_levels, spec, initial_level)
    n_paths, n_obs = obs_levels.shape
    dollar_position = torch.zeros(n_paths, dtype=obs_levels.dtype)
    trading_pnl = torch.zeros(n_paths, dtype=obs_levels.dtype)
    total_turnover = torch.zeros(n_paths, dtype=obs_levels.dtype)
    cost_rate = cost_bps / 10_000.0

    for i in range(n_obs - 1):
        level_i = obs_levels[:, i]
        # Gate by "alive going INTO i+1" (not "alive going into i"): if a path
        # redeems exactly AT observation i, there is zero exposure to hedge
        # over the i -> i+1 interval regardless of what was known at i.
        alive_next = alive_schedule[:, i + 1]
        tau = max(T - float(obs_times[i]), 0.0)
        log_moneyness = torch.log(level_i / initial_level)
        tau_feature = torch.full_like(level_i, tau / T if T > 0 else 0.0)
        features = torch.stack([log_moneyness, tau_feature], dim=1)
        raw = net(features)
        desired_position = torch.where(alive_next, raw * spec.notional, torch.zeros_like(raw))
        trade = desired_position - dollar_position
        total_turnover = total_turnover + torch.abs(trade) / initial_level  # in "shares" units
        cost = cost_rate * torch.abs(trade)
        level_next = obs_levels[:, i + 1]
        pct_return = (level_next - level_i) / level_i
        trading_pnl = trading_pnl + desired_position * pct_return - cost
        dollar_position = desired_position

    unwind_cost = cost_rate * torch.abs(dollar_position)  # close out whatever remains at maturity
    total_turnover = total_turnover + torch.abs(dollar_position) / initial_level
    trading_pnl = trading_pnl - unwind_cost

    hedged_pnl = trading_pnl - payoff
    if return_turnover:
        return hedged_pnl, total_turnover
    return hedged_pnl
