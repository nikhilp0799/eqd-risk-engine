"""Training objectives for deep hedging (`planning/deep_hedging_plan.md`): all
three objectives the project owner chose — variance, CVaR, and cost-adjusted.
"""

from __future__ import annotations

import torch

DEFAULT_CVAR_ALPHA = 0.95
DEFAULT_COST_LAMBDA = 0.1


def variance_loss(hedged_pnl: torch.Tensor) -> torch.Tensor:
    """Minimize the variance of the hedged P&L — the simplest, most standard
    deep-hedging objective (a perfect hedge has zero variance)."""
    return torch.var(hedged_pnl, unbiased=True)


def cvar_loss(hedged_pnl: torch.Tensor, alpha: float = DEFAULT_CVAR_ALPHA) -> torch.Tensor:
    """Minimize the empirical CVaR (expected shortfall) of hedged-P&L LOSSES —
    the mean of the worst `1 - alpha` fraction of outcomes in the batch. A
    standard, fully differentiable sample-based CVaR proxy used in the deep-
    hedging literature: `torch.topk` backprops gradients only to the selected
    tail elements, exactly the ones the objective cares about."""
    losses = -hedged_pnl
    n_tail = max(1, int(len(losses) * (1.0 - alpha)))
    worst = torch.topk(losses, n_tail, largest=True).values
    return worst.mean()


def cost_adjusted_loss(
    hedged_pnl: torch.Tensor, turnover: torch.Tensor, lambda_turnover: float = DEFAULT_COST_LAMBDA
) -> torch.Tensor:
    """Variance PLUS an explicit turnover penalty — the bps transaction cost is
    already deducted from `hedged_pnl` by the rollout (so every objective is
    compared apples-to-apples on realized, cost-inclusive P&L), but this
    objective additionally penalizes trading activity directly during
    training, pushing the learned policy toward a lower-turnover hedge than
    variance-minimization alone would find."""
    return torch.var(hedged_pnl, unbiased=True) + lambda_turnover * torch.mean(turnover)
