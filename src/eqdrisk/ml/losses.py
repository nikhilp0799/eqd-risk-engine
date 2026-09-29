"""Training objectives for deep hedging (`planning/deep_hedging_plan.md`): all
three objectives the project owner chose — variance, CVaR, and cost-adjusted.
"""

from __future__ import annotations

import torch

DEFAULT_CVAR_ALPHA = 0.95
# Phase 6: a dollar saved in expected trading cost is worth a dollar more P&L
# risk (std). Dimensionless, since both terms of `cost_adjusted_loss` are dollars.
DEFAULT_COST_LAMBDA = 1.0


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
    hedged_pnl: torch.Tensor,
    trading_cost: torch.Tensor,
    lambda_cost: float = DEFAULT_COST_LAMBDA,
) -> torch.Tensor:
    """P&L risk (std) plus `lambda_cost` times the expected dollar trading cost.

    Both terms are dollars, so the trade-off does not depend on the size of the
    position: scaling every price by a constant scales both terms equally.
    Phase 6 replaced `Var + 0.1 * E[turnover]`, whose penalty was measured
    (2026-09-28) to be negligible against the variance term — about 1e3 against
    1.8e11 for the $5M note — so the "cost-aware" policy was identical to the
    variance one. At `lambda_cost = 0` this has the same minimizer as
    `variance_loss`.

    The bps cost is also already deducted inside `hedged_pnl`, so every
    objective is compared on realized, cost-inclusive P&L; this term adds an
    explicit incentive to trade less.
    """
    return torch.std(hedged_pnl, unbiased=True) + lambda_cost * torch.mean(trading_cost)
