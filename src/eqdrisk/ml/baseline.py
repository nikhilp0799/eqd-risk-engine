"""The existing-technique baseline every learned policy is compared against:
a practitioner Black-Scholes delta hedge, discretely rebalanced on the SAME
simulated paths and SAME bps transaction-cost assumption as the learned
policy (`hedge_model.rollout_hedged_pnl`) — the only fair comparison. Uses the
real calibrated ATM implied vol (`market.HedgingMarketInputs.atm_implied_vol`)
as its constant vol input, since that mismatch against the true (non-constant)
local-vol dynamics actually simulated is precisely what a learned policy has
the opportunity to improve on.
"""

from __future__ import annotations

import numpy as np
import torch

from eqdrisk.ml.payoffs import autocallable_payoff_and_alive_schedule
from eqdrisk.pricing.autocallable import AutocallableSpec
from eqdrisk.pricing.blackscholes import delta_spot


def bs_delta_hedge_pnl(
    paths: np.ndarray,
    t_grid: np.ndarray,
    strike: float,
    is_call: bool,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cost_bps: float,
) -> np.ndarray:
    """One hedged-P&L value per path, same accounting convention as
    `hedge_model.rollout_hedged_pnl`: accumulated trading P&L (financed by
    holding the BS delta between each rebalance, net of bps costs) minus the
    option payoff owed at expiry."""
    n_paths, n_steps_plus1 = paths.shape
    n_steps = n_steps_plus1 - 1
    holdings = np.zeros(n_paths)
    trading_pnl = np.zeros(n_paths)
    cost_rate = cost_bps / 10_000.0

    for i in range(n_steps):
        s_t = paths[:, i]
        tau = max(T - float(t_grid[i]), 1e-8)  # avoid delta_spot's T=0 singularity
        forward = s_t * np.exp((r - q) * tau)
        discount_factor = np.exp(-r * tau)
        new_holdings = np.array(
            [
                delta_spot(is_call, f, strike, tau, sigma, discount_factor, s)
                for f, s in zip(forward, s_t, strict=True)
            ]
        )
        trade = new_holdings - holdings
        trading_pnl = trading_pnl - trade * s_t - cost_rate * np.abs(trade) * s_t
        holdings = new_holdings

    s_T = paths[:, -1]
    trading_pnl = trading_pnl + holdings * s_T
    payoff = np.clip(s_T - strike, 0.0, None) if is_call else np.clip(strike - s_T, 0.0, None)
    return trading_pnl - payoff


def autocallable_static_delta_hedge_pnl(
    obs_levels: np.ndarray,
    spec: AutocallableSpec,
    initial_level: float,
    static_delta_shares: float,
    cost_bps: float,
) -> np.ndarray:
    """The autocallable baseline: a hedge established ONCE at inception (at
    `static_delta_shares` — the note's real bump-and-reval MC delta,
    `pricing/autocallable.py::autocallable_greeks`, computed once at t=0) and
    held, UNREBALANCED, until the path's own actual redemption date (early
    autocall or maturity), then unwound. A documented simplification: fully
    dynamic MC-Greeks re-hedging at every observation date is prohibitively
    expensive to run per-path per-step for a comparison of this kind (each
    re-hedge would itself be a bump-and-reval Monte Carlo re-price); a static
    initial hedge is also a real, recognized simplified baseline for
    autocallables in practice, precisely because dynamically re-hedging this
    instrument type is itself known to be hard (the same difficulty Steps
    12/13 already found and documented for this project's own book)."""
    payoff_t, alive_schedule_t = autocallable_payoff_and_alive_schedule(
        torch.from_numpy(obs_levels), spec, initial_level
    )
    payoff = payoff_t.numpy()
    alive_schedule = alive_schedule_t.numpy()

    n_paths, n_obs = obs_levels.shape
    exit_index = np.clip(alive_schedule.sum(axis=1) - 1, 0, n_obs - 1)
    exit_level = obs_levels[np.arange(n_paths), exit_index]

    cost_rate = cost_bps / 10_000.0
    entry_cost = cost_rate * abs(static_delta_shares) * initial_level
    exit_cost = cost_rate * abs(static_delta_shares) * exit_level
    trading_pnl = static_delta_shares * (exit_level - initial_level) - entry_cost - exit_cost
    return trading_pnl - payoff
