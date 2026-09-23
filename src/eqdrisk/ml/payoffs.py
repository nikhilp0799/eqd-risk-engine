"""Payoffs for deep hedging: vanilla (Phase 1), the autocallable (Phase 3), and
the down-and-in put (Phase 5) — `planning/deep_hedging_plan.md`. None needs
autograd through its own arithmetic — the simulated path/observation levels
are fixed data (see the plan's architecture note); only the hedge network's
decisions do.
"""

from __future__ import annotations

import torch

from eqdrisk.pricing.autocallable import AutocallableSpec


def vanilla_payoff(paths: torch.Tensor, strike: float, is_call: bool) -> torch.Tensor:
    """One value per path — the option holder's payoff at expiry, per unit
    notional. `paths`: (n_paths, n_steps+1)."""
    terminal = paths[:, -1]
    if is_call:
        return torch.clamp(terminal - strike, min=0.0)
    return torch.clamp(strike - terminal, min=0.0)


def down_and_in_put_payoff(paths: torch.Tensor, strike: float, barrier: float) -> torch.Tensor:
    """Knocked-in (and thus worth a plain put payoff) if the FULL simulated
    path ever touches/breaches `barrier`, else worthless. Discrete monitoring
    at the simulation grid's own resolution — a real, documented simplification
    versus Step 6.4's Brownian-bridge-corrected barrier pricer, see
    `planning/deep_hedging_plan.md`'s Phase 5 section for why."""
    terminal = paths[:, -1]
    knocked_in = (paths <= barrier).any(dim=1)
    put_payoff = torch.clamp(strike - terminal, min=0.0)
    return torch.where(knocked_in, put_payoff, torch.zeros_like(put_payoff))


def autocallable_payoff_and_alive_schedule(
    obs_levels: torch.Tensor, spec: AutocallableSpec, initial_level: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same autocall/coupon-memory/put-barrier RULES as the real, already-tested
    `pricing/autocallable.py::autocallable_payoff` (same barrier levels, same
    coupon-memory logic, same priority order) — reused exactly, just without
    that function's discounting, to match this hedging framework's simplified
    undiscounted P&L accounting elsewhere (`hedge_model.rollout_hedged_pnl`
    doesn't discount either; both sides of the comparison are undiscounted
    consistently, not a hidden mismatch).

    `obs_levels`: (n_paths, n_obs) simulated spot at each of `spec.obs_times`
    (excludes t=0). Returns `(payoff, alive_schedule)` — `payoff` is one
    undiscounted value per path; `alive_schedule[:, i]` is True for paths that
    have NOT yet autocalled/redeemed entering observation `i` (the hedge must
    stop trading a path the moment it's no longer True)."""
    n_paths, n_obs = obs_levels.shape
    autocall_level = spec.autocall_barrier * initial_level
    coupon_level = spec.coupon_barrier * initial_level
    put_level = spec.put_barrier * initial_level

    pv = torch.zeros(n_paths, dtype=obs_levels.dtype)
    alive = torch.ones(n_paths, dtype=torch.bool)
    memory = torch.zeros(n_paths, dtype=obs_levels.dtype)
    alive_schedule = torch.ones((n_paths, n_obs), dtype=torch.bool)

    for i in range(n_obs):
        level = obs_levels[:, i]
        is_last = i == n_obs - 1
        alive_schedule[:, i] = alive

        autocalled = alive & (level >= autocall_level)
        pv = pv + torch.where(
            autocalled,
            spec.notional * (1.0 + spec.coupon_rate * (1.0 + memory)),
            torch.zeros_like(pv),
        )
        alive = alive & ~autocalled

        if not is_last:
            paid_coupon = alive & (level >= coupon_level)
            pv = pv + torch.where(
                paid_coupon,
                spec.notional * spec.coupon_rate * (1.0 + memory),
                torch.zeros_like(pv),
            )
            memory = torch.where(
                paid_coupon, torch.zeros_like(memory), torch.where(alive, memory + 1.0, memory)
            )
        else:
            paid_coupon = alive & (level >= coupon_level)
            above_put = alive & ~paid_coupon & (level >= put_level)
            breached_put = alive & ~paid_coupon & (level < put_level)

            pv = pv + torch.where(
                paid_coupon,
                spec.notional * (1.0 + spec.coupon_rate * (1.0 + memory)),
                torch.zeros_like(pv),
            )
            pv = pv + torch.where(
                above_put, torch.full_like(pv, spec.notional), torch.zeros_like(pv)
            )
            pv = pv + torch.where(
                breached_put, spec.notional * (level / initial_level), torch.zeros_like(pv)
            )

    return pv, alive_schedule
