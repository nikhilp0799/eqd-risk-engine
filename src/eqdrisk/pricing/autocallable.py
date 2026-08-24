"""Autocallable structured note (README 6.5): quarterly-observation reverse
convertible with early redemption, a memory coupon, and a down-and-in put
observed only at maturity. Path-dependent with no useful closed form — priced
by Monte Carlo, reusing Step 6.2's local-vol path simulator directly (the
payoff only needs the simulated LEVEL at each observation date, which is just a
handful of columns out of the already-computed path matrix).

Structure, at each quarterly observation date T_i, in priority order:
1. Autocall: if S(T_i) >= autocall_barrier * S0, redeem immediately at par plus
   this period's coupon (and any unpaid "memory" coupon from earlier periods).
2. Memory coupon: else if S(T_i) >= coupon_barrier * S0, pay this period's
   coupon plus any accumulated unpaid coupon from earlier periods, and reset
   the memory counter.
3. Otherwise: no payment this period; increment the memory counter.

At the FINAL observation date (maturity), if the note hasn't already
autocalled: pay the coupon (with memory) if still above the coupon barrier;
return par with no coupon if between the put barrier and the coupon barrier;
otherwise the investor is short a down-and-in put struck at S0 (European —
only observed at maturity, per the README's own wording) and receives
par * S(T)/S0 instead of par.

Greeks are bump-and-reval WITH common random numbers (the *same* seed, hence
the same driving Sobol/Brownian-bridge draws, for the base case and every
bump) — path-dependent payoffs make independently-reseeded bump-and-reval
Greeks pure noise, per the README's own warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqdrisk.pricing.monte_carlo import simulate_local_vol_paths
from eqdrisk.vol.local_vol import LocalVolGrid


@dataclass
class AutocallableSpec:
    notional: float
    autocall_barrier: float  # fraction of initial level, e.g. 1.00
    coupon_barrier: float  # fraction of initial level, e.g. 0.75
    put_barrier: float  # fraction of initial level, e.g. 0.65
    coupon_rate: float  # per-observation-period coupon, e.g. 0.0225
    obs_times: np.ndarray  # (n_obs,) year-fractions from today, increasing; last = maturity


def autocallable_payoff(
    obs_levels: np.ndarray, spec: AutocallableSpec, initial_level: float, r: float
) -> np.ndarray:
    """`obs_levels`: (n_paths, n_obs) simulated spot at each observation date.
    Returns the discounted-to-today total payoff per path (same units as
    `spec.notional`), not yet averaged across paths."""
    n_paths, n_obs = obs_levels.shape
    autocall_level = spec.autocall_barrier * initial_level
    coupon_level = spec.coupon_barrier * initial_level
    put_level = spec.put_barrier * initial_level
    discount_factors = np.exp(-r * spec.obs_times)

    pv = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    memory = np.zeros(n_paths)

    for i in range(n_obs):
        level = obs_levels[:, i]
        df = discount_factors[i]
        is_last = i == n_obs - 1

        autocalled = alive & (level >= autocall_level)
        pv += np.where(
            autocalled, df * spec.notional * (1.0 + spec.coupon_rate * (1.0 + memory)), 0.0
        )
        alive = alive & ~autocalled

        if not is_last:
            paid_coupon = alive & (level >= coupon_level)
            pv += np.where(
                paid_coupon, df * spec.notional * spec.coupon_rate * (1.0 + memory), 0.0
            )
            memory = np.where(paid_coupon, 0.0, np.where(alive, memory + 1.0, memory))
        else:
            paid_coupon = alive & (level >= coupon_level)
            above_put = alive & ~paid_coupon & (level >= put_level)
            breached_put = alive & ~paid_coupon & (level < put_level)

            pv += np.where(
                paid_coupon, df * spec.notional * (1.0 + spec.coupon_rate * (1.0 + memory)), 0.0
            )
            pv += np.where(above_put, df * spec.notional, 0.0)
            pv += np.where(breached_put, df * spec.notional * (level / initial_level), 0.0)

    return pv


def price_autocallable(
    spec: AutocallableSpec,
    s0: float,
    grid: LocalVolGrid,
    r: float,
    q: float,
    n_paths: int,
    n_steps_per_period: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """Simulates to the note's maturity (the last `obs_times` entry) and applies
    `autocallable_payoff` to the levels at each observation date (nearest
    simulated grid point — exact when `n_steps_per_period` is a power of two and
    `len(obs_times)` is too, as in the README's own quarterly/2yr example)."""
    T = float(spec.obs_times[-1])
    n_steps_total = n_steps_per_period * len(spec.obs_times)
    result = simulate_local_vol_paths(s0, T, grid, r, q, n_paths, n_steps_total, seed=seed)

    obs_indices = [int(np.argmin(np.abs(result.t_grid - t))) for t in spec.obs_times]
    obs_levels = result.paths[:, obs_indices]

    discounted_payoff = autocallable_payoff(obs_levels, spec, s0, r)
    price = float(np.mean(discounted_payoff))
    stderr = float(np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff)))
    return price, stderr


@dataclass
class AutocallableGreeks:
    price: float
    delta: float
    gamma: float
    vega: float


def autocallable_greeks(
    spec: AutocallableSpec,
    s0: float,
    grid: LocalVolGrid,
    r: float,
    q: float,
    n_paths: int,
    n_steps_per_period: int,
    seed: int,
    spot_bump_frac: float = 0.01,
    vol_bump: float = 0.01,
) -> AutocallableGreeks:
    """Bump-and-reval, all under the SAME seed (common random numbers) — the
    only way path-dependent bump-and-reval Greeks aren't pure noise."""
    h = s0 * spot_bump_frac
    price0, _ = price_autocallable(spec, s0, grid, r, q, n_paths, n_steps_per_period, seed)
    price_up, _ = price_autocallable(spec, s0 + h, grid, r, q, n_paths, n_steps_per_period, seed)
    price_dn, _ = price_autocallable(spec, s0 - h, grid, r, q, n_paths, n_steps_per_period, seed)
    delta = (price_up - price_dn) / (2 * h)
    gamma = (price_up - 2 * price0 + price_dn) / h**2

    bumped_grid = LocalVolGrid(
        s_grid=grid.s_grid,
        t_grid=grid.t_grid,
        sigma_loc=grid.sigma_loc + vol_bump,
        n_floored=grid.n_floored,
    )
    price_vol_up, _ = price_autocallable(
        spec, s0, bumped_grid, r, q, n_paths, n_steps_per_period, seed
    )
    vega = (price_vol_up - price0) / vol_bump

    return AutocallableGreeks(price=price0, delta=delta, gamma=gamma, vega=vega)
