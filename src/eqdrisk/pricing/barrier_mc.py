"""Local-vol Monte Carlo down-and-in put with a Brownian-bridge continuity
correction (README 6.4).

Naive discrete monitoring only checks the barrier at the simulated grid points,
so it systematically MISSES paths that dipped below the barrier and recovered
between two adjacent steps — this is the textbook "discrete monitoring
underprices knock-ins" bias, and it decays slowly (empirically ~1/sqrt(n_steps)
here, matching the closed-form-vs-discretely-monitored-MC gap this module's own
tests measure directly against `barrier_closed_form.down_and_in_put_price`).

The fix (Brownian-bridge continuity correction): between two consecutive
simulated points S_i, S_{i+1} that are BOTH still above the barrier H, there is
still a known, closed-form probability that a continuous Brownian bridge between
them dipped below H:

    p_cross = exp(-2 * ln(S_i/H) * ln(S_{i+1}/H) / (sigma_i^2 * dt))

(the classic reflection-principle result for a Brownian bridge's minimum). Each
step draws one uniform per path and marks a knock-in if that uniform falls below
`p_cross`, in addition to the naive discrete check — this recovers most of the
missed crossings analytically, without needing more time steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from eqdrisk.pricing.monte_carlo import (
    _bilinear,
    brownian_bridge_paths,
    next_power_of_two,
    sobol_normals,
)
from eqdrisk.vol.local_vol import LocalVolGrid


@numba.njit(cache=True)
def _euler_step_with_barrier(
    s0: float,
    dW: np.ndarray,
    t_grid_sim: np.ndarray,
    r: float,
    q: float,
    s_grid: np.ndarray,
    t_grid_lv: np.ndarray,
    sigma_loc: np.ndarray,
    barrier: float,
    bridge_uniforms: np.ndarray,
    use_bridge_correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_paths, n_steps = dW.shape
    terminal = np.empty(n_paths)
    knocked_in = np.zeros(n_paths, dtype=np.bool_)
    for p in range(n_paths):
        log_s = np.log(s0)
        s_prev = s0
        knocked = s_prev <= barrier
        for i in range(n_steps):
            t = t_grid_sim[i]
            dt = t_grid_sim[i + 1] - t
            sigma = _bilinear(s_grid, t_grid_lv, sigma_loc, s_prev, t)
            log_s += (r - q - 0.5 * sigma * sigma) * dt + sigma * dW[p, i]
            s_next = np.exp(log_s)
            if s_next <= barrier:
                knocked = True
            elif use_bridge_correction and not knocked:
                dist_prev = np.log(s_prev / barrier)
                dist_next = np.log(s_next / barrier)
                p_cross = np.exp(-2.0 * dist_prev * dist_next / (sigma * sigma * dt))
                if bridge_uniforms[p, i] < p_cross:
                    knocked = True
            s_prev = s_next
        terminal[p] = s_prev
        knocked_in[p] = knocked
    return terminal, knocked_in


@dataclass
class BarrierSimulationResult:
    terminal: np.ndarray
    knocked_in: np.ndarray

    def price_and_stderr(self, strike: float, discount_factor: float) -> tuple[float, float]:
        payoff = self.knocked_in * np.maximum(strike - self.terminal, 0.0)
        discounted = discount_factor * payoff
        price = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        return price, stderr


def simulate_down_and_in_put(
    s0: float,
    barrier: float,
    T: float,
    grid: LocalVolGrid,
    r: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
    use_bridge_correction: bool = True,
) -> BarrierSimulationResult:
    """Mirrors `monte_carlo.simulate_local_vol_paths`'s Sobol + Brownian-bridge
    path construction for the driving normals; adds a second, independent stream
    of plain pseudo-random uniforms (not low-discrepancy — this is a Bernoulli
    correction term, not a payoff driver) for the barrier continuity correction.
    """
    n_steps_pow2 = next_power_of_two(n_steps)
    t_grid_sim = np.linspace(0.0, T, n_steps_pow2 + 1)

    if antithetic:
        n_half = next_power_of_two((n_paths + 1) // 2)
        z_half = sobol_normals(n_half, n_steps_pow2, seed)
        z = np.concatenate([z_half, -z_half], axis=0)
        n_paths_total = 2 * n_half
    else:
        n_paths_total = next_power_of_two(n_paths)
        z = sobol_normals(n_paths_total, n_steps_pow2, seed)

    W = brownian_bridge_paths(z, T)
    dW = np.diff(W, axis=1)

    bridge_seed = None if seed is None else seed + 1
    # Shape passed as an explicit (int, int) tuple, not `dW.shape` — mypy's
    # numpy-stub overload resolution for `Generator.random(size=...)` was
    # observed to infer the `size=None` scalar (float-returning) overload when
    # given `dW.shape` (typed `tuple[Any, ...]` since `dW` traces back through
    # `np.diff`'s unparameterised `ndarray` return) — a stub-resolution quirk,
    # not a runtime issue, but avoided outright by using concrete int types.
    bridge_uniforms = np.random.default_rng(bridge_seed).random((n_paths_total, n_steps_pow2))

    terminal, knocked_in = _euler_step_with_barrier(
        s0,
        dW,
        t_grid_sim,
        r,
        q,
        grid.s_grid,
        grid.t_grid,
        grid.sigma_loc,
        barrier,
        bridge_uniforms,
        use_bridge_correction,
    )
    return BarrierSimulationResult(terminal=terminal, knocked_in=knocked_in)


@dataclass
class BarrierGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    vanna: float
    volga: float


def down_and_in_put_greeks(
    s0: float,
    strike: float,
    barrier: float,
    T: float,
    grid: LocalVolGrid,
    r: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    spot_bump_frac: float = 0.01,
    vol_bump: float = 0.01,
) -> BarrierGreeks:
    """Bump-and-reval delta/gamma/vega/vanna/volga under common random numbers
    — same pattern as `autocallable.autocallable_greeks`, for the same reason:
    a path-dependent (here, barrier-dependent) payoff makes independently
    reseeded bump-and-reval Greeks pure noise. Vanna/volga close the same
    Step 13 incident gap for barrier positions, which share the identical
    architectural limitation (see `autocallable_greeks`'s docstring)."""
    discount_factor = np.exp(-r * T)
    h = s0 * spot_bump_frac

    def _price(spot: float, g: LocalVolGrid) -> float:
        result = simulate_down_and_in_put(spot, barrier, T, g, r, q, n_paths, n_steps, seed)
        price, _ = result.price_and_stderr(strike, discount_factor)
        return price

    def _bumped_grid(dvol: float) -> LocalVolGrid:
        return LocalVolGrid(
            s_grid=grid.s_grid,
            t_grid=grid.t_grid,
            sigma_loc=grid.sigma_loc + dvol,
            n_floored=grid.n_floored,
        )

    grid_vol_up = _bumped_grid(vol_bump)
    grid_vol_dn = _bumped_grid(-vol_bump)

    price0 = _price(s0, grid)
    price_up = _price(s0 + h, grid)
    price_dn = _price(s0 - h, grid)
    price_vol_up = _price(s0, grid_vol_up)
    price_vol_dn = _price(s0, grid_vol_dn)
    price_pp = _price(s0 + h, grid_vol_up)
    price_pm = _price(s0 + h, grid_vol_dn)
    price_mp = _price(s0 - h, grid_vol_up)
    price_mm = _price(s0 - h, grid_vol_dn)

    delta = (price_up - price_dn) / (2 * h)
    gamma = (price_up - 2 * price0 + price_dn) / h**2
    vega = (price_vol_up - price_vol_dn) / (2 * vol_bump)
    volga = (price_vol_up - 2 * price0 + price_vol_dn) / vol_bump**2
    vanna = (price_pp - price_pm - price_mp + price_mm) / (4 * h * vol_bump)

    return BarrierGreeks(
        price=price0, delta=delta, gamma=gamma, vega=vega, vanna=vanna, volga=volga
    )
