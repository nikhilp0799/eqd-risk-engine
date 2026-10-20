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

from eqdrisk.pricing.monte_carlo import next_power_of_two, simulate_local_vol_paths
from eqdrisk.vol.local_vol import LocalVolGrid


@dataclass
class AutocallableSpec:
    notional: float
    autocall_barrier: float  # fraction of initial level, e.g. 1.00
    coupon_barrier: float  # fraction of initial level, e.g. 0.75
    put_barrier: float  # fraction of initial level, e.g. 0.65
    coupon_rate: float  # per-observation-period coupon, e.g. 0.0225
    obs_times: np.ndarray  # (n_obs,) year-fractions from today, increasing; last = maturity
    # The note's initial fixing, which the barriers are fractions of. None: a note
    # issued today, struck at today's spot (deep hedging simulates notes from
    # issue). A seasoned note must pass its fixing: before 2026-10-20 the daily mark
    # used today's spot, i.e. re-issued P008 at the money every day
    # (planning/position_lifecycle_plan.md).
    initial_level: float | None = None
    # Coupons missed at past observations that the memory feature still owes.
    initial_memory: int = 0


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
    memory = np.full(n_paths, float(spec.initial_memory))

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
            pv += np.where(paid_coupon, df * spec.notional * spec.coupon_rate * (1.0 + memory), 0.0)
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


def observation_time_grid(
    obs_times: np.ndarray, n_steps_per_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """A simulation time grid with every observation date exactly on a grid
    point, and the grid index of each observation.

    `next_power_of_two(n_steps_per_period * len(obs_times))` steps in total (the
    Brownian bridge needs a power of two), split across the periods in
    proportion to their length (largest remainder, at least one each), evenly
    spaced within a period. Replaced (2026-10-14) an evenly spaced grid from 0
    to maturity with each observation snapped to its nearest point: exact only
    when observations are evenly spaced from today, which a seasoned note's are
    not (its first period is a stub). For P008 that put observations up to 5.5
    days off their real dates, and the error jumped by ~10 days on some nights
    (planning/input_stability_plan.md). For evenly spaced observations the two
    grids coincide."""
    obs = np.asarray(obs_times, dtype=float)
    edges = np.concatenate([[0.0], obs])
    lengths = np.diff(edges)
    total = next_power_of_two(n_steps_per_period * len(obs))
    raw = total * lengths / edges[-1]
    counts = np.maximum(np.floor(raw).astype(int), 1)
    while counts.sum() < total:
        counts[int(np.argmax(raw - counts))] += 1
    while counts.sum() > total:
        counts[int(np.argmax(np.where(counts > 1, counts - raw, -np.inf)))] -= 1
    pieces = [np.linspace(edges[i], edges[i + 1], c + 1)[:-1] for i, c in enumerate(counts)]
    grid = np.concatenate([*pieces, [edges[-1]]])
    return grid, np.cumsum(counts)


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
    `autocallable_payoff` to the levels at each observation date, each of which
    is exactly a simulated grid point (`observation_time_grid`)."""
    T = float(spec.obs_times[-1])
    t_grid, obs_indices = observation_time_grid(spec.obs_times, n_steps_per_period)
    result = simulate_local_vol_paths(
        s0, T, grid, r, q, n_paths, len(t_grid) - 1, seed=seed, t_grid=t_grid
    )
    obs_levels = result.paths[:, obs_indices]

    initial_level = spec.initial_level if spec.initial_level is not None else s0
    discounted_payoff = autocallable_payoff(obs_levels, spec, initial_level, r)
    price = float(np.mean(discounted_payoff))
    stderr = float(np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff)))
    return price, stderr


@dataclass
class AutocallableGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    vanna: float
    volga: float


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
    only way path-dependent bump-and-reval Greeks aren't pure noise.

    Vanna and volga close the gap this incident found (README Step 13): a
    vega-only Greek set badly overstates the true vol sensitivity of a
    barrier-laden payoff whose effective exposure itself changes with spot and
    vol. Volga is a second central difference in vol, free to add once vega is
    computed as a central difference too. Vanna needs the four cross-bumped
    (spot, vol) corners on top of the five single-variable prices already
    computed for delta/gamma/vega — nine full reprices in total, all under the
    same seed so the mixed finite difference isn't noise."""
    h = s0 * spot_bump_frac

    def _price(s0_: float, grid_: LocalVolGrid) -> float:
        return price_autocallable(spec, s0_, grid_, r, q, n_paths, n_steps_per_period, seed)[0]

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

    return AutocallableGreeks(
        price=price0, delta=delta, gamma=gamma, vega=vega, vanna=vanna, volga=volga
    )
