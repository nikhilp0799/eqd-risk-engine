"""Local-vol Monte Carlo engine (README 6.2).

Log-spot Euler scheme, `numba`-jitted, driven by Sobol quasi-random normals mapped
through a Brownian bridge path construction (not naive sequential Sobol draws —
the bridge order is what makes Sobol's low-discrepancy property pay off along a
whole path: the first, best-equidistributed Sobol dimension sets the path's
terminal value, the highest-impact single number in the whole path; later
dimensions fill in progressively less important intermediate detail).

Deliberate, documented simplification: the bridge construction implemented here is
the classic power-of-two recursive-bisection algorithm (Glasserman, "Monte Carlo
Methods in Financial Engineering", 4.4) — simpler and more standard to implement
correctly than the general-n bridge index bookkeeping, at the cost of requiring
`n_steps` to be a power of two (`next_power_of_two` rounds up for the caller).
Antithetic variates are applied to the driving Sobol normals BEFORE the bridge
construction (negate z, not the resulting path) — valid because the bridge
construction is linear in z, so negating z exactly negates the constructed
Brownian path, preserving the paired-variance-reduction property.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from scipy.stats import norm, qmc

from eqdrisk.vol.local_vol import LocalVolGrid


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def sobol_normals(n_paths: int, n_steps: int, seed: int | None) -> np.ndarray:
    """(n_paths, n_steps) iid-marginal standard normals from a scrambled Sobol
    sequence — scrambling avoids the unscrambled sequence's degenerate all-zeros
    first point (which would map to -inf under the inverse normal CDF)."""
    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    uniforms = sampler.random(n_paths)
    uniforms = np.clip(uniforms, 1e-10, 1 - 1e-10)  # guard the exact 0/1 endpoints too
    return norm.ppf(uniforms)


def brownian_bridge_paths(z: np.ndarray, T: float) -> np.ndarray:
    """z: (n_paths, n_steps) iid N(0,1) in bridge-construction order, n_steps a
    power of two. Returns W: (n_paths, n_steps+1) standard Brownian motion paths
    over [0, T], W[:, 0] = 0, vectorised across paths (the bridge index pattern is
    identical for every path, only the driving normals differ).
    """
    n_paths, n = z.shape
    m = n.bit_length() - 1
    if (1 << m) != n:
        raise ValueError(f"n_steps must be a power of two, got {n}")

    W = np.zeros((n_paths, n + 1))
    W[:, n] = np.sqrt(T) * z[:, 0]
    idx = 1
    for k in range(1, m + 1):
        step = n // (1 << k)
        n_mid = 1 << (k - 1)
        lefts = np.arange(n_mid) * 2 * step
        mids = lefts + step
        rights = lefts + 2 * step
        t_left = lefts / n * T
        t_mid = mids / n * T
        t_right = rights / n * T
        w_left = W[:, lefts]
        w_right = W[:, rights]
        frac_right = (t_right - t_mid) / (t_right - t_left)
        frac_left = (t_mid - t_left) / (t_right - t_left)
        variance = (t_right - t_mid) * (t_mid - t_left) / (t_right - t_left)
        z_slice = z[:, idx : idx + n_mid]
        W[:, mids] = w_left * frac_right + w_right * frac_left + np.sqrt(variance) * z_slice
        idx += n_mid
    return W


@numba.njit(cache=True)
def _bilinear(
    s_grid: np.ndarray, t_grid: np.ndarray, sigma_loc: np.ndarray, s: float, t: float
) -> float:
    s_c = min(max(s, s_grid[0]), s_grid[-1])
    t_c = min(max(t, t_grid[0]), t_grid[-1])

    si = np.searchsorted(s_grid, s_c) - 1
    si = min(max(si, 0), len(s_grid) - 2)
    ti = np.searchsorted(t_grid, t_c) - 1
    ti = min(max(ti, 0), len(t_grid) - 2)

    s0, s1 = s_grid[si], s_grid[si + 1]
    t0, t1 = t_grid[ti], t_grid[ti + 1]
    fs = 0.0 if s1 == s0 else (s_c - s0) / (s1 - s0)
    ft = 0.0 if t1 == t0 else (t_c - t0) / (t1 - t0)

    v00 = sigma_loc[ti, si]
    v01 = sigma_loc[ti, si + 1]
    v10 = sigma_loc[ti + 1, si]
    v11 = sigma_loc[ti + 1, si + 1]
    return v00 * (1 - fs) * (1 - ft) + v01 * fs * (1 - ft) + v10 * (1 - fs) * ft + v11 * fs * ft


@numba.njit(cache=True)
def _euler_step_paths(
    s0: float,
    dW: np.ndarray,
    t_grid_sim: np.ndarray,
    r: float,
    q: float,
    s_grid: np.ndarray,
    t_grid_lv: np.ndarray,
    sigma_loc: np.ndarray,
) -> np.ndarray:
    """Log-spot Euler scheme: d(logS) = (r - q - 0.5*sigma_loc^2)dt + sigma_loc*dW.
    `dW`: (n_paths, n_steps) Brownian increments (already bridge-constructed
    upstream). Returns the full path matrix (n_paths, n_steps+1), S[:, 0] = s0."""
    n_paths, n_steps = dW.shape
    paths = np.empty((n_paths, n_steps + 1))
    for p in range(n_paths):
        log_s = np.log(s0)
        paths[p, 0] = s0
        for i in range(n_steps):
            t = t_grid_sim[i]
            dt = t_grid_sim[i + 1] - t_grid_sim[i]
            s = np.exp(log_s)
            sigma = _bilinear(s_grid, t_grid_lv, sigma_loc, s, t)
            log_s += (r - q - 0.5 * sigma * sigma) * dt + sigma * dW[p, i]
            paths[p, i + 1] = np.exp(log_s)
    return paths


@dataclass
class SimulationResult:
    paths: np.ndarray  # (n_paths, n_steps+1), including antithetic pairs if used
    t_grid: np.ndarray  # (n_steps+1,) simulation times, 0..T

    @property
    def terminal(self) -> np.ndarray:
        return self.paths[:, -1]

    def price_and_stderr(self, payoff: np.ndarray, discount_factor: float) -> tuple[float, float]:
        """`payoff` must be one value per simulated path (same order as `self.paths`).
        Returns (price, standard_error) — standard error uses the sample std of the
        (already antithetic-paired, if applicable) discounted payoffs / sqrt(n)."""
        discounted = discount_factor * payoff
        price = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        return price, stderr


def simulate_local_vol_paths(
    s0: float,
    T: float,
    grid: LocalVolGrid,
    r: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> SimulationResult:
    """`n_steps` is rounded up to the next power of two for the Brownian bridge.
    If `antithetic`, half of `n_paths` (also rounded to a power of two — Sobol's
    own balance property requires it, `scipy` warns otherwise) is drawn and negated
    to form mirrored pairs."""
    n_steps_pow2 = next_power_of_two(n_steps)
    t_grid_sim = np.linspace(0.0, T, n_steps_pow2 + 1)

    if antithetic:
        n_half = next_power_of_two((n_paths + 1) // 2)
        z_half = sobol_normals(n_half, n_steps_pow2, seed)
        z = np.concatenate([z_half, -z_half], axis=0)
    else:
        n_paths_pow2 = next_power_of_two(n_paths)
        z = sobol_normals(n_paths_pow2, n_steps_pow2, seed)

    W = brownian_bridge_paths(z, T)
    dW = np.diff(W, axis=1)

    paths = _euler_step_paths(s0, dW, t_grid_sim, r, q, grid.s_grid, grid.t_grid, grid.sigma_loc)
    return SimulationResult(paths=paths, t_grid=t_grid_sim)
