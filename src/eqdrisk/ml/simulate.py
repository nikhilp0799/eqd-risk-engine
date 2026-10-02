"""Training/evaluation path generation for deep hedging — a thin wrapper around
the EXISTING, already-tested `pricing/monte_carlo.py::simulate_local_vol_paths`
(Sobol + Brownian-bridge + Euler under the real calibrated local-vol grid), not a
reimplementation. See `planning/deep_hedging_plan.md`'s architecture note for why
no gradients are needed here: the hedge network is trained against a FIXED,
already-simulated price path (`requires_grad=False`), the same "simulate the
world once, then learn a policy against it" separation the deep-hedging
literature itself uses — only the hedge network's decisions and the resulting
P&L arithmetic need to be part of the autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from eqdrisk.ml.market import HedgingMarketInputs
from eqdrisk.pricing.monte_carlo import next_power_of_two, simulate_local_vol_paths


@dataclass
class SimulatedPaths:
    paths: torch.Tensor  # (n_paths, n_steps+1), requires_grad=False
    t_grid: np.ndarray  # (n_steps+1,) simulation times, 0..T


def simulate_training_paths(
    inputs: HedgingMarketInputs,
    n_paths: int,
    n_steps: int,
    seed: int,
    substeps: int = 1,
) -> SimulatedPaths:
    """Paths on a grid of `n_steps` (rounded up to a power of two) rebalancing
    times. With `substeps > 1`, the price is simulated `substeps` times finer
    and only the rebalancing times are kept: the hedge still acts on the coarse
    grid, but the path between those times follows the model properly.

    Why this exists (Phase 7, found 2026-10-01): one Euler step per quarter
    applies the local vol at the START of each quarter to the whole quarter.
    For NVDA on 2026-09-22 that gave the autocallable's first quarter a
    simulated vol of 17% against a converged 39% (64 steps per year).
    """
    if substeps < 1 or substeps & (substeps - 1):
        raise ValueError(f"substeps={substeps} must be a power of two")
    n_coarse = next_power_of_two(n_steps)
    result = simulate_local_vol_paths(
        s0=inputs.spot,
        T=inputs.T,
        grid=inputs.grid,
        r=inputs.r,
        q=inputs.q,
        n_paths=n_paths,
        n_steps=n_coarse * substeps,
        seed=seed,
    )
    # Both counts are powers of two, so the coarse times sit exactly on the fine grid.
    stride = substeps
    paths = torch.from_numpy(result.paths[:, ::stride].copy()).to(dtype=torch.float64)
    paths.requires_grad_(False)
    return SimulatedPaths(paths=paths, t_grid=result.t_grid[::stride])
