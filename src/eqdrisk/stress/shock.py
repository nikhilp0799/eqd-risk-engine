"""A market shock: spot move + vol-surface move, applied to a `MarketState`
before repricing (README 11.1/11.2). Used by both the historical-replay
scenarios and the hypothetical spot/vol grid — a shock is just a
transformation of today's real market state, so the exact same per-position
pricers Steps 5-7 already built and tested run unmodified underneath it.

**Vol-surface shock model on `LocalVolGrid`s — a real approximation, measured,
not assumed (a first version of this module claimed the parallel-shock case
was exact; it wasn't, see below):**

Scaling total variance `w(k,T)` by a constant `C` at every point (`C =
(1+vol_shock_pct)^2` for a pure parallel shock) scales `w`, `dw/dk`, `d^2w/dk^2`
and `dw/dT` all EXACTLY by `C` — that part is provable and confirmed
numerically (SVI's `w(k) = a + b(...)` is linear in `(a,b)`, so scaling both by
`C` scales `w` and its `k`-derivatives by `C` at fixed shape). But Durrleman's
`g(k)` — Dupire's denominator — is NOT scale-invariant under this rescaling:
`g` contains an additive `1/4` constant (`(dk_w^2/4)*(1/w + 1/4)`) that does
NOT scale with `w`, so `g` changes shape, not just magnitude, and local
variance (`dT_w / g`) does not scale by exactly `C`. Measured directly: for a
realistic SVI slice, `sigma_loc*sqrt(C)` (this module's cheap shortcut) misses
the fully-correct value (rebuild the grid from a `C`-rescaled surface) by
~7% at a 10%-relative vol shock, ~18% at 25%, and ~40% at the ladder's most
extreme 50% shock — growing with shock size, as expected since the
approximation is a small-perturbation argument being pushed past where it's
accurate.

Rebuilding the grid exactly (re-run Dupire from a `C`-rescaled surface) is
cheap for ONE shock but 11.2's hypothetical grid needs 42 combinations per
MC-priced position — this module accepts the cheap, quantified-above
approximation in exchange for that shock sweep finishing in seconds rather
than minutes, and reports it as exactly that: a fast-sweep approximation,
larger at extreme shocks, not a substitute for a full re-calibration. Skew-
steepening and term-structure shocks use the identical mechanism and carry the
same caveat (they were never claimed exact in the first place, since `C`
itself varies with `k`/`T` for those).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.vol.local_vol import LocalVolGrid, local_variance_at

TERM_REFERENCE_T = 1.0  # 1y: shocks front-load vol shorter than this, per README 11.2


@dataclass
class MarketShock:
    """All fields default to 0.0 (a pure no-op) — `mark_with_state` given the
    default `MarketShock()` reproduces the plain, unshocked daily mark exactly."""

    spot_shock_pct: float = 0.0  # e.g. -0.20 for a 20% spot decline
    vol_shock_pct: float = 0.0  # e.g. +0.25 for a 25% relative parallel vol increase
    skew_shock: float = 0.0  # additional multiplicative tilt vs. |k| (steepening)
    term_shock: float = 0.0  # additional tilt front-loaded below TERM_REFERENCE_T

    @property
    def is_noop(self) -> bool:
        return (
            self.spot_shock_pct == 0.0
            and self.vol_shock_pct == 0.0
            and self.skew_shock == 0.0
            and self.term_shock == 0.0
        )

    def w_multiplier(self, k: float, T: float) -> float:
        vol_mult = (1 + self.vol_shock_pct) ** 2
        skew_mult = 1 + self.skew_shock * abs(k)
        term_mult = 1 + self.term_shock * max(0.0, (TERM_REFERENCE_T - T) / TERM_REFERENCE_T)
        return vol_mult * skew_mult * term_mult


def shocked_spot(spot: float, shock: MarketShock) -> float:
    return spot * (1 + shock.spot_shock_pct)


def shocked_w(surface: pd.DataFrame, k: float, T: float, shock: MarketShock) -> float:
    """`surface`: a `vol_surface` DataFrame for one underlying (see
    `vol.local_vol.local_variance_at`). Raises if there isn't enough calibrated
    data to interpolate at this (k, T) at all."""
    lv = local_variance_at(surface, k, T)
    if lv is None:
        raise ValueError(f"not enough calibrated expiries to shock-price at T={T:.3f}")
    return lv.w * shock.w_multiplier(k, T)


def shock_local_vol_grid(
    grid: LocalVolGrid, forward_curve: ForwardCurve, shock: MarketShock
) -> LocalVolGrid:
    """Rescale a precomputed local-vol grid in place of re-stripping Dupire — a
    fast, quantified APPROXIMATION for every shock type, including a pure
    parallel vol shock (see module docstring for the measured error size)."""
    if shock.is_noop:
        return grid
    sigma_loc = np.empty_like(grid.sigma_loc)
    for ti, t in enumerate(grid.t_grid):
        t_eff = max(t, 1e-6)
        forward_t = forward_curve.forward(t_eff)
        for si, s in enumerate(grid.s_grid):
            k = float(np.log(s / forward_t))
            multiplier = shock.w_multiplier(k, t_eff)
            sigma_loc[ti, si] = grid.sigma_loc[ti, si] * np.sqrt(multiplier)
    return LocalVolGrid(
        s_grid=grid.s_grid, t_grid=grid.t_grid, sigma_loc=sigma_loc, n_floored=grid.n_floored
    )
