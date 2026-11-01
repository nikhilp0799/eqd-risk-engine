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
MIN_SHOCKED_VOL_POINTS = 1.0  # a vol-point shock never takes a node's implied vol below this


@dataclass
class NodeVolShock:
    """A vol move given per grid node (README 9.1's per-node factor moves): vol
    points added to the implied vol at each (k, T) on a rectangular grid,
    interpolated bilinearly between nodes and held flat beyond them.

    `multiplier` is the same move as a total-variance multiplier,
    ((iv + dsigma) / iv)^2 against the base day's implied vol at each node, so it
    plugs into `MarketShock.w_multiplier` and with it every pricer the other
    shocks already reach, vanillas and local-vol exotics alike."""

    t_values: np.ndarray  # ascending, years
    k_values: np.ndarray  # ascending
    dsigma: np.ndarray  # (len(t_values), len(k_values)), vol points
    multiplier: np.ndarray  # same shape

    @classmethod
    def from_vol_points(
        cls, t_values: np.ndarray, k_values: np.ndarray, base_iv: np.ndarray, dsigma: np.ndarray
    ) -> NodeVolShock:
        """`base_iv`: the base day's implied vol (vol points) at each node."""
        shocked = np.maximum(base_iv + dsigma, MIN_SHOCKED_VOL_POINTS)
        return cls(
            t_values=np.asarray(t_values, dtype=float),
            k_values=np.asarray(k_values, dtype=float),
            dsigma=shocked - base_iv,
            multiplier=(shocked / base_iv) ** 2,
        )

    def _interp(self, field: np.ndarray, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Bilinear in (k, T), flat beyond the outermost nodes; broadcasts."""
        k, T = np.broadcast_arrays(
            np.clip(k, self.k_values[0], self.k_values[-1]),
            np.clip(T, self.t_values[0], self.t_values[-1]),
        )
        kf, tf = k.ravel(), T.ravel()
        rows = np.stack([np.interp(kf, self.k_values, f) for f in field])  # (n_T, n)
        j = np.clip(np.searchsorted(self.t_values, tf, side="right") - 1, 0, len(self.t_values) - 2)
        t_lo, t_hi = self.t_values[j], self.t_values[j + 1]
        weight = (tf - t_lo) / (t_hi - t_lo)
        n = np.arange(kf.size)
        return (rows[j, n] * (1 - weight) + rows[j + 1, n] * weight).reshape(k.shape)

    def dsigma_at(self, k: float, T: float) -> float:
        return float(self._interp(self.dsigma, np.asarray(k), np.asarray(T)))

    def multiplier_at(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self._interp(self.multiplier, k, T)


@dataclass
class MarketShock:
    """All fields default to 0.0 (a pure no-op) — `mark_with_state` given the
    default `MarketShock()` reproduces the plain, unshocked daily mark exactly."""

    spot_shock_pct: float = 0.0  # e.g. -0.20 for a 20% spot decline
    vol_shock_pct: float = 0.0  # e.g. +0.25 for a 25% relative parallel vol increase
    skew_shock: float = 0.0  # additional multiplicative tilt vs. |k| (steepening)
    term_shock: float = 0.0  # additional tilt front-loaded below TERM_REFERENCE_T
    node_vol: NodeVolShock | None = None  # per-node vol-point move (VaR scenarios)

    @property
    def is_noop(self) -> bool:
        return (
            self.spot_shock_pct == 0.0
            and self.vol_shock_pct == 0.0
            and self.skew_shock == 0.0
            and self.term_shock == 0.0
            and self.node_vol is None
        )

    def w_multiplier(self, k: float, T: float) -> float:
        return float(self.w_multiplier_array(np.asarray(k), np.asarray(T)))

    def w_multiplier_array(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """`w_multiplier` over arrays of k and T (broadcast together)."""
        vol_mult = (1 + self.vol_shock_pct) ** 2
        skew_mult = 1 + self.skew_shock * np.abs(k)
        term_mult = 1 + self.term_shock * np.maximum(0.0, (TERM_REFERENCE_T - T) / TERM_REFERENCE_T)
        node_mult = self.node_vol.multiplier_at(k, T) if self.node_vol is not None else 1.0
        return np.asarray(vol_mult * skew_mult * term_mult * node_mult)


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
    t_eff = np.maximum(grid.t_grid, 1e-6)
    forward_t = np.array([forward_curve.forward(float(t)) for t in t_eff])
    k = np.log(grid.s_grid[None, :] / forward_t[:, None])
    sigma_loc = grid.sigma_loc * np.sqrt(shock.w_multiplier_array(k, t_eff[:, None]))
    return LocalVolGrid(
        s_grid=grid.s_grid, t_grid=grid.t_grid, sigma_loc=sigma_loc, n_floored=grid.n_floored
    )
