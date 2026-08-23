"""Dupire local volatility, stripped from the CALIBRATED surface (README 6.1) —
never from raw quotes, since differentiating noisy market data twice is hopeless.

    sigma_loc^2(k,T) = dT_w / g(k,T)

where g(k,T) is exactly Durrleman's butterfly condition from Step 4:

    g(k,T) = (1 - k*dk_w/(2*w))^2 - (dk_w^2/4)*(1/w + 0.25) + dkk_w/2

This is why Step 4's arbitrage enforcement is a prerequisite, not a nicety: if a
slice admits butterfly arbitrage, g(k,T) <= 0 and local variance blows up or goes
negative.

`w`, `dk_w`, `dkk_w` come from each calibrated expiry's own model — closed-form for
SVI (`SVIParams.first_derivative`/`second_derivative`), or a numeric central
finite difference for SSVI (no closed-form k-derivative in this codebase; the same
finite-difference pattern already used and tested for SSVI's own Durrleman check in
`vol/surface.py`). Only `dT_w` needs numerical treatment across expiries, per the
README's own framing: we only have a handful of calibrated `T` pillars.

**Real-data finding, investigated and fixed, not papered over:** an exact
interpolant (PCHIP) through every pillar reproduces each pillar's own value
exactly but its *derivative* amplifies ordinary pillar-to-pillar calibration
noise — real SPX data checked here showed `dT_w` at neighbouring pillars swinging
between roughly 0.006 and 0.035 with no smooth trend, purely from independent
per-expiry calibration noise (each expiry's SVI/SSVI fit is calibrated separately,
with no cross-expiry smoothness constraint). Feeding that directly into MC pricing
produced a real, measured local-vol MC reprice bias of tens of standard errors on
real data, even though the exact same Dupire formula reproduced closed-form prices
to <1 standard error on smooth synthetic surfaces — i.e. the *formula* was right,
the *raw derivative estimator* wasn't robust to real calibration noise, exactly
the "differentiating noisy data is hopeless" trap this module's own docstring
warns about, just one level removed (across expiries instead of across strikes).
Fixed by replacing the exact PCHIP interpolant with `scipy`'s automatic
generalised-cross-validation smoothing spline (`make_smoothing_spline`) for the
T-direction fit of `w`, `dk_w`, and `dkk_w` — trades exact pillar reproduction for
a smooth, noise-robust derivative, which is what `dT_w` actually needs. Falls back
to the exact PCHIP interpolant when there are too few pillars for the smoothing
spline to fit at all (`scipy` requires >= 5 points) — documented as a real,
reduced-accuracy regime for thin single names, not silently masked.

**Second real-data finding, also investigated and fixed:** even after the T-axis
smoothing fix above, MC reprice was still biased ~10-13% high on real SPX data.
Root cause: the MC engine's price grid necessarily spans a wide moneyness range
(paths can wander far from spot), but the calibrated SVI/SSVI smile has no real
market support that far out — extrapolating a curved smile in log-moneyness `k`
into deep wings amplifies without bound (SPX's real smile produced a local vol of
~80% at 50% moneyness, vs ~14% ATM, purely from parametric extrapolation past
where any real quote exists). Paths that wander into that region pick up hugely
inflated variance, which then contaminates the price of every option, not just
deep OTM ones. Fixed the same way Step 3 already draws this exact line for raw
quotes (`vol.implied.EXTREME_K_MULTIPLE`, "don't trust the smile past
`EXTREME_K_MULTIPLE * atm_vol * sqrt(T)`"): the smile is evaluated flat beyond
that same per-`T` cap, reusing the established threshold rather than inventing a
new one.

**Third finding — a real fix that was tried, measured, and reverted:** total
variance is exactly zero at T=0 for every k (a hard fact, not an approximation),
so folding a synthetic (T=0, w=0) point into the SAME T-interpolant used for the
real pillars looked like a clean way to fix the "no data below the first pillar"
problem outright. Measured effect: it did fix that specific problem, but a
smoothing spline is a *global* fit, and adding that anchor point measurably
distorted the *interior* fit too — a synthetic case that reproduced a calibrated
vanilla to <1 standard error with an anchor-free interior fit regressed to several
standard errors once the anchor was folded in. Reverted in favour of the explicit,
local, `T <= T_pillars[0]` special case below, which fixes the boundary without
touching the interior fit at all. Worth remembering: a "more correct" boundary
condition for a *global* smoothing method can still be a net loss if it perturbs
everything else the method was already fitting well — locality would have needed
verifying either way, and here it mattered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, make_smoothing_spline

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.vol.implied import EXTREME_K_MULTIPLE
from eqdrisk.vol.ssvi import SSVIParams
from eqdrisk.vol.svi import SVIParams

MIN_PILLARS_FOR_LOCAL_VOL = 2  # need >= 2 expiries to get any dT_w at all
MIN_PILLARS_FOR_SMOOTHING = 5  # scipy's make_smoothing_spline's own minimum
K_DERIVATIVE_H = 1e-4  # finite-difference step for SSVI's numeric k-derivatives
LOCAL_VARIANCE_FLOOR = 1e-6  # sigma_loc^2 floor when interpolation dips non-positive


def _t_interpolant(T_pillars: np.ndarray, values: np.ndarray):
    """Smoothing spline (GCV-selected smoothness, robust to per-expiry calibration
    noise) when there are enough pillars, else an exact PCHIP fallback — see the
    module docstring for why an exact interpolant's derivative isn't safe to use
    here in general, and why a fallback is still needed for thin names."""
    if len(T_pillars) >= MIN_PILLARS_FOR_SMOOTHING:
        return make_smoothing_spline(T_pillars, values)
    return PchipInterpolator(T_pillars, values, extrapolate=True)


def _slice_w_dk_dkk(
    surface_row: pd.Series, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, dk_w, dkk_w) for one calibrated expiry slice, at log-moneyness `k`
    relative to THAT expiry's own forward."""
    if surface_row["model"] == "SVI":
        params = SVIParams(
            a=float(surface_row["a"]),
            b=float(surface_row["b"]),
            rho=float(surface_row["rho"]),
            m=float(surface_row["m"]),
            sigma=float(surface_row["sigma"]),
        )
        return (
            np.asarray(params.total_variance(k)),
            np.asarray(params.first_derivative(k)),
            np.asarray(params.second_derivative(k)),
        )

    ssvi = SSVIParams(rho=float(surface_row["rho"]), eta=float(surface_row["eta"]))
    theta = float(surface_row["theta"])
    h = K_DERIVATIVE_H
    w = ssvi.total_variance(k, theta)
    w_up = ssvi.total_variance(k + h, theta)
    w_dn = ssvi.total_variance(k - h, theta)
    dk_w = (w_up - w_dn) / (2 * h)
    dkk_w = (w_up - 2 * w + w_dn) / h**2
    return np.asarray(w), np.asarray(dk_w), np.asarray(dkk_w)


@dataclass
class LocalVarianceResult:
    w: float
    dk_w: float
    dkk_w: float
    dT_w: float
    g: float
    local_variance_raw: float  # before flooring — negative means an honest arb-adjacent finding
    local_variance: float  # floored at LOCAL_VARIANCE_FLOOR


def local_variance_at(
    surface_for_underlying: pd.DataFrame, k: float, T: float
) -> LocalVarianceResult | None:
    """`surface_for_underlying`: that day's `vol_surface` rows for ONE underlying
    (any number of expiries, any mix of SVI/SSVI per Step 4's own per-slice choice).
    Returns None if fewer than `MIN_PILLARS_FOR_LOCAL_VOL` expiries are calibrated —
    there is genuinely no T-direction information to strip a local vol from.
    """
    pillars = surface_for_underlying.sort_values("T")
    if len(pillars) < MIN_PILLARS_FOR_LOCAL_VOL:
        return None

    T_pillars = pillars["T"].to_numpy(dtype=float)
    w_vals = np.empty(len(pillars))
    dk_w_vals = np.empty(len(pillars))
    dkk_w_vals = np.empty(len(pillars))
    for i, (_, row) in enumerate(pillars.iterrows()):
        w_vals[i], dk_w_vals[i], dkk_w_vals[i] = _slice_w_dk_dkk(row, np.asarray(k))

    # T-interpolant is fit to the REAL pillars only. Adding a synthetic (T=0, w=0)
    # anchor point was tried and rejected: total variance genuinely is exactly zero
    # at T=0 (no time has elapsed to accumulate any), but a smoothing spline is a
    # GLOBAL fit, and folding that anchor into the same fit measurably distorted
    # the interior curve too (confirmed on real data: a case that reproduced a
    # calibrated vanilla to <1 standard error with an anchor-free fit regressed to
    # several standard errors once the T=0 point was added to the same spline).
    # Below the first real pillar there is genuinely no data to interpolate at all
    # — handled explicitly below as a flat-local-vol assumption, not folded into
    # the interior fit.
    T_clamped = float(np.clip(T, T_pillars[0], T_pillars[-1]))
    w_interp = _t_interpolant(T_pillars, w_vals)
    dk_w_interp = _t_interpolant(T_pillars, dk_w_vals)
    dkk_w_interp = _t_interpolant(T_pillars, dkk_w_vals)

    w = float(w_interp(T_clamped))
    dk_w = float(dk_w_interp(T_clamped))
    dkk_w = float(dkk_w_interp(T_clamped))
    dT_w = float(w_interp.derivative()(T_clamped))

    if T <= T_pillars[0]:
        # No data before the first calibrated expiry: assume local vol is flat
        # from 0 to T_1, at exactly the level that reproduces the first pillar's
        # own total variance (dT_w = w(k, T_1) / T_1, not the interior fit's
        # derivative AT T_1, which reflects the curve's shape *after* T_1 too).
        dT_w = w / T_pillars[0]

    g = (1 - k * dk_w / (2 * w)) ** 2 - (dk_w**2 / 4) * (1 / w + 0.25) + dkk_w / 2
    local_variance_raw = dT_w / g if g != 0 else np.nan
    local_variance = max(local_variance_raw, LOCAL_VARIANCE_FLOOR)

    return LocalVarianceResult(
        w=w,
        dk_w=dk_w,
        dkk_w=dkk_w,
        dT_w=dT_w,
        g=g,
        local_variance_raw=local_variance_raw,
        local_variance=local_variance,
    )


@dataclass
class LocalVolGrid:
    """A precomputed sigma_loc(S, t) grid — the handoff point to Step 6.2's
    numba-jitted MC engine, which cannot call scipy interpolators directly inside a
    jitted loop. MC does cheap bilinear lookups into `sigma_loc` at simulation time;
    all the scipy/PCHIP work happens once, up front, here.
    """

    s_grid: np.ndarray
    t_grid: np.ndarray
    sigma_loc: np.ndarray  # shape (len(t_grid), len(s_grid))
    n_floored: int  # how many grid points needed the non-positive-variance floor


def build_local_vol_grid(
    surface_for_underlying: pd.DataFrame,
    forward_curve: ForwardCurve,
    s_grid: np.ndarray,
    t_grid: np.ndarray,
) -> LocalVolGrid | None:
    """Evaluate local vol on an (s_grid x t_grid) rectangle. Returns None if the
    underlying has too few calibrated expiries (see `local_variance_at`)."""
    pillars = surface_for_underlying.sort_values("T")
    if len(pillars) < MIN_PILLARS_FOR_LOCAL_VOL:
        return None

    sigma_loc = np.empty((len(t_grid), len(s_grid)))
    n_floored = 0
    for ti, t in enumerate(t_grid):
        t_eff = max(t, 1e-6)  # t=0 has no forward-implied k; treat as the first instant after
        forward_t = forward_curve.forward(t_eff)

        atm = local_variance_at(surface_for_underlying, 0.0, t_eff)
        assert atm is not None  # already checked len(pillars) >= MIN_PILLARS_FOR_LOCAL_VOL
        atm_iv = float(np.sqrt(max(atm.w, 1e-12) / t_eff))
        k_cap = EXTREME_K_MULTIPLE * atm_iv * np.sqrt(t_eff)

        for si, s in enumerate(s_grid):
            k = float(np.log(s / forward_t))
            k_clamped = float(np.clip(k, -k_cap, k_cap))
            result = local_variance_at(surface_for_underlying, k_clamped, t_eff)
            assert result is not None
            if result.local_variance_raw < LOCAL_VARIANCE_FLOOR:
                n_floored += 1
            sigma_loc[ti, si] = np.sqrt(result.local_variance)

    return LocalVolGrid(s_grid=s_grid, t_grid=t_grid, sigma_loc=sigma_loc, n_floored=n_floored)


def bilinear_lookup(grid: LocalVolGrid, s: float, t: float) -> float:
    """sigma_loc(s, t) via bilinear interpolation on the precomputed grid, clamped
    to the grid's own range (flat extrapolation) — a pure-numpy reference
    implementation; the numba MC kernel in 6.2 inlines the same logic on raw arrays
    for jit-compatibility."""
    s_c = min(max(s, grid.s_grid[0]), grid.s_grid[-1])
    t_c = min(max(t, grid.t_grid[0]), grid.t_grid[-1])

    si = int(np.searchsorted(grid.s_grid, s_c, side="right") - 1)
    si = min(max(si, 0), len(grid.s_grid) - 2)
    ti = int(np.searchsorted(grid.t_grid, t_c, side="right") - 1)
    ti = min(max(ti, 0), len(grid.t_grid) - 2)

    s0, s1 = grid.s_grid[si], grid.s_grid[si + 1]
    t0, t1 = grid.t_grid[ti], grid.t_grid[ti + 1]
    fs = 0.0 if s1 == s0 else (s_c - s0) / (s1 - s0)
    ft = 0.0 if t1 == t0 else (t_c - t0) / (t1 - t0)

    v00 = grid.sigma_loc[ti, si]
    v01 = grid.sigma_loc[ti, si + 1]
    v10 = grid.sigma_loc[ti + 1, si]
    v11 = grid.sigma_loc[ti + 1, si + 1]
    return float(
        v00 * (1 - fs) * (1 - ft) + v01 * fs * (1 - ft) + v10 * (1 - fs) * ft + v11 * fs * ft
    )
