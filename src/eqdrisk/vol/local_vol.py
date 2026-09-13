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

**Fourth finding — a real performance bottleneck, root-caused (not guessed at)
and fixed twice:** building a full local-vol grid for one underlying calls
`local_variance_at` once per (strike, time) grid point (~6,000 points), and each
call independently smoothing-spline-fits THREE quantities (`w`, `dk_w`, `dkk_w`)
across the T-pillars — profiled (`cProfile`) at ~90% of a full portfolio mark's
wall time. First fix: each grid ROW is a pure, independent function of its own
inputs, so rows now run across a process pool rather than sequentially (`README
Step 16`) — a pure engineering change, no math difference. Second fix (this
one): researched what production Dupire-formula implementations actually do
(analytic derivatives from the calibrated smile's own closed form wherever
possible, rather than independently re-fitting related quantities) and found
this module was smoothing `dk_w`/`dkk_w` separately from `w`, when they're
derivatives of the SAME underlying total-variance surface. `_sigma_loc_row` (the
grid-building hot path only — `local_variance_at` itself is UNCHANGED, for every
other caller) now smooths only `w` across T per grid point (one spline fit
instead of three) and derives `dk_w`/`dkk_w` via finite differences across
NEIGHBORING grid points already being computed for the same row, rather than
independently T-smoothing each pillar's own closed-form k-derivative. A
from-scratch reimplementation of scipy's private auto-GCV smoothing-parameter
search was tried FIRST (to reuse one selected parameter across nearby strikes)
and measured to be ~42x slower per reference point than scipy's own private
path before being integrated any further — abandoned once measured, replaced by
this simpler, lower-risk, better-motivated fix instead.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

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


def slice_total_variance(surface_row: pd.Series, k: np.ndarray | float) -> np.ndarray:
    """Public wrapper around `_slice_w_dk_dkk` for callers that only need `w`
    (total variance), not its k-derivatives — e.g. `pricing/varswap.py`, which
    prices a static replication strip off one calibrated expiry's own smile."""
    w, _, _ = _slice_w_dk_dkk(surface_row, np.asarray(k))
    return w


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


def _row_w_and_dT_w(
    pillars: pd.DataFrame, T_pillars: np.ndarray, k: float, T: float
) -> tuple[float, float]:
    """`w(k, T)` and `dT_w(k, T)` via ONLY a w-across-T smoothing spline — the
    grid-building hot path's analog of `local_variance_at`, deliberately
    skipping that function's separate `dk_w`/`dkk_w` T-smoothing splines.
    `_sigma_loc_row` derives those instead from finite differences across
    neighboring, already-computed grid points in the SAME row (see the module
    docstring's "Fourth finding"): one smoothed w-surface, differentiated once,
    rather than three independently-smoothed quantities that are supposed to be
    related in the first place. Same `T <= T_pillars[0]` flat-local-vol special
    case as `local_variance_at`."""
    k_arr = np.asarray(k)
    w_vals = np.array(
        [float(np.asarray(_slice_w_dk_dkk(row, k_arr)[0])) for _, row in pillars.iterrows()]
    )
    T_clamped = float(np.clip(T, T_pillars[0], T_pillars[-1]))
    w_interp = _t_interpolant(T_pillars, w_vals)
    w = float(w_interp(T_clamped))
    dT_w = float(w_interp.derivative()(T_clamped))
    if T <= T_pillars[0]:
        dT_w = w / T_pillars[0]
    return w, dT_w


def _finite_diff_k(k: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of `w` w.r.t. `k`, via a standard 3-point
    NON-uniform finite-difference stencil across adjacent entries of an
    already-sorted-ascending `k` array (guaranteed by construction: `k` comes
    from `log(s / forward)` at increasing `s`, and clamping preserves
    monotonicity). Reuses grid points already being computed for the row — no
    extra evaluations, unlike a fresh central-difference bump.

    Interior points use the exact 3-point formula (confirmed exact on a
    synthetic quadratic before being trusted here — see
    `tests/unit/test_local_vol.py`); the first/last points fall back to a
    one-sided 2-point difference for `dk_w` (a 2-point stencil can't estimate
    curvature, so `dkk_w` is left at 0 there). Any pair of adjacent points
    sharing the same `k` (both clamped into the flat `EXTREME_K_MULTIPLE` wing)
    gives `dk_w = dkk_w = 0` there by explicit guard, not division-by-zero —
    consistent with the wing being flat by construction, not a special case
    invented for this stencil.
    """
    n = len(k)
    dk_w = np.zeros(n)
    dkk_w = np.zeros(n)
    for i in range(n):
        if i == 0:
            h1 = k[1] - k[0]
            dk_w[i] = 0.0 if h1 == 0 else (w[1] - w[0]) / h1
            continue
        if i == n - 1:
            h0 = k[i] - k[i - 1]
            dk_w[i] = 0.0 if h0 == 0 else (w[i] - w[i - 1]) / h0
            continue
        h0 = k[i] - k[i - 1]
        h1 = k[i + 1] - k[i]
        if h0 == 0 and h1 == 0:
            continue
        if h0 == 0:
            dk_w[i] = 0.0 if h1 == 0 else (w[i + 1] - w[i]) / h1
            continue
        if h1 == 0:
            dk_w[i] = (w[i] - w[i - 1]) / h0
            continue
        dk_w[i] = (
            (-h1 / (h0 * (h0 + h1))) * w[i - 1]
            + ((h1 - h0) / (h0 * h1)) * w[i]
            + (h0 / (h1 * (h0 + h1))) * w[i + 1]
        )
        dkk_w[i] = (
            (2 / (h0 * (h0 + h1))) * w[i - 1]
            - (2 / (h0 * h1)) * w[i]
            + (2 / (h1 * (h0 + h1))) * w[i + 1]
        )
    return dk_w, dkk_w


def _sigma_loc_row(
    surface_for_underlying: pd.DataFrame,
    forward_curve: ForwardCurve,
    s_grid: np.ndarray,
    t: float,
) -> tuple[np.ndarray, int]:
    """One `t_grid` row of `build_local_vol_grid`'s (s_grid x t_grid) rectangle —
    pulled out to a plain, picklable function so it can run in a separate process.
    Each row is a pure function of its own inputs (no shared state with any other
    row), which is what makes this embarrassingly parallel across rows, on top
    of the 3x-fewer-fits-per-point win from `_row_w_and_dT_w`/`_finite_diff_k`
    within a row."""
    t_eff = max(t, 1e-6)  # t=0 has no forward-implied k; treat as the first instant after
    forward_t = forward_curve.forward(t_eff)

    pillars = surface_for_underlying.sort_values("T")
    T_pillars = pillars["T"].to_numpy(dtype=float)

    atm_w, _ = _row_w_and_dT_w(pillars, T_pillars, 0.0, t_eff)
    atm_iv = float(np.sqrt(max(atm_w, 1e-12) / t_eff))
    k_cap = EXTREME_K_MULTIPLE * atm_iv * np.sqrt(t_eff)

    k_arr = np.array([float(np.clip(np.log(s / forward_t), -k_cap, k_cap)) for s in s_grid])
    w_arr = np.empty(len(s_grid))
    dT_w_arr = np.empty(len(s_grid))
    for si, k in enumerate(k_arr):
        w_arr[si], dT_w_arr[si] = _row_w_and_dT_w(pillars, T_pillars, float(k), t_eff)

    dk_w_arr, dkk_w_arr = _finite_diff_k(k_arr, w_arr)

    # Points clamped into the flat `EXTREME_K_MULTIPLE` wing (k == +/- k_cap)
    # share an identical k with their clamped neighbors, so the finite-
    # difference stencil above correctly reads their local slope as zero — but
    # that's not the same as the true smile's curvature AT the cap boundary
    # itself, which is what `local_variance_at`'s own per-point method uses
    # there. Measured, not assumed: up to ~69% relative local-variance
    # difference at the cap on a synthetic test surface, vs. <1% everywhere in
    # the interior (`tests/unit/test_local_vol.py`). Recomputed exactly once
    # per distinct capped boundary value actually present in this row (not once
    # per point sharing it) via the same closed-form-per-pillar-then-T-smooth
    # approach `local_variance_at` uses, to match its already-validated wing
    # behavior exactly rather than silently changing it.
    T_clamped = float(np.clip(t_eff, T_pillars[0], T_pillars[-1]))
    for boundary_k in (-k_cap, k_cap):
        capped_mask = k_arr == boundary_k
        if not np.any(capped_mask):
            continue
        dk_w_vals = np.empty(len(pillars))
        dkk_w_vals = np.empty(len(pillars))
        for i, (_, prow) in enumerate(pillars.iterrows()):
            _, dk_w_vals[i], dkk_w_vals[i] = _slice_w_dk_dkk(prow, np.asarray(boundary_k))
        dk_w_boundary = float(_t_interpolant(T_pillars, dk_w_vals)(T_clamped))
        dkk_w_boundary = float(_t_interpolant(T_pillars, dkk_w_vals)(T_clamped))
        dk_w_arr[capped_mask] = dk_w_boundary
        dkk_w_arr[capped_mask] = dkk_w_boundary

    row = np.empty(len(s_grid))
    n_floored = 0
    for si in range(len(s_grid)):
        w, dk_w, dkk_w, dT_w, k = w_arr[si], dk_w_arr[si], dkk_w_arr[si], dT_w_arr[si], k_arr[si]
        g = (1 - k * dk_w / (2 * w)) ** 2 - (dk_w**2 / 4) * (1 / w + 0.25) + dkk_w / 2
        local_variance_raw = dT_w / g if g != 0 else np.nan
        local_variance = max(local_variance_raw, LOCAL_VARIANCE_FLOOR)
        if local_variance_raw < LOCAL_VARIANCE_FLOOR:
            n_floored += 1
        row[si] = np.sqrt(local_variance)
    return row, n_floored


def build_local_vol_grid(
    surface_for_underlying: pd.DataFrame,
    forward_curve: ForwardCurve,
    s_grid: np.ndarray,
    t_grid: np.ndarray,
) -> LocalVolGrid | None:
    """Evaluate local vol on an (s_grid x t_grid) rectangle. Returns None if the
    underlying has too few calibrated expiries (see `local_variance_at`).

    **Performance (README Step 16):** each `t_grid` row's smoothing-spline fits
    (the dominant real cost — measured at ~90% of a full portfolio mark's wall
    time before this fix) are completely independent of every other row, so rows
    are computed across a process pool rather than sequentially. This changes
    nothing about the computed values (same pure function, same inputs, same
    floating-point arithmetic) — it is purely an engineering speedup, not a
    modeling change, and doesn't touch `local_variance_at`'s own math at all.
    """
    pillars = surface_for_underlying.sort_values("T")
    if len(pillars) < MIN_PILLARS_FOR_LOCAL_VOL:
        return None

    worker = partial(_sigma_loc_row, surface_for_underlying, forward_curve, s_grid)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        rows = list(executor.map(worker, t_grid))

    sigma_loc = np.array([row for row, _ in rows])
    n_floored = sum(n for _, n in rows)
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
