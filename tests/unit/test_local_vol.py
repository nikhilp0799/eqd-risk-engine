import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.vol.local_vol import (
    MIN_PILLARS_FOR_LOCAL_VOL,
    build_local_vol_grid,
    local_variance_at,
)
from eqdrisk.vol.svi import SVIParams, durrleman_g


def _svi_surface(Ts: list[float], rho: float = -0.4) -> pd.DataFrame:
    """A smooth, realistic (non-degenerate) synthetic SVI term structure: flat 20%
    ATM vol, mild constant skew — the well-behaved fixture every debugging step in
    this module's real-data investigation converged back to as the ground truth."""
    rows = []
    for T in Ts:
        b, m, sigma = 0.10 * np.sqrt(T), 0.0, 0.10
        offset = b * sigma * np.sqrt(1 - rho**2)
        a = (0.20**2) * T - offset
        rows.append(
            {
                "T": T,
                "model": "SVI",
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "eta": None,
                "theta": None,
            }
        )
    return pd.DataFrame(rows)


def test_g_matches_durrleman_g_exactly():
    """The Dupire denominator must be bit-identical to the already-tested
    Durrleman g — it's the same quantity, just evaluated through the T-interpolated
    (k, T) surface instead of a single fixed-T SVIParams object. Uses only 3 real
    pillars (+ the T=0 anchor = 4 total) to stay below `MIN_PILLARS_FOR_SMOOTHING`
    and exercise the exact-interpolation (PCHIP) path — the smoothing-spline path
    trades exact pillar reproduction for noise robustness by design (see the
    module docstring) and is checked separately, not here."""
    surface = _svi_surface([0.3, 0.6, 1.0])
    pillar_row = surface.iloc[1]  # T=0.6 is an exact pillar
    params = SVIParams(
        a=pillar_row["a"],
        b=pillar_row["b"],
        rho=pillar_row["rho"],
        m=pillar_row["m"],
        sigma=pillar_row["sigma"],
    )
    for k in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        g_ref = float(durrleman_g(params, np.array([k]))[0])
        result = local_variance_at(surface, k, 0.6)
        assert result is not None
        assert result.g == pytest.approx(g_ref)


def test_local_variance_at_returns_none_below_minimum_pillars():
    surface = _svi_surface([0.5])
    assert len(surface) < MIN_PILLARS_FOR_LOCAL_VOL
    assert local_variance_at(surface, 0.0, 0.5) is None


def test_before_first_pillar_uses_flat_vol_at_first_pillar_level():
    """There is genuinely no calibrated data before the first expiry. Rather than
    extrapolate the interior T-interpolant's fitted derivative there (which
    reflects the curve's shape *after* the first pillar too, and was found on real
    NVDA/SPX data to be wildly inconsistent with the first pillar's own implied
    vol — see the module docstring), `dT_w` for T <= T_1 is defined as exactly
    w(k, T_1) / T_1: a flat local-vol assumption from 0 to T_1 that reproduces the
    first pillar's own total variance. This must hold for ANY T in (0, T_1],
    not just in the T -> 0 limit."""
    surface = _svi_surface([0.1, 0.3, 0.6, 1.0, 1.5])
    k = 0.05
    at_pillar = local_variance_at(surface, k, 0.1)
    assert at_pillar is not None
    expected_dT_w = at_pillar.w / 0.1

    for T in [1e-8, 0.03, 0.07, 0.1]:
        result = local_variance_at(surface, k, T)
        assert result is not None
        assert result.dT_w == pytest.approx(expected_dT_w)


def test_local_variance_positive_for_realistic_smooth_surface():
    surface = _svi_surface([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    for T in [0.02, 0.15, 0.4, 0.9]:
        for k in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            result = local_variance_at(surface, k, T)
            assert result is not None
            assert result.local_variance > 0


def test_build_local_vol_grid_none_when_too_few_pillars():
    surface = _svi_surface([0.5])
    fc = ForwardCurve(pillar_T=np.array([0.5]), pillar_log_forward=np.array([np.log(100.0)]))
    grid = build_local_vol_grid(surface, fc, np.linspace(50, 150, 5), np.linspace(0.1, 0.5, 5))
    assert grid is None


def test_build_local_vol_grid_reports_floored_points_honestly():
    """A deliberately butterfly-arbitrage-violating surface (not repaired, unlike
    every real slice this module actually consumes downstream of Step 4) should
    produce grid points needing the non-positive-variance floor, and the grid must
    report that count rather than silently hide it."""
    bad = SVIParams(a=0.01, b=5.0, rho=0.0, m=0.0, sigma=0.05)
    rows = []
    for T in [0.2, 0.5]:
        rows.append(
            {
                "T": T,
                "model": "SVI",
                "a": bad.a,
                "b": bad.b,
                "rho": bad.rho,
                "m": bad.m,
                "sigma": bad.sigma,
                "eta": None,
                "theta": None,
            }
        )
    surface = pd.DataFrame(rows)
    fc = ForwardCurve(pillar_T=np.array([0.2, 0.5]), pillar_log_forward=np.log([100.0, 101.0]))
    grid = build_local_vol_grid(surface, fc, np.linspace(60, 160, 15), np.linspace(0.05, 0.5, 10))
    assert grid is not None
    assert grid.n_floored > 0
