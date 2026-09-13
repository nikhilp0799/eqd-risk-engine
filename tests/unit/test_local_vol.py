import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.vol.implied import EXTREME_K_MULTIPLE
from eqdrisk.vol.local_vol import (
    MIN_PILLARS_FOR_LOCAL_VOL,
    _finite_diff_k,
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


def test_finite_diff_k_exact_on_a_known_quadratic():
    """`_finite_diff_k`'s 3-point non-uniform stencil must be exact (not just
    approximate) on a quadratic, since a 3-point stencil is derived to be exact
    for up to 2nd-degree polynomials — a strong, deterministic check that
    doesn't depend on tolerances, same style as `test_g_matches_durrleman_g_exactly`."""
    k = np.array([-0.31, -0.12, 0.05, 0.22, 0.30])  # deliberately non-uniform spacing
    a, b, c = 2.0, 3.0, 5.0
    w = a + b * k + c * k**2
    dk_w, dkk_w = _finite_diff_k(k, w)

    # interior points (indices 1..3): exact first and second derivative
    for i in [1, 2, 3]:
        assert dk_w[i] == pytest.approx(b + 2 * c * k[i])
        assert dkk_w[i] == pytest.approx(2 * c)


def test_finite_diff_k_handles_clamped_flat_region_without_dividing_by_zero():
    """Adjacent points sharing the same k (both clamped into a flat wing) must
    give dk_w = dkk_w = 0, not a crash or a NaN from 0/0."""
    k = np.array([-0.5, -0.5, -0.5, 0.0, 0.3])
    w = np.array([1.0, 1.0, 1.0, 1.2, 1.5])
    dk_w, dkk_w = _finite_diff_k(k, w)
    assert np.all(np.isfinite(dk_w))
    assert np.all(np.isfinite(dkk_w))
    assert dk_w[0] == 0.0
    assert dk_w[1] == 0.0
    assert dkk_w[1] == 0.0


def test_build_local_vol_grid_row_based_derivatives_are_close_to_per_point_reference():
    """`build_local_vol_grid`'s grid-building hot path now derives dk_w/dkk_w from
    finite differences across neighboring grid points (smoothing only `w`
    across T), instead of `local_variance_at`'s per-point three-independent-
    splines approach. This is a real algorithm change (see the module
    docstring's "Fourth finding"), so it must be validated against the
    unchanged per-point reference, not just assumed equivalent: local vol at
    each grid point must be CLOSE (same order of magnitude, not wildly off) to
    what `local_variance_at` itself would compute at the exact same (k, T)."""
    surface = _svi_surface([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    s0 = 100.0
    fc = ForwardCurve(pillar_T=np.array([0.05, 1.0]), pillar_log_forward=np.log([s0, s0]))
    s_grid = np.linspace(60.0, 160.0, 40)
    t_grid = np.linspace(0.05, 0.9, 15)

    grid = build_local_vol_grid(surface, fc, s_grid, t_grid)
    assert grid is not None

    max_rel_diff = 0.0
    for ti, t in enumerate(t_grid):
        t_eff = max(t, 1e-6)
        forward_t = fc.forward(t_eff)
        atm = local_variance_at(surface, 0.0, t_eff)
        assert atm is not None
        atm_iv = float(np.sqrt(max(atm.w, 1e-12) / t_eff))
        k_cap = EXTREME_K_MULTIPLE * atm_iv * np.sqrt(t_eff)
        for si, s in enumerate(s_grid):
            # Clamped exactly like `_sigma_loc_row` itself does — comparing
            # against the reference at an UNCLAMPED k would be apples-to-oranges
            # in the deep wings, not a real test of the row-based method.
            k = float(np.clip(np.log(s / forward_t), -k_cap, k_cap))
            reference = local_variance_at(surface, k, t_eff)
            assert reference is not None
            row_based = grid.sigma_loc[ti, si] ** 2
            rel_diff = abs(row_based - reference.local_variance) / max(
                reference.local_variance, 1e-8
            )
            max_rel_diff = max(max_rel_diff, rel_diff)

    # A real algorithm change should differ only modestly from the per-point
    # reference on a smooth, well-behaved surface — not an exact-equality bar
    # (the two methods are mathematically different), but not wildly off either.
    assert max_rel_diff < 0.05
