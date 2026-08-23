import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.pricing.blackscholes import call_price
from eqdrisk.pricing.monte_carlo import (
    brownian_bridge_paths,
    next_power_of_two,
    simulate_local_vol_paths,
    sobol_normals,
)
from eqdrisk.vol.local_vol import LocalVolGrid, build_local_vol_grid


def test_next_power_of_two():
    assert next_power_of_two(1) == 1
    assert next_power_of_two(50) == 64
    assert next_power_of_two(64) == 64
    assert next_power_of_two(65) == 128


def test_sobol_normals_shape_and_finiteness():
    z = sobol_normals(n_paths=32, n_steps=8, seed=0)
    assert z.shape == (32, 8)
    assert np.all(np.isfinite(z))


def test_brownian_bridge_paths_start_at_zero():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 16))
    W = brownian_bridge_paths(z, T=1.0)
    assert W.shape == (100, 17)
    assert np.all(W[:, 0] == 0.0)


def test_brownian_bridge_paths_have_correct_second_moments():
    """Statistical validation of the bridge construction: terminal variance ~ T,
    per-step increment variance ~ dt (equal spacing here), and non-overlapping
    increments are ~uncorrelated — the defining properties of a Brownian motion."""
    rng = np.random.default_rng(1)
    n_steps = 16
    T = 1.0
    n_paths = 40_000
    z = rng.standard_normal((n_paths, n_steps))
    W = brownian_bridge_paths(z, T)

    assert W[:, -1].var() == pytest.approx(T, rel=0.05)
    increments = np.diff(W, axis=1)
    dt = T / n_steps
    assert increments.var(axis=0).mean() == pytest.approx(dt, rel=0.1)
    corr = np.corrcoef(increments[:, 0], increments[:, 1])[0, 1]
    assert abs(corr) < 0.05


def test_brownian_bridge_rejects_non_power_of_two():
    z = np.zeros((10, 6))
    with pytest.raises(ValueError):
        brownian_bridge_paths(z, T=1.0)


def _flat_vol_grid(sigma: float) -> LocalVolGrid:
    s_grid = np.linspace(1.0, 500.0, 20)
    t_grid = np.linspace(0.0, 2.0, 10)
    sigma_loc = np.full((len(t_grid), len(s_grid)), sigma)
    return LocalVolGrid(s_grid=s_grid, t_grid=t_grid, sigma_loc=sigma_loc, n_floored=0)


def test_flat_local_vol_reduces_to_black_scholes():
    """The single most important test in this module, per the README: MC off a
    local-vol grid must reproduce the closed-form price within its own reported
    standard error. Flat local vol is the special case where the closed form is
    exactly Black-76 — the cleanest possible version of this check."""
    s0, r, q, sigma, T, K = 100.0, 0.03, 0.01, 0.2, 1.0, 100.0
    grid = _flat_vol_grid(sigma)

    result = simulate_local_vol_paths(s0, T, grid, r, q, n_paths=100_000, n_steps=64, seed=42)
    payoff = np.maximum(result.terminal - K, 0.0)
    discount_factor = np.exp(-r * T)
    price, se = result.price_and_stderr(payoff, discount_factor)

    forward = s0 * np.exp((r - q) * T)
    bs_price = call_price(forward, K, T, sigma, discount_factor)
    assert price == pytest.approx(bs_price, abs=3 * se)


def _svi_surface(Ts: list[float], rho: float = -0.3) -> pd.DataFrame:
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


def test_local_vol_mc_reprices_vanilla_within_two_standard_errors():
    """README's own Step 6 acceptance bar, on a smooth, realistic (dense-enough
    T-pillar) synthetic surface — the honest real-data investigation documented in
    `vol/local_vol.py` found that real, sparse, single-day free-tier calibrations
    (esp. short-dated slices) can miss this bar; this test validates the
    ALGORITHM under conditions where the theory's own preconditions are met."""
    Ts = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    surface = _svi_surface(Ts)
    s0, r, q = 100.0, 0.03, 0.01
    log_forwards = [np.log(s0 * np.exp((r - q) * T)) for T in Ts]
    fc = ForwardCurve(pillar_T=np.array(Ts), pillar_log_forward=np.array(log_forwards))

    # Grid resolution matters, not just path count: bilinear interpolation of the
    # precomputed sigma_loc grid is its own source of discretization error,
    # measured here to shrink from ~9 SE at a coarse (40x25) grid down to <2 SE at
    # this resolution, on an otherwise identical setup — the same kind of
    # convergence-with-resolution property as the MC path count itself, not a
    # free correctness guarantee at any resolution.
    s_grid = np.linspace(20.0, 300.0, 90)
    t_grid = np.linspace(0.01, 1.0, 60)
    grid = build_local_vol_grid(surface, fc, s_grid, t_grid)
    assert grid is not None
    assert grid.n_floored == 0

    T_test = 0.5
    forward_test = fc.forward(T_test)
    discount_factor = np.exp(-r * T_test)
    result = simulate_local_vol_paths(s0, T_test, grid, r, q, n_paths=150_000, n_steps=64, seed=11)

    from scipy.interpolate import make_smoothing_spline

    from eqdrisk.vol.local_vol import _slice_w_dk_dkk

    for K in [85.0, 100.0, 115.0]:
        k = np.log(K / forward_test)
        T_p = []
        ws = []
        for _, row in surface.sort_values("T").iterrows():
            w, _, _ = _slice_w_dk_dkk(row, np.array(k))
            ws.append(float(w))
            T_p.append(float(row["T"]))
        spl = make_smoothing_spline(np.array(T_p), np.array(ws))
        iv = np.sqrt(max(float(spl(T_test)), 1e-12) / T_test)
        bs = call_price(forward_test, K, T_test, iv, discount_factor)

        payoff = np.maximum(result.terminal - K, 0.0)
        price, se = result.price_and_stderr(payoff, discount_factor)
        assert price == pytest.approx(bs, abs=2 * se)
