import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.stress.shock import MarketShock, shock_local_vol_grid, shocked_spot, shocked_w
from eqdrisk.vol.local_vol import build_local_vol_grid, local_variance_at


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


def test_noop_shock_leaves_spot_and_w_unchanged():
    shock = MarketShock()
    assert shocked_spot(100.0, shock) == 100.0

    surface = _svi_surface([0.25, 0.5])
    base = local_variance_at(surface, 0.05, 0.4).w
    assert shocked_w(surface, 0.05, 0.4, shock) == pytest.approx(base)


def test_noop_shock_returns_the_same_grid_object():
    s_grid, t_grid = np.linspace(50, 150, 10), np.linspace(0.1, 1.0, 5)
    from eqdrisk.vol.local_vol import LocalVolGrid

    grid = LocalVolGrid(s_grid=s_grid, t_grid=t_grid, sigma_loc=np.ones((5, 10)), n_floored=0)
    fc = ForwardCurve(pillar_T=np.array([1.0]), pillar_log_forward=np.array([np.log(100.0)]))
    assert shock_local_vol_grid(grid, fc, MarketShock()) is grid


def test_spot_shock_scales_multiplicatively():
    assert shocked_spot(100.0, MarketShock(spot_shock_pct=-0.20)) == pytest.approx(80.0)
    assert shocked_spot(100.0, MarketShock(spot_shock_pct=0.10)) == pytest.approx(110.0)


def test_parallel_vol_shock_scales_w_by_multiplier_squared():
    surface = _svi_surface([0.25, 0.5])
    base = local_variance_at(surface, -0.1, 0.4).w
    shock = MarketShock(vol_shock_pct=0.25)
    shocked = shocked_w(surface, -0.1, 0.4, shock)
    assert shocked == pytest.approx(base * 1.25**2)


def test_skew_shock_moves_wings_more_than_atm():
    surface = _svi_surface([0.25, 0.5])
    shock = MarketShock(skew_shock=0.5)
    atm = shocked_w(surface, 0.0, 0.4, shock)
    atm_base = local_variance_at(surface, 0.0, 0.4).w
    wing = shocked_w(surface, -0.3, 0.4, shock)
    wing_base = local_variance_at(surface, -0.3, 0.4).w
    assert atm == pytest.approx(atm_base)  # |k|=0 -> skew multiplier is a no-op
    assert wing / wing_base > 1.0


def test_term_shock_front_loads_short_dated_more_than_long_dated():
    surface = _svi_surface([0.1, 0.25, 0.5, 1.0, 2.0])
    shock = MarketShock(term_shock=0.5)
    short_ratio = shocked_w(surface, 0.0, 0.1, shock) / local_variance_at(surface, 0.0, 0.1).w
    long_ratio = shocked_w(surface, 0.0, 2.0, shock) / local_variance_at(surface, 0.0, 2.0).w
    assert short_ratio > long_ratio
    assert long_ratio == pytest.approx(1.0)  # T >= TERM_REFERENCE_T (1y) -> no term shock


def test_pure_parallel_vol_shock_on_grid_is_a_bounded_approximation_not_exact():
    """Measures, rather than assumes, how close `shock_local_vol_grid`'s cheap
    post-hoc `sigma_loc -> sigma_loc*sqrt(C)` shortcut comes to the fully
    correct answer (rebuild the grid from a genuinely `C`-rescaled SVI surface
    — exact for a,b scaled by C, since w and its k-derivatives scale by C
    exactly). An earlier version of this module claimed the shortcut was
    EXACT for a pure parallel shock; it wasn't (Durrleman's g has an additive
    1/4 term that doesn't scale with w), and this test is what caught it.
    Bounds are the actually-measured error at this shock size, not a
    made-up-looking round number, so a real regression still fails it.
    """
    Ts = [0.3, 0.5, 0.75, 1.0]
    surface = _svi_surface(Ts)
    C = 1.5**2  # vol up 50% relative, the ladder's most extreme point

    scaled_surface = surface.copy()
    scaled_surface["a"] = scaled_surface["a"] * C
    scaled_surface["b"] = scaled_surface["b"] * C

    s0 = 100.0
    fc = ForwardCurve(pillar_T=np.array(Ts), pillar_log_forward=np.log([s0] * len(Ts)))
    s_grid = np.linspace(90.0, 111.0, 20)  # near-the-money, away from the wing cap
    t_grid = np.linspace(0.3, 1.0, 15)

    base_grid = build_local_vol_grid(surface, fc, s_grid, t_grid)
    exact_scaled_grid = build_local_vol_grid(scaled_surface, fc, s_grid, t_grid)
    assert base_grid is not None and exact_scaled_grid is not None

    shocked_grid = shock_local_vol_grid(base_grid, fc, MarketShock(vol_shock_pct=0.5))

    rel_err = (
        np.abs(shocked_grid.sigma_loc - exact_scaled_grid.sigma_loc) / exact_scaled_grid.sigma_loc
    )
    assert rel_err.max() < 0.20  # measured ~19% at this shock size; catches a regression, not tiny


def test_approximation_error_shrinks_for_smaller_shocks():
    """The shortcut is a small-perturbation approximation, so its error should
    grow with shock size — confirms that qualitative claim on real numbers."""
    Ts = [0.3, 0.5, 0.75, 1.0]
    surface = _svi_surface(Ts)
    s0 = 100.0
    fc = ForwardCurve(pillar_T=np.array(Ts), pillar_log_forward=np.log([s0] * len(Ts)))
    s_grid = np.linspace(90.0, 111.0, 20)
    t_grid = np.linspace(0.3, 1.0, 15)
    base_grid = build_local_vol_grid(surface, fc, s_grid, t_grid)
    assert base_grid is not None

    errs = []
    for vol_shock_pct in [0.1, 0.5]:
        C = (1 + vol_shock_pct) ** 2
        scaled_surface = surface.copy()
        scaled_surface["a"] *= C
        scaled_surface["b"] *= C
        exact_grid = build_local_vol_grid(scaled_surface, fc, s_grid, t_grid)
        shocked_grid = shock_local_vol_grid(base_grid, fc, MarketShock(vol_shock_pct=vol_shock_pct))
        assert exact_grid is not None
        rel_err = np.abs(shocked_grid.sigma_loc - exact_grid.sigma_loc) / exact_grid.sigma_loc
        errs.append(rel_err.max())

    assert errs[0] < errs[1]
