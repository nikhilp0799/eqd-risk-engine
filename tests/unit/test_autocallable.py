import numpy as np
import pytest

from eqdrisk.pricing.autocallable import (
    AutocallableSpec,
    autocallable_greeks,
    autocallable_payoff,
    price_autocallable,
)
from eqdrisk.vol.local_vol import LocalVolGrid

NOTIONAL = 1_000_000.0
OBS_TIMES = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])


def _spec(**overrides) -> AutocallableSpec:
    defaults = dict(
        notional=NOTIONAL,
        autocall_barrier=1.00,
        coupon_barrier=0.75,
        put_barrier=0.65,
        coupon_rate=0.0225,
        obs_times=OBS_TIMES,
    )
    defaults.update(overrides)
    return AutocallableSpec(**defaults)


def test_immediate_autocall_pays_par_plus_first_coupon():
    spec = _spec()
    levels = np.full((1, 8), 105.0)
    pv = autocallable_payoff(levels, spec, initial_level=100.0, r=0.03)
    df0 = np.exp(-0.03 * OBS_TIMES[0])
    expected = df0 * NOTIONAL * (1 + spec.coupon_rate)
    assert pv[0] == pytest.approx(expected)


def test_final_coupon_backpays_all_accumulated_memory():
    spec = _spec()
    levels = np.array([[70, 70, 70, 70, 70, 70, 70, 80.0]])  # below coupon barrier until maturity
    pv = autocallable_payoff(levels, spec, initial_level=100.0, r=0.03)
    df_last = np.exp(-0.03 * OBS_TIMES[-1])
    expected = df_last * NOTIONAL * (1 + spec.coupon_rate * (1 + 7))
    assert pv[0] == pytest.approx(expected)


def test_breach_of_put_barrier_at_maturity_gives_full_downside_participation():
    spec = _spec()
    levels = np.array([[70, 70, 70, 70, 70, 70, 70, 60.0]])  # 60 < put_barrier*100=65
    pv = autocallable_payoff(levels, spec, initial_level=100.0, r=0.03)
    df_last = np.exp(-0.03 * OBS_TIMES[-1])
    assert pv[0] == pytest.approx(df_last * NOTIONAL * 0.60)


def test_survival_between_put_and_coupon_barrier_returns_principal_only():
    spec = _spec()
    levels = np.array([[70, 70, 70, 70, 70, 70, 70, 70.0]])  # 65 <= 70 < 75
    pv = autocallable_payoff(levels, spec, initial_level=100.0, r=0.03)
    df_last = np.exp(-0.03 * OBS_TIMES[-1])
    assert pv[0] == pytest.approx(df_last * NOTIONAL)


def test_memory_coupon_paid_then_reset_then_reaccumulated_before_autocall():
    """Period 0: below coupon barrier (memory 0->1). Period 1: above coupon barrier
    (pays this period + the 1 memorized period, resets to 0). Periods 2-3: below
    again (memory 0->1->2). Period 4: autocalls, backpaying the 2 memorized periods."""
    spec = _spec()
    levels = np.array([[70, 80, 70, 70, 105, 105, 105, 105]])
    pv = autocallable_payoff(levels, spec, initial_level=100.0, r=0.03)
    df1, df4 = np.exp(-0.03 * OBS_TIMES[1]), np.exp(-0.03 * OBS_TIMES[4])
    expected = df1 * NOTIONAL * spec.coupon_rate * (1 + 1) + df4 * NOTIONAL * (
        1 + spec.coupon_rate * (1 + 2)
    )
    assert pv[0] == pytest.approx(expected)


def _flat_vol_grid(sigma: float) -> LocalVolGrid:
    s_grid = np.linspace(1.0, 400.0, 80)
    t_grid = np.linspace(0.0, 2.0, 20)
    return LocalVolGrid(
        s_grid=s_grid,
        t_grid=t_grid,
        sigma_loc=np.full((len(t_grid), len(s_grid)), sigma),
        n_floored=0,
    )


def test_price_is_a_plausible_fraction_of_notional():
    spec = _spec()
    grid = _flat_vol_grid(0.25)
    price, se = price_autocallable(
        spec, 100.0, grid, r=0.03, q=0.01, n_paths=50_000, n_steps_per_period=8, seed=1
    )
    assert 0.5 * NOTIONAL < price < 1.2 * NOTIONAL
    assert se > 0


def test_crn_delta_and_gamma_are_exactly_zero_under_flat_scale_invariant_vol():
    """A subtle but genuine correctness check: under FLAT local vol, this payoff
    is scale-invariant in spot (every barrier and the downside-participation
    payoff are defined as a FRACTION of the initial level, and log-spot GBM
    under constant vol scales proportionally under common random numbers) — so
    bumping s0 must leave the discrete autocall/coupon/put decisions, and hence
    the price, EXACTLY unchanged. Delta and gamma should come out as exactly
    0.0, not just "small" — a strong, deterministic-under-CRN sanity check that
    doesn't depend on statistical tolerances.

    The same scale-invariance argument extends one derivative further: since
    price(S+h) == price(S) == price(S-h) EXACTLY at ANY fixed flat vol level
    (not just the base one), it holds at the vol-bumped levels too, so the four
    vanna corner prices collapse pairwise (price_pp == price_mp, price_pm ==
    price_mm) and vanna must come out exactly 0.0 as well. Volga is NOT
    expected to vanish — it's a pure vol-convexity effect, unrelated to the
    spot scale-invariance identity.
    """
    spec = _spec()
    grid = _flat_vol_grid(0.25)
    g = autocallable_greeks(
        spec, 100.0, grid, r=0.03, q=0.01, n_paths=30_000, n_steps_per_period=8, seed=1
    )
    assert g.delta == 0.0
    assert g.gamma == 0.0
    assert g.vanna == 0.0
    assert np.isfinite(g.volga)


def test_crn_greeks_reuse_the_same_paths_across_bumps():
    """Common random numbers means the SAME seed must be used for the base case
    and every bump — verified indirectly: two independent `autocallable_greeks`
    calls with the same seed must be bit-identical (deterministic), which would
    fail if any internal randomness weren't fully seeded end-to-end."""
    spec = _spec()
    grid = _flat_vol_grid(0.22)
    g1 = autocallable_greeks(
        spec, 105.0, grid, r=0.03, q=0.01, n_paths=20_000, n_steps_per_period=8, seed=42
    )
    g2 = autocallable_greeks(
        spec, 105.0, grid, r=0.03, q=0.01, n_paths=20_000, n_steps_per_period=8, seed=42
    )
    assert g1 == g2
