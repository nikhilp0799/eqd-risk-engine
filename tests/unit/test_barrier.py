import numpy as np
import pytest

from eqdrisk.pricing.barrier_closed_form import down_and_in_put_price, down_and_out_put_price
from eqdrisk.pricing.barrier_mc import simulate_down_and_in_put
from eqdrisk.pricing.blackscholes import put_price
from eqdrisk.vol.local_vol import LocalVolGrid


def _vanilla_put(spot: float, strike: float, T: float, sigma: float, r: float, q: float) -> float:
    b = r - q
    forward = spot * np.exp(b * T)
    discount = np.exp(-r * T)
    return put_price(forward, strike, T, sigma, discount)


@pytest.mark.parametrize(
    "spot,strike,barrier,T,sigma,r,q",
    [
        (100.0, 100.0, 80.0, 1.0, 0.25, 0.03, 0.01),  # strike > barrier (README's own case)
        (100.0, 70.0, 90.0, 0.5, 0.30, 0.02, 0.00),  # strike < barrier
        (100.0, 100.0, 99.0, 0.1, 0.40, 0.05, 0.02),  # barrier close to spot
        (100.0, 120.0, 60.0, 2.0, 0.15, 0.01, 0.03),
    ],
)
def test_down_and_in_plus_down_and_out_equals_vanilla_parity(spot, strike, barrier, T, sigma, r, q):
    """Model-independent static identity: knock-in + knock-out = vanilla, for
    ANY barrier/strike configuration — a fast, deterministic correctness check
    on both closed-form formulas without needing Monte Carlo."""
    di = down_and_in_put_price(spot, strike, barrier, T, sigma, r, q)
    do = down_and_out_put_price(spot, strike, barrier, T, sigma, r, q)
    vanilla = _vanilla_put(spot, strike, T, sigma, r, q)
    assert di + do == pytest.approx(vanilla, abs=1e-8)


def test_down_and_in_vanishes_as_barrier_to_zero():
    price = down_and_in_put_price(100.0, 100.0, 1e-6, 1.0, 0.25, 0.03, 0.01)
    assert price == pytest.approx(0.0, abs=1e-8)


def test_down_and_in_approaches_vanilla_as_barrier_approaches_spot():
    price = down_and_in_put_price(100.0, 100.0, 99.999, 1.0, 0.25, 0.03, 0.01)
    vanilla = _vanilla_put(100.0, 100.0, 1.0, 0.25, 0.03, 0.01)
    assert price == pytest.approx(vanilla, rel=1e-4)


def test_down_and_in_rejects_barrier_at_or_above_spot():
    with pytest.raises(ValueError):
        down_and_in_put_price(100.0, 100.0, 100.0, 1.0, 0.25, 0.03, 0.01)


def _flat_vol_grid(sigma: float) -> LocalVolGrid:
    s_grid = np.linspace(1.0, 400.0, 80)
    t_grid = np.linspace(0.0, 1.0, 10)
    return LocalVolGrid(
        s_grid=s_grid,
        t_grid=t_grid,
        sigma_loc=np.full((len(t_grid), len(s_grid)), sigma),
        n_floored=0,
    )


def test_bridge_correction_reprices_closed_form_far_better_than_naive_monitoring():
    """The README's own headline finding for this sub-step: naive discrete
    monitoring systematically underprices a down-and-in put (it misses paths that
    dip below the barrier and recover between two sampled points), and the bias
    shrinks only slowly as the step count grows. The Brownian-bridge continuity
    correction recovers the true (closed-form, continuous-monitoring) price even
    at a coarse step count, without needing more steps at all.
    """
    spot, strike, barrier, T, sigma, r, q = 100.0, 100.0, 80.0, 1.0, 0.25, 0.03, 0.01
    closed_form = down_and_in_put_price(spot, strike, barrier, T, sigma, r, q)
    grid = _flat_vol_grid(sigma)
    discount_factor = np.exp(-r * T)
    n_steps = 16  # deliberately coarse — the bridge correction should still work here

    naive = simulate_down_and_in_put(
        spot,
        barrier,
        T,
        grid,
        r,
        q,
        n_paths=200_000,
        n_steps=n_steps,
        seed=1,
        use_bridge_correction=False,
    )
    price_naive, se_naive = naive.price_and_stderr(strike, discount_factor)

    bridge = simulate_down_and_in_put(
        spot,
        barrier,
        T,
        grid,
        r,
        q,
        n_paths=200_000,
        n_steps=n_steps,
        seed=1,
        use_bridge_correction=True,
    )
    price_bridge, se_bridge = bridge.price_and_stderr(strike, discount_factor)

    # Naive monitoring at this coarse a step count is biased low by many standard
    # errors — this assertion documents the bias exists, it isn't a bug to "fix"
    # by loosening it.
    assert closed_form - price_naive > 10 * se_naive
    # The bridge-corrected price matches the closed form within a few standard
    # errors, at the SAME (coarse) step count.
    assert price_bridge == pytest.approx(closed_form, abs=4 * se_bridge)


def test_bridge_correction_bias_shrinks_with_more_steps_even_without_correction():
    """Sanity check on the naive path itself: the well-known discretization bias
    should shrink (not stay flat or grow) as the step count increases, confirming
    it's a genuine discretization effect and not an unrelated implementation bug."""
    spot, strike, barrier, T, sigma, r, q = 100.0, 100.0, 80.0, 1.0, 0.25, 0.03, 0.01
    closed_form = down_and_in_put_price(spot, strike, barrier, T, sigma, r, q)
    grid = _flat_vol_grid(sigma)
    discount_factor = np.exp(-r * T)

    gaps = []
    for n_steps in [16, 64, 256]:
        naive = simulate_down_and_in_put(
            spot,
            barrier,
            T,
            grid,
            r,
            q,
            n_paths=150_000,
            n_steps=n_steps,
            seed=3,
            use_bridge_correction=False,
        )
        price, _ = naive.price_and_stderr(strike, discount_factor)
        gaps.append(closed_form - price)

    assert gaps[0] > gaps[1] > gaps[2] > 0
