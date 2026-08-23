import numpy as np
import pandas as pd
import pytest

from eqdrisk.pricing.varswap import atm_implied_vol, fair_variance_strike, wide_k_cap


def _svi_row(a: float, b: float, rho: float = 0.0, m: float = 0.0, sigma: float = 0.1) -> pd.Series:
    return pd.Series({"model": "SVI", "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma})


def test_fair_strike_recovers_flat_vol_exactly():
    """For a flat (b=0) smile, every strike has the same implied vol, so the
    Carr-Madan replication must recover that same vol as the fair variance
    swap strike — a textbook closed-form check, not just a plausibility test."""
    T = 0.5
    flat_vol = 0.20
    row = _svi_row(a=flat_vol**2 * T, b=0.0)
    k_cap = wide_k_cap(row, T)

    k_var_sq = fair_variance_strike(
        row, forward=100.0, T=T, discount_factor=0.98, k_min=-k_cap, k_max=k_cap
    )

    assert np.sqrt(k_var_sq) == pytest.approx(flat_vol, rel=1e-3)


def test_fair_strike_positive_for_skewed_smile():
    row = _svi_row(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15)
    T = 0.5
    k_cap = wide_k_cap(row, T)
    k_var_sq = fair_variance_strike(
        row, forward=100.0, T=T, discount_factor=0.97, k_min=-k_cap, k_max=k_cap
    )
    assert k_var_sq > 0


def test_wider_range_gives_higher_or_equal_fair_strike_for_curved_smile():
    """A curved (skewed) smile has higher implied vol away from the money, so
    integrating a wider strike range should never produce a LOWER fair strike
    than a narrower one nested inside it — the wings can only add variance."""
    row = _svi_row(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15)
    T = 0.5
    forward, discount_factor = 100.0, 0.97
    k_cap = wide_k_cap(row, T)

    narrow = fair_variance_strike(row, forward, T, discount_factor, -0.1, 0.1)
    wide = fair_variance_strike(row, forward, T, discount_factor, -k_cap, k_cap)

    assert wide >= narrow


def test_atm_implied_vol_matches_svi_atm_total_variance():
    """`atm_implied_vol` must agree with the SVI closed form for w(0) — note `a`
    alone is NOT w(0) unless b=0, since w(0) = a + b*sigma*sqrt(1-rho^2)."""
    from eqdrisk.vol.svi import SVIParams

    row = _svi_row(a=0.03, b=0.1, rho=-0.2, m=0.0, sigma=0.12)
    T = 0.25
    iv = atm_implied_vol(row, T)

    params = SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])
    expected_w0 = float(params.total_variance(0.0))
    assert iv == pytest.approx(np.sqrt(expected_w0 / T))
