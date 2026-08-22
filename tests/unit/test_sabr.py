import numpy as np
import pytest

from eqdrisk.vol.sabr import SABRParams, fit_sabr_slice, sabr_implied_vol


def test_sabr_implied_vol_atm_matches_off_atm_limit():
    """ATM branch (F==K, direct formula) and the general branch should agree in
    the limit as K -> F (continuity check on the Hagan formula's two code paths)."""
    F, T = 100.0, 0.5
    params = SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.6)

    atm = sabr_implied_vol(F, F, T, params)
    near_atm = sabr_implied_vol(F, F + 1e-6, T, params)

    assert atm == pytest.approx(near_atm, rel=1e-4)


def test_fit_sabr_slice_recovers_known_parameters():
    F, T = 100.0, 0.25
    true = SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.6)
    strikes = np.linspace(80, 120, 15)
    ivs = np.array([sabr_implied_vol(F, k, T, true) for k in strikes])
    weights = np.ones_like(strikes)

    fit = fit_sabr_slice(F, strikes, ivs, T, weights, beta=0.5)
    fitted_ivs = np.array([sabr_implied_vol(F, k, T, fit) for k in strikes])

    assert np.max(np.abs(fitted_ivs - ivs)) < 1e-6


def test_sabr_implied_vol_increases_with_nu_away_from_atm():
    F, T = 100.0, 0.5
    low_nu = SABRParams(alpha=0.2, beta=0.5, rho=0.0, nu=0.2)
    high_nu = SABRParams(alpha=0.2, beta=0.5, rho=0.0, nu=1.0)

    iv_low = sabr_implied_vol(F, 130.0, T, low_nu)
    iv_high = sabr_implied_vol(F, 130.0, T, high_nu)

    assert iv_high > iv_low  # more vol-of-vol -> fatter wings
