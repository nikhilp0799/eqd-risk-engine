import numpy as np
import pytest

from eqdrisk.pricing.blackscholes import delta_spot, vega
from eqdrisk.pricing.stickiness import (
    STICKY_LOCAL_VOL_MULTIPLIER,
    compute_stickiness_deltas,
    dsigma_dk,
)
from eqdrisk.vol.svi import SVIParams


def test_sticky_strike_equals_plain_spot_delta():
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)
    forward, strike, T, discount_factor, spot = 100.0, 95.0, 0.5, 0.98, 100.0
    k = np.log(strike / forward)
    sigma = float(np.sqrt(params.total_variance(k) / T))

    result = compute_stickiness_deltas(
        True, forward, strike, T, sigma, discount_factor, spot, params
    )

    assert result.sticky_strike == pytest.approx(
        delta_spot(True, forward, strike, T, sigma, discount_factor, spot)
    )


def test_dsigma_dk_matches_finite_difference_of_sigma():
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.01, sigma=0.2)
    T = 0.5
    k0 = 0.05
    h = 1e-6

    def sigma_at(k: float) -> float:
        return float(np.sqrt(params.total_variance(k) / T))

    numeric = (sigma_at(k0 + h) - sigma_at(k0 - h)) / (2 * h)
    analytic = dsigma_dk(params, k0, T, sigma_at(k0))
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_sticky_delta_and_local_vol_differ_by_the_documented_multiplier():
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)
    forward, strike, T, discount_factor, spot = 100.0, 90.0, 1.0, 0.97, 100.0
    k = np.log(strike / forward)
    sigma = float(np.sqrt(params.total_variance(k) / T))

    result = compute_stickiness_deltas(
        False, forward, strike, T, sigma, discount_factor, spot, params
    )
    option_vega = vega(forward, strike, T, sigma, discount_factor)

    sticky_delta_adjustment = result.sticky_delta - result.sticky_strike
    sticky_local_vol_adjustment = result.sticky_local_vol - result.sticky_strike
    assert sticky_local_vol_adjustment == pytest.approx(
        STICKY_LOCAL_VOL_MULTIPLIER * sticky_delta_adjustment
    )
    # Adjustment should be non-trivial for a skewed slice with nonzero vega — otherwise
    # the fixture wouldn't actually be testing anything.
    assert abs(sticky_delta_adjustment) > 1e-6
    assert option_vega > 0
