import numpy as np
import pytest

from eqdrisk.pricing.blackscholes import (
    call_price,
    delta_forward,
    delta_spot,
    dividend_rho,
    gamma_forward,
    gamma_spot,
    put_price,
    rho,
    theta,
    vanna_forward,
    vanna_spot,
    vega,
    volga,
)

# A handful of (forward, strike, T, sigma, discount_factor, spot) scenarios spanning
# ITM/ATM/OTM, short/long-dated, low/high vol — matches the plan's "not just one
# arbitrary point" intent for the analytic-vs-finite-difference check.
CASES = [
    (100.0, 100.0, 1.0, 0.20, 0.97, 100.0),
    (100.0, 90.0, 0.5, 0.30, 0.99, 105.0),
    (100.0, 120.0, 2.0, 0.15, 0.90, 95.0),
    (50.0, 55.0, 0.1, 0.40, 0.995, 49.0),
    (200.0, 180.0, 3.0, 0.25, 0.85, 210.0),
]

REL_TOL = 1e-5


def _rel_close(analytic: float, numeric: float, atol: float = 1e-8) -> bool:
    return abs(analytic - numeric) <= atol + REL_TOL * max(abs(analytic), abs(numeric), 1.0)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_delta_forward_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = forward * 1e-5
    pricer = call_price if is_call else put_price
    numeric = (
        pricer(forward + h, strike, T, sigma, df) - pricer(forward - h, strike, T, sigma, df)
    ) / (2 * h)
    analytic = delta_forward(is_call, forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_gamma_forward_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = forward * 1e-4
    pricer = call_price if is_call else put_price
    numeric = (
        pricer(forward + h, strike, T, sigma, df)
        - 2 * pricer(forward, strike, T, sigma, df)
        + pricer(forward - h, strike, T, sigma, df)
    ) / h**2
    analytic = gamma_forward(forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_vega_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = sigma * 1e-5
    pricer = call_price if is_call else put_price
    numeric = (
        pricer(forward, strike, T, sigma + h, df) - pricer(forward, strike, T, sigma - h, df)
    ) / (2 * h)
    analytic = vega(forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_vanna_forward_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = sigma * 1e-5
    numeric = (
        delta_forward(is_call, forward, strike, T, sigma + h, df)
        - delta_forward(is_call, forward, strike, T, sigma - h, df)
    ) / (2 * h)
    analytic = vanna_forward(forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_volga_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = sigma * 1e-5
    numeric = (
        vega(forward, strike, T, sigma + h, df) - vega(forward, strike, T, sigma - h, df)
    ) / (2 * h)
    analytic = volga(forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_theta_matches_finite_difference(forward, strike, T, sigma, df, spot, is_call):
    h = T * 1e-6
    pricer = call_price if is_call else put_price
    numeric_dv_dt = (
        pricer(forward, strike, T + h, sigma, df) - pricer(forward, strike, T - h, sigma, df)
    ) / (2 * h)
    analytic_per_day = theta(is_call, forward, strike, T, sigma, df)
    numeric_per_day = -numeric_dv_dt / 365.0
    assert _rel_close(analytic_per_day, numeric_per_day, atol=1e-6)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_rho_matches_finite_difference_via_discount_factor(
    forward, strike, T, sigma, df, spot, is_call
):
    # rho = dV/dr; discount_factor = exp(-rT), so bump r directly through discount_factor.
    r = -np.log(df) / T
    h = 1e-6
    df_up = np.exp(-(r + h) * T)
    df_dn = np.exp(-(r - h) * T)
    pricer = call_price if is_call else put_price
    numeric = (
        pricer(forward, strike, T, sigma, df_up) - pricer(forward, strike, T, sigma, df_dn)
    ) / (2 * h)
    analytic = rho(is_call, forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric, atol=1e-4)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_dividend_rho_matches_finite_difference_via_forward(
    forward, strike, T, sigma, df, spot, is_call
):
    # dividend_rho = dV/dq; forward = S*exp((r-q)T), so bumping q by h changes forward by
    # a factor of exp(-h*T) — bump the forward directly through that relationship.
    h = 1e-6
    forward_up = forward * np.exp(-h * T)
    forward_dn = forward * np.exp(h * T)
    pricer = call_price if is_call else put_price
    numeric = (
        pricer(forward_up, strike, T, sigma, df) - pricer(forward_dn, strike, T, sigma, df)
    ) / (2 * h)
    analytic = dividend_rho(is_call, forward, strike, T, sigma, df)
    assert _rel_close(analytic, numeric, atol=1e-2)


@pytest.mark.parametrize("forward,strike,T,sigma,df,spot", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_spot_delta_and_gamma_and_vanna_scale_by_forward_over_spot(
    forward, strike, T, sigma, df, spot, is_call
):
    ratio = forward / spot
    assert delta_spot(is_call, forward, strike, T, sigma, df, spot) == pytest.approx(
        delta_forward(is_call, forward, strike, T, sigma, df) * ratio
    )
    assert gamma_spot(forward, strike, T, sigma, df, spot) == pytest.approx(
        gamma_forward(forward, strike, T, sigma, df) * ratio**2
    )
    assert vanna_spot(forward, strike, T, sigma, df, spot) == pytest.approx(
        vanna_forward(forward, strike, T, sigma, df) * ratio
    )


def test_greeks_zero_at_expiry_except_delta_and_price():
    from eqdrisk.pricing.blackscholes import compute_greeks

    g = compute_greeks(True, 100.0, 90.0, 0.0, 0.2, 1.0, 100.0)
    assert g.price == pytest.approx(10.0)
    assert g.gamma_forward == 0.0
    assert g.vega == 0.0
    assert g.theta == 0.0
    assert g.vanna_forward == 0.0
    assert g.volga == 0.0
