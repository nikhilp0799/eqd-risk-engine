"""Black-76 pricing on forwards, plus the full Greek set (Step 5, README 5.1-5.2).

Working in forward space throughout is the project-wide convention:

C = P(0,T)[F*N(d1) - K*N(d2)],  d1,2 = (log(F/K) ± 0.5*sigma^2*T) / (sigma*sqrt(T))

Every analytic Greek here differentiates this closed form with respect to its own
literal arguments (F, K, T, sigma, discount_factor) — F and the discount factor are
market-observed/curve-derived inputs to this module, not re-derived from a constant
(r, q) as T or S move. That is what makes the finite-difference validation in
`tests/unit/test_greeks.py` a fair test: bumping one literal argument and re-calling
the same price function must match the corresponding analytic derivative exactly,
with no hidden chain-rule terms through other arguments.

Two exceptions need an explicit spot-vs-forward chain rule, since risk is hedged in
spot, not in the forward: delta and vanna. Given F = S * exp((r-q)T), dF/dS = F/S,
so `X_spot = X_forward * (F/S)` for delta and vanna (gamma needs (F/S)^2, since it's
a second derivative in F). Vega, theta, rho, dividend_rho, and volga don't need this
conversion — sigma doesn't depend on S, and T/discount_factor/dividend-yield
sensitivities are already defined in terms of literal arguments only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_TRADING_DAYS_PER_YEAR_CONVENTION = 365.0  # theta reported per calendar day, desk convention


def _d1_d2(forward: float, strike: float, T: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def call_price(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    if T <= 0 or sigma <= 0:
        return discount_factor * max(forward - strike, 0.0)
    d1, d2 = _d1_d2(forward, strike, T, sigma)
    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def put_price(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    if T <= 0 or sigma <= 0:
        return discount_factor * max(strike - forward, 0.0)
    d1, d2 = _d1_d2(forward, strike, T, sigma)
    return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def vega(forward: float, strike: float, T: float, sigma: float, discount_factor: float) -> float:
    """Same for calls and puts (put-call parity has no vol dependence)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(forward, strike, T, sigma)
    return discount_factor * forward * norm.pdf(d1) * np.sqrt(T)


def delta_forward(
    is_call: bool, forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """dV/dF, holding T, sigma, discount_factor fixed."""
    if T <= 0 or sigma <= 0:
        if is_call:
            return discount_factor if forward > strike else 0.0
        return -discount_factor if forward < strike else 0.0
    d1, _ = _d1_d2(forward, strike, T, sigma)
    if is_call:
        return discount_factor * norm.cdf(d1)
    return discount_factor * (norm.cdf(d1) - 1.0)


def gamma_forward(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """d^2V/dF^2 — same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(forward, strike, T, sigma)
    return discount_factor * norm.pdf(d1) / (forward * sigma * np.sqrt(T))


def vanna_forward(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """d^2V/dF dsigma = d(vega)/dF — same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(forward, strike, T, sigma)
    return -discount_factor * norm.pdf(d1) * d2 / sigma


def volga(forward: float, strike: float, T: float, sigma: float, discount_factor: float) -> float:
    """d^2V/dsigma^2 = vega * d1 * d2 / sigma — same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(forward, strike, T, sigma)
    return vega(forward, strike, T, sigma, discount_factor) * d1 * d2 / sigma


def theta(
    is_call: bool, forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """-dV/dT per calendar day, holding F, sigma, discount_factor fixed (desk convention:
    positive time-to-expiry increases value, so theta itself is <= 0). This is the pure
    volatility/time-value decay component only — no carry term, since F and the discount
    factor are already the market-implied values at each T, not re-derived from a constant
    (r, q) as T shrinks. Identical for calls and puts: c0 - p0 = F - K is T-independent
    (given F, K fixed), so d(c0)/dT = d(p0)/dT exactly.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(forward, strike, T, sigma)
    dv_dT = discount_factor * forward * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    return -dv_dT / _TRADING_DAYS_PER_YEAR_CONVENTION


def rho(
    is_call: bool, forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """dV/dr via discount_factor = exp(-rT): dV/dr = dV/d(discount_factor) * d(discount_factor)/dr
    = (V/discount_factor) * (-T*discount_factor) = -T*V."""
    pricer = call_price if is_call else put_price
    price = pricer(forward, strike, T, sigma, discount_factor)
    return -T * price


def dividend_rho(
    is_call: bool, forward: float, strike: float, T: float, sigma: float, discount_factor: float
) -> float:
    """dV/dq via forward = S*exp((r-q)T): dF/dq = -T*F, so
    dV/dq = dV/dF * dF/dq = -T*F*delta_forward."""
    return -T * forward * delta_forward(is_call, forward, strike, T, sigma, discount_factor)


def delta_spot(
    is_call: bool,
    forward: float,
    strike: float,
    T: float,
    sigma: float,
    discount_factor: float,
    spot: float,
) -> float:
    """dV/dS = dV/dF * dF/dS = delta_forward * (F/S)."""
    return delta_forward(is_call, forward, strike, T, sigma, discount_factor) * (forward / spot)


def gamma_spot(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float, spot: float
) -> float:
    """d^2V/dS^2 = gamma_forward * (F/S)^2."""
    return gamma_forward(forward, strike, T, sigma, discount_factor) * (forward / spot) ** 2


def vanna_spot(
    forward: float, strike: float, T: float, sigma: float, discount_factor: float, spot: float
) -> float:
    """d^2V/dS dsigma = d(vega)/dS = vanna_forward * (F/S)."""
    return vanna_forward(forward, strike, T, sigma, discount_factor) * (forward / spot)


@dataclass
class Greeks:
    price: float
    delta_forward: float
    delta_spot: float
    gamma_forward: float
    gamma_spot: float
    vega: float
    theta: float
    rho: float
    dividend_rho: float
    vanna_forward: float
    vanna_spot: float
    volga: float


def compute_greeks(
    is_call: bool,
    forward: float,
    strike: float,
    T: float,
    sigma: float,
    discount_factor: float,
    spot: float,
) -> Greeks:
    """Bundle the full Greek set for one option in one call — the natural unit of
    output for a pricing run (one row per quote in the `greeks` curated table)."""
    pricer = call_price if is_call else put_price
    return Greeks(
        price=pricer(forward, strike, T, sigma, discount_factor),
        delta_forward=delta_forward(is_call, forward, strike, T, sigma, discount_factor),
        delta_spot=delta_spot(is_call, forward, strike, T, sigma, discount_factor, spot),
        gamma_forward=gamma_forward(forward, strike, T, sigma, discount_factor),
        gamma_spot=gamma_spot(forward, strike, T, sigma, discount_factor, spot),
        vega=vega(forward, strike, T, sigma, discount_factor),
        theta=theta(is_call, forward, strike, T, sigma, discount_factor),
        rho=rho(is_call, forward, strike, T, sigma, discount_factor),
        dividend_rho=dividend_rho(is_call, forward, strike, T, sigma, discount_factor),
        vanna_forward=vanna_forward(forward, strike, T, sigma, discount_factor),
        vanna_spot=vanna_spot(forward, strike, T, sigma, discount_factor, spot),
        volga=volga(forward, strike, T, sigma, discount_factor),
    )
