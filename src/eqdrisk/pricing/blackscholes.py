"""Black-76 pricing on forwards.

Price only, plus vega (needed now for Step 3's vega/spread quote-weighting,
README 3.3) — the rest of the Greeks arrive in Step 5. Working in forward
space throughout is the project-wide convention (README 3.1, 5.1):

C = P(0,T)[F*N(d1) - K*N(d2)],  d1,2 = (log(F/K) ± 0.5*sigma^2*T) / (sigma*sqrt(T))
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


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
