"""Closed-form single-barrier option prices (Reiner & Rubinstein, 1991), constant
vol — used ONLY as a benchmark for the local-vol Monte Carlo engine (README 6.4),
never as the actual pricer (a real book has skew, which this formula ignores
entirely). Implements the standard building-block construction reproduced in
Haug's "The Complete Guide to Option Pricing Formulas" and widely cross-checked
elsewhere; only the down-and-in put case this project actually needs is exposed
publicly, but the shared A/B/C/D/E/F blocks are the general single-barrier family.

Notation: `b = r - q` is the cost of carry. `eta = +1` for a "down" barrier,
`-1` for "up"; `phi = +1` for a call, `-1` for a put.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

N = norm.cdf


def _blocks(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r: float,
    b: float,
    phi: float,
    eta: float,
) -> tuple[float, float, float, float]:
    """Haug's A, B, C, D building blocks (E, F omitted — this project only prices
    zero-rebate barriers)."""
    vol_sqrt_t = sigma * np.sqrt(T)
    mu = (b - sigma**2 / 2) / sigma**2
    x1 = np.log(spot / strike) / vol_sqrt_t + (1 + mu) * vol_sqrt_t
    x2 = np.log(spot / barrier) / vol_sqrt_t + (1 + mu) * vol_sqrt_t
    y1 = np.log(barrier**2 / (spot * strike)) / vol_sqrt_t + (1 + mu) * vol_sqrt_t
    y2 = np.log(barrier / spot) / vol_sqrt_t + (1 + mu) * vol_sqrt_t

    carry_factor = spot * np.exp((b - r) * T)
    discounted_strike = strike * np.exp(-r * T)
    hs_pow_2mu_plus_2 = (barrier / spot) ** (2 * (mu + 1))
    hs_pow_2mu = (barrier / spot) ** (2 * mu)

    a = phi * carry_factor * N(phi * x1) - phi * discounted_strike * N(phi * x1 - phi * vol_sqrt_t)
    b_ = phi * carry_factor * N(phi * x2) - phi * discounted_strike * N(phi * x2 - phi * vol_sqrt_t)
    c = phi * carry_factor * hs_pow_2mu_plus_2 * N(
        eta * y1
    ) - phi * discounted_strike * hs_pow_2mu * N(eta * y1 - eta * vol_sqrt_t)
    d = phi * carry_factor * hs_pow_2mu_plus_2 * N(
        eta * y2
    ) - phi * discounted_strike * hs_pow_2mu * N(eta * y2 - eta * vol_sqrt_t)
    return float(a), float(b_), float(c), float(d)


def down_and_in_put_price(
    spot: float, strike: float, barrier: float, T: float, sigma: float, r: float, q: float
) -> float:
    """Zero-rebate down-and-in put, Reiner-Rubinstein closed form. `barrier` must
    be below `spot` (a down barrier that hasn't already knocked in)."""
    if barrier >= spot:
        raise ValueError("down_and_in_put_price requires barrier < spot (not yet knocked in)")
    b = r - q
    a, b_block, c, d = _blocks(spot, strike, barrier, T, sigma, r, b, phi=-1.0, eta=1.0)
    if strike > barrier:
        return b_block - c + d
    return a


def down_and_out_put_price(
    spot: float, strike: float, barrier: float, T: float, sigma: float, r: float, q: float
) -> float:
    """Zero-rebate down-and-out put — exists mainly so
    `down_and_in + down_and_out == vanilla` (a model-independent static identity)
    can be used as a fast, deterministic correctness check on both formulas,
    without needing Monte Carlo in the unit test suite."""
    if barrier >= spot:
        raise ValueError("down_and_out_put_price requires barrier < spot (not yet knocked out)")
    b = r - q
    a, b_block, c, d = _blocks(spot, strike, barrier, T, sigma, r, b, phi=-1.0, eta=1.0)
    if strike > barrier:
        return a - b_block + c - d
    return 0.0
