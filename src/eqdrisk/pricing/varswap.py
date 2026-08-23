"""Variance swap fair strike via Carr-Madan static replication (README 6.3):

    K_var^2 = (2/T) * [ int_0^F P(K)/K^2 dK + int_F^inf C(K)/K^2 dK ] / P(0,T)

Priced off the CALIBRATED surface's own smile (one expiry's SVI or SSVI slice,
whichever Step 4 selected), not raw quotes — a discrete strip of put/call prices
evaluated on a dense synthetic strike grid, then integrated numerically
(trapezoidal rule in strike space). This mirrors Step 6.1's own "strip from the
model, not from noisy quotes" principle.

**The practical point the README explicitly wants written up, not hand-waved:**
any real strip is truncated at some finite strike — you cannot trade options at
every strike out to infinity. Two ranges are computed and compared for every
expiry: the REAL, tradeable range (bounded by what the day's quality-filtered
`implied_vols` actually observed for that expiry) and a WIDE range (bounded by
the same `EXTREME_K_MULTIPLE * atm_vol * sqrt(T)` cap `vol/local_vol.py` already
uses to decide how far the parametric smile can be trusted at all). The
difference between the two fair strikes IS the wing-truncation replication error
— a real, measured number, not an assumption.

**Jump/gap risk (README's other explicit write-up point):** static replication
assumes the strike grid can be continuously delta-hedged as spot moves through
it — Carr-Madan's derivation relies on trading an infinitesimal quantity of every
strike as a barrier is crossed. A real market gaps (overnight, on news, at the
open) rather than passing through every intermediate price continuously, so the
realised replication error also includes a genuine jump-risk component this
strip can't see or correct for — it is a property of the real underlying process,
not a numerical artefact fixable by a finer strike grid or a wider integration
range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqdrisk.pricing.blackscholes import call_price, put_price
from eqdrisk.vol.implied import EXTREME_K_MULTIPLE
from eqdrisk.vol.local_vol import slice_total_variance

N_INTEGRATION_POINTS = 400


def _price_strip(
    surface_row: pd.Series, forward: float, T: float, discount_factor: float, k_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    w = slice_total_variance(surface_row, k_grid)
    iv = np.sqrt(np.clip(w, 1e-12, None) / T)
    strikes = forward * np.exp(k_grid)
    return np.array(
        [
            call_price(forward, float(strike), T, float(sigma), discount_factor)
            if strike >= forward
            else put_price(forward, float(strike), T, float(sigma), discount_factor)
            for strike, sigma in zip(strikes, iv, strict=True)
        ]
    ), strikes


def fair_variance_strike(
    surface_row: pd.Series,
    forward: float,
    T: float,
    discount_factor: float,
    k_min: float,
    k_max: float,
    n_points: int = N_INTEGRATION_POINTS,
) -> float:
    """K_var^2 (annualised variance units, not vol points) integrating the
    replication strip over log-moneyness [k_min, k_max]."""
    k_grid = np.linspace(k_min, k_max, n_points)
    prices, strikes = _price_strip(surface_row, forward, T, discount_factor, k_grid)
    integrand = prices / strikes**2
    integral = float(np.trapezoid(integrand, strikes))
    return (2.0 / T) * integral / discount_factor


def atm_implied_vol(surface_row: pd.Series, T: float) -> float:
    w0 = float(slice_total_variance(surface_row, 0.0))
    return float(np.sqrt(max(w0, 1e-12) / T))


def wide_k_cap(surface_row: pd.Series, T: float) -> float:
    """Same cap `vol/local_vol.py` uses to stop trusting the parametric smile's
    extrapolation — reused here rather than inventing a second threshold."""
    return EXTREME_K_MULTIPLE * atm_implied_vol(surface_row, T) * np.sqrt(T)
