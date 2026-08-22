"""Hagan (2002) lognormal SABR — fit to the shortest available expiry per underlying,
for the SVI/SSVI/SABR comparison the README calls out explicitly (4.3).

Deliberately the plain Hagan asymptotic formula, not the arbitrage-free PDE variant —
its known failure mode (admits negative densities for low strikes / long maturities) is
part of the point of the comparison, not something to engineer around.

beta is fixed (not calibrated) at a market-convention default: a single day's smile
can't identify beta separately from rho/nu without a strike/vol time series, so real
desks fix it by convention too — same situation we're in with one day of data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

DEFAULT_BETA = 0.5


@dataclass
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol(forward: float, strike: float, T: float, params: SABRParams) -> float:
    """Hagan's lognormal-vol asymptotic approximation."""
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    f, k = forward, strike

    if abs(f - k) < 1e-12:
        f_mid_pow = f ** (1 - beta)
        term = (
            1
            + (
                ((1 - beta) ** 2 / 24) * alpha**2 / f_mid_pow**2
                + 0.25 * rho * beta * nu * alpha / f_mid_pow
                + (2 - 3 * rho**2) / 24 * nu**2
            )
            * T
        )
        return float(alpha / f_mid_pow * term)

    log_fk = np.log(f / k)
    fk_pow = (f * k) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_pow * log_fk
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    denom = fk_pow * (1 + ((1 - beta) ** 2 / 24) * log_fk**2 + ((1 - beta) ** 4 / 1920) * log_fk**4)
    term = (
        1
        + (
            ((1 - beta) ** 2 / 24) * alpha**2 / fk_pow**2
            + 0.25 * rho * beta * nu * alpha / fk_pow
            + (2 - 3 * rho**2) / 24 * nu**2
        )
        * T
    )
    return float((alpha / denom) * (z / x_z) * term)


def fit_sabr_slice(
    forward: float,
    strikes: np.ndarray,
    ivs: np.ndarray,
    T: float,
    weights: np.ndarray,
    beta: float = DEFAULT_BETA,
) -> SABRParams:
    """Calibrate (alpha, rho, nu) via weighted least squares on implied vol directly
    (not total variance — Hagan's formula is naturally in vol space)."""
    alpha0 = float(np.interp(forward, strikes, ivs))

    def objective(params: np.ndarray) -> float:
        log_alpha, rho_raw, log_nu = params
        alpha = np.exp(log_alpha)
        rho = np.tanh(rho_raw)
        nu = np.exp(log_nu)
        model = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        fitted = np.array([sabr_implied_vol(forward, k, T, model) for k in strikes])
        return float(np.sum(weights * (fitted - ivs) ** 2))

    result = minimize(
        objective,
        x0=[np.log(alpha0), 0.0, np.log(0.5)],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000},
    )
    log_alpha, rho_raw, log_nu = result.x
    return SABRParams(
        alpha=float(np.exp(log_alpha)),
        beta=beta,
        rho=float(np.tanh(rho_raw)),
        nu=float(np.exp(log_nu)),
    )
