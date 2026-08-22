"""SSVI — arbitrage-free-by-construction global surface, used as the calendar-arbitrage
fallback when per-slice SVI fits violate the across-expiry condition (README 4.2).

w(k, theta) = theta/2 * {1 + rho*phi(theta)*k + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2)}

Power-law phi(theta) = eta * theta^(-1/2) (Gatheral-Jacquier 2014): a single shared (rho,
eta) across all expiries, with theta_t the ATM total variance term structure taken
directly from the data. Sufficient condition for no static arbitrage with this specific
phi: eta*(1+|rho|) <= 2 — enforced as a constraint, not just hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SSVIParams:
    rho: float
    eta: float

    def phi(self, theta: float) -> float:
        return self.eta / np.sqrt(theta)

    def total_variance(self, k: np.ndarray | float, theta: float) -> np.ndarray | float:
        k = np.asarray(k, dtype=float)
        phi = self.phi(theta)
        return (theta / 2) * (
            1 + self.rho * phi * k + np.sqrt((phi * k + self.rho) ** 2 + 1 - self.rho**2)
        )


def fit_ssvi(slices: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]) -> SSVIParams:
    """`slices`: list of (theta, k, w, weights) per expiry, theta = ATM total variance for
    that expiry. Fits a single (rho, eta) minimising total weighted squared error across
    all slices at once, subject to eta*(1+|rho|) <= 2 (no static arbitrage, sufficient cond.)."""

    def objective(params: np.ndarray) -> float:
        rho_raw, eta_raw = params
        rho = np.tanh(rho_raw)  # unconstrained -> (-1, 1)
        eta = np.exp(eta_raw)  # unconstrained -> (0, inf)
        eta = min(eta, 2.0 / (1 + abs(rho)))  # project onto the no-arb feasible region
        model = SSVIParams(rho=rho, eta=eta)
        total = 0.0
        for theta, k, w, weights in slices:
            fitted = model.total_variance(k, theta)
            total += float(np.sum(weights * (fitted - w) ** 2))
        return total

    result = minimize(
        objective, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10}
    )
    rho_raw, eta_raw = result.x
    rho = float(np.tanh(rho_raw))
    eta = float(min(np.exp(eta_raw), 2.0 / (1 + abs(rho))))
    return SSVIParams(rho=rho, eta=eta)
