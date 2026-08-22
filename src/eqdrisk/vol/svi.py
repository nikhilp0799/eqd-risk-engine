"""Raw SVI per-expiry-slice fit, plus Durrleman's butterfly no-arbitrage check.

w(k) = a + b*{rho*(k-m) + sqrt((k-m)^2 + sigma^2)}   (total implied variance)

Zeliade's quasi-explicit method (README 4.1): for FIXED (m, sigma), the problem
is linear in (a, b*rho, b) — solved exactly by least squares. Only (m, sigma)
need a nonlinear (Nelder-Mead) search over the much smaller 2D outer problem.

Simplification vs. the Zeliade paper's exact constrained QP, documented rather
than silently adopted: the inner linear solve here is unconstrained OLS, then
projected onto the feasible region (b>=0, |rho|<1, a>=0-variance-at-ATM) rather
than solved as a constrained QP directly. Adequate for calibrating a handful of
sparse, already quality-filtered slices; would need revisiting for larger,
noisier slices where the projection could matter more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

SIGMA_MIN = 1e-4
SIGMA_MAX = 10.0  # k is typically in [-1, 1]; guards Nelder-Mead against exp() overflow


def _clip_sigma(log_sigma: float) -> float:
    """exp(log_sigma) can overflow for Nelder-Mead's wilder trial points — that's
    expected and harmless since the result gets clipped to SIGMA_MAX regardless;
    suppress the warning rather than let it fire on every such trial."""
    with np.errstate(over="ignore"):
        return float(np.clip(np.exp(log_sigma), SIGMA_MIN, SIGMA_MAX))


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x**2 + self.sigma**2))

    def first_derivative(self, k: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(k, dtype=float) - self.m
        s = np.sqrt(x**2 + self.sigma**2)
        return self.b * (self.rho + x / s)

    def second_derivative(self, k: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(k, dtype=float) - self.m
        s = np.sqrt(x**2 + self.sigma**2)
        return self.b * self.sigma**2 / s**3


def durrleman_g(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """Durrleman's butterfly condition: g(k) >= 0 everywhere <=> no butterfly arbitrage.

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    """
    w = params.total_variance(k)
    wp = params.first_derivative(k)
    wpp = params.second_derivative(k)
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def _inner_linear_solve(
    k: np.ndarray, w: np.ndarray, weights: np.ndarray, m: float, sigma: float
) -> tuple[float, float, float]:
    """Given (m, sigma), solve for (a, b, rho) via weighted least squares, then
    project onto the feasible region (b>=0, |rho|<1, non-negative ATM variance).

    Clips (m, sigma) into a sane range and falls back to a safe degenerate answer on
    numerical failure — Nelder-Mead's simplex search (and its returned "best" point)
    can wander into degenerate territory (huge sigma, m far outside the data) that
    makes this SVD-based solve fail outright rather than just fit badly. A caller
    should never see a raw LinAlgError from a slice-fitting call.
    """
    spread = max(float(k.max() - k.min()), 0.1)
    m = float(np.clip(m, k.min() - 3 * spread, k.max() + 3 * spread))
    sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))

    x = k - m
    y = np.sqrt(x**2 + sigma**2)
    X = np.column_stack([np.ones_like(k), x, y])
    sqrt_w = np.sqrt(weights)
    try:
        coeffs, *_ = np.linalg.lstsq(X * sqrt_w[:, None], w * sqrt_w, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.max(w)), 1e-6, 0.0  # safe, clearly-bad-fit fallback, never a crash
    a, c1, b = coeffs  # w ~ a + c1*x + b*y, c1 = b*rho

    b = max(b, 1e-8)
    rho = np.clip(c1 / b, -0.999, 0.999)
    min_variance = a + b * sigma * np.sqrt(1 - rho**2)
    if min_variance < 0:
        a -= min_variance  # shift up just enough to touch zero, not below
    return float(a), float(b), float(rho)


def fit_svi_slice(k: np.ndarray, w: np.ndarray, weights: np.ndarray) -> SVIParams:
    """Fit one expiry slice. `k` = log-moneyness, `w` = total implied variance
    (iv^2 * T), `weights` = calibration weights (vega/spread from Step 3)."""
    m0 = float(np.average(k, weights=weights))
    sigma0 = max(float(np.std(k)), 0.1)

    def objective(params: np.ndarray) -> float:
        m, log_sigma = params
        sigma = _clip_sigma(log_sigma)
        try:
            a, b, rho = _inner_linear_solve(k, w, weights, m, sigma)
            fitted = SVIParams(a, b, rho, m, sigma).total_variance(k)
            sse = float(np.sum(weights * (fitted - w) ** 2))
        except np.linalg.LinAlgError:
            return np.inf
        return sse if np.isfinite(sse) else np.inf

    result = minimize(
        objective,
        x0=[m0, np.log(sigma0)],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10},
    )
    m, log_sigma = result.x
    sigma = _clip_sigma(log_sigma)
    a, b, rho = _inner_linear_solve(k, w, weights, m, sigma)
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def count_butterfly_violations(params: SVIParams, k_grid: np.ndarray) -> int:
    return int(np.sum(durrleman_g(params, k_grid) < 0))


BUTTERFLY_MARGIN = 1e-6  # require g(k) >= margin, not just >= 0 — see repair_butterfly_violation
MAX_REPAIR_ATTEMPTS = 3


def repair_butterfly_violation(
    k: np.ndarray, w: np.ndarray, weights: np.ndarray, k_grid: np.ndarray, initial: SVIParams
) -> SVIParams:
    """SLSQP repair, constraining g(k_grid) >= 0 (README 4.2: "re-fit with the
    constraint imposed via penalty or SLSQP"). Returns the repaired params — may
    still have residual violations if SLSQP can't fully satisfy the constraint; the
    caller re-checks and reports honestly rather than assuming success.

    Optimises all 5 SVI params directly (not via the quasi-explicit inner-solve used
    for the initial fit) — that inner solve clips/projects (b, rho, a) discontinuously,
    which breaks SLSQP's gradient-based line search ("positive directional derivative"
    failures were observed in practice). Direct parameterisation with proper `bounds`
    is smooth and well-behaved for a constrained local repair.

    Retries up to MAX_REPAIR_ATTEMPTS times from the previous attempt's result, and
    enforces a small positive margin (not literal >=0) on the constraint — SLSQP's
    convergence is a local, platform-dependent numerical result (a run that fully
    satisfies the constraint on one BLAS/LAPACK backend left a few residual
    violations on another in practice), so a single pass to the exact boundary isn't
    reliable across environments.
    """

    def unpack(params: np.ndarray) -> SVIParams:
        a, b, rho, m, sigma = params
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    def objective(params: np.ndarray) -> float:
        model = unpack(params)
        return float(np.sum(weights * (model.total_variance(k) - w) ** 2))

    def butterfly_constraint(params: np.ndarray) -> np.ndarray:
        return durrleman_g(unpack(params), k_grid) - BUTTERFLY_MARGIN

    def atm_variance_constraint(params: np.ndarray) -> float:
        a, b, rho, _, sigma = params
        return a + b * sigma * np.sqrt(max(1 - rho**2, 0.0)) - BUTTERFLY_MARGIN

    x0 = [initial.a, initial.b, initial.rho, initial.m, initial.sigma]
    best = initial
    for _ in range(MAX_REPAIR_ATTEMPTS):
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=[(None, None), (1e-8, None), (-0.999, 0.999), (None, None), (SIGMA_MIN, None)],
            constraints=[
                {"type": "ineq", "fun": butterfly_constraint},
                {"type": "ineq", "fun": atm_variance_constraint},
            ],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        best = unpack(result.x)
        if count_butterfly_violations(best, k_grid) == 0:
            break
        x0 = list(result.x)
    return best
