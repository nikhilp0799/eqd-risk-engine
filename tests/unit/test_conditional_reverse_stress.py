import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.mark import PortfolioMarkResult, PositionMark, revaluation_pnl
from eqdrisk.risk.var import VolProxy
from eqdrisk.stress.conditional import (
    calm_term_ratio,
    conditional_moves,
    crash_term_ratio,
    fit_crash_response,
    scale_long_end,
)
from eqdrisk.stress.reverse import (
    ReverseProblem,
    factor_bounds,
    factor_covariance,
    solve,
)
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

N_T, N_K = len(T_GRID), len(K_GRID)


def _levels(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """SPX a random walk; AAPL and VIX tied to it with known log betas that
    differ in sell-offs (SPX 10-day move in its worst 10%) and otherwise."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n).date
    spx = np.cumsum(rng.normal(0, 0.012, n))
    return pd.DataFrame(
        {
            "SPX": 100 * np.exp(spx),
            "AAPL": 50 * np.exp(1.3 * spx),
            "VIX": 20 * np.exp(-4.0 * spx),
            "VXV": 20 * np.exp(-4.0 * spx) * 0.75 + 5.0,
        },
        index=pd.Index(days, name="asof_date"),
    )


def test_crash_fit_recovers_exact_log_betas_and_applies_them():
    fits = fit_crash_response(_levels(), ["AAPL", "VIX"])
    assert fits["AAPL"].beta == pytest.approx(1.3, abs=1e-9)
    assert fits["VIX"].beta == pytest.approx(-4.0, abs=1e-9)
    assert fits["VIX"].alpha == pytest.approx(0.0, abs=1e-9)
    moves = conditional_moves(fits, -0.20)
    assert moves["SPX"] == -0.20
    assert moves["VIX"] == pytest.approx(0.8**-4.0 - 1)
    assert moves["AAPL"] == pytest.approx(0.8**1.3 - 1)


def test_crash_term_ratio_is_the_points_slope_in_sell_offs():
    assert crash_term_ratio(_levels()) == pytest.approx(0.75, abs=1e-9)


def test_long_end_scaling_leaves_one_month_alone():
    beta = np.tile(np.array([1.0, 0.6, 0.4, 0.2, 0.1])[:, None], (1, N_K))
    proxy = VolProxy("VIX", 20.0, np.full((N_T, N_K), 20.0), beta)
    assert calm_term_ratio(proxy) == pytest.approx(0.6)
    scaled = scale_long_end(proxy, 1.5)
    np.testing.assert_allclose(scaled.beta[0], beta[0])
    np.testing.assert_allclose(scaled.beta[1:], 1.5 * beta[1:])


def _mark(pid: str, type_: str, **kw) -> PositionMark:
    base = {"underlying": "SPX", "expiry": dt.date(2027, 1, 1), "T": 0.5, "price": 0.0}
    return PositionMark(position_id=pid, type=type_, **{**base, **kw})


def test_revaluation_pnl_holds_the_varswap_at_its_base_fair_strike():
    asof = dt.date(2026, 10, 29)
    base = PortfolioMarkResult(
        asof=asof,
        marks=[
            _mark("O", "vanilla", price=10.0),
            _mark("S", "varswap", vega=1000.0, fair_strike=20.0),
            _mark("D", "vanilla", price=5.0, status=lifecycle.SETTLED),
        ],
    )
    shocked = PortfolioMarkResult(
        asof=asof,
        marks=[
            _mark("O", "vanilla", price=7.0),
            _mark("S", "varswap", vega=1000.0, fair_strike=25.0),
            _mark("D", "vanilla", price=5.0, status=lifecycle.SETTLED),
        ],
    )
    pnl = revaluation_pnl(base, shocked)
    assert pnl["O"] == pytest.approx(-3.0)
    assert pnl["S"] == pytest.approx(1000.0 * (625.0 - 400.0) / 40.0)
    assert pnl["D"] == 0.0


class _LinearProblem(ReverseProblem):
    """P&L linear in the factors: the reverse stress has a closed form."""

    def __init__(self, cov: np.ndarray, g: np.ndarray):
        self.cov_inv = np.linalg.inv(cov)
        self.bounds = [(-5.0, 5.0)] * len(g)
        self.g = g

    def pnl(self, x: np.ndarray) -> float:
        return float(self.g @ x)


def test_reverse_solve_matches_the_closed_form_for_a_linear_book():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    g = np.array([1e6, -2e6])
    problem = _LinearProblem(cov, g)
    target = 5e5
    direction = -cov @ g
    sol = solve(
        problem, target, "linear", direction / np.sqrt(direction @ problem.cov_inv @ direction)
    )
    assert sol is not None
    expected_x = -target * cov @ g / (g @ cov @ g)
    np.testing.assert_allclose(sol.x, expected_x, rtol=1e-3)
    assert sol.distance == pytest.approx(target / np.sqrt(g @ cov @ g), rel=1e-3)
    assert sol.pnl <= -target * (1 - 1e-3)


def test_reverse_solve_finds_the_shorter_way_from_a_poor_start():
    cov = np.diag([0.04, 0.09])
    g = np.array([1e6, 1e6])
    problem = _LinearProblem(cov, g)
    sol = solve(problem, 5e5, "axis", np.array([-1.0, 0.0]) / 0.2)
    assert sol is not None
    assert sol.distance == pytest.approx(5e5 / np.sqrt(g @ cov @ g), rel=1e-2)


def test_covariance_is_zero_mean_and_bounds_use_the_index_range():
    levels = _levels()
    cov = factor_covariance(levels, ["SPX", "VIX"])
    moves = np.log(levels[["SPX", "VIX"]] / levels[["SPX", "VIX"]].shift(10)).dropna().to_numpy()
    np.testing.assert_allclose(cov, moves.T @ moves / len(moves))
    asof = levels.index[-1]
    bounds = factor_bounds(levels, ["SPX", "VIX"], ["SPX"], asof)
    assert bounds[0] == pytest.approx((np.log(0.4), np.log(1.6)))
    vix = levels["VIX"]
    assert bounds[1] == pytest.approx(
        (np.log(vix.min() / vix.loc[asof]), np.log(vix.max() / vix.loc[asof]))
    )
