import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    RISK_FACTOR_REQUIRED_NOT_NULL,
    RISK_FACTOR_SCHEMA,
    VOL_SURFACE_REQUIRED_NOT_NULL,
    VOL_SURFACE_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import trading_days
from eqdrisk.vol.pca import (
    ROLLING_WINDOW,
    analyse_underlying,
    daily_vol_changes,
    fit_pca,
    node_coverage,
    rolling_stability,
    run_vol_pca,
)
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

UNDERLYING = "TEST"
NODES = [(t, k) for t in T_GRID for k in K_GRID]


def _risk_factors(days: list[dt.date], iv_by_day: dict[dt.date, np.ndarray]) -> pd.DataFrame:
    """Grid rows in the stored form (total variance w), from implied vols in %."""
    rows = []
    for day in days:
        for (t_label, k), iv in zip(NODES, iv_by_day[day], strict=True):
            T = T_GRID[t_label]
            rows.append(
                {
                    "asof_date": day,
                    "underlying": UNDERLYING,
                    "k": k,
                    "T": T,
                    "T_label": t_label,
                    "w": (iv / 100.0) ** 2 * T,
                }
            )
    return pd.DataFrame(rows)


def _surface(days: list[dt.date], k_min: float = -0.35, k_max: float = 0.25) -> pd.DataFrame:
    rows = []
    for day in days:
        for T in (0.05, 0.3, 0.8, 2.5):
            rows.append(
                {
                    "asof_date": day,
                    "underlying": UNDERLYING,
                    "expiry": day + dt.timedelta(days=int(T * 365)),
                    "T": T,
                    "model": "SVI",
                    "a": 0.01,
                    "b": 0.1,
                    "rho": -0.3,
                    "m": 0.0,
                    "sigma": 0.1,
                    "eta": None,
                    "theta": None,
                    "n_points": 9,
                    "rmse_vol_points": 0.01,
                    "max_abs_error_vol_points": 0.02,
                    "max_abs_error_k": 0.0,
                    "butterfly_violations": 0,
                    "calendar_violated": False,
                    "n_carried": 0,
                    "k_min": k_min,
                    "k_max": k_max,
                }
            )
    return pd.DataFrame(rows)


def _one_factor_history(n_days: int, noise: float = 0.01, seed: int = 0):
    """A parallel vol shift each day plus small node noise."""
    rng = np.random.default_rng(seed)
    days = trading_days(dt.date(2026, 9, 1), dt.date(2026, 12, 31))[:n_days]
    level = 20.0 + np.cumsum(rng.normal(0.0, 1.0, n_days))
    iv = {d: lvl + rng.normal(0.0, noise, len(NODES)) for d, lvl in zip(days, level, strict=True)}
    return days, iv


def test_vol_changes_are_equal_across_maturities_for_a_parallel_vol_move():
    """The reason for working in vol, not w: 20% -> 21% is one vol point at
    every maturity, while dw is 24x bigger at 2y than at 1m."""
    days = [dt.date(2026, 9, 1), dt.date(2026, 9, 2)]
    iv = {days[0]: np.full(len(NODES), 20.0), days[1]: np.full(len(NODES), 21.0)}
    changes = daily_vol_changes(_risk_factors(days, iv), NODES)
    assert list(changes.index) == [days[1]]
    assert np.allclose(changes.to_numpy(), 1.0)


def test_vol_changes_skip_non_trading_days_and_gaps():
    """2026-09-05 is a Saturday (a carried surface) and 09-09 -> 09-11 skips
    09-10: neither may appear as a one-day move."""
    days = [
        dt.date(2026, 9, 3),
        dt.date(2026, 9, 4),
        dt.date(2026, 9, 5),
        dt.date(2026, 9, 8),
        dt.date(2026, 9, 9),
        dt.date(2026, 9, 11),
    ]
    iv = {d: np.full(len(NODES), 20.0 + i) for i, d in enumerate(days)}
    changes = daily_vol_changes(_risk_factors(days, iv), NODES)
    # 09-04 -> 09-08 is consecutive (09-07 is Labor Day) once the Saturday is dropped.
    assert list(changes.index) == [dt.date(2026, 9, 4), dt.date(2026, 9, 8), dt.date(2026, 9, 9)]
    assert np.allclose(changes.loc[dt.date(2026, 9, 8)].to_numpy(), 2.0)


def test_fit_pca_recovers_a_planted_level_factor():
    days, iv = _one_factor_history(40)
    changes = daily_vol_changes(_risk_factors(days, iv), NODES)
    fit = fit_pca(changes.to_numpy())
    assert fit.explained_share[0] > 0.99
    assert np.allclose(fit.loadings[:, 0], 1.0 / np.sqrt(len(NODES)), atol=0.01)
    assert np.isclose(fit.explained_share.sum(), 1.0)


def test_fit_pca_loadings_sum_positive_and_are_orthonormal():
    rng = np.random.default_rng(3)
    fit = fit_pca(rng.normal(size=(50, 6)))
    assert (fit.loadings.sum(axis=0) > 0).all()
    assert np.allclose(fit.loadings.T @ fit.loadings, np.eye(6), atol=1e-10)
    assert (np.diff(fit.eigenvalues) <= 0).all()


def test_rolling_stability_is_one_for_a_stationary_single_factor():
    days, iv = _one_factor_history(40)
    changes = daily_vol_changes(_risk_factors(days, iv), NODES)
    stability = rolling_stability(changes, fit_pca(changes.to_numpy()))
    pc1 = stability[stability["pc"] == 1]
    assert len(pc1) == len(changes) - ROLLING_WINDOW + 1
    assert (pc1["abs_cosine"] > 0.99).all()


def test_node_coverage_requires_both_bracketing_expiries_to_quote_the_strike():
    days = [dt.date(2026, 9, 1), dt.date(2026, 9, 2)]
    surface = _surface(days, k_min=-0.15, k_max=0.15)
    # One day the 0.8y expiry quotes deeper puts: the 6m and 1y -0.2 nodes need
    # both neighbours (0.3y and 0.8y; 0.8y and 2.5y), so they stay uncovered.
    surface.loc[(surface["asof_date"] == days[0]) & (surface["T"] == 0.8), "k_min"] = -0.25
    cov = node_coverage(surface)
    assert cov[("3m", 0.0)] == 1.0
    assert cov[("6m", -0.2)] == 0.0
    assert cov[("3m", 0.2)] == 0.0


def test_node_coverage_counts_a_node_outside_the_expiry_range_as_uncovered():
    surface = _surface([dt.date(2026, 9, 1)])
    surface = surface[surface["T"] >= 0.3]  # first expiry after the 1m and 3m nodes
    cov = node_coverage(surface)
    assert cov[("1m", 0.0)] == 0.0 and cov[("3m", 0.0)] == 0.0
    assert cov[("6m", 0.0)] == 1.0


def test_node_coverage_treats_a_missing_strike_range_as_uncovered():
    surface = _surface([dt.date(2026, 9, 1)])
    surface["k_min"] = None
    assert (node_coverage(surface) == 0.0).all()


def test_analyse_underlying_drops_nodes_below_the_coverage_bar():
    days, iv = _one_factor_history(30)
    result = analyse_underlying(
        UNDERLYING, _surface(days, k_min=-0.15, k_max=0.25), _risk_factors(days, iv)
    )
    assert not isinstance(result, str)
    assert all(k >= -0.1 for _, k in result.nodes)
    assert len(result.nodes) == 4 * len(T_GRID)


def test_analyse_underlying_needs_a_full_window_of_changes():
    days, iv = _one_factor_history(ROLLING_WINDOW)  # 20 days -> 19 changes
    result = analyse_underlying(UNDERLYING, _surface(days), _risk_factors(days, iv))
    assert isinstance(result, str) and "need 20" in result


@pytest.fixture
def stored_history(tmp_path):
    days, iv = _one_factor_history(30)
    curated = tmp_path / "curated"
    store.write_partitioned(
        validate(_surface(days), VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL),
        curated / "vol_surface",
        ["asof_date", "underlying"],
    )
    store.write_partitioned(
        validate(_risk_factors(days, iv), RISK_FACTOR_SCHEMA, RISK_FACTOR_REQUIRED_NOT_NULL),
        curated / "risk_factors",
        ["asof_date", "underlying"],
    )
    cfg = BaseConfig(
        run_date=days[-1],
        universe=Universe(index=[UNDERLYING], single_names=["MISSING"]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(curated)),
        calendar="NYSE",
        daycount="ACT/365F",
    )
    return cfg, days


def test_run_vol_pca_writes_all_three_tables(stored_history):
    cfg, days = stored_history
    result = run_vol_pca(cfg, days[-1])

    assert [u.underlying for u in result.fitted] == [UNDERLYING]
    assert "MISSING" in result.skipped
    text = result.render()
    assert "29 daily changes" in text and "rolling 20-day windows" in text

    views = {
        name: str(cfg.paths.curated) + f"/{name}"
        for name in ("vol_pca", "vol_pca_loadings", "vol_pca_stability")
    }
    summary = store.query("SELECT * FROM vol_pca ORDER BY pc", views=views).to_pandas()
    assert len(summary) == len(NODES)
    assert summary["explained_share"].iloc[0] > 0.99
    loadings = store.query("SELECT * FROM vol_pca_loadings", views=views).to_pandas()
    assert len(loadings) == 3 * len(NODES)
    assert loadings["coverage"].eq(1.0).all()
    stability = store.query("SELECT * FROM vol_pca_stability", views=views).to_pandas()
    assert set(stability["pc"]) == {1, 2, 3}


def test_run_vol_pca_only_uses_history_up_to_the_date(stored_history):
    cfg, days = stored_history
    result = run_vol_pca(cfg, days[24])
    assert len(result.fitted[0].changes) == 24
