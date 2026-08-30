import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    CURVE_REQUIRED_NOT_NULL,
    CURVE_SCHEMA,
    FORWARD_REQUIRED_NOT_NULL,
    FORWARD_SCHEMA,
    UNDERLYING_REQUIRED_NOT_NULL,
    UNDERLYING_SCHEMA,
    VOL_SURFACE_REQUIRED_NOT_NULL,
    VOL_SURFACE_SCHEMA,
    validate,
)
from eqdrisk.stress.hypothetical_grid import (
    SPOT_LADDER,
    VOL_LADDER,
    run_hypothetical_grid,
)

ASOF = dt.date(2026, 8, 20)
UNDERLYING = "TEST"
SPOT = 100.0
PILLAR_TS = [0.25, 0.5]
PILLAR_EXPIRIES = [ASOF + dt.timedelta(days=int(t * 365)) for t in PILLAR_TS]


def _write_curve(tmp_path):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF, ASOF],
            "tenor": ["3M", "1Y"],
            "rate": [3.0, 3.0],
            "source": ["test", "test"],
        }
    )
    table = validate(df, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "curves", ["asof_date"])


def _write_underlying(tmp_path):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": [UNDERLYING],
            "open": [SPOT],
            "high": [SPOT],
            "low": [SPOT],
            "close": [SPOT],
            "volume": [0],
        }
    )
    table = validate(df, UNDERLYING_SCHEMA, UNDERLYING_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "underlyings", ["asof_date"])


def _write_forwards(tmp_path, r=0.03, q=0.01):
    rows = []
    for T, expiry in zip(PILLAR_TS, PILLAR_EXPIRIES, strict=True):
        forward = SPOT * np.exp((r - q) * T)
        rows.append(
            {
                "asof_date": ASOF,
                "underlying": UNDERLYING,
                "expiry": expiry,
                "T": T,
                "n_strikes": 10,
                "forward": forward,
                "discount_factor_implied": np.exp(-r * T),
                "discount_factor_curve": np.exp(-r * T),
                "discount_factor_diff_bp": 0.0,
                "r_squared": 0.9999,
                "implied_dividend_yield": q,
                "announced_dividend_yield": None,
                "dividend_yield_diff": None,
                "flag_r2": False,
                "flag_discount_factor_bp": False,
            }
        )
    table = validate(pd.DataFrame(rows), FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "forwards", ["asof_date", "underlying"])


def _write_surface(tmp_path, rho=-0.3):
    rows = []
    for T, expiry in zip(PILLAR_TS, PILLAR_EXPIRIES, strict=True):
        b, m, sigma = 0.10 * np.sqrt(T), 0.0, 0.10
        offset = b * sigma * np.sqrt(1 - rho**2)
        a = (0.20**2) * T - offset
        rows.append(
            {
                "asof_date": ASOF,
                "underlying": UNDERLYING,
                "expiry": expiry,
                "T": T,
                "model": "SVI",
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "eta": None,
                "theta": None,
                "n_points": 9,
                "rmse_vol_points": 0.01,
                "max_abs_error_vol_points": 0.02,
                "max_abs_error_k": 0.0,
                "butterfly_violations": 0,
                "calendar_violated": False,
            }
        )
    table = validate(pd.DataFrame(rows), VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "vol_surface", ["asof_date", "underlying"])


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _write_portfolio_yaml(tmp_path) -> str:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        yaml.dump(
            {"positions": [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 100}]}
        )
    )
    return str(path)


def test_run_hypothetical_grid_covers_full_ladder_and_matches_equity_arithmetic(tmp_path):
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)
    portfolio_path = _write_portfolio_yaml(tmp_path)

    result = run_hypothetical_grid(_cfg(tmp_path), ASOF, portfolio_path)

    assert not result.skipped
    assert result.base_value == pytest.approx(100 * SPOT)
    assert len(result.cells) == len(SPOT_LADDER) * len(VOL_LADDER)

    # A pure equity book only cares about the spot shock, not the vol shock —
    # every cell at the same spot shock (any vol shock) must have the same P&L.
    by_spot: dict[float, set[float]] = {}
    for c in result.cells:
        by_spot.setdefault(c.spot_shock_pct, set()).add(round(c.pnl, 6))
    for spot_shock_pct, pnls in by_spot.items():
        assert len(pnls) == 1, f"spot shock {spot_shock_pct} showed vol-dependent P&L for equity"
        expected_pnl = 100 * SPOT * spot_shock_pct
        assert next(iter(pnls)) == pytest.approx(expected_pnl)

    assert result.skew_scenario_pnl == pytest.approx(0.0)  # equity has no vol/skew exposure
    assert result.term_scenario_pnl == pytest.approx(0.0)


def test_pnl_matrix_shape_matches_ladders(tmp_path):
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)
    portfolio_path = _write_portfolio_yaml(tmp_path)

    result = run_hypothetical_grid(_cfg(tmp_path), ASOF, portfolio_path)
    matrix = result.pnl_matrix()

    assert matrix.shape == (len(SPOT_LADDER), len(VOL_LADDER))
    assert list(matrix.index) == sorted(SPOT_LADDER, reverse=True)
    assert list(matrix.columns) == sorted(VOL_LADDER)


def test_run_hypothetical_grid_reports_skip_when_no_curated_rates(tmp_path):
    portfolio_path = _write_portfolio_yaml(tmp_path)
    result = run_hypothetical_grid(_cfg(tmp_path), ASOF, portfolio_path)
    assert "_all_" in result.skipped
