import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import VOL_SURFACE_REQUIRED_NOT_NULL, VOL_SURFACE_SCHEMA, validate
from eqdrisk.vol.risk_factors import (
    K_GRID,
    T_GRID,
    evaluate_risk_factor_grid,
    run_risk_factor_grid,
)
from eqdrisk.vol.svi import SVIParams

ASOF = dt.date(2026, 8, 20)
UNDERLYING = "TEST"


def _surface_df(pillar_Ts: list[float], rho: float = -0.3) -> pd.DataFrame:
    rows = []
    for T in pillar_Ts:
        b, m, sigma = 0.10 * np.sqrt(T), 0.0, 0.10
        offset = b * sigma * np.sqrt(1 - rho**2)
        a = (0.20**2) * T - offset
        rows.append(
            {
                "asof_date": ASOF,
                "underlying": UNDERLYING,
                "expiry": ASOF + dt.timedelta(days=int(T * 365)),
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
    return pd.DataFrame(rows)


def test_evaluate_risk_factor_grid_has_expected_shape_and_column_values():
    surface = _surface_df([0.25, 0.5])
    grid = evaluate_risk_factor_grid(surface)
    assert grid is not None
    assert len(grid) == len(K_GRID) * len(T_GRID)
    assert set(grid["T_label"]) == set(T_GRID)
    assert set(grid["k"]) == set(K_GRID)
    assert np.all(np.isfinite(grid["w"]))


def test_evaluate_risk_factor_grid_at_a_pillar_matches_that_slices_own_svi_params():
    T = 0.5
    surface = _surface_df([0.25, T])
    grid = evaluate_risk_factor_grid(surface)
    assert grid is not None

    row = surface[surface["T"] == T].iloc[0]
    params = SVIParams(a=row.a, b=row.b, rho=row.rho, m=row.m, sigma=row.sigma)

    node = grid[(grid["T_label"] == "6m") & (grid["k"] == 0.0)].iloc[0]
    assert node["w"] == pytest.approx(float(params.total_variance(0.0)), rel=1e-9)


def test_evaluate_risk_factor_grid_none_with_too_few_pillars():
    surface = _surface_df([0.5])
    assert evaluate_risk_factor_grid(surface) is None


def test_run_risk_factor_grid_end_to_end(tmp_path):
    surface = _surface_df([0.25, 0.5])
    table = validate(surface, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "vol_surface", ["asof_date", "underlying"])

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_risk_factor_grid(cfg, ASOF, underlyings=[UNDERLYING])
    assert result.n_underlyings == 1
    assert not result.skipped

    out = store.query("SELECT * FROM rf", views={"rf": str(tmp_path / "risk_factors")}).to_pandas()
    assert len(out) == len(K_GRID) * len(T_GRID)
    assert set(out["underlying"]) == {UNDERLYING}


def test_run_risk_factor_grid_reports_skip_when_too_thin(tmp_path):
    surface = _surface_df([0.5])  # only 1 pillar
    table = validate(surface, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "vol_surface", ["asof_date", "underlying"])

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_risk_factor_grid(cfg, ASOF, underlyings=[UNDERLYING])
    assert result.n_underlyings == 0
    assert UNDERLYING in result.skipped
