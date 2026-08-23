import datetime as dt

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    CURVE_REQUIRED_NOT_NULL,
    CURVE_SCHEMA,
    FORWARD_REQUIRED_NOT_NULL,
    FORWARD_SCHEMA,
    IMPLIED_VOL_REQUIRED_NOT_NULL,
    IMPLIED_VOL_SCHEMA,
    VOL_SURFACE_REQUIRED_NOT_NULL,
    VOL_SURFACE_SCHEMA,
    validate,
)
from eqdrisk.pricing.varswap_engine import run_varswap
from eqdrisk.vol.svi import SVIParams

ASOF = dt.date(2026, 8, 20)
EXPIRY = dt.date(2026, 11, 18)
UNDERLYING = "TEST"
T = 0.25
FORWARD = 101.0


def _write_curve(tmp_path):
    df = pd.DataFrame({"asof_date": [ASOF], "tenor": ["3M"], "rate": [4.0], "source": ["test"]})
    table = validate(df, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "curves", ["asof_date"])


def _write_forward(tmp_path):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": [UNDERLYING],
            "expiry": [EXPIRY],
            "T": [T],
            "n_strikes": [10],
            "forward": [FORWARD],
            "discount_factor_implied": [0.99],
            "discount_factor_curve": [0.99],
            "discount_factor_diff_bp": [1.0],
            "r_squared": [0.9999],
            "implied_dividend_yield": [0.01],
            "announced_dividend_yield": [None],
            "dividend_yield_diff": [None],
            "flag_r2": [False],
            "flag_discount_factor_bp": [False],
        }
    )
    table = validate(df, FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "forwards", ["asof_date", "underlying"])


def _write_surface_and_iv(tmp_path, params: SVIParams):
    surface_df = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": [UNDERLYING],
            "expiry": [EXPIRY],
            "T": [T],
            "model": ["SVI"],
            "a": [params.a],
            "b": [params.b],
            "rho": [params.rho],
            "m": [params.m],
            "sigma": [params.sigma],
            "eta": [None],
            "theta": [None],
            "n_points": [9],
            "rmse_vol_points": [0.01],
            "max_abs_error_vol_points": [0.02],
            "max_abs_error_k": [0.0],
            "butterfly_violations": [0],
            "calendar_violated": [False],
        }
    )
    surface_table = validate(surface_df, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(surface_table, tmp_path / "vol_surface", ["asof_date", "underlying"])

    k = np.linspace(-0.15, 0.15, 9)
    w = params.total_variance(k)
    iv = np.sqrt(w / T)
    n = len(k)
    iv_df = pd.DataFrame(
        {
            "asof_date": [ASOF] * n,
            "underlying": [UNDERLYING] * n,
            "expiry": [EXPIRY] * n,
            "strike": FORWARD * np.exp(k),
            "cp": np.where(k > 0, "C", "P"),
            "T": [T] * n,
            "k": k,
            "iv": iv,
            "total_variance": w,
            "vega": np.ones(n),
            "weight": np.ones(n),
            "reason": ["OK"] * n,
        }
    )
    iv_table = validate(iv_df, IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
    store.write_partitioned(iv_table, tmp_path / "implied_vols", ["asof_date", "underlying"])


def test_run_varswap_end_to_end(tmp_path):
    params = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    _write_curve(tmp_path)
    _write_forward(tmp_path)
    _write_surface_and_iv(tmp_path, params)

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_varswap(cfg, ASOF, underlyings=[UNDERLYING])

    assert result.n_priced[UNDERLYING] == 1
    assert not result.skipped

    out = store.query("SELECT * FROM v", views={"v": str(tmp_path / "varswap")}).to_pandas()
    assert len(out) == 1
    row = out.iloc[0]
    assert row["fair_strike_vol_wide"] > 0
    assert row["fair_strike_vol_narrow"] > 0
    assert row["n_strikes_observed"] == 9
    # The narrow (observed-strike) range sits inside the wide (extrapolated) range,
    # and this is a curved smile, so wide >= narrow (truncation error >= 0).
    assert row["truncation_error_vol_points"] >= 0


def test_run_varswap_reports_skip_when_no_surface(tmp_path):
    _write_curve(tmp_path)

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_varswap(cfg, ASOF, underlyings=[UNDERLYING])

    assert "_all_" in result.skipped
