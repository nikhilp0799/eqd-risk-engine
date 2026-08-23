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
    UNDERLYING_REQUIRED_NOT_NULL,
    UNDERLYING_SCHEMA,
    VOL_SURFACE_REQUIRED_NOT_NULL,
    VOL_SURFACE_SCHEMA,
    validate,
)
from eqdrisk.pricing.engine import run_pricing
from eqdrisk.vol.svi import SVIParams

ASOF = dt.date(2026, 8, 20)
EXPIRY = dt.date(2026, 11, 18)
UNDERLYING = "TEST"
T = 0.25
FORWARD = 101.0
SPOT = 100.0


def _write_curve(tmp_path):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "tenor": ["3M"],
            "rate": [4.0],
            "source": ["test"],
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
            "volume": [1000],
        }
    )
    table = validate(df, UNDERLYING_SCHEMA, UNDERLYING_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "underlyings", ["asof_date"])


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


def _write_iv_and_surface(tmp_path, params: SVIParams):
    k = np.linspace(-0.2, 0.2, 9)
    w = params.total_variance(k)
    iv = np.sqrt(w / T)
    strikes = FORWARD * np.exp(k)
    cp = np.where(k > 0, "C", "P")
    iv_df = pd.DataFrame(
        {
            "asof_date": [ASOF] * len(k),
            "underlying": [UNDERLYING] * len(k),
            "expiry": [EXPIRY] * len(k),
            "strike": strikes,
            "cp": cp,
            "T": [T] * len(k),
            "k": k,
            "iv": iv,
            "total_variance": w,
            "vega": np.ones(len(k)),
            "weight": np.ones(len(k)),
            "reason": ["OK"] * len(k),
        }
    )
    iv_table = validate(iv_df, IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
    store.write_partitioned(iv_table, tmp_path / "implied_vols", ["asof_date", "underlying"])

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
            "n_points": [len(k)],
            "rmse_vol_points": [0.01],
            "max_abs_error_vol_points": [0.02],
            "max_abs_error_k": [0.0],
            "butterfly_violations": [0],
            "calendar_violated": [False],
        }
    )
    surface_table = validate(surface_df, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(surface_table, tmp_path / "vol_surface", ["asof_date", "underlying"])


def test_run_pricing_end_to_end(tmp_path):
    params = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forward(tmp_path)
    _write_iv_and_surface(tmp_path, params)

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_pricing(cfg, ASOF, underlyings=[UNDERLYING])

    assert result.n_priced[UNDERLYING] == 9
    assert not result.skipped

    out = store.query("SELECT * FROM g", views={"g": str(tmp_path / "greeks")}).to_pandas()
    assert len(out) == 9
    assert (out["price"] >= 0).all()
    assert out["vega"].gt(0).all()
    # Every row has a well-defined stickiness comparison since the whole slice is SVI.
    assert out["delta_sticky_strike"].notna().all()
    assert out["delta_sticky_delta"].notna().all()
    assert out["delta_sticky_local_vol"].notna().all()
    # Sticky-strike must equal the plain spot delta exactly (no vol-surface adjustment).
    assert np.allclose(out["delta_sticky_strike"], out["delta_spot"])
    # For a skewed (rho != 0) slice, the sticky-delta adjustment should be non-trivial
    # for at least some strikes away from the money.
    assert not np.allclose(out["delta_sticky_delta"], out["delta_sticky_strike"])


def test_run_pricing_reports_skip_when_no_surface(tmp_path):
    _write_curve(tmp_path)

    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_pricing(cfg, ASOF, underlyings=[UNDERLYING])

    assert "_all_" in result.skipped
