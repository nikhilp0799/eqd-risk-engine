import datetime as dt

import numpy as np
import pandas as pd
import yaml

import eqdrisk.ml.run as run_module
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
from eqdrisk.ml.run import run_deep_hedge
from eqdrisk.ml.train import TrainConfig

UNDERLYING = "TEST"
ASOF = dt.date(2026, 8, 20)
SPOT = 100.0
PILLAR_TS = [0.25, 0.5, 1.5]  # last pillar comfortably beyond the autocall's T=1.0
PILLAR_EXPIRIES = [ASOF + dt.timedelta(days=int(t * 365)) for t in PILLAR_TS]
PLACEHOLDER_EXPIRY = ASOF + dt.timedelta(days=400)  # >1y out -> t_grid covers T=1.0


def _write_curve(tmp_path, rate=3.0):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF, ASOF],
            "tenor": ["3M", "1Y"],
            "rate": [rate, rate],
            "source": ["t", "t"],
        }
    )
    table = validate(df, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "curves", ["asof_date"])


def _write_underlying(tmp_path, spot=SPOT):
    df = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": [UNDERLYING],
            "open": [spot],
            "high": [spot],
            "low": [spot],
            "close": [spot],
            "volume": [0],
        }
    )
    table = validate(df, UNDERLYING_SCHEMA, UNDERLYING_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "underlyings", ["asof_date"])


def _write_forwards(tmp_path, spot=SPOT, r=3.0, q=0.01):
    r_frac = r / 100.0
    rows = []
    for tt, expiry in zip(PILLAR_TS, PILLAR_EXPIRIES, strict=True):
        forward = spot * np.exp((r_frac - q) * tt)
        rows.append(
            {
                "asof_date": ASOF,
                "underlying": UNDERLYING,
                "expiry": expiry,
                "T": tt,
                "n_strikes": 10,
                "forward": forward,
                "discount_factor_implied": np.exp(-r_frac * tt),
                "discount_factor_curve": np.exp(-r_frac * tt),
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
    for tt, expiry in zip(PILLAR_TS, PILLAR_EXPIRIES, strict=True):
        b, m, sigma = 0.10 * np.sqrt(tt), 0.0, 0.10
        offset = b * sigma * np.sqrt(1 - rho**2)
        a = 0.20**2 * tt - offset
        rows.append(
            {
                "asof_date": ASOF,
                "underlying": UNDERLYING,
                "expiry": expiry,
                "T": tt,
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


def _write_portfolio(tmp_path) -> None:
    (tmp_path / "configs").mkdir(exist_ok=True)
    positions = [
        {
            "id": "B1",
            "type": "barrier",
            "underlying": UNDERLYING,
            "sub": "down_and_in_put",
            "strike": SPOT,
            "barrier": SPOT * 0.7,
            "expiry": PLACEHOLDER_EXPIRY.isoformat(),
            "qty": 1,
        }
    ]
    (tmp_path / "configs" / "portfolio.yaml").write_text(yaml.dump({"positions": positions}))


def _seed_real_data(tmp_path):
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)
    _write_portfolio(tmp_path)


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _tiny_train_cfg(instrument):
    n_steps = 4 if instrument == "vanilla" else len(run_module.AUTOCALL_SPEC.obs_times)
    return TrainConfig(n_paths_train=32, n_steps=n_steps, epochs=2, hidden=4, cost_bps=5.0)


def test_run_deep_hedge_returns_none_when_no_curated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _write_portfolio(tmp_path)  # market data intentionally absent
    cfg = _cfg(tmp_path)
    result = run_deep_hedge(
        cfg, ASOF, "vanilla", "variance", underlying=UNDERLYING, project_root=tmp_path
    )
    assert result is None


def test_run_deep_hedge_vanilla_end_to_end_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    cfg = _cfg(tmp_path)

    result = run_deep_hedge(
        cfg,
        ASOF,
        "vanilla",
        "variance",
        underlying=UNDERLYING,
        n_paths_eval=32,
        project_root=tmp_path,
    )

    assert result is not None
    assert result.instrument == "vanilla"
    assert np.isfinite(result.comparison.learned.std)
    assert np.isfinite(result.comparison.baseline.std)

    df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_results")}
    ).to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["instrument"] == "vanilla"
    assert df.iloc[0]["loss_type"] == "variance"


def test_run_deep_hedge_autocall_end_to_end_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    cfg = _cfg(tmp_path)

    result = run_deep_hedge(
        cfg,
        ASOF,
        "autocall",
        "cvar",
        underlying=UNDERLYING,
        n_paths_eval=32,
        project_root=tmp_path,
    )

    assert result is not None
    assert result.instrument == "autocall"
    assert np.isfinite(result.comparison.learned.std)
    assert np.isfinite(result.comparison.baseline.std)

    df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_results")}
    ).to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["instrument"] == "autocall"
    assert df.iloc[0]["loss_type"] == "cvar"


def test_run_deep_hedge_second_run_overwrites_same_day_partition(tmp_path, monkeypatch):
    """`store.write_partitioned` is idempotent per partition — running twice for
    the same asof_date must not duplicate rows."""
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    cfg = _cfg(tmp_path)

    run_deep_hedge(
        cfg,
        ASOF,
        "vanilla",
        "variance",
        underlying=UNDERLYING,
        n_paths_eval=32,
        project_root=tmp_path,
    )
    run_deep_hedge(
        cfg,
        ASOF,
        "vanilla",
        "cvar",
        underlying=UNDERLYING,
        n_paths_eval=32,
        project_root=tmp_path,
    )

    df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_results")}
    ).to_pandas()
    assert len(df) == 1  # second run overwrote the first (same asof_date partition)
    assert df.iloc[0]["loss_type"] == "cvar"
