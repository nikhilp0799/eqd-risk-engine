import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

import eqdrisk.ml.run as run_module
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    CURVE_REQUIRED_NOT_NULL,
    CURVE_SCHEMA,
    DEEP_HEDGE_RESULT_SCHEMA,
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

LEGACY_MISSING = {"seed", "learned_turnover", "baseline_turnover"}

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
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": True,
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


def _read_results(tmp_path):
    return store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_results")}
    ).to_pandas()


def _run(tmp_path, instrument="vanilla", loss="variance", seeds=(0,)):
    return run_deep_hedge(
        _cfg(tmp_path),
        ASOF,
        instrument,
        loss,
        underlying=UNDERLYING,
        n_paths_eval=32,
        project_root=tmp_path,
        seeds=seeds,
    )


def test_run_deep_hedge_returns_none_when_no_curated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _write_portfolio(tmp_path)  # market data intentionally absent
    assert _run(tmp_path) is None


def test_run_deep_hedge_vanilla_end_to_end_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    results = _run(tmp_path, "vanilla", "variance")

    assert results is not None and len(results) == 1
    result = results[0]
    assert result.instrument == "vanilla" and result.seed == 0
    assert np.isfinite(result.comparison.learned.std)
    assert np.isfinite(result.comparison.baseline.std)

    df = _read_results(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["instrument"] == "vanilla"
    assert df.iloc[0]["loss_type"] == "variance"
    assert df.iloc[0]["seed"] == 0
    assert np.isfinite(df.iloc[0]["learned_turnover"])
    assert np.isfinite(df.iloc[0]["baseline_turnover"])


def test_run_deep_hedge_autocall_end_to_end_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    results = _run(tmp_path, "autocall", "cvar")

    assert results is not None
    assert results[0].instrument == "autocall"
    assert np.isfinite(results[0].comparison.learned.std)
    assert np.isfinite(results[0].comparison.baseline.std)

    df = _read_results(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["instrument"] == "autocall"
    assert df.iloc[0]["loss_type"] == "cvar"


def test_run_deep_hedge_different_combinations_same_day_accumulate(tmp_path, monkeypatch):
    """`store.write_partitioned` replaces a WHOLE `asof_date` partition per call
    — since `deephedge` is invoked once PER combination, two different
    combinations run on the same day must both survive (not have the second
    silently erase the first), unlike every other table in this project which
    writes a full day's rows in one call."""
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    _run(tmp_path, "vanilla", "variance")
    _run(tmp_path, "vanilla", "cvar")

    df = _read_results(tmp_path)
    assert len(df) == 2
    assert set(df["loss_type"]) == {"variance", "cvar"}


def test_run_deep_hedge_same_combination_rerun_replaces_not_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    for _ in range(2):
        _run(tmp_path, "vanilla", "variance")

    df = _read_results(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["loss_type"] == "variance"


def test_run_deep_hedge_default_trains_three_seeds_one_row_each(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    results = _run(tmp_path, seeds=run_module.DEFAULT_SEEDS)

    assert results is not None and [r.seed for r in results] == [0, 1, 2]
    df = _read_results(tmp_path)
    assert sorted(df["seed"]) == [0, 1, 2]
    # Different seeds really are different training runs.
    assert df["learned_std"].nunique() == 3


def test_run_deep_hedge_separate_seed_runs_accumulate(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)

    _run(tmp_path, seeds=(0,))
    _run(tmp_path, seeds=(1,))

    assert sorted(_read_results(tmp_path)["seed"]) == [0, 1]


def _write_pre_phase6_row(tmp_path, loss_type):
    """A row as written before Phase 6: no seed or turnover columns."""
    legacy_fields = [f for f in DEEP_HEDGE_RESULT_SCHEMA if f.name not in LEGACY_MISSING]
    row = {
        "asof_date": ASOF,
        "underlying": UNDERLYING,
        "instrument": "vanilla",
        "loss_type": loss_type,
        "cost_bps": 5.0,
        "n_paths_train": 32,
        "n_steps": 4,
        "epochs": 300,
        "n_eval_paths": 32,
        "learned_mean": 0.0,
        "learned_std": 1.0,
        "learned_cvar": -1.0,
        "baseline_mean": 0.0,
        "baseline_std": 1.0,
        "baseline_cvar": -1.0,
    }
    table = pa.Table.from_pandas(pd.DataFrame([row]), schema=pa.schema(legacy_fields))
    store.write_partitioned(table, tmp_path / "deep_hedge_results", ["asof_date"])


def test_run_deep_hedge_replaces_pre_phase6_row_of_same_combination_only(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    _write_pre_phase6_row(tmp_path, "variance")

    _run(tmp_path, "vanilla", "variance", seeds=(0,))

    df = _read_results(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["seed"] == 0


def test_run_deep_hedge_keeps_pre_phase6_row_of_other_combination(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    _write_pre_phase6_row(tmp_path, "cvar")

    _run(tmp_path, "vanilla", "variance", seeds=(0,))

    df = _read_results(tmp_path)
    assert len(df) == 2
    legacy = df[df["loss_type"] == "cvar"].iloc[0]
    assert pd.isna(legacy["seed"])


def test_autocallable_training_simulates_finely_between_quarterly_rebalances():
    """Guard for the 2026-10-01 bug: the note rebalances quarterly, but one
    simulation step per quarter understated its first-quarter vol by more than
    half on real data."""
    cfg = run_module._train_cfg("autocall")
    assert cfg.n_steps == len(run_module.AUTOCALL_SPEC.obs_times)
    assert cfg.n_steps * cfg.sim_substeps >= 64


def test_every_instrument_trains_under_a_position_limit():
    for instrument in run_module.INSTRUMENTS:
        assert run_module._train_cfg(instrument).hedge_limit is not None
