import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

import eqdrisk.risk.var as var_module
import eqdrisk.stress.replay as replay_module
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
from eqdrisk.risk.proxy import ProxyFit
from eqdrisk.stress.historical_scenarios import HistoricalEpisode
from eqdrisk.stress.replay import run_historical_replay
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

ASOF = dt.date(2026, 8, 20)
UNDERLYING = "SPX"
SPOT = 100.0
PILLAR_TS = [0.25, 0.5]
PILLAR_EXPIRIES = [ASOF + dt.timedelta(days=int(t * 365)) for t in PILLAR_TS]
EPISODE = HistoricalEpisode("fake_crash", dt.date(2020, 3, 2), dt.date(2020, 3, 16), "synthetic")
N_T, N_K = len(T_GRID), len(K_GRID)


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
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": True,
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
                "n_carried": 0,
                "k_min": None,
                "k_max": None,
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
            {
                "positions": [
                    {"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 100},
                    {
                        "id": "S1",
                        "type": "varswap",
                        "underlying": UNDERLYING,
                        "expiry": PILLAR_EXPIRIES[1].isoformat(),
                        "vega_notional": 100,
                    },
                ]
            }
        )
    )
    return str(path)


def _levels(with_vix: bool = True) -> pd.DataFrame:
    """The episode: SPX -20%, VIX +100%; today SPX at SPOT, VIX at 20."""
    levels = pd.DataFrame(
        {"SPX": [SPOT, 0.8 * SPOT, SPOT], "VIX": [20.0, 40.0, 20.0]},
        index=pd.Index([EPISODE.before, EPISODE.after, ASOF], name="asof_date"),
    )
    if not with_vix:
        levels.loc[[EPISODE.before, EPISODE.after], "VIX"] = np.nan
    return levels


def _mock_proxy(monkeypatch) -> None:
    """Node betas falling with maturity, as fitted on real days."""
    beta = np.tile(np.linspace(0.5, 0.05, N_T)[:, None], (1, N_K))
    fit = ProxyFit(
        underlying=UNDERLYING,
        index="VIX",
        nodes=[],
        coverage=pd.Series(dtype=float),
        changes=pd.DataFrame(),
        index_change=pd.Series(dtype=float),
        alpha=np.array([]),
        beta=np.array([]),
        r2=np.array([]),
    )
    monkeypatch.setattr(fit, "beta_field", lambda: beta)
    monkeypatch.setattr(var_module, "fit_proxies", lambda *a, **k: ({UNDERLYING: fit}, {}))
    monkeypatch.setattr(var_module, "base_iv_grid", lambda *a, **k: np.full((N_T, N_K), 20.0))
    monkeypatch.setattr(replay_module, "EPISODES", [EPISODE])


def _write_market(tmp_path) -> None:
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)


def test_run_historical_replay_runs_both_variants_and_persists(tmp_path, monkeypatch):
    _write_market(tmp_path)
    _mock_proxy(monkeypatch)
    result = run_historical_replay(
        _cfg(tmp_path), ASOF, _write_portfolio_yaml(tmp_path), levels=_levels(), max_workers=1
    )

    assert not result.skipped, result.skipped
    by_variant = {er.variant: er for er in result.episodes}
    assert set(by_variant) == {"proxy", "naive"}
    for er in by_variant.values():
        assert er.moves == pytest.approx({"SPX": -0.2, "VIX": 1.0})
        assert er.shocked_value == pytest.approx(result.base_value + er.pnl)
        # E1: 100 shares, spot -20%.
        assert er.pnl - er.varswap_pnl == pytest.approx(-0.2 * 100 * SPOT, rel=1e-12)
    # Naive doubles 6-month vol; the proxy lifts it by beta x 20 index points.
    assert 0 < by_variant["proxy"].varswap_pnl < by_variant["naive"].varswap_pnl / 3

    stored = pd.read_parquet(tmp_path / "historical_replay")
    assert sorted(stored["variant"]) == ["naive", "proxy"]


def test_run_historical_replay_skips_an_episode_missing_a_factor(tmp_path, monkeypatch):
    _write_market(tmp_path)
    _mock_proxy(monkeypatch)
    result = run_historical_replay(
        _cfg(tmp_path),
        ASOF,
        _write_portfolio_yaml(tmp_path),
        levels=_levels(with_vix=False),
        max_workers=1,
    )
    assert "no VIX price within" in result.skipped["fake_crash"]
    assert not result.episodes


def test_run_historical_replay_reports_skip_when_no_proxy_or_rates(tmp_path):
    result = run_historical_replay(
        _cfg(tmp_path), ASOF, _write_portfolio_yaml(tmp_path), levels=_levels(), max_workers=1
    )
    assert "_all_" in result.skipped
