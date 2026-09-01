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
from eqdrisk.portfolio.mark import load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.pricing.pnl_explain import run_pnl_explain

UNDERLYING = "TEST"
DAY0 = dt.date(2026, 8, 20)
DAY1 = dt.date(2026, 8, 21)
PILLAR_TS_DAY0 = [0.25, 0.5]
PILLAR_EXPIRIES = [DAY0 + dt.timedelta(days=int(t * 365)) for t in PILLAR_TS_DAY0]

SPOT_DAY0 = 100.0
SPOT_DAY1 = 102.0
RATE_DAY0 = 3.0
RATE_DAY1 = 3.2


def _write_curve(tmp_path, asof, rate):
    df = pd.DataFrame(
        {
            "asof_date": [asof, asof],
            "tenor": ["3M", "1Y"],
            "rate": [rate, rate],
            "source": ["test", "test"],
        }
    )
    table = validate(df, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "curves", ["asof_date"])


def _write_underlying(tmp_path, asof, spot):
    df = pd.DataFrame(
        {
            "asof_date": [asof],
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


def _write_forwards(tmp_path, asof, spot, r, q=0.01):
    r_frac = r / 100.0
    rows = []
    for T, expiry in zip(PILLAR_TS_DAY0, PILLAR_EXPIRIES, strict=True):
        forward = spot * np.exp((r_frac - q) * T)
        rows.append(
            {
                "asof_date": asof,
                "underlying": UNDERLYING,
                "expiry": expiry,
                "T": T,
                "n_strikes": 10,
                "forward": forward,
                "discount_factor_implied": np.exp(-r_frac * T),
                "discount_factor_curve": np.exp(-r_frac * T),
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


def _write_surface(tmp_path, asof, rho=-0.3, vol_bump=0.0):
    rows = []
    for T, expiry in zip(PILLAR_TS_DAY0, PILLAR_EXPIRIES, strict=True):
        b, m, sigma = 0.10 * np.sqrt(T), 0.0, 0.10
        offset = b * sigma * np.sqrt(1 - rho**2)
        a = (0.20 + vol_bump) ** 2 * T - offset
        rows.append(
            {
                "asof_date": asof,
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
        run_date=DAY0,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _write_both_days(tmp_path, vol_bump_day1=0.02):
    _write_curve(tmp_path, DAY0, RATE_DAY0)
    _write_curve(tmp_path, DAY1, RATE_DAY1)
    _write_underlying(tmp_path, DAY0, SPOT_DAY0)
    _write_underlying(tmp_path, DAY1, SPOT_DAY1)
    _write_forwards(tmp_path, DAY0, SPOT_DAY0, RATE_DAY0)
    _write_forwards(tmp_path, DAY1, SPOT_DAY1, RATE_DAY1)
    _write_surface(tmp_path, DAY0, vol_bump=0.0)
    _write_surface(tmp_path, DAY1, vol_bump=vol_bump_day1)


def _write_portfolio_yaml(tmp_path, positions) -> str:
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.dump({"positions": positions}))
    return str(path)


def test_total_actual_pnl_equals_true_end_to_end_full_reval(tmp_path):
    """The 4-step waterfall's actual P&L must telescope to exactly the true,
    direct full-reval P&L (today's real mark minus yesterday's real mark) —
    the one thing that must hold regardless of how the Greeks-based
    'explained' side is designed, since every intermediate state's full
    reval is exact by construction."""
    _write_both_days(tmp_path)
    portfolio_path = _write_portfolio_yaml(
        tmp_path,
        [
            {
                "id": "V1",
                "type": "vanilla",
                "underlying": UNDERLYING,
                "cp": "C",
                "strike": 100.0,
                "expiry": PILLAR_EXPIRIES[0].isoformat(),
                "qty": 10,
            },
            {"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": -3},
        ],
    )
    cfg = _cfg(tmp_path)

    result = run_pnl_explain(cfg, DAY0, DAY1, portfolio_path)
    assert not result.skipped

    portfolio = Portfolio.from_yaml(portfolio_path)
    state0 = load_market_state(cfg, DAY0, portfolio)
    state1 = load_market_state(cfg, DAY1, portfolio)
    true_pnl = (
        mark_with_state(cfg, DAY1, portfolio, state1).total_value()
        - mark_with_state(cfg, DAY0, portfolio, state0).total_value()
    )
    assert result.total_actual() == pytest.approx(true_pnl, rel=1e-9)


def test_equity_position_shows_zero_residual_in_every_step(tmp_path):
    _write_both_days(tmp_path)
    portfolio_path = _write_portfolio_yaml(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": -3}]
    )
    result = run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, portfolio_path)

    assert not result.skipped
    # Equity delta = qty exactly, gamma = 0 exactly, no time/rate/vol dependence at
    # all -> the Greeks-based explanation must be EXACT, not just close.
    for s in result.steps:
        assert s.residual == pytest.approx(0.0, abs=1e-8)
    assert result.by_position_residual["E1"] == pytest.approx(0.0, abs=1e-8)
    # And the only nonzero step should be "spot": qty * (SPOT_DAY1 - SPOT_DAY0).
    spot_step = next(s for s in result.steps if s.step == "spot")
    assert spot_step.actual_pnl == pytest.approx(-3 * (SPOT_DAY1 - SPOT_DAY0))
    for s in result.steps:
        if s.step != "spot":
            assert s.actual_pnl == pytest.approx(0.0, abs=1e-8)


def test_varswap_position_is_zero_everywhere_by_construction(tmp_path):
    _write_both_days(tmp_path)
    portfolio_path = _write_portfolio_yaml(
        tmp_path,
        [
            {
                "id": "VS1",
                "type": "varswap",
                "underlying": UNDERLYING,
                "expiry": PILLAR_EXPIRIES[0].isoformat(),
                "vega_notional": 250000,
            }
        ],
    )
    result = run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, portfolio_path)

    assert not result.skipped
    for s in result.steps:
        assert s.actual_pnl == 0.0
        assert s.explained_pnl == 0.0
        assert s.residual == 0.0
    assert result.by_position_residual["VS1"] == 0.0


def test_vanilla_vega_materially_explains_the_vol_step(tmp_path):
    """With a real (non-tiny) vol bump between the two days, the vol step's
    vega-based explanation should account for most of that step's actual P&L
    — a real, not just internally-consistent, sanity check on the design."""
    _write_both_days(tmp_path, vol_bump_day1=0.03)
    portfolio_path = _write_portfolio_yaml(
        tmp_path,
        [
            {
                "id": "V1",
                "type": "vanilla",
                "underlying": UNDERLYING,
                "cp": "C",
                "strike": 100.0,
                "expiry": PILLAR_EXPIRIES[0].isoformat(),
                "qty": 100,
            }
        ],
    )
    result = run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, portfolio_path)

    assert not result.skipped
    vol_step = next(s for s in result.steps if s.step == "vol")
    assert abs(vol_step.actual_pnl) > 0
    assert abs(vol_step.explained_pnl) > 0
    # residual should be a modest fraction of the actual move, not comparable to it
    assert abs(vol_step.residual) < 0.5 * abs(vol_step.actual_pnl)


def test_run_pnl_explain_reports_skip_when_no_curated_rates_at_all(tmp_path):
    """No curve data for either day at all -> `load_market_state` returns None
    for both -> the global '_all_' skip. (A day1 that merely lacks its OWN spot/
    surface/forward rows, with day0's curve still findable via `latest_available_
    date`'s on-or-before semantics, is a different, per-position skip case —
    covered by Step 7's own tests, not re-tested here.)"""
    portfolio_path = _write_portfolio_yaml(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 1}]
    )
    result = run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, portfolio_path)
    assert "_all_" in result.skipped
