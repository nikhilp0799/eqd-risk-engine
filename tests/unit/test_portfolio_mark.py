import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

import eqdrisk.portfolio.mark as mark_module
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
from eqdrisk.portfolio.mark import _expiry_bucket, _moneyness_bucket, mark_portfolio
from eqdrisk.pricing.blackscholes import call_price
from eqdrisk.vol.svi import SVIParams

ASOF = dt.date(2026, 8, 20)
UNDERLYING = "TEST"
SPOT = 100.0
PILLAR_TS = [0.25, 0.5]
PILLAR_EXPIRIES = [ASOF + dt.timedelta(days=int(t * 365)) for t in PILLAR_TS]


def test_expiry_bucket_boundaries():
    assert _expiry_bucket(0.1) == "0-3m"
    assert _expiry_bucket(0.25) == "0-3m"
    assert _expiry_bucket(0.4) == "3-6m"
    assert _expiry_bucket(0.9) == "6-12m"
    assert _expiry_bucket(1.5) == "1-2y"
    assert _expiry_bucket(5.0) == "2y+"


def test_moneyness_bucket_boundaries_and_none_is_structured():
    assert _moneyness_bucket(None) == "structured"
    assert _moneyness_bucket(-0.3) == "deep_otm_put"
    assert _moneyness_bucket(-0.1) == "otm_put"
    assert _moneyness_bucket(0.0) == "atm"
    assert _moneyness_bucket(0.1) == "otm_call"
    assert _moneyness_bucket(0.3) == "deep_otm_call"


def _write_curve(tmp_path):
    # Flat 3% at both a short and a long pillar: log-linear interpolation between
    # two EQUAL rates is exactly exp(-r*T) for any T in between (and a single
    # pillar would instead flat-extrapolate the FULL-tenor discount factor to
    # every T, which is not the same thing) — needed so tests can independently
    # recompute an expected discount factor as `exp(-r*T)` and have it actually
    # match what `bootstrap_curve` produces.
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


def _write_forwards(tmp_path, r: float = 0.03, q: float = 0.01):
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


def _write_surface(tmp_path, rho: float = -0.3):
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


def _write_portfolio_yaml(tmp_path, positions: list[dict]) -> str:
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.dump({"positions": positions}))
    return str(path)


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def test_vanilla_and_equity_marks_match_independent_computation(tmp_path):
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)

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

    result = mark_portfolio(_cfg(tmp_path), ASOF, portfolio_path)
    assert not result.skipped

    equity_mark = next(m for m in result.marks if m.position_id == "E1")
    assert equity_mark.price == pytest.approx(-3 * SPOT)
    assert equity_mark.delta == pytest.approx(-3)

    vanilla_mark = next(m for m in result.marks if m.position_id == "V1")
    T = PILLAR_TS[0]
    r, q = 0.03, 0.01
    forward = SPOT * np.exp((r - q) * T)
    discount_factor = np.exp(-r * T)
    # T=0.25 is exactly the first calibrated pillar, so the T-interpolant must
    # reproduce that slice's own SVI params exactly -> independently recompute
    # the same closed-form price at the pillar's own params.
    rho, b, m_, sigma = -0.3, 0.10 * np.sqrt(T), 0.0, 0.10
    a = (0.20**2) * T - b * sigma * np.sqrt(1 - rho**2)
    params = SVIParams(a=a, b=b, rho=rho, m=m_, sigma=sigma)
    iv = float(np.sqrt(params.total_variance(np.log(100.0 / forward)) / T))
    expected_price = 10 * call_price(forward, 100.0, T, iv, discount_factor)
    assert vanilla_mark.price == pytest.approx(expected_price, rel=1e-6)


def test_varswap_mark_has_zero_price_and_vega_equal_to_notional(tmp_path):
    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)

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

    result = mark_portfolio(_cfg(tmp_path), ASOF, portfolio_path)
    assert not result.skipped
    mark = result.marks[0]
    assert mark.price == 0.0
    assert mark.vega == 250000
    assert mark.note is not None and "fair_strike" in mark.note


def test_barrier_and_autocall_positions_price_without_crashing(tmp_path, monkeypatch):
    """Keep MC cost tiny for the test — correctness of the underlying pricers is
    already covered by Steps 6.4/6.5's own extensive test suites; this only
    verifies the portfolio orchestration wires spot/curve/forward/grid through
    correctly and doesn't skip or crash."""
    monkeypatch.setattr(mark_module, "MC_N_PATHS", 2_000)
    monkeypatch.setattr(mark_module, "BARRIER_N_STEPS", 8)
    monkeypatch.setattr(mark_module, "AUTOCALL_N_STEPS_PER_PERIOD", 2)

    _write_curve(tmp_path)
    _write_underlying(tmp_path)
    _write_forwards(tmp_path)
    _write_surface(tmp_path)

    portfolio_path = _write_portfolio_yaml(
        tmp_path,
        [
            {
                "id": "B1",
                "type": "barrier",
                "underlying": UNDERLYING,
                "sub": "down_and_in_put",
                "strike": 105.0,
                "barrier": 85.0,
                "expiry": PILLAR_EXPIRIES[1].isoformat(),
                "qty": -5,
            },
            {
                "id": "A1",
                "type": "autocall",
                "underlying": UNDERLYING,
                "notional": 1_000_000,
                "autocall_barrier": 1.00,
                "coupon_barrier": 0.75,
                "put_barrier": 0.65,
                "coupon": 0.02,
                "obs": "quarterly",
                "expiry": PILLAR_EXPIRIES[1].isoformat(),
            },
        ],
    )

    result = mark_portfolio(_cfg(tmp_path), ASOF, portfolio_path)
    assert not result.skipped
    assert len(result.marks) == 2
    for m in result.marks:
        assert np.isfinite(m.price)
        assert np.isfinite(m.delta)


def test_skips_positions_with_no_market_data_for_their_underlying(tmp_path):
    _write_curve(tmp_path)

    portfolio_path = _write_portfolio_yaml(
        tmp_path,
        [{"id": "E1", "type": "equity", "underlying": "NOPE", "qty": 1}],
    )

    result = mark_portfolio(_cfg(tmp_path), ASOF, portfolio_path)
    assert "E1" in result.skipped
    assert not result.marks
