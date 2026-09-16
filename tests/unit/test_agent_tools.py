import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from eqdrisk.agent.tools import (
    MODEL_DOC_SECTIONS,
    SPOT_SHOCK_BOUNDS,
    execute_tool_call,
    read_model_doc_section,
    what_if_reprice,
)
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

UNDERLYING = "TEST"
DAY0 = dt.date(2026, 8, 20)
EXPIRY = dt.date(2027, 2, 20)  # ~0.5y out — matches PILLAR below
SPOT = 100.0


def _write_curve(tmp_path, asof, rate=3.0):
    df = pd.DataFrame(
        {
            "asof_date": [asof, asof],
            "tenor": ["3M", "1Y"],
            "rate": [rate, rate],
            "source": ["t", "t"],
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


def _write_forwards(tmp_path, asof, spot, r=3.0, q=0.01):
    r_frac = r / 100.0
    T = 0.5
    forward = spot * np.exp((r_frac - q) * T)
    df = pd.DataFrame(
        [
            {
                "asof_date": asof,
                "underlying": UNDERLYING,
                "expiry": EXPIRY,
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
        ]
    )
    table = validate(df, FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "forwards", ["asof_date", "underlying"])


def _write_surface(tmp_path, asof, rho=-0.3):
    T = 0.5
    b, m, sigma = 0.10 * np.sqrt(T), 0.0, 0.10
    offset = b * sigma * np.sqrt(1 - rho**2)
    a = 0.20**2 * T - offset
    df = pd.DataFrame(
        [
            {
                "asof_date": asof,
                "underlying": UNDERLYING,
                "expiry": EXPIRY,
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
        ]
    )
    table = validate(df, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "vol_surface", ["asof_date", "underlying"])


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=DAY0,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _write_portfolio(tmp_path, positions) -> str:
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.dump({"positions": positions}))
    return str(path)


def _seed_market(tmp_path):
    _write_curve(tmp_path, DAY0)
    _write_underlying(tmp_path, DAY0, SPOT)
    _write_forwards(tmp_path, DAY0, SPOT)
    _write_surface(tmp_path, DAY0)


# --- what_if_reprice ---------------------------------------------------


def test_what_if_reprice_equity_matches_exact_expected_delta(tmp_path):
    """Equity is priced as qty * spot exactly (Step 5's own design) — a real,
    hand-computable check that the tool's shock and reprice machinery is
    wired correctly, not just "returns something"."""
    _seed_market(tmp_path)
    portfolio_path = _write_portfolio(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 10}]
    )
    cfg = _cfg(tmp_path)

    out = what_if_reprice(cfg, portfolio_path, DAY0, "E1", spot_shock_pct=0.10, vol_shock_pct=0.0)

    assert "error" not in out
    assert out["base_price"] == pytest.approx(1000.0)
    assert out["shocked_price"] == pytest.approx(1100.0)
    assert out["price_delta"] == pytest.approx(100.0)
    assert out["shocked_delta"] == pytest.approx(10.0)


def test_what_if_reprice_unknown_position_returns_error_not_exception(tmp_path):
    _seed_market(tmp_path)
    portfolio_path = _write_portfolio(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 1}]
    )
    cfg = _cfg(tmp_path)

    out = what_if_reprice(cfg, portfolio_path, DAY0, "NOPE", spot_shock_pct=0.0, vol_shock_pct=0.0)
    assert "error" in out


def test_what_if_reprice_clamps_extreme_shocks_to_documented_bounds(tmp_path):
    _seed_market(tmp_path)
    portfolio_path = _write_portfolio(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 1}]
    )
    cfg = _cfg(tmp_path)

    out = what_if_reprice(cfg, portfolio_path, DAY0, "E1", spot_shock_pct=10.0, vol_shock_pct=0.0)
    assert out["spot_shock_pct"] == SPOT_SHOCK_BOUNDS[1]
    assert out["shocked_price"] == pytest.approx(100.0 * (1 + SPOT_SHOCK_BOUNDS[1]))


def test_what_if_reprice_no_market_state_returns_error(tmp_path):
    portfolio_path = _write_portfolio(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 1}]
    )
    cfg = _cfg(tmp_path)

    out = what_if_reprice(cfg, portfolio_path, DAY0, "E1", spot_shock_pct=0.0, vol_shock_pct=0.0)
    assert "error" in out


# --- read_model_doc_section --------------------------------------------


def test_read_model_doc_section_returns_real_repo_section():
    from pathlib import Path

    import eqdrisk

    project_root = Path(list(eqdrisk.__path__)[0]).parent.parent
    text = read_model_doc_section(project_root, 4)
    assert "vega" in text.lower() or "limitation" in text.lower()


def test_read_model_doc_section_unknown_section_number(tmp_path):
    assert "does not exist" in read_model_doc_section(tmp_path, 99)


def test_read_model_doc_section_missing_file(tmp_path):
    assert "not found" in read_model_doc_section(tmp_path, 4)


def test_read_model_doc_section_truncates_long_section(tmp_path):
    (tmp_path / "docs").mkdir()
    body = "x" * 5000
    (tmp_path / "docs" / "model_documentation.md").write_text(
        f"## 4. Assumptions and limitations\n{body}\n## 5. Data\nmore"
    )
    text = read_model_doc_section(tmp_path, 4)
    assert len(text) <= 3000


def test_model_doc_sections_covers_1_through_9():
    assert set(MODEL_DOC_SECTIONS.keys()) == set(range(1, 10))


# --- execute_tool_call dispatch -----------------------------------------


def test_execute_tool_call_dispatches_what_if_reprice(tmp_path):
    _seed_market(tmp_path)
    portfolio_path = _write_portfolio(
        tmp_path, [{"id": "E1", "type": "equity", "underlying": UNDERLYING, "qty": 1}]
    )
    cfg = _cfg(tmp_path)

    out = execute_tool_call(
        "what_if_reprice",
        {"position_id": "E1", "spot_shock_pct": 0.0, "vol_shock_pct": 0.0},
        cfg,
        portfolio_path,
        DAY0,
        tmp_path,
    )
    assert out["price_delta"] == 0.0


def test_execute_tool_call_unknown_tool_name_returns_error(tmp_path):
    out = execute_tool_call("does_not_exist", {}, _cfg(tmp_path), "x.yaml", DAY0, tmp_path)
    assert "error" in out
    assert "unknown tool" in out["error"]
