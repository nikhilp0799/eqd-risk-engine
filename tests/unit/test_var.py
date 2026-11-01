import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

import eqdrisk.risk.var as var_module
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
from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.mark import PositionMark
from eqdrisk.risk.proxy import ProxyFit
from eqdrisk.risk.var import (
    LIQUIDITY_HORIZONS,
    VolProxy,
    build_shocks,
    full_pnl,
    liquidity_horizon_es,
    liquidity_horizon_pnl,
    run_var,
    scenario_moves,
    tail_count,
    taylor_pnl,
    var_es,
)
from eqdrisk.stress.shock import MarketShock
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

N_T, N_K = len(T_GRID), len(K_GRID)


def _mark(pid: str, type_: str, **kw) -> PositionMark:
    base = {"underlying": "SPX", "expiry": dt.date(2027, 1, 1), "T": 0.5, "price": 0.0}
    return PositionMark(position_id=pid, type=type_, **{**base, **kw})


def _proxy(beta: float = 0.5, index_today: float = 20.0, index: str = "VIX") -> VolProxy:
    return VolProxy(index, index_today, np.full((N_T, N_K), 20.0), np.full((N_T, N_K), beta))


# --- risk measures ----------------------------------------------------------


def test_var_is_the_kth_worst_and_es_the_mean_of_the_k_worst():
    pnl = np.arange(1, 501, dtype=float) - 250.0  # -249 .. 250
    assert tail_count(500, 0.99) == 5
    assert tail_count(250, 0.99) == 2
    assert tail_count(500, 0.975) == 12
    var, es = var_es(np.random.default_rng(0).permutation(pnl), 0.99)
    assert var == pytest.approx(245.0)
    assert es == pytest.approx(np.mean([249, 248, 247, 246, 245]))


def test_liquidity_horizon_es_follows_the_readme_formula():
    assert LIQUIDITY_HORIZONS == (10, 20, 60)
    es = liquidity_horizon_es([100.0, 30.0, 20.0])
    assert es == pytest.approx(np.sqrt(100**2 + 30**2 * 1.0 + 20**2 * 4.0))


def test_liquidity_horizon_pnl_nests_the_factor_sets():
    marks = [_mark("V", "vanilla"), _mark("A", "autocall"), _mark("B", "barrier")]
    pnl = {
        "all": np.array([[1.0, 2.0, 3.0]]),
        "vol": np.array([[10.0, 20.0, 30.0]]),
        "single_name_vol": np.array([[100.0, 200.0, 300.0]]),
    }
    lh10, lh20, lh60 = liquidity_horizon_pnl(pnl, marks)
    assert lh10[0] == 6.0
    assert lh20[0] == 100.0 + 20.0 + 30.0  # single-name vol, plus all vol for exotics
    assert lh60[0] == 50.0


# --- scenarios ----------------------------------------------------------------


def test_scenario_moves_are_relative_overlapping_and_drop_gaps():
    idx = pd.Index([dt.date(2026, 1, d) for d in range(1, 13)])
    levels = pd.DataFrame(
        {"SPX": np.arange(100.0, 112.0), "VIX": np.linspace(10.0, 21.0, 12)}, index=idx
    )
    levels.loc[idx[5], "VIX"] = np.nan  # a FRED gap: that day drops out
    one = scenario_moves(levels, ["SPX", "VIX"], 1, window=100)
    assert idx[5] not in one.index
    assert len(one) == 10
    assert one.loc[idx[6], "SPX"] == pytest.approx(106.0 / 104.0 - 1)  # across the gap
    ten = scenario_moves(levels, ["SPX", "VIX"], 3, window=2)
    assert list(ten.index) == list(idx[-2:])
    assert ten.loc[idx[-1], "SPX"] == pytest.approx(111.0 / 108.0 - 1)


def test_build_shocks_moves_the_requested_factors_only():
    move = pd.Series({"SPX": -0.02, "AAPL": 0.01, "VIX": 0.10, "VXAPL": 0.05})
    proxies = {"SPX": _proxy(0.5, 20.0, "VIX"), "AAPL": _proxy(0.4, 30.0, "VXAPL")}
    und, singles = ["AAPL", "SPX"], ["AAPL"]

    full = build_shocks(move, proxies, und, singles, "all")
    assert full["SPX"].spot_shock_pct == -0.02
    # Index return rescaled to today's level: 10% of 20 = 2 points, times beta 0.5.
    assert full["SPX"].node_vol.dsigma_at(0.0, 0.5) == pytest.approx(1.0)
    assert full["AAPL"].node_vol.dsigma_at(0.0, 0.5) == pytest.approx(0.4 * 0.05 * 30.0)

    spot = build_shocks(move, proxies, und, singles, "spot")
    assert spot["SPX"].node_vol is None and spot["SPX"].spot_shock_pct == -0.02

    vol = build_shocks(move, proxies, und, singles, "vol")
    assert vol["SPX"].spot_shock_pct == 0.0 and vol["SPX"].node_vol is not None

    single = build_shocks(move, proxies, und, singles, "single_name_vol")
    assert single["SPX"].node_vol is None and single["AAPL"].node_vol is not None
    assert single["AAPL"].spot_shock_pct == 0.0


# --- the two P&L methods ------------------------------------------------------


def test_full_pnl_revalues_the_varswap_against_its_fair_strike():
    marks = [_mark("V", "vanilla"), _mark("S", "varswap", vega=1000.0)]
    base = np.array([[10.0, 0.0], [np.nan, 20.0]])
    scen = np.array([[[12.0, 0.0], [np.nan, 22.0]]])
    pnl = full_pnl(base, scen, marks)
    assert pnl[0, 0] == pytest.approx(2.0)
    assert pnl[0, 1] == pytest.approx(1000.0 * (22.0**2 - 20.0**2) / 40.0)  # 2.1 points


def test_taylor_pnl_uses_sticky_moneyness_greeks_and_the_vol_move_at_vol_times():
    proxy = _proxy(0.5, 20.0)
    shock = MarketShock(spot_shock_pct=0.01, node_vol=proxy.node_vol(0.1))  # +1 vol point
    van = _mark(
        "V",
        "vanilla",
        delta=9.0,
        gamma=9.0,
        delta_sm=2.0,
        gamma_sm=0.1,
        vega=50.0,
        vanna=3.0,
        volga=40.0,
    )
    eq = _mark("E", "equity", delta=-5.0)
    vs = _mark("S", "varswap", vega=1000.0, fair_strike=20.0)
    done = _mark("D", "vanilla", delta=100.0, status=lifecycle.SETTLED)
    pnl = taylor_pnl([van, eq, vs, done], {"SPX": shock}, {"SPX": 100.0})
    d_s, d_vol = 1.0, 0.01
    assert pnl[0] == pytest.approx(
        2.0 * d_s + 0.5 * 0.1 * d_s**2 + 50.0 * d_vol + 3.0 * d_s * d_vol + 0.5 * 40.0 * d_vol**2
    )
    assert pnl[1] == pytest.approx(-5.0)
    assert pnl[2] == pytest.approx(1000.0 * (1.0 + 1.0 / 40.0))
    assert pnl[3] == 0.0


# --- end to end on a synthetic store ------------------------------------------

ASOF = dt.date(2026, 8, 20)
SPOT = 100.0
PILLARS = [(0.25, ASOF + dt.timedelta(days=91)), (0.5, ASOF + dt.timedelta(days=182))]


def _write_market(root) -> None:
    curve = pd.DataFrame(
        {"asof_date": [ASOF] * 2, "tenor": ["3M", "1Y"], "rate": [3.0, 3.0], "source": ["t"] * 2}
    )
    store.write_partitioned(
        validate(curve, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL), root / "curves", ["asof_date"]
    )
    und = pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": ["SPX"],
            "open": [SPOT],
            "high": [SPOT],
            "low": [SPOT],
            "close": [SPOT],
            "volume": [0],
        }
    )
    store.write_partitioned(
        validate(und, UNDERLYING_SCHEMA, UNDERLYING_REQUIRED_NOT_NULL),
        root / "underlyings",
        ["asof_date"],
    )
    fwd, surf = [], []
    for T, expiry in PILLARS:
        fwd.append(
            {
                "asof_date": ASOF,
                "underlying": "SPX",
                "expiry": expiry,
                "T": T,
                "n_strikes": 10,
                "forward": SPOT * np.exp(0.02 * T),
                "discount_factor_implied": np.exp(-0.03 * T),
                "discount_factor_curve": np.exp(-0.03 * T),
                "discount_factor_diff_bp": 0.0,
                "r_squared": 0.9999,
                "implied_dividend_yield": 0.01,
                "announced_dividend_yield": None,
                "dividend_yield_diff": None,
                "flag_r2": False,
                "flag_discount_factor_bp": False,
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": True,
            }
        )
        b, rho, sigma = 0.1 * np.sqrt(T), -0.3, 0.1
        surf.append(
            {
                "asof_date": ASOF,
                "underlying": "SPX",
                "expiry": expiry,
                "T": T,
                "model": "SVI",
                "a": 0.04 * T - b * sigma * np.sqrt(1 - rho**2),
                "b": b,
                "rho": rho,
                "m": 0.0,
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
    store.write_partitioned(
        validate(pd.DataFrame(fwd), FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL),
        root / "forwards",
        ["asof_date", "underlying"],
    )
    store.write_partitioned(
        validate(pd.DataFrame(surf), VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL),
        root / "vol_surface",
        ["asof_date", "underlying"],
    )


def test_run_var_end_to_end(tmp_path, monkeypatch):
    _write_market(tmp_path)
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(
        yaml.dump(
            {
                "positions": [
                    {"id": "E1", "type": "equity", "underlying": "SPX", "qty": -10},
                    {
                        "id": "O1",
                        "type": "vanilla",
                        "underlying": "SPX",
                        "cp": "P",
                        "strike": 95.0,
                        "expiry": PILLARS[1][1].isoformat(),
                        "qty": 50,
                    },
                    {
                        "id": "S1",
                        "type": "varswap",
                        "underlying": "SPX",
                        "expiry": PILLARS[1][1].isoformat(),
                        "vega_notional": 100,
                    },
                ]
            }
        )
    )
    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=["SPX"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )
    days = pd.bdate_range(end=ASOF, periods=80).date
    rng = np.random.default_rng(1)
    levels = pd.DataFrame(
        {
            "SPX": SPOT * np.exp(np.cumsum(rng.normal(0, 0.01, 80)) - 0.0),
            "VIX": 20.0 * np.exp(np.cumsum(rng.normal(0, 0.05, 80))),
        },
        index=pd.Index(days, name="asof_date"),
    )
    levels.loc[ASOF, "SPX"] = SPOT
    fit = ProxyFit(
        underlying="SPX",
        index="VIX",
        nodes=[(t, k) for t in T_GRID for k in K_GRID],
        coverage=pd.Series(1.0, index=pd.MultiIndex.from_product([list(T_GRID), K_GRID])),
        changes=pd.DataFrame(),
        index_change=pd.Series(dtype=float),
        alpha=np.zeros(N_T * N_K),
        beta=np.full(N_T * N_K, 0.5),
        r2=np.full(N_T * N_K, 0.8),
    )
    monkeypatch.setattr(var_module, "WINDOWS", (20, 30, 40))
    monkeypatch.setattr(var_module, "PRIMARY_WINDOW", 30)
    monkeypatch.setattr(var_module, "fit_proxies", lambda *a, **k: ({"SPX": fit}, {}))
    monkeypatch.setattr(var_module, "proxy_error", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(var_module, "base_iv_grid", lambda *a, **k: np.full((N_T, N_K), 20.0))

    result = run_var(cfg, ASOF, str(portfolio), full=True, levels=levels, max_workers=1)
    assert not result.skipped, result.skipped

    sc = result.scenarios
    one = sc[(sc["horizon_days"] == 1) & (sc["factors"] == "all")]
    eq = one[one["position_id"] == "E1"].set_index("scenario_date")
    moves = scenario_moves(levels, ["SPX", "VIX"], 1, 40)
    # Equity: full and Taylor are both exactly qty x spot x return.
    np.testing.assert_allclose(eq["pnl_full"], -10 * SPOT * moves["SPX"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(eq["pnl_taylor"], eq["pnl_full"], rtol=1e-12)
    # The variance swap moves with vol only, never with spot alone.
    spot_only = sc[(sc["factors"] == "spot") & (sc["position_id"] == "S1")]
    assert np.abs(spot_only["pnl_full"]).max() < 1e-6

    book = one.groupby("scenario_date")["pnl_full"].sum().to_numpy()
    assert result.value("VaR", "full") == pytest.approx(var_es(book[-30:], 0.99)[0])
    assert result.value("VaR", "full", window_days=40) == pytest.approx(var_es(book, 0.99)[0])
    ten = result.value("VaR", "sqrt10_full", 10)
    assert ten == pytest.approx(np.sqrt(10) * result.value("VaR", "full"))
    assert result.summary["measure"].eq("ES_LH").sum() == 1

    stored = store.query("SELECT * FROM s", views={"s": str(tmp_path / "var_summary")}).to_pandas()
    assert len(stored) == len(result.summary)
    assert (tmp_path / "var_scenarios").exists() and (tmp_path / "proxy_betas").exists()
    assert "1-day 99%" in result.render()


def test_run_var_refuses_when_a_name_has_no_vol_proxy(tmp_path, monkeypatch):
    _write_market(tmp_path)
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(
        yaml.dump({"positions": [{"id": "E1", "type": "equity", "underlying": "SPX", "qty": 1}]})
    )
    cfg = BaseConfig(
        run_date=ASOF,
        universe=Universe(index=["SPX"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )
    levels = pd.DataFrame({"SPX": [SPOT], "VIX": [20.0]}, index=pd.Index([ASOF]))
    monkeypatch.setattr(var_module, "fit_proxies", lambda *a, **k: ({}, {"SPX": "too short"}))
    result = run_var(cfg, ASOF, str(portfolio), levels=levels, max_workers=1)
    assert "SPX" in result.skipped["_all_"]
    assert result.summary.empty
