"""Quote noise of MC exotics' marks (portfolio/quote_noise.py) and its use in the
P&L explain's alert rule (planning/p008_noise_band_plan.md)."""

import datetime as dt
import functools

import numpy as np
import pandas as pd
import pytest
from test_pnl_explain import (
    DAY0,
    DAY1,
    PILLAR_EXPIRIES,
    PILLAR_TS_DAY0,
    UNDERLYING,
    _cfg,
    _write_both_days,
    _write_portfolio_yaml,
)

import eqdrisk.portfolio.quote_noise as qn
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    CHAIN_REQUIRED_NOT_NULL,
    CHAIN_SCHEMA,
    IMPLIED_VOL_REQUIRED_NOT_NULL,
    IMPLIED_VOL_SCHEMA,
    VOL_SURFACE_REQUIRED_NOT_NULL,
    VOL_SURFACE_SCHEMA,
    validate,
)
from eqdrisk.portfolio.mark import MCSettings, PositionMark, load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.pricing.pnl_explain import _classify, _combine_quote_noise, run_pnl_explain

CHEAP_MC = MCSettings(
    n_paths=2_048, barrier_n_steps=16, greeks=False, exotic_theta=False, n_batches=1
)
BARRIER = {
    "id": "B1",
    "type": "barrier",
    "underlying": UNDERLYING,
    "sub": "down_and_in_put",
    "strike": 100.0,
    "barrier": 80.0,
    "expiry": PILLAR_EXPIRIES[0].isoformat(),
    "qty": 10,
}
VANILLA = {
    "id": "V1",
    "type": "vanilla",
    "underlying": UNDERLYING,
    "cp": "C",
    "strike": 100.0,
    "expiry": PILLAR_EXPIRIES[0].isoformat(),
    "qty": 10,
}


def _quotes(asof: dt.date, half_spread_vol: float, vega: float = 20.0) -> pd.DataFrame:
    """OK quotes read exactly off the fixture's day0 SVI surface, with a bid-ask
    of +- `half_spread_vol` in vol around each mid."""
    rows = []
    rho, sigma = -0.3, 0.10
    for T, expiry in zip(PILLAR_TS_DAY0, PILLAR_EXPIRIES, strict=True):
        b = 0.10 * np.sqrt(T)
        a = 0.20**2 * T - b * sigma * np.sqrt(1 - rho**2)
        for k in np.linspace(-0.3, 0.3, 15):
            w = a + b * (rho * k + np.sqrt(k**2 + sigma**2))
            mid = 10.0
            rows.append(
                {
                    "asof_date": asof,
                    "underlying": UNDERLYING,
                    "expiry": expiry,
                    "strike": round(100.0 * np.exp(k), 6),
                    "cp": "C",
                    "T": T,
                    "k": k,
                    "iv": np.sqrt(w / T),
                    "total_variance": w,
                    "vega": vega,
                    "weight": 1.0,
                    "reason": "OK",
                    "carried_from": None,
                    "forward_source": "fit",
                    "bid": mid - half_spread_vol * vega,
                    "ask": mid + half_spread_vol * vega,
                }
            )
    return pd.DataFrame(rows)


def _write_quotes(tmp_path, asof, half_spread_vol):
    q = _quotes(asof, half_spread_vol)
    iv = validate(q.drop(columns=["bid", "ask"]), IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
    store.write_partitioned(iv, tmp_path / "implied_vols", ["asof_date", "underlying"])
    chain = q[["asof_date", "underlying", "expiry", "strike", "cp", "bid", "ask"]].assign(
        asof_ts=pd.Timestamp(f"{asof} 16:00", tz="America/New_York"),
        source="test",
        bid_size=1,
        ask_size=1,
        volume=1,
        open_interest=1,
        underlying_px=100.0,
        last_trade_ts=pd.NaT,
    )
    table = validate(chain, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "chains", ["asof_date", "underlying"])


def test_half_spread_is_in_vol_and_unusable_quotes_take_the_median():
    q = pd.DataFrame(
        {
            "bid": [1.0, 1.0, np.nan, 1.0, 1.2],
            "ask": [1.2, 1.4, 1.5, 1.6, 1.0],  # last one crossed
            "vega": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    hs = qn.half_spread_vol(q)
    assert hs[:2] == pytest.approx([0.01, 0.02])
    assert hs[3] == pytest.approx(0.03)
    assert hs[2] == pytest.approx(0.02) and hs[4] == pytest.approx(0.02)


def test_perturbation_stays_inside_the_spread_and_is_reproducible():
    q = _quotes(DAY0, 0.01)
    hs = qn.half_spread_vol(q)
    a = qn.perturb_quotes(q, hs, np.random.default_rng(1))
    b = qn.perturb_quotes(q, hs, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)
    moved = (a["iv"] - q["iv"]).abs()
    assert moved.max() <= 0.01 + 1e-12 and moved.max() > 0.005
    assert np.allclose(a["total_variance"], a["iv"] ** 2 * a["T"])
    assert qn.perturb_quotes(q, np.zeros(len(q)), np.random.default_rng(1))["iv"].equals(q["iv"])


def test_refit_recovers_the_surface_and_has_the_stored_columns():
    surf = qn.refit_surface(UNDERLYING, _quotes(DAY0, 0.0), DAY0)
    validate(surf, VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
    assert qn.surface_model(surf) == "SVI"
    assert surf["rmse_vol_points"].max() < 0.05


def _state_and_book(tmp_path):
    _write_both_days(tmp_path)
    book = Portfolio.model_validate({"positions": [BARRIER]})
    return _cfg(tmp_path), book, load_market_state(_cfg(tmp_path), DAY0, book)


def test_zero_spread_gives_exactly_zero_noise(tmp_path, monkeypatch):
    """Every draw refits the same quotes and prices on the same seed: the SD is
    zero to rounding, so Monte Carlo noise cannot leak into the quote-noise band.
    Not exactly zero: on Linux CI one run gave 2e-15 (2026-11-24), last-bit
    differences between refits of identical quotes, against marks in the dollars."""
    monkeypatch.setattr(qn, "QUOTE_NOISE_MC", CHEAP_MC)
    cfg, book, state = _state_and_book(tmp_path)
    rows = qn.measure_underlying(cfg, DAY0, book, state, _quotes(DAY0, 0.0), n_draws=3)
    assert [r.position_id for r in rows] == ["B1"]
    assert rows[0].sd < 1e-9 and rows[0].model_flip_share == 0.0


def test_a_real_spread_gives_positive_noise_that_grows_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(qn, "QUOTE_NOISE_MC", CHEAP_MC)
    cfg, book, state = _state_and_book(tmp_path)
    narrow = qn.measure_underlying(cfg, DAY0, book, state, _quotes(DAY0, 0.002), n_draws=4)
    wide = qn.measure_underlying(cfg, DAY0, book, state, _quotes(DAY0, 0.02), n_draws=4)
    assert 0.0 < narrow[0].sd < wide[0].sd
    assert wide[0].half_spread_vol_points == pytest.approx(2.0)


def test_run_quote_noise_measures_only_mc_exotics_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(qn, "QUOTE_NOISE_MC", CHEAP_MC)
    _write_both_days(tmp_path)
    _write_quotes(tmp_path, DAY0, 0.01)
    book = _write_portfolio_yaml(tmp_path, [BARRIER, VANILLA])
    result = qn.run_quote_noise(_cfg(tmp_path), DAY0, book, n_draws=3)
    assert [r.position_id for r in result.rows] == ["B1"]
    assert "B1 (TEST): mark SD" in result.render()
    stored = qn.load_quote_noise(tmp_path, DAY0)
    assert stored == {"B1": pytest.approx(result.rows[0].sd)} and stored["B1"] > 0
    assert qn.load_quote_noise(tmp_path, DAY1) == {}


def test_band_needs_both_days_and_only_mc_exotics_get_one():
    def mark(pid, kind):
        return PositionMark(
            position_id=pid, type=kind, underlying="X", expiry=DAY1, T=1.0, price=1.0
        )

    marks = {"B": mark("B", "barrier"), "A": mark("A", "autocall"), "V": mark("V", "vanilla")}
    out = _combine_quote_noise(marks, {"B": 3.0, "A": 1.0, "V": 9.0}, {"B": 4.0, "V": 9.0})
    assert out == {"B": pytest.approx(5.0)}


def test_quote_noise_widens_the_band_before_an_alert():
    nav = 1_000_000.0  # 5bp threshold = $500
    assert _classify(600.0, 100.0, nav, "p") == (None, None)  # 600 - 200 < 500
    alert, unclear = _classify(600.0, 100.0, nav, "p", quote_noise=300.0)  # 2 x 316 > 500
    assert alert is None and "Monte Carlo and quote noise are wider" in unclear
    alert, unclear = _classify(2_000.0, 100.0, nav, "p", quote_noise=300.0)
    assert unclear is None and "MC and quote noise" in alert


def test_explain_adds_both_days_quote_noise(tmp_path, monkeypatch):
    import eqdrisk.pricing.pnl_explain as pe

    _write_both_days(tmp_path, vol_bump_day1=0.02)
    for day, sd in ((DAY0, 300.0), (DAY1, 400.0)):
        result = qn.QuoteNoiseResult(
            asof=day, rows=[qn.QuoteNoiseRow("B1", UNDERLYING, sd, 30, 1.0, 0.0)]
        )
        qn._persist(result, tmp_path)
    book = _write_portfolio_yaml(tmp_path, [BARRIER, VANILLA])
    cheap = MCSettings(n_paths=2_048, barrier_n_steps=16, n_batches=4)
    monkeypatch.setattr(
        pe, "mark_with_state", functools.partial(mark_with_state, mc_settings=cheap)
    )

    result = run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, book)

    assert result.by_position_quote_noise == {"B1": pytest.approx(500.0)}
    assert result.total_quote_noise == pytest.approx(500.0)
    assert "quote noise +- 1,000.00" in result.render()
    stored = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "pnl_explain_by_position")}
    ).to_pandas()
    by_id = stored.set_index("position_id")["quote_noise"]
    assert by_id["B1"] == pytest.approx(500.0) and pd.isna(by_id["V1"])
    totals = store.query("SELECT * FROM t", views={"t": str(tmp_path / "pnl_explain")}).to_pandas()
    assert totals["total_quote_noise"].iloc[0] == pytest.approx(500.0)
