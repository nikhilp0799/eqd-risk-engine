"""Carry and smile-shape revaluation lines for MC exotics in the P&L explain
(planning/p008_alert_days_plan.md)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from test_pnl_explain import (
    DAY0,
    DAY1,
    PILLAR_EXPIRIES,
    SPOT_DAY0,
    SPOT_DAY1,
    UNDERLYING,
    _cfg,
    _write_both_days,
    _write_portfolio_yaml,
)

import eqdrisk.pricing.pnl_explain as pe
from eqdrisk.io import store
from eqdrisk.marketdata.forward import ForwardCurve
from eqdrisk.portfolio.mark import MCSettings, load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.vol.local_vol import local_variance_at, slice_total_variance

K = np.linspace(-0.4, 0.4, 17)
BARRIER = {
    "id": "B1",
    "type": "barrier",
    "underlying": UNDERLYING,
    "sub": "down_and_in_put",
    "strike": 100.0,
    "barrier": 80.0,
    "expiry": PILLAR_EXPIRIES[1].isoformat(),
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


def test_carry_held_forward_moves_every_forward_with_spot_alone():
    fc = ForwardCurve(np.array([0.25, 0.5]), np.log([101.0, 102.5]), long_end_carry=0.03)
    held = pe._carry_held_forward(fc, 100.0, 95.0)
    for T in (0.0, 0.25, 0.4, 0.5, 2.0):  # before, on, between and beyond the pillars
        assert held.forward(T) / 95.0 == pytest.approx(fc.forward(T) / 100.0, rel=1e-12)


def test_ssvi_slice_as_raw_svi_is_the_same_smile():
    row = pd.Series({"model": "SSVI", "T": 0.5, "theta": 0.03, "rho": -0.6, "eta": 1.1})
    for col in ("a", "b", "m", "sigma", "k_min", "k_max"):
        row[col] = np.nan
    raw = pe._ssvi_as_raw_svi(row)
    assert raw["model"] == "SVI" and np.isnan(raw["k_min"])
    np.testing.assert_allclose(
        slice_total_variance(raw, K), slice_total_variance(row, K), rtol=1e-12
    )


def _surfaces(tmp_path):
    _write_both_days(tmp_path, vol_bump_day1=0.02)
    book = Portfolio.model_validate({"positions": [BARRIER]})
    s0 = load_market_state(_cfg(tmp_path), DAY0, book)
    s1 = load_market_state(_cfg(tmp_path), DAY1, book)
    return s0.surface[UNDERLYING], s1.surface[UNDERLYING]


def test_relevelling_to_the_same_day_changes_nothing(tmp_path):
    surf0, _ = _surfaces(tmp_path)
    same = pe._relevelled_surface(surf0, surf0)
    for (_, a), (_, b) in zip(same.iterrows(), surf0.iterrows(), strict=True):
        np.testing.assert_allclose(slice_total_variance(a, K), slice_total_variance(b, K))


def test_relevelling_keeps_day0_shape_and_takes_day1_atm_level(tmp_path):
    surf0, surf1 = _surfaces(tmp_path)
    level = pe._relevelled_surface(surf0, surf1)
    for (_, lv), (_, d0) in zip(level.iterrows(), surf0.iterrows(), strict=True):
        shift = slice_total_variance(lv, K) - slice_total_variance(d0, K)
        assert np.ptp(shift) < 1e-12  # the same smile, moved in total variance
        T = float(d0["T"])
        atm_level = local_variance_at(level, 0.0, T).w
        assert atm_level == pytest.approx(local_variance_at(surf1, 0.0, T).w, rel=1e-9)


def _cheap(monkeypatch):
    """Every state on the same cheap paths and sub-runs, flags kept per state."""

    def mark(cfg, asof, portfolio, state, shock=None, mc_settings=None):
        settings = dataclasses.replace(
            mc_settings or MCSettings(), n_paths=2_048, barrier_n_steps=16, n_batches=4
        )
        return mark_with_state(cfg, asof, portfolio, state, shock, settings)

    monkeypatch.setattr(pe, "mark_with_state", mark)


def test_explain_books_carry_and_shape_by_revaluation_for_mc_exotics_only(tmp_path, monkeypatch):
    """The fixture's rate rises 3.0% -> 3.2% (the carry changes) and its vol
    bump moves only the SVI level (no shape change)."""
    _cheap(monkeypatch)
    _write_both_days(tmp_path, vol_bump_day1=0.02)
    book = _write_portfolio_yaml(tmp_path, [BARRIER, VANILLA])

    result = pe.run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, book)

    assert not result.skipped
    assert set(result.by_position_carry) == {"B1"} == set(result.by_position_shape)
    assert abs(result.by_position_carry["B1"]) > 0.01
    # A pure level move: the re-levelled surface is day1's, so no shape P&L.
    assert result.by_position_shape["B1"] == pytest.approx(0.0, abs=1e-6)
    steps = {s.step: s for s in result.steps}
    assert steps["spot"].reval_pnl == pytest.approx(result.by_position_carry["B1"])
    assert steps["time"].reval_pnl == 0.0 and steps["rates_divs"].reval_pnl == 0.0
    assert "explained by revaluation: carry" in result.render()

    # The actual side is untouched: the steps still telescope to the full reval.
    portfolio = Portfolio.from_yaml(book)
    cfg = _cfg(tmp_path)
    marks = [
        sum(
            m.price
            for m in pe.mark_with_state(
                cfg, d, portfolio, load_market_state(cfg, d, portfolio)
            ).marks
        )
        for d in (DAY0, DAY1)
    ]
    assert result.total_actual() == pytest.approx(marks[1] - marks[0], abs=1e-6)

    stored = store.query("SELECT * FROM t", views={"t": str(tmp_path / "pnl_explain")}).to_pandas()
    by_step = stored.set_index("step")["reval_pnl"]
    assert by_step["spot"] == pytest.approx(result.by_position_carry["B1"])
    pos = (
        store.query("SELECT * FROM t", views={"t": str(tmp_path / "pnl_explain_by_position")})
        .to_pandas()
        .set_index("position_id")
    )
    assert pos.loc["B1", "carry_pnl"] == pytest.approx(result.by_position_carry["B1"])
    assert pd.isna(pos.loc["V1", "carry_pnl"]) and pd.isna(pos.loc["V1", "shape_pnl"])


def test_the_carry_line_is_what_spot_alone_leaves_out(tmp_path, monkeypatch):
    """With the forward moving exactly with spot (no rate change), the carry
    line is zero to rounding: the mid state is state 3."""
    _cheap(monkeypatch)
    import test_pnl_explain as fx

    monkeypatch.setattr(fx, "RATE_DAY1", fx.RATE_DAY0)
    _write_both_days(tmp_path, vol_bump_day1=0.0)
    book = _write_portfolio_yaml(tmp_path, [BARRIER])
    result = pe.run_pnl_explain(_cfg(tmp_path), DAY0, DAY1, book)
    assert SPOT_DAY1 != SPOT_DAY0
    assert result.by_position_carry["B1"] == pytest.approx(0.0, abs=1e-6)
