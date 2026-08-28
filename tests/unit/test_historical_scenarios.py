import datetime as dt

import pandas as pd
import pytest

import eqdrisk.stress.historical_scenarios as hs
from eqdrisk.stress.historical_scenarios import (
    HistoricalEpisode,
    _nearest_close,
    compute_episode_shocks,
)


def test_nearest_close_finds_closest_within_tolerance():
    df = pd.DataFrame(
        {
            "asof_date": [dt.date(2020, 3, 10), dt.date(2020, 3, 13), dt.date(2020, 3, 20)],
            "close": [100.0, 90.0, 60.0],
        }
    )
    assert _nearest_close(df, dt.date(2020, 3, 14), "TEST") == pytest.approx(90.0)


def test_nearest_close_raises_when_nothing_within_tolerance():
    df = pd.DataFrame({"asof_date": [dt.date(2020, 1, 1)], "close": [100.0]})
    with pytest.raises(ValueError, match="no TEST price within"):
        _nearest_close(df, dt.date(2020, 3, 14), "TEST")


def test_nearest_close_raises_on_empty_data():
    df = pd.DataFrame({"asof_date": [], "close": []})
    with pytest.raises(ValueError, match="no TEST data available"):
        _nearest_close(df, dt.date(2020, 3, 14), "TEST")


def test_compute_episode_shocks_uses_real_pulled_moves(monkeypatch):
    episode = HistoricalEpisode("test_ep", dt.date(2020, 3, 1), dt.date(2020, 3, 15), "test")

    def fake_ohlc(underlyings, start, end):
        return pd.DataFrame(
            {
                "asof_date": [dt.date(2020, 3, 1), dt.date(2020, 3, 15)] * len(underlyings),
                "underlying": [u for u in underlyings for _ in range(2)],
                "close": [100.0, 80.0] * len(underlyings),
            }
        )

    def fake_vix(start, end):
        return pd.DataFrame(
            {
                "asof_date": [dt.date(2020, 3, 1), dt.date(2020, 3, 15)],
                "index": ["VIX", "VIX"],
                "value": [15.0, 45.0],
            }
        )

    monkeypatch.setattr(hs, "fetch_underlying_ohlc", fake_ohlc)
    monkeypatch.setattr(hs, "fetch_vol_indices", fake_vix)

    shocks = compute_episode_shocks(episode, ["SPX", "AAPL"])

    assert set(shocks) == {"SPX", "AAPL"}
    for shock in shocks.values():
        assert shock.spot_shock_pct == pytest.approx(80.0 / 100.0 - 1.0)
        assert shock.vol_shock_pct == pytest.approx(45.0 / 15.0 - 1.0)


def test_compute_episode_shocks_skips_underlyings_with_no_data(monkeypatch):
    episode = HistoricalEpisode("test_ep", dt.date(2020, 3, 1), dt.date(2020, 3, 15), "test")

    def fake_ohlc(underlyings, start, end):
        return pd.DataFrame(
            {
                "asof_date": [dt.date(2020, 3, 1), dt.date(2020, 3, 15)],
                "underlying": ["SPX", "SPX"],
                "close": [100.0, 90.0],
            }
        )

    def fake_vix(start, end):
        return pd.DataFrame(
            {
                "asof_date": [dt.date(2020, 3, 1), dt.date(2020, 3, 15)],
                "index": ["VIX"] * 2,
                "value": [15.0, 20.0],
            }
        )

    monkeypatch.setattr(hs, "fetch_underlying_ohlc", fake_ohlc)
    monkeypatch.setattr(hs, "fetch_vol_indices", fake_vix)

    shocks = compute_episode_shocks(episode, ["SPX", "NOPE"])
    assert set(shocks) == {"SPX"}
