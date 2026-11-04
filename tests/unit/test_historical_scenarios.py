import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.stress.historical_scenarios import (
    HistoricalEpisode,
    _nearest_close,
    episode_moves,
)


def _series(dates: list[dt.date], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.Index(dates, name="asof_date"))


def test_nearest_close_finds_closest_within_tolerance():
    s = _series([dt.date(2020, 3, 10), dt.date(2020, 3, 13), dt.date(2020, 3, 20)], [100, 90, 60])
    assert _nearest_close(s, dt.date(2020, 3, 14), "TEST") == pytest.approx(90.0)


def test_nearest_close_raises_when_nothing_within_tolerance():
    s = _series([dt.date(2020, 1, 1)], [100.0])
    with pytest.raises(ValueError, match="no TEST price within"):
        _nearest_close(s, dt.date(2020, 3, 14), "TEST")


def test_nearest_close_ignores_missing_values_and_raises_on_empty_data():
    s = _series([dt.date(2020, 3, 13), dt.date(2020, 3, 14)], [90.0, np.nan])
    assert _nearest_close(s, dt.date(2020, 3, 14), "TEST") == pytest.approx(90.0)
    with pytest.raises(ValueError, match="no TEST data available"):
        _nearest_close(_series([], []), dt.date(2020, 3, 14), "TEST")


def test_episode_moves_are_each_columns_own_relative_move():
    episode = HistoricalEpisode("test_ep", dt.date(2020, 3, 1), dt.date(2020, 3, 15), "test")
    levels = pd.DataFrame(
        {"SPX": [100.0, 80.0], "VIX": [15.0, 45.0], "VXAPL": [30.0, 60.0]},
        index=pd.Index([dt.date(2020, 3, 2), dt.date(2020, 3, 16)], name="asof_date"),
    )
    moves = episode_moves(episode, levels, ["SPX", "VIX", "VXAPL"])
    assert moves == pytest.approx({"SPX": -0.2, "VIX": 2.0, "VXAPL": 1.0})


def test_episode_moves_refuse_a_factor_with_no_data():
    episode = HistoricalEpisode("test_ep", dt.date(2020, 3, 1), dt.date(2020, 3, 15), "test")
    levels = pd.DataFrame(
        {"SPX": [100.0, 80.0]},
        index=pd.Index([dt.date(2020, 3, 2), dt.date(2020, 3, 16)], name="asof_date"),
    )
    with pytest.raises(ValueError, match="no VXN data"):
        episode_moves(episode, levels, ["SPX", "VXN"])
