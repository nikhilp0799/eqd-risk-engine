import datetime as dt

import pandas as pd
import pytest


def _make_chain_df(asof_date: dt.date, underlying: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": [asof_date] * n,
            "asof_ts": [pd.Timestamp(asof_date, tz="America/New_York").replace(hour=16)] * n,
            "underlying": [underlying] * n,
            "expiry": [asof_date + dt.timedelta(days=30)] * n,
            "strike": [100.0 + i for i in range(n)],
            "cp": ["C"] * n,
            "bid": [1.0] * n,
            "ask": [1.1] * n,
            "bid_size": pd.array([None] * n, dtype="Int64"),
            "ask_size": pd.array([None] * n, dtype="Int64"),
            "volume": [10] * n,
            "open_interest": [100] * n,
            "underlying_px": [100.0] * n,
            "last_trade_ts": [pd.Timestamp(asof_date, tz="America/New_York").replace(hour=16)] * n,
            "source": ["test"] * n,
        }
    )


@pytest.fixture
def make_chain_df():
    return _make_chain_df
