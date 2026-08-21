import datetime as dt

import pandas as pd
import pytest

from eqdrisk.io.schemas import (
    CHAIN_REQUIRED_NOT_NULL,
    CHAIN_SCHEMA,
    DIVIDEND_REQUIRED_NOT_NULL,
    DIVIDEND_SCHEMA,
    SchemaViolation,
    validate,
)


def _valid_chain_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": [dt.date(2026, 8, 11)] * n,
            "asof_ts": [pd.Timestamp("2026-08-11 16:00", tz="America/New_York")] * n,
            "underlying": ["SPX"] * n,
            "expiry": [dt.date(2026, 9, 19)] * n,
            "strike": [5000.0 + i for i in range(n)],
            "cp": ["C", "P"][:n] or ["C"] * n,
            "bid": [10.0] * n,
            "ask": [10.5] * n,
            "bid_size": pd.array([None] * n, dtype="Int64"),
            "ask_size": pd.array([None] * n, dtype="Int64"),
            "volume": [100] * n,
            "open_interest": [1000] * n,
            "underlying_px": [5050.0] * n,
            "last_trade_ts": [pd.Timestamp("2026-08-11 16:00", tz="America/New_York")] * n,
            "source": ["test"] * n,
        }
    )


def test_valid_chain_passes():
    table = validate(_valid_chain_df(), CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    assert table.num_rows == 2


def test_missing_column_raises():
    df = _valid_chain_df().drop(columns=["strike"])
    with pytest.raises(SchemaViolation, match="missing required columns"):
        validate(df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)


def test_null_identity_column_raises():
    df = _valid_chain_df()
    df.loc[0, "underlying"] = None
    with pytest.raises(SchemaViolation, match="required"):
        validate(df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)


def test_bad_type_raises():
    df = _valid_chain_df()
    df["strike"] = ["not-a-number"] * len(df)
    with pytest.raises(SchemaViolation, match="type mismatch"):
        validate(df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)


def test_bid_ask_nulls_allowed():
    """bid/ask are market observations, not identity columns — nulls are a Step 3 quality
    concern (reason codes), not a Step 1 structural violation."""
    df = _valid_chain_df()
    df.loc[0, "bid"] = None
    table = validate(df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    assert table.num_rows == 2


def test_dividend_schema_valid():
    df = pd.DataFrame(
        {
            "underlying": ["AAPL", "AAPL"],
            "ex_date": [dt.date(2026, 5, 11), dt.date(2026, 8, 10)],
            "amount": [0.27, 0.27],
        }
    )
    table = validate(df, DIVIDEND_SCHEMA, DIVIDEND_REQUIRED_NOT_NULL)
    assert table.num_rows == 2


def test_dividend_schema_rejects_null_amount():
    df = pd.DataFrame(
        {
            "underlying": ["AAPL"],
            "ex_date": [dt.date(2026, 5, 11)],
            "amount": [None],
        }
    )
    with pytest.raises(SchemaViolation):
        validate(df, DIVIDEND_SCHEMA, DIVIDEND_REQUIRED_NOT_NULL)
