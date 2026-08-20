"""Schema enforcement for curated tables — fail loudly on structural violations.

Business-level quote quality (zero bids, crossed markets, wide spreads) is
Step 3's job and gets reason codes, not exceptions. What raises here is
structural: a missing column, a type that won't cast, or a null in a column
that must always be populated (an identity column, not a market observation).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

CHAIN_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("asof_ts", pa.timestamp("ns", tz="America/New_York")),
        ("underlying", pa.string()),
        ("expiry", pa.date32()),
        ("strike", pa.float64()),
        ("cp", pa.string()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_size", pa.int64()),
        ("ask_size", pa.int64()),
        ("volume", pa.int64()),
        ("open_interest", pa.int64()),
        ("underlying_px", pa.float64()),
        ("source", pa.string()),
    ]
)
CHAIN_REQUIRED_NOT_NULL = ["asof_date", "asof_ts", "underlying", "expiry", "strike", "cp", "source"]

UNDERLYING_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("underlying", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
    ]
)
UNDERLYING_REQUIRED_NOT_NULL = ["asof_date", "underlying", "close"]

CURVE_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("tenor", pa.string()),
        ("rate", pa.float64()),
        ("source", pa.string()),
    ]
)
CURVE_REQUIRED_NOT_NULL = ["asof_date", "tenor", "rate", "source"]

DIVIDEND_SCHEMA = pa.schema(
    [
        ("underlying", pa.string()),
        ("ex_date", pa.date32()),
        ("amount", pa.float64()),
    ]
)
DIVIDEND_REQUIRED_NOT_NULL = ["underlying", "ex_date", "amount"]


class SchemaViolation(ValueError):
    pass


def validate(df: pd.DataFrame, schema: pa.Schema, required_not_null: list[str]) -> pa.Table:
    """Cast `df` to `schema` and enforce not-null identity columns. Raises SchemaViolation."""
    missing = set(schema.names) - set(df.columns)
    if missing:
        raise SchemaViolation(f"missing required columns: {sorted(missing)}")

    try:
        table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise SchemaViolation(f"type mismatch casting to schema: {exc}") from exc

    for col in required_not_null:
        null_count = table.column(col).null_count
        if null_count:
            raise SchemaViolation(f"column '{col}' has {null_count} nulls but is required")

    return table
