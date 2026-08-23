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
        ("last_trade_ts", pa.timestamp("ns", tz="America/New_York")),
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

VOL_INDEX_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("index", pa.string()),
        ("value", pa.float64()),
        ("source", pa.string()),
    ]
)
VOL_INDEX_REQUIRED_NOT_NULL = ["asof_date", "index", "value", "source"]

DIVIDEND_SCHEMA = pa.schema(
    [
        ("underlying", pa.string()),
        ("ex_date", pa.date32()),
        ("amount", pa.float64()),
    ]
)
DIVIDEND_REQUIRED_NOT_NULL = ["underlying", "ex_date", "amount"]

DISCOUNT_CURVE_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("tenor", pa.string()),
        ("T", pa.float64()),
        ("rate", pa.float64()),
        ("discount_factor", pa.float64()),
    ]
)
DISCOUNT_CURVE_REQUIRED_NOT_NULL = ["asof_date", "tenor", "T", "rate", "discount_factor"]

FORWARD_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("underlying", pa.string()),
        ("expiry", pa.date32()),
        ("T", pa.float64()),
        ("n_strikes", pa.int64()),
        ("forward", pa.float64()),
        ("discount_factor_implied", pa.float64()),
        ("discount_factor_curve", pa.float64()),
        ("discount_factor_diff_bp", pa.float64()),
        ("r_squared", pa.float64()),
        ("implied_dividend_yield", pa.float64()),
        ("announced_dividend_yield", pa.float64()),
        ("dividend_yield_diff", pa.float64()),
        ("flag_r2", pa.bool_()),
        ("flag_discount_factor_bp", pa.bool_()),
    ]
)
FORWARD_REQUIRED_NOT_NULL = [
    "asof_date",
    "underlying",
    "expiry",
    "T",
    "n_strikes",
    "forward",
    "discount_factor_implied",
    "discount_factor_curve",
    "discount_factor_diff_bp",
    "r_squared",
]

IMPLIED_VOL_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("underlying", pa.string()),
        ("expiry", pa.date32()),
        ("strike", pa.float64()),
        ("cp", pa.string()),
        ("T", pa.float64()),
        ("k", pa.float64()),
        ("iv", pa.float64()),
        ("total_variance", pa.float64()),
        ("vega", pa.float64()),
        ("weight", pa.float64()),
        ("reason", pa.string()),
    ]
)
IMPLIED_VOL_REQUIRED_NOT_NULL = [
    "asof_date",
    "underlying",
    "expiry",
    "strike",
    "cp",
    "T",
    "reason",
]

VOL_SURFACE_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("underlying", pa.string()),
        ("expiry", pa.date32()),
        ("T", pa.float64()),
        ("model", pa.string()),  # "SVI" or "SSVI"
        ("a", pa.float64()),
        ("b", pa.float64()),
        ("rho", pa.float64()),
        ("m", pa.float64()),
        ("sigma", pa.float64()),
        ("eta", pa.float64()),
        ("theta", pa.float64()),
        ("n_points", pa.int64()),
        ("rmse_vol_points", pa.float64()),
        ("max_abs_error_vol_points", pa.float64()),
        ("max_abs_error_k", pa.float64()),
        ("butterfly_violations", pa.int64()),
        ("calendar_violated", pa.bool_()),
    ]
)
VOL_SURFACE_REQUIRED_NOT_NULL = [
    "asof_date",
    "underlying",
    "expiry",
    "T",
    "model",
    "n_points",
    "rmse_vol_points",
    "butterfly_violations",
    "calendar_violated",
]


GREEKS_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("underlying", pa.string()),
        ("expiry", pa.date32()),
        ("strike", pa.float64()),
        ("cp", pa.string()),
        ("T", pa.float64()),
        ("forward", pa.float64()),
        ("spot", pa.float64()),
        ("discount_factor", pa.float64()),
        ("model", pa.string()),  # "SVI" or "SSVI" — which surface produced `sigma`
        ("sigma", pa.float64()),
        ("price", pa.float64()),
        ("delta_spot", pa.float64()),
        ("delta_forward", pa.float64()),
        ("gamma_spot", pa.float64()),
        ("vega", pa.float64()),
        ("theta", pa.float64()),
        ("rho", pa.float64()),
        ("dividend_rho", pa.float64()),
        ("vanna_spot", pa.float64()),
        ("volga", pa.float64()),
        ("delta_sticky_strike", pa.float64()),
        ("delta_sticky_delta", pa.float64()),
        ("delta_sticky_local_vol", pa.float64()),
    ]
)
GREEKS_REQUIRED_NOT_NULL = [
    "asof_date",
    "underlying",
    "expiry",
    "strike",
    "cp",
    "T",
    "forward",
    "spot",
    "discount_factor",
    "model",
    "sigma",
    "price",
]


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
