"""Partitioned Parquet read/write, queried through DuckDB."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds


def write_partitioned(table: pa.Table, base_path: str | Path, partition_cols: list[str]) -> None:
    """Write `table` under `base_path`, hive-partitioned by `partition_cols`.

    Idempotent: re-running for the same partition values overwrites that
    partition's file(s) rather than appending duplicates.
    """
    partitioning = ds.partitioning(
        pa.schema([table.schema.field(c) for c in partition_cols]), flavor="hive"
    )
    ds.write_dataset(
        table,
        base_dir=str(base_path),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior="delete_matching",
    )


def query(sql: str, views: dict[str, str] | None = None) -> pa.Table:
    """Run `sql` against DuckDB, exposing each `views` entry (name -> parquet root) as a view."""
    con = duckdb.connect()
    if views:
        for name, path in views.items():
            glob = f"{path}/**/*.parquet"
            con.execute(
                f"CREATE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
            )
    return con.execute(sql).to_arrow_table()


def latest_available_date(root: str | Path, asof: dt.date) -> dt.date | None:
    """Latest `asof_date=YYYY-MM-DD` hive partition <= `asof` under `root`.

    Reads partition directory names directly rather than a DuckDB aggregate over
    the parquet glob — the latter hit a DuckDB optimizer internal exception on
    small single-partition datasets (see Step 2 planning notes). Also sidesteps
    FRED-style 1-2 day publication lag: rates/vol-index tables often don't have a
    row for the exact `asof` date yet, so callers need "most recent on or before,"
    not an exact match.
    """
    root = Path(root)
    if not root.exists():
        return None
    dates = []
    for p in root.glob("asof_date=*"):
        if not p.is_dir():
            continue
        try:
            d = dt.date.fromisoformat(p.name.split("=", 1)[1])
        except ValueError:
            continue
        if d <= asof:
            dates.append(d)
    return max(dates) if dates else None
