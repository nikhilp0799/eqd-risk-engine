import datetime as dt
import time

from eqdrisk.io import store
from eqdrisk.io.schemas import CHAIN_REQUIRED_NOT_NULL, CHAIN_SCHEMA, validate


def test_write_partitioned_matches_hive_layout(tmp_path, make_chain_df):
    table = validate(
        make_chain_df(dt.date(2026, 8, 11), "SPX", 2), CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, tmp_path / "chains", ["asof_date", "underlying"])

    expected = tmp_path / "chains" / "asof_date=2026-08-11" / "underlying=SPX"
    assert expected.exists()
    assert list(expected.glob("*.parquet"))


def test_write_partitioned_is_idempotent(tmp_path, make_chain_df):
    base = tmp_path / "chains"
    for _ in range(2):
        table = validate(
            make_chain_df(dt.date(2026, 8, 11), "SPX", 3), CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL
        )
        store.write_partitioned(table, base, ["asof_date", "underlying"])

    result = store.query("SELECT count(*) AS n FROM chains", views={"chains": str(base)})
    assert result.column("n")[0].as_py() == 3


def test_query_over_200_partitions_is_fast(tmp_path, make_chain_df):
    """Synthetic stand-in for the README's '>=200 dates in <2s' target.

    Real 200-day coverage needs a paid historical chain archive (see
    feedback_operating_mode / project notes) — this proves the DuckDB
    read-path meets the performance bar against schema-valid synthetic data.
    """
    base = tmp_path / "chains"
    start = dt.date(2025, 1, 1)
    for i in range(200):
        asof = start + dt.timedelta(days=i)
        table = validate(make_chain_df(asof, "SPX", 50), CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
        store.write_partitioned(table, base, ["asof_date", "underlying"])

    t0 = time.perf_counter()
    result = store.query(
        "SELECT asof_date, count(*) AS n FROM chains GROUP BY asof_date",
        views={"chains": str(base)},
    )
    elapsed = time.perf_counter() - t0

    assert result.num_rows == 200
    assert elapsed < 2.0
