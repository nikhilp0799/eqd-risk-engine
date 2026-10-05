"""Remove stored P&L-explain results for day pairs that lacked complete market
data (planning/decisions.md, 2026-10-04).

Before the completeness gate in `pricing/pnl_explain.run_pnl_explain`, days
whose option-chain ingest failed were still explained on a half-loaded market
and persisted: all-zero pairs, and one -$676K residual against a NAV of 0. This
finds every stored pair where either day fails the same check the pipeline now
applies (`missing_market_data`), and deletes it from the P&L-explain tables and
from the AI-investigation tables built on top of them.

Dry run by default; pass `--apply` to delete. Run from the project root:
    python scripts/prune_incomplete_pnl_explain.py [--apply]
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.pricing.pnl_explain import missing_market_data

# Every table keyed by (day0, asof_date=day1) for one explain run.
PAIR_TABLES = (
    "pnl_explain",
    "pnl_explain_by_position",
    "ai_investigations",
    "ai_flagged_positions",
    "ai_proposed_changes",
    "ai_investigation_trace",
)


def _read(root: Path) -> pd.DataFrame:
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame()
    return store.query("SELECT * FROM t", views={"t": str(root)}).to_pandas()


def invalid_pairs(cfg: BaseConfig, portfolio: Portfolio) -> dict[tuple[dt.date, dt.date], str]:
    pairs = _read(Path(cfg.paths.curated) / "pnl_explain")
    if pairs.empty:
        return {}
    out = {}
    for day0, day1 in pairs[["day0", "asof_date"]].drop_duplicates().itertuples(index=False):
        gaps = {d: missing_market_data(cfg, d, portfolio) for d in (day0, day1)}
        if any(gaps.values()):
            out[(day0, day1)] = "; ".join(
                f"{d} missing {', '.join(m)}" for d, m in gaps.items() if m
            )
    return out


def _prune_table(root: Path, bad: set[tuple[dt.date, dt.date]], apply: bool) -> int:
    """Rewrite each affected `asof_date` partition without the bad pairs, keeping
    the partition's own stored schema (older partitions predate some columns),
    or remove the partition if nothing is left. Returns rows removed."""
    removed = 0
    for day1 in sorted({d1 for _, d1 in bad}):
        partition = root / f"asof_date={day1.isoformat()}"
        if not partition.exists():
            continue
        table = ds.dataset(partition, format="parquet").to_table()
        bad_day0 = pa.array([d0 for d0, d1 in bad if d1 == day1], type=pa.date32())
        keep = table.filter(pc.invert(pc.is_in(table["day0"], value_set=bad_day0)))
        removed += table.num_rows - keep.num_rows
        if not apply or keep.num_rows == table.num_rows:
            continue
        shutil.rmtree(partition)
        if keep.num_rows:
            partition.mkdir(parents=True)
            pq.write_table(keep, partition / "part-0.parquet")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--portfolio", default="configs/portfolio.yaml")
    args = parser.parse_args()

    cfg = BaseConfig.from_yaml(args.config)
    bad = invalid_pairs(cfg, Portfolio.from_yaml(args.portfolio))
    if not bad:
        print("No incomplete P&L-explain pairs stored.")
        return
    print(f"{len(bad)} stored pair(s) with incomplete market data:")
    for (day0, day1), why in sorted(bad.items()):
        print(f"  {day0} -> {day1}: {why}")
    curated = Path(cfg.paths.curated)
    verb = "Removed" if args.apply else "Would remove"
    for table in PAIR_TABLES:
        n = _prune_table(curated / table, set(bad), args.apply)
        print(f"{verb} {n} row(s) from {table}")
    if not args.apply:
        print("Dry run. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
