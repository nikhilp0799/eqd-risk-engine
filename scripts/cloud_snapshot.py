"""The daily snapshot from a GitHub Actions runner, as a test
(planning/sleep_safe_snapshot_plan.md, `.github/workflows/cloud_snapshot.yml`).

On 2026-11-10 the SPX day was lost because the Mac was asleep at the close. A
cloud runner does not sleep; whether Yahoo answers one is the open question,
so this runs the daily run's own `run_snapshot` into `--out` (not the repo's
data/) and prints, per name, what a usable chain needs: quotes, quotes with a
bid, and quotes traded inside the staleness window. Exits 0 either way, so the
workflow still uploads what it got.

    python scripts/cloud_snapshot.py --out cloud_snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from eqdrisk.config import BaseConfig, Paths
from eqdrisk.io.snapshot import NY_TZ, run_snapshot
from eqdrisk.marketdata.calendar import is_trading_day
from eqdrisk.marketdata.quality import DEFAULT_STALE_MINUTES, staleness_reference_ts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cloud_snapshot")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    out = Path(args.out)
    base = BaseConfig.from_yaml(args.config)
    cfg = base.model_copy(
        update={"paths": Paths(raw=str(out / "raw"), curated=str(out / "curated"))}
    )
    now = pd.Timestamp.now(tz=NY_TZ)
    asof = now.date()
    print(f"cloud snapshot at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if not is_trading_day(asof):
        print(f"{asof} is not a trading day: chains fetched anyway, as a reachability test")

    result = run_snapshot(cfg, asof)
    print(result.qc.render())

    reference = staleness_reference_ts(now, asof, cfg.canonical_snap_time)
    print(f"\nusable quotes (traded within {DEFAULT_STALE_MINUTES} min of {reference:%H:%M} ET):")
    for underlying in cfg.universe.index + cfg.universe.single_names:
        print("  " + _usable(out / "raw", asof, underlying, reference))


def _usable(raw_root: Path, asof: dt.date, underlying: str, reference: pd.Timestamp) -> str:
    path = raw_root / "chains" / f"asof_date={asof}" / f"underlying={underlying}" / "part-0.parquet"
    if not path.exists():
        return f"{underlying}: no chain"
    df = pd.read_parquet(path)
    traded = pd.to_datetime(df["last_trade_ts"]).dt.tz_convert(NY_TZ)
    fresh = (reference - traded).dt.total_seconds() / 60.0 <= DEFAULT_STALE_MINUTES
    latest = traded.max()
    return (
        f"{underlying}: {len(df)} quotes, {int((df['bid'] > 0).sum())} with a bid, "
        f"{int(fresh.sum())} fresh, latest trade "
        f"{latest.strftime('%H:%M') if pd.notna(latest) else 'none'} ET"
    )


if __name__ == "__main__":
    main()
