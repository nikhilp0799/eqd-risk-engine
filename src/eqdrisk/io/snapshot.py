"""The daily ingest job: pull -> validate schema -> write raw -> write curated -> emit QC report."""

from __future__ import annotations

import datetime as dt
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from eqdrisk.config import BaseConfig
from eqdrisk.io import sources, store
from eqdrisk.io.schemas import (
    CHAIN_REQUIRED_NOT_NULL,
    CHAIN_SCHEMA,
    CURVE_REQUIRED_NOT_NULL,
    CURVE_SCHEMA,
    DIVIDEND_REQUIRED_NOT_NULL,
    DIVIDEND_SCHEMA,
    UNDERLYING_REQUIRED_NOT_NULL,
    UNDERLYING_SCHEMA,
    VOL_INDEX_REQUIRED_NOT_NULL,
    VOL_INDEX_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import last_n_trading_days

NY_TZ = "America/New_York"


# Three attempts per fetch: the daily ingest crashed on 6 of 26 trading days
# (2026-09-11..30) from transient Yahoo failures (`^SPX` KeyError
# 'currentTradingPeriod') and network timeouts; option chains are live-only, so
# a lost day can never be backfilled.
RETRY_DELAYS_SECONDS: tuple[float, ...] = (5.0, 30.0)
_sleep = time.sleep  # patched out in tests

T = TypeVar("T")


def _with_retries(label: str, fn: Callable[[], T], retries: list[str]) -> T:
    """Call `fn`, retrying after each delay in `RETRY_DELAYS_SECONDS`. Every
    data-source call here is network I/O whose failures (timeouts, Yahoo
    returning a malformed payload as a KeyError, ...) are transient, so any
    exception is retried; the last one propagates to the caller."""
    for attempt, delay in enumerate((*RETRY_DELAYS_SECONDS, None), start=1):
        try:
            return fn()
        except Exception as exc:
            if delay is None:
                raise
            retries.append(f"{label}: attempt {attempt} failed ({_describe(exc)}), retrying")
            _sleep(delay)
    raise AssertionError("unreachable")


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


@dataclass
class QCReport:
    asof: dt.date
    # What failed even after retries (one entry per underlying or data feed):
    # the run carries on without it rather than aborting everything.
    failures: dict[str, str] = field(default_factory=dict)
    retries: list[str] = field(default_factory=list)
    chain_rows: dict[str, int] = field(default_factory=dict)
    null_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    prior_day_row_delta: dict[str, int | None] = field(default_factory=dict)
    rejections: dict[str, dict[str, int]] = field(default_factory=dict)
    snap_alerts: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"QC report — {self.asof}"]
        for underlying, n in self.chain_rows.items():
            lines.append(f"  {underlying}: {n} quotes")
            noteworthy = {k: v for k, v in self.null_rates.get(underlying, {}).items() if v > 0}
            if noteworthy:
                lines.append(f"    null rates: {noteworthy}")
            delta = self.prior_day_row_delta.get(underlying)
            if delta is not None:
                lines.append(f"    vs prior day: {delta:+d} rows")
            rej = self.rejections.get(underlying)
            if rej:
                lines.append(f"    rejected: {rej}")
        if self.retries:
            lines.append("  RETRIES:")
            lines.extend(f"    {r}" for r in self.retries)
        alerts = self.snap_alerts + [
            f"FAILED after retries: {name} ({why})" for name, why in self.failures.items()
        ]
        if alerts:
            lines.append("  ALERTS:")
            lines.extend(f"    {a}" for a in alerts)
        return "\n".join(lines)


@dataclass
class SnapshotResult:
    asof: dt.date
    qc: QCReport
    curated_root: Path


def _prior_day_count(chains_root: Path, underlying: str, asof: dt.date) -> int | None:
    if not chains_root.exists() or not any(chains_root.rglob("*.parquet")):
        return None
    partitioning = ds.partitioning(
        pa.schema([("asof_date", pa.date32()), ("underlying", pa.string())]), flavor="hive"
    )
    dataset = ds.dataset(str(chains_root), format="parquet", partitioning=partitioning)
    filt = (ds.field("underlying") == underlying) & (ds.field("asof_date") < asof)
    table = dataset.to_table(filter=filt, columns=["asof_date"])
    if table.num_rows == 0:
        return None
    dates = table.column("asof_date").to_pandas()
    prior_date = dates.max()
    return int((dates == prior_date).sum())


def _clean_chain(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Row-level, reason-coded rejection for structurally-unusable rows.

    This is distinct from Step 3's market-quality filters (ZERO_BID, WIDE_SPREAD,
    ...): those judge whether a quote is fit to calibrate on. This judges whether
    a row can even be identified as a contract at all — a single malformed row
    here should not fail the whole day's ingestion the way a missing column does.
    """
    rejections: dict[str, int] = {}
    df = raw_df

    null_identity = df["expiry"].isna() | df["strike"].isna() | df["cp"].isna()
    if null_identity.any():
        rejections["NULL_IDENTITY"] = int(null_identity.sum())
        df = df.loc[~null_identity]

    dup = df.duplicated(subset=["expiry", "strike", "cp"], keep="last")
    if dup.any():
        rejections["DUPLICATE_CONTRACT"] = int(dup.sum())
        df = df.loc[~dup]

    return df.reset_index(drop=True), rejections


def _snap_offset_minutes(
    asof_ts: pd.Timestamp, asof: dt.date, canonical_snap_time: dt.time
) -> float:
    canonical = pd.Timestamp.combine(asof, canonical_snap_time).tz_localize(NY_TZ)
    return (asof_ts - canonical).total_seconds() / 60.0


def _snapshot_chain(
    underlying: str,
    asof: dt.date,
    asof_ts: pd.Timestamp,
    raw_root: Path,
    chains_curated_root: Path,
    qc: QCReport,
) -> None:
    raw_df = _with_retries(
        f"{underlying} option chain",
        lambda: sources.fetch_option_chain(underlying, asof_ts),
        qc.retries,
    )

    raw_dir = raw_root / "chains" / f"asof_date={asof.isoformat()}" / f"underlying={underlying}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_dir / "part-0.parquet", index=False)

    prior_count = _prior_day_count(chains_curated_root, underlying, asof)

    clean_df, rejections = _clean_chain(raw_df)
    if rejections:
        qc.rejections[underlying] = rejections

    table = validate(clean_df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, chains_curated_root, ["asof_date", "underlying"])

    qc.chain_rows[underlying] = table.num_rows
    qc.null_rates[underlying] = {
        col: (table.column(col).null_count / table.num_rows if table.num_rows else 0.0)
        for col in table.schema.names
    }
    qc.prior_day_row_delta[underlying] = (
        None if prior_count is None else table.num_rows - prior_count
    )


def run_snapshot(cfg: BaseConfig, asof: dt.date) -> SnapshotResult:
    raw_root = Path(cfg.paths.raw)
    curated_root = Path(cfg.paths.curated)
    asof_ts = pd.Timestamp.now(tz=NY_TZ)

    universe = cfg.universe.index + cfg.universe.single_names
    qc = QCReport(asof=asof)
    chains_curated_root = curated_root / "chains"

    offset = _snap_offset_minutes(asof_ts, asof, cfg.canonical_snap_time)
    if abs(offset) > cfg.snap_tolerance_minutes:
        qc.snap_alerts.append(
            f"capture time {asof_ts.strftime('%H:%M:%S %Z')} is {offset:+.1f} min from "
            f"canonical snap {cfg.canonical_snap_time} ET "
            f"(tolerance {cfg.snap_tolerance_minutes}min)"
        )

    for underlying in universe:
        try:
            _snapshot_chain(underlying, asof, asof_ts, raw_root, chains_curated_root, qc)
        except Exception as exc:
            qc.failures[f"{underlying} option chain"] = _describe(exc)
            continue
        try:
            div_df = _with_retries(
                f"{underlying} dividends",
                functools.partial(sources.fetch_dividends, underlying),
                qc.retries,
            )
            if not div_df.empty:
                div_table = validate(div_df, DIVIDEND_SCHEMA, DIVIDEND_REQUIRED_NOT_NULL)
                store.write_partitioned(div_table, curated_root / "dividends", ["underlying"])
        except Exception as exc:
            qc.failures[f"{underlying} dividends"] = _describe(exc)

    # Spot and rates re-fetch a trailing window every day, so a failure here is
    # recovered by the next run; it is still reported, not fatal.
    ohlc_start = last_n_trading_days(asof, 10, cfg.calendar)[0]
    rates_start = last_n_trading_days(asof, 15, cfg.calendar)[0]
    feeds: list[tuple[str, Callable[[], pd.DataFrame], pa.Schema, list[str], str]] = [
        (
            "underlyings",
            lambda: sources.fetch_underlying_ohlc(
                universe, ohlc_start, asof + dt.timedelta(days=1)
            ),
            UNDERLYING_SCHEMA,
            UNDERLYING_REQUIRED_NOT_NULL,
            "underlyings",
        ),
        (
            "rates",
            lambda: sources.fetch_rates(rates_start, asof),
            CURVE_SCHEMA,
            CURVE_REQUIRED_NOT_NULL,
            "curves",
        ),
        (
            "vol indices",
            lambda: sources.fetch_vol_indices(rates_start, asof),
            VOL_INDEX_SCHEMA,
            VOL_INDEX_REQUIRED_NOT_NULL,
            "vol_indices",
        ),
    ]
    for label, fetch, schema, required, table_name in feeds:
        try:
            df = _with_retries(label, fetch, qc.retries)
            if not df.empty:
                table = validate(df, schema, required)
                store.write_partitioned(table, curated_root / table_name, ["asof_date"])
        except Exception as exc:
            qc.failures[label] = _describe(exc)

    return SnapshotResult(asof=asof, qc=qc, curated_root=curated_root)
