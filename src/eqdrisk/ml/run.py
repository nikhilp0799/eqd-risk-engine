"""Phase 4 (`planning/deep_hedging_plan.md`): the CLI-facing entry point that
trains, evaluates, and persists one deep-hedging comparison. Deliberately NOT
called from `run_daily_pipeline`/`daily_ingest.sh` — deep hedging is a
separate, additive research comparison, run on demand, not a new daily
pricing stage (it does not replace or touch the existing pricing/Greeks
engine at all).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    DEEP_HEDGE_RESULT_REQUIRED_NOT_NULL,
    DEEP_HEDGE_RESULT_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import year_fraction
from eqdrisk.ml.evaluate import (
    ComparisonResult,
    autocall_benchmark_delta,
    barrier_benchmark_delta,
    evaluate_autocall_hedge,
    evaluate_barrier_hedge,
    evaluate_vanilla_hedge,
)
from eqdrisk.ml.market import HedgingMarketInputs, load_hedging_inputs
from eqdrisk.ml.registry import save_model
from eqdrisk.ml.train import (
    TrainConfig,
    train_autocall_hedge,
    train_barrier_hedge,
    train_vanilla_hedge_general,
)
from eqdrisk.pricing.autocallable import AutocallableSpec

Instrument = Literal["vanilla", "autocall", "barrier"]
LossType = Literal["variance", "cvar", "cost"]

INSTRUMENTS: tuple[Instrument, ...] = ("vanilla", "autocall", "barrier")
LOSS_TYPES: tuple[LossType, ...] = ("variance", "cvar", "cost")

VANILLA_T = 0.25  # ATM call, 3 months — matches Phase 1/2's own validation

# Matches P008's REAL position terms (configs/portfolio.yaml) — the same
# instrument whose vega-only Greek gap Steps 12/13 already found and
# documented, deliberately reused rather than an arbitrary made-up note.
AUTOCALL_SPEC = AutocallableSpec(
    notional=5_000_000.0,
    autocall_barrier=1.00,
    coupon_barrier=0.75,
    put_barrier=0.65,
    coupon_rate=0.0225,
    obs_times=np.array([0.25, 0.5, 0.75, 1.0]),
)

# Matches P007's REAL barrier-to-strike ratio (4500/5500), applied to an ATM
# strike at whatever spot is real on the given `asof` — see
# `planning/deep_hedging_plan.md`'s Phase 5 section for why the ratio, not the
# absolute levels, is what's reused.
BARRIER_RATIO = 4500.0 / 5500.0
BARRIER_REAL_EXPIRY = dt.date(2027, 6, 17)  # P007's actual real expiry

DEFAULT_UNDERLYING: dict[Instrument, str] = {
    "vanilla": "NVDA",
    "autocall": "NVDA",
    "barrier": "SPX",  # P007's real underlying
}


# Phase 6: every combination trains these seeds by default. A seed changes both
# the initial weights and the training paths; validation and held-out paths stay
# fixed across seeds, so seed results are directly comparable.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass
class DeepHedgeRunResult:
    asof: dt.date
    underlying: str
    instrument: Instrument
    loss_type: LossType
    seed: int
    comparison: ComparisonResult
    epochs_run: int
    stopped_early: bool


def _train_cfg(instrument: Instrument) -> TrainConfig:
    """Phase 6: early stopping on a 4,000-path validation set (seed 500,
    distinct from training seeds and from the held-out seed 999), capped at
    3,000 epochs. A fixed 300 epochs was measured (2026-09-28) to stop well
    short of convergence."""
    if instrument == "vanilla":
        n_steps = 32
    elif instrument == "barrier":
        n_steps = 64  # finer monitoring partially mitigates the discretization bias
    else:
        n_steps = len(AUTOCALL_SPEC.obs_times)
    return TrainConfig(
        n_steps=n_steps,
        n_paths_train=8_000,
        epochs=3_000,
        cost_bps=5.0,
        hidden=64,
        lr=2e-3,
        n_paths_val=4_000,
        seed_val=500,
        eval_every=10,
        patience=100,
        min_rel_improvement=1e-3,
    )


@dataclass
class InstrumentSetup:
    """Everything about one instrument that depends only on the as-of market,
    shared by training (`run_deep_hedge`) and the Phase 7 robustness test."""

    underlying: str
    inputs: HedgingMarketInputs
    strike: float
    barrier: float  # only meaningful for the barrier instrument
    static_delta: float | None  # the static benchmark's hedge ratio (exotics only)


def instrument_setup(
    cfg: BaseConfig,
    asof: dt.date,
    instrument: Instrument,
    underlying: str | None = None,
    project_root: Path | None = None,
) -> InstrumentSetup | None:
    """`None` if real curated market data isn't available for that day."""
    underlying = underlying or DEFAULT_UNDERLYING[instrument]
    if instrument == "vanilla":
        T = VANILLA_T
    elif instrument == "barrier":
        T = year_fraction(asof, BARRIER_REAL_EXPIRY, cfg.daycount)
    else:
        T = float(AUTOCALL_SPEC.obs_times[-1])
    inputs = load_hedging_inputs(cfg, asof, T, underlying, project_root)
    if inputs is None:
        return None

    strike = round(inputs.spot)
    barrier = strike * BARRIER_RATIO
    # The static benchmarks' hedge ratios depend only on the market: once per call.
    static_delta = (
        barrier_benchmark_delta(inputs, strike, barrier)
        if instrument == "barrier"
        else autocall_benchmark_delta(inputs, AUTOCALL_SPEC)
        if instrument == "autocall"
        else None
    )
    return InstrumentSetup(underlying, inputs, strike, barrier, static_delta)


def run_deep_hedge(
    cfg: BaseConfig,
    asof: dt.date,
    instrument: Instrument,
    loss_type: LossType,
    underlying: str | None = None,
    n_paths_eval: int = 8_000,
    seed_eval: int = 999,
    project_root: Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[DeepHedgeRunResult] | None:
    """Trains, evaluates and persists one result per seed. Returns `None` if
    real curated market data isn't available for `underlying` on `asof` —
    honest skip, same contract as `market.load_hedging_inputs`, not a crash.
    `underlying=None` picks each instrument's own real underlying
    (`DEFAULT_UNDERLYING`)."""
    setup = instrument_setup(cfg, asof, instrument, underlying, project_root)
    if setup is None:
        return None
    underlying, inputs, strike, barrier, static_delta = (
        setup.underlying,
        setup.inputs,
        setup.strike,
        setup.barrier,
        setup.static_delta,
    )
    base_cfg = _train_cfg(instrument)

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        train_cfg = replace(base_cfg, seed_train=base_cfg.seed_train + seed)
        if instrument == "vanilla":
            trained = train_vanilla_hedge_general(inputs, strike, True, loss_type, train_cfg)
            comparison = evaluate_vanilla_hedge(
                inputs,
                trained.net,
                strike,
                True,
                n_paths_eval,
                train_cfg.n_steps,
                train_cfg.cost_bps,
                seed_eval,
            )
        elif instrument == "barrier":
            trained = train_barrier_hedge(inputs, strike, barrier, loss_type, train_cfg)
            comparison = evaluate_barrier_hedge(
                inputs,
                trained.net,
                strike,
                barrier,
                n_paths_eval,
                train_cfg.n_steps,
                train_cfg.cost_bps,
                seed_eval,
                static_delta=static_delta,
            )
        else:
            trained = train_autocall_hedge(inputs, AUTOCALL_SPEC, loss_type, train_cfg)
            comparison = evaluate_autocall_hedge(
                inputs,
                AUTOCALL_SPEC,
                trained.net,
                n_paths_eval,
                train_cfg.cost_bps,
                seed_eval,
                static_delta=static_delta,
            )
        save_model(trained.net, Path(cfg.paths.curated), asof, instrument, loss_type, seed)
        results.append(
            DeepHedgeRunResult(
                asof=asof,
                underlying=underlying,
                instrument=instrument,
                loss_type=loss_type,
                seed=seed,
                comparison=comparison,
                epochs_run=trained.epochs_run,
                stopped_early=trained.stopped_early,
            )
        )

    _persist(results, Path(cfg.paths.curated), base_cfg)
    return results


def _persist(results: list[DeepHedgeRunResult], curated_root: Path, train_cfg: TrainConfig) -> None:
    """`store.write_partitioned` replaces a WHOLE `asof_date` partition per
    call (by design, matching every other table in this project, which always
    writes a full day's rows in one call) — but `deephedge` is invoked once
    PER combination, so a naive write here would silently erase every other
    combination already persisted for the same day. Read any existing rows for
    this `asof_date` first, drop only the rows this call replaces, and write
    the full set back.

    Rows are keyed by (instrument, loss_type, seed) since Phase 6. Rows written
    before Phase 6 have no seed; a new run of the same combination replaces
    those too, rather than leaving an unseeded duplicate next to seeded rows."""
    if not results:
        return
    table_root = curated_root / "deep_hedge_results"
    first = results[0]
    rows = [
        {
            "asof_date": r.asof,
            "underlying": r.underlying,
            "instrument": r.instrument,
            "loss_type": r.loss_type,
            "seed": r.seed,
            "cost_bps": train_cfg.cost_bps,
            "n_paths_train": train_cfg.n_paths_train,
            "n_steps": train_cfg.n_steps,
            "epochs": r.epochs_run,
            "n_eval_paths": r.comparison.n_eval_paths,
            "learned_mean": r.comparison.learned.mean,
            "learned_std": r.comparison.learned.std,
            "learned_cvar": r.comparison.learned.cvar,
            "learned_turnover": r.comparison.learned.turnover,
            "baseline_mean": r.comparison.baseline.mean,
            "baseline_std": r.comparison.baseline.std,
            "baseline_cvar": r.comparison.baseline.cvar,
            "baseline_turnover": r.comparison.baseline.turnover,
        }
        for r in results
    ]
    new = pd.DataFrame(rows)

    existing = pd.DataFrame()
    if table_root.exists() and any(table_root.rglob("*.parquet")):
        existing = store.query(
            f"SELECT * FROM t WHERE asof_date = DATE '{first.asof.isoformat()}'",
            views={"t": str(table_root)},
        ).to_pandas()
        for col in new.columns:
            if col not in existing.columns:
                existing[col] = pd.NA  # older partitions predate Phase 6's columns
        existing["seed"] = existing["seed"].astype("Int64")
        same_combo = (existing["instrument"] == first.instrument) & (
            existing["loss_type"] == first.loss_type
        )
        replaced = same_combo & (
            existing["seed"].isna() | existing["seed"].isin([r.seed for r in results])
        )
        existing = existing[~replaced.fillna(False)]

    combined = pd.concat([existing, new], ignore_index=True)
    combined["seed"] = combined["seed"].astype("Int64")
    table = validate(combined, DEEP_HEDGE_RESULT_SCHEMA, DEEP_HEDGE_RESULT_REQUIRED_NOT_NULL)
    store.write_partitioned(table, table_root, ["asof_date"])
