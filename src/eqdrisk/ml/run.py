"""Phase 4 (`planning/deep_hedging_plan.md`): the CLI-facing entry point that
trains, evaluates, and persists one deep-hedging comparison. Deliberately NOT
called from `run_daily_pipeline`/`daily_ingest.sh` — deep hedging is a
separate, additive research comparison, run on demand, not a new daily
pricing stage (it does not replace or touch the existing pricing/Greeks
engine at all).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

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
    evaluate_autocall_hedge,
    evaluate_barrier_hedge,
    evaluate_vanilla_hedge,
)
from eqdrisk.ml.market import load_hedging_inputs
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


@dataclass
class DeepHedgeRunResult:
    asof: dt.date
    underlying: str
    instrument: Instrument
    loss_type: LossType
    comparison: ComparisonResult


def _train_cfg(instrument: Instrument) -> TrainConfig:
    if instrument == "vanilla":
        n_steps = 32
    elif instrument == "barrier":
        n_steps = 64  # finer monitoring partially mitigates the discretization bias
    else:
        n_steps = len(AUTOCALL_SPEC.obs_times)
    return TrainConfig(
        n_steps=n_steps, n_paths_train=8_000, epochs=300, cost_bps=5.0, hidden=64, lr=2e-3
    )


def run_deep_hedge(
    cfg: BaseConfig,
    asof: dt.date,
    instrument: Instrument,
    loss_type: LossType,
    underlying: str | None = None,
    n_paths_eval: int = 8_000,
    seed_eval: int = 999,
    project_root: Path | None = None,
) -> DeepHedgeRunResult | None:
    """Returns `None` if real curated market data isn't available for
    `underlying` on `asof` — honest skip, same contract as
    `market.load_hedging_inputs`, not a crash. `underlying=None` picks each
    instrument's own real underlying (`DEFAULT_UNDERLYING`)."""
    underlying = underlying or DEFAULT_UNDERLYING[instrument]
    train_cfg = _train_cfg(instrument)

    if instrument == "vanilla":
        inputs = load_hedging_inputs(cfg, asof, VANILLA_T, underlying, project_root)
        if inputs is None:
            return None
        strike = round(inputs.spot)
        result = train_vanilla_hedge_general(inputs, strike, True, loss_type, train_cfg)
        comparison = evaluate_vanilla_hedge(
            inputs,
            result.net,
            strike,
            True,
            n_paths_eval,
            train_cfg.n_steps,
            train_cfg.cost_bps,
            seed_eval,
        )
    elif instrument == "barrier":
        T = year_fraction(asof, BARRIER_REAL_EXPIRY, cfg.daycount)
        inputs = load_hedging_inputs(cfg, asof, T, underlying, project_root)
        if inputs is None:
            return None
        strike = round(inputs.spot)
        barrier = strike * BARRIER_RATIO
        result = train_barrier_hedge(inputs, strike, barrier, loss_type, train_cfg)
        comparison = evaluate_barrier_hedge(
            inputs,
            result.net,
            strike,
            barrier,
            n_paths_eval,
            train_cfg.n_steps,
            train_cfg.cost_bps,
            seed_eval,
        )
    else:
        T = float(AUTOCALL_SPEC.obs_times[-1])
        inputs = load_hedging_inputs(cfg, asof, T, underlying, project_root)
        if inputs is None:
            return None
        result = train_autocall_hedge(inputs, AUTOCALL_SPEC, loss_type, train_cfg)
        comparison = evaluate_autocall_hedge(
            inputs, AUTOCALL_SPEC, result.net, n_paths_eval, train_cfg.cost_bps, seed_eval
        )

    run_result = DeepHedgeRunResult(
        asof=asof,
        underlying=underlying,
        instrument=instrument,
        loss_type=loss_type,
        comparison=comparison,
    )
    _persist(run_result, Path(cfg.paths.curated), train_cfg)
    return run_result


def _persist(result: DeepHedgeRunResult, curated_root: Path, train_cfg: TrainConfig) -> None:
    """`store.write_partitioned` replaces a WHOLE `asof_date` partition per
    call (by design, matching every other table in this project, which always
    writes a full day's rows in one call) — but `deephedge` is invoked once
    PER combination, so a naive single-row write here would silently erase
    every other combination already persisted for the same day. Read any
    existing rows for this `asof_date` first, replace only this exact
    (instrument, loss_type) row if present, and write the full set back."""
    table_root = curated_root / "deep_hedge_results"
    row = {
        "asof_date": result.asof,
        "underlying": result.underlying,
        "instrument": result.instrument,
        "loss_type": result.loss_type,
        "cost_bps": train_cfg.cost_bps,
        "n_paths_train": train_cfg.n_paths_train,
        "n_steps": train_cfg.n_steps,
        "epochs": train_cfg.epochs,
        "n_eval_paths": result.comparison.n_eval_paths,
        "learned_mean": result.comparison.learned.mean,
        "learned_std": result.comparison.learned.std,
        "learned_cvar": result.comparison.learned.cvar,
        "baseline_mean": result.comparison.baseline.mean,
        "baseline_std": result.comparison.baseline.std,
        "baseline_cvar": result.comparison.baseline.cvar,
    }

    existing = pd.DataFrame()
    if table_root.exists() and any(table_root.rglob("*.parquet")):
        existing = store.query(
            f"SELECT * FROM t WHERE asof_date = DATE '{result.asof.isoformat()}'",
            views={"t": str(table_root)},
        ).to_pandas()
        same_combo = (existing["instrument"] == result.instrument) & (
            existing["loss_type"] == result.loss_type
        )
        existing = existing[~same_combo]

    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    table = validate(combined, DEEP_HEDGE_RESULT_SCHEMA, DEEP_HEDGE_RESULT_REQUIRED_NOT_NULL)
    store.write_partitioned(table, table_root, ["asof_date"])
