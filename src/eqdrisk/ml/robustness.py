"""Out-of-model robustness for the saved deep-hedging policies (Phase 7,
`planning/deep_hedging_plan.md`).

Every earlier result trains AND tests on the same calibrated local-vol
simulator — fresh paths, same dynamics. This module asks whether a learned
policy's advantage over its benchmark survives when the market does not behave
like that model: simulator stresses (volatility scaled up/down, crash-like
jumps) and real price history replayed through the hedges. Evaluation only:
nothing is retrained on any scenario, and both hedgers stay exactly as
calibrated on the as-of date.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    DEEP_HEDGE_ROBUSTNESS_REQUIRED_NOT_NULL,
    DEEP_HEDGE_ROBUSTNESS_SCHEMA,
    validate,
)
from eqdrisk.io.sources import fetch_underlying_ohlc
from eqdrisk.ml.evaluate import (
    ComparisonResult,
    compare_autocall_on_paths,
    compare_barrier_on_paths,
    compare_vanilla_on_paths,
)
from eqdrisk.ml.hedge_model import HedgeNet
from eqdrisk.ml.market import HedgingMarketInputs
from eqdrisk.ml.registry import load_model
from eqdrisk.ml.run import (
    AUTOCALL_SPEC,
    DEFAULT_SEEDS,
    INSTRUMENTS,
    LOSS_TYPES,
    Instrument,
    InstrumentSetup,
    _train_cfg,
    instrument_setup,
)
from eqdrisk.ml.simulate import SimulatedPaths, simulate_training_paths
from eqdrisk.pricing.monte_carlo import next_power_of_two

SCENARIOS = ("in_sample", "vol_up_25", "vol_down_25", "jumps", "history")

EVAL_SEED = 999  # the same held-out seed Phase 6's results were measured on
TRADING_DAYS_PER_YEAR = 252

# Crash-like compensated Merton jumps: on average one jump a year, each a 10%
# fall give or take 5%. Drift-compensated, so the expected price is unchanged
# and only the shape of the distribution moves.
JUMP_INTENSITY = 1.0
JUMP_MEAN = -0.10
JUMP_STD = 0.05

# A one-day move this large in split-adjusted closes means the data is NOT
# split-adjusted (or is corrupt): fail loudly rather than replay a fake crash.
MAX_PLAUSIBLE_DAILY_LOG_RETURN = 0.5


# --- scenario path generators -------------------------------------------------


def scaled_vol_paths(
    inputs: HedgingMarketInputs,
    factor: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    substeps: int = 1,
) -> SimulatedPaths:
    """The calibrated local-vol model with every local vol multiplied by
    `factor` — a market that turns out more (or less) volatile than priced."""
    grid = dataclasses.replace(inputs.grid, sigma_loc=inputs.grid.sigma_loc * factor)
    return simulate_training_paths(
        dataclasses.replace(inputs, grid=grid), n_paths, n_steps, seed, substeps
    )


def jump_paths(
    inputs: HedgingMarketInputs,
    n_paths: int,
    n_steps: int,
    seed: int,
    intensity: float = JUMP_INTENSITY,
    mean: float = JUMP_MEAN,
    std: float = JUMP_STD,
    substeps: int = 1,
) -> SimulatedPaths:
    """Compensated Merton jumps overlaid on calibrated local-vol paths.

    The jumps are added to each step's log return after the diffusion is
    simulated, so the local vol after a jump does not react to the new level:
    an overlay approximation, but a deliberate one, since the point is a shock
    the hedgers' model never saw. With `intensity = 0` the paths are identical
    to the unshocked local-vol paths. Jumps are drawn per rebalancing step; a
    Poisson count over a step has the same distribution however finely the
    diffusion underneath was simulated.
    """
    base = simulate_training_paths(inputs, n_paths, n_steps, seed, substeps)
    paths = base.paths.numpy()
    dt_steps = np.diff(base.t_grid)
    log_ret = np.diff(np.log(paths), axis=1)

    rng = np.random.default_rng(seed + 7_919)
    n_jumps = rng.poisson(intensity * dt_steps, size=log_ret.shape)
    jump_sizes = mean * n_jumps + std * np.sqrt(n_jumps) * rng.standard_normal(log_ret.shape)
    compensator = intensity * (np.exp(mean + 0.5 * std**2) - 1.0) * dt_steps
    shocked = log_ret + jump_sizes - compensator

    log_paths = np.log(paths[:, :1]) + np.concatenate(
        [np.zeros((paths.shape[0], 1)), np.cumsum(shocked, axis=1)], axis=1
    )
    out = torch.from_numpy(np.exp(log_paths)).to(dtype=torch.float64)
    out.requires_grad_(False)
    return SimulatedPaths(paths=out, t_grid=base.t_grid)


def training_time_grid(T: float, n_steps: int) -> np.ndarray:
    """The exact time grid `simulate_training_paths` uses, so a historical path
    is sampled at the same moments the policy was trained to act at."""
    return np.linspace(0.0, T, next_power_of_two(n_steps) + 1)


@dataclass
class HistoryWindows:
    paths: SimulatedPaths
    n_windows: int
    n_independent: float  # history length / instrument life: overlapping windows are correlated


def history_paths(
    closes: np.ndarray, spot: float, T: float, n_steps: int, stride: int = 5
) -> HistoryWindows:
    """Rolling windows of real daily closes, each as long as the instrument's
    life (`T * 252` trading days), rescaled to start at `spot` and sampled at
    the training grid's times. Windows start every `stride` trading days."""
    closes = np.asarray(closes, dtype=float)
    log_ret = np.diff(np.log(closes))
    if np.any(np.abs(log_ret) > MAX_PLAUSIBLE_DAILY_LOG_RETURN):
        raise ValueError(
            "a one-day move above 50% in the price history: closes look unadjusted for splits"
        )

    t_grid = training_time_grid(T, n_steps)
    window = int(round(T * TRADING_DAYS_PER_YEAR))
    offsets = np.rint(t_grid / T * window).astype(int)
    starts = np.arange(0, len(closes) - window, stride)
    if len(starts) == 0:
        raise ValueError(f"price history too short for a {window}-day window")

    rel = closes[starts[:, None] + offsets[None, :]] / closes[starts][:, None]
    paths = torch.from_numpy(spot * rel).to(dtype=torch.float64)
    paths.requires_grad_(False)
    years = len(closes) / TRADING_DAYS_PER_YEAR
    return HistoryWindows(
        paths=SimulatedPaths(paths=paths, t_grid=t_grid),
        n_windows=len(starts),
        n_independent=years / T,
    )


HistoryFetcher = Callable[[list[str], dt.date, dt.date], pd.DataFrame]


def load_history_closes(
    underlying: str,
    start: dt.date,
    end: dt.date,
    cache_dir: Path,
    fetch: HistoryFetcher = fetch_underlying_ohlc,
) -> np.ndarray:
    """Daily closes over [start, end), from a local cache when present, so a
    rerun replays exactly the same history instead of whatever the free data
    source returns that day."""
    path = cache_dir / f"{underlying}_{start.isoformat()}_{end.isoformat()}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = fetch([underlying], start, end)
        if df.empty:
            raise ValueError(f"no price history returned for {underlying}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df.sort_values("asof_date")["close"].to_numpy(dtype=float)


# --- runner ---------------------------------------------------------------------


@dataclass
class ScenarioSet:
    paths: dict[str, SimulatedPaths]
    history_independent: float


def build_scenarios(
    setup: InstrumentSetup,
    n_steps: int,
    n_paths: int,
    history_closes: np.ndarray,
    substeps: int = 1,
) -> ScenarioSet:
    """`substeps` must match training (`TrainConfig.sim_substeps`), so the
    in-sample scenario reproduces the stored training-run results exactly."""
    inputs = setup.inputs
    k = substeps
    history = history_paths(history_closes, inputs.spot, inputs.T, n_steps)
    return ScenarioSet(
        paths={
            "in_sample": simulate_training_paths(inputs, n_paths, n_steps, EVAL_SEED, k),
            "vol_up_25": scaled_vol_paths(inputs, 1.25, n_paths, n_steps, EVAL_SEED, k),
            "vol_down_25": scaled_vol_paths(inputs, 0.75, n_paths, n_steps, EVAL_SEED, k),
            "jumps": jump_paths(inputs, n_paths, n_steps, EVAL_SEED, substeps=k),
            "history": history.paths,
        },
        history_independent=history.n_independent,
    )


def _compare(
    instrument: Instrument,
    setup: InstrumentSetup,
    net: HedgeNet,
    sim: SimulatedPaths,
    cost_bps: float,
) -> ComparisonResult:
    if instrument == "vanilla":
        return compare_vanilla_on_paths(setup.inputs, net, setup.strike, True, sim, cost_bps)
    assert setup.static_delta is not None
    if instrument == "barrier":
        return compare_barrier_on_paths(
            setup.inputs,
            net,
            setup.strike,
            setup.barrier,
            sim,
            cost_bps,
            setup.static_delta,
        )
    return compare_autocall_on_paths(
        setup.inputs,
        AUTOCALL_SPEC,
        net,
        sim,
        cost_bps,
        setup.static_delta,
    )


def run_robustness(
    cfg: BaseConfig,
    asof: dt.date,
    history_years: int = 20,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_paths: int = 8_000,
    project_root: Path | None = None,
    fetch: HistoryFetcher = fetch_underlying_ohlc,
    underlying: str | None = None,
) -> pd.DataFrame | None:
    """Scores every saved (instrument, loss, seed) policy on every scenario and
    persists one row each. `None` if the as-of day has no curated market data.
    Raises if a policy was never saved (train it first with `deephedge --all`).
    `underlying=None` uses each instrument's own real underlying, exactly as
    `run_deep_hedge` does."""
    curated = Path(cfg.paths.curated)
    root = project_root or Path(".")
    cache_dir = root / cfg.paths.raw / "history"
    start = asof.replace(year=asof.year - history_years)

    rows = []
    for instrument in INSTRUMENTS:
        setup = instrument_setup(cfg, asof, instrument, underlying, project_root)
        if setup is None:
            return None
        train_cfg = _train_cfg(instrument)
        closes = load_history_closes(setup.underlying, start, asof, cache_dir, fetch)
        scenarios = build_scenarios(
            setup, train_cfg.n_steps, n_paths, closes, train_cfg.sim_substeps
        )

        for loss_type in LOSS_TYPES:
            for seed in seeds:
                net = load_model(curated, asof, instrument, loss_type, seed)
                if net is None:
                    raise FileNotFoundError(
                        f"no saved policy for {instrument}/{loss_type}/seed {seed} on {asof}; "
                        "run `eqdrisk deephedge --all` first"
                    )
                for scenario, sim in scenarios.paths.items():
                    cmp = _compare(instrument, setup, net, sim, train_cfg.cost_bps)
                    rows.append(
                        {
                            "asof_date": asof,
                            "underlying": setup.underlying,
                            "instrument": instrument,
                            "loss_type": loss_type,
                            "seed": seed,
                            "scenario": scenario,
                            "n_paths": cmp.n_eval_paths,
                            "n_independent": (
                                scenarios.history_independent if scenario == "history" else None
                            ),
                            "learned_std": cmp.learned.std,
                            "learned_cvar": cmp.learned.cvar,
                            "learned_turnover": cmp.learned.turnover,
                            "baseline_std": cmp.baseline.std,
                            "baseline_cvar": cmp.baseline.cvar,
                            "baseline_turnover": cmp.baseline.turnover,
                        }
                    )

    df = pd.DataFrame(rows)
    table = validate(df, DEEP_HEDGE_ROBUSTNESS_SCHEMA, DEEP_HEDGE_ROBUSTNESS_REQUIRED_NOT_NULL)
    # One call writes the whole day (every instrument, loss, seed and scenario),
    # the same convention as every other curated table.
    store.write_partitioned(table, curated / "deep_hedge_robustness", ["asof_date"])
    return df
