"""Orchestrates Step 11.1: today's book repriced under each historical
episode's real factor moves, two ways:

- `proxy`: each name's real spot move, and its own vol index's real move
  (rescaled to today's index level) mapped onto every surface node by the 8.3
  betas, as VaR does.
- `naive`: the same spot moves, every node scaled by VIX's relative move (the
  original 11.1 method). It lifts 1-2 year vol as much as 1-month vol, which
  history does not, and gives the variance swap gains no market has produced
  (COVID +$128.7M on the 2026-10-29 book). Kept to show how far it is off.

One episode missing a factor's data is reported and skipped, not allowed to
kill the whole report — these are five independent scenarios, not a pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    HISTORICAL_REPLAY_REQUIRED_NOT_NULL,
    HISTORICAL_REPLAY_SCHEMA,
    validate,
)
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.risk.proxy import VOL_INDEX_FOR
from eqdrisk.risk.var import (
    PreparedRevaluation,
    active_underlyings,
    full_pnl,
    load_factor_history,
    prepare_revaluation,
    revalue,
)
from eqdrisk.stress.historical_scenarios import (
    EPISODES,
    HISTORY_START,
    HistoricalEpisode,
    episode_moves,
)
from eqdrisk.stress.shock import MarketShock

VARIANTS = ("proxy", "naive")
NAIVE_INDEX = "VIX"


@dataclass
class EpisodeResult:
    episode: HistoricalEpisode
    variant: str
    moves: dict[str, float]  # relative move per spot and vol index
    shocked_value: float
    pnl: float
    varswap_pnl: float = 0.0  # the variance swap's part of `pnl`


@dataclass
class HistoricalReplayResult:
    asof: dt.date
    base_value: float = 0.0
    episodes: list[EpisodeResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"Historical replay — {self.asof}",
            f"  base portfolio value: {self.base_value:,.2f}",
        ]
        by_name: dict[str, list[EpisodeResult]] = {}
        for er in self.episodes:
            by_name.setdefault(er.episode.name, []).append(er)
        for results in by_name.values():
            ep = results[0].episode
            lines.append(f"  {ep.name} ({ep.description}):")
            moves = results[0].moves
            lines.append("    moves: " + ", ".join(f"{f} {m:+.1%}" for f, m in moves.items()))
            for er in results:
                lines.append(
                    f"    {er.variant:>5}: P&L {er.pnl:+,.0f}  (variance swap "
                    f"{er.varswap_pnl:+,.0f}; without it {er.pnl - er.varswap_pnl:+,.0f})"
                )
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def episode_shocks(
    moves: dict[str, float], prep: PreparedRevaluation
) -> dict[str, dict[str, MarketShock]]:
    """Per variant, the shock for each active name."""
    return {
        "proxy": {
            u: MarketShock(
                spot_shock_pct=moves[u],
                node_vol=prep.proxies[u].node_vol(moves[VOL_INDEX_FOR[u]]),
            )
            for u in prep.underlyings
        },
        "naive": {
            u: MarketShock(spot_shock_pct=moves[u], vol_shock_pct=moves[NAIVE_INDEX])
            for u in prep.underlyings
        },
    }


def run_historical_replay(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio_path: str,
    levels: pd.DataFrame | None = None,
    max_workers: int | None = None,
    persist: bool = True,
) -> HistoricalReplayResult:
    result = HistoricalReplayResult(asof=asof)
    portfolio = Portfolio.from_yaml(portfolio_path)
    if levels is None:
        levels = load_factor_history(
            active_underlyings(portfolio, asof), HISTORY_START, asof, extra_indices=[NAIVE_INDEX]
        )
    levels = levels.loc[levels.index <= asof]
    prep = prepare_revaluation(cfg, asof, portfolio, levels)
    if isinstance(prep, str):
        result.skipped["_all_"] = prep
        return result
    result.base_value = float(prep.base[0].sum())

    factors = list(
        dict.fromkeys(
            prep.underlyings + [VOL_INDEX_FOR[u] for u in prep.underlyings] + [NAIVE_INDEX]
        )
    )
    shocks: list[dict[str, MarketShock]] = []
    labels: list[tuple[HistoricalEpisode, str, dict[str, float]]] = []
    for episode in EPISODES:
        try:
            moves = episode_moves(episode, levels, factors)
        except ValueError as exc:
            result.skipped[episode.name] = str(exc)
            continue
        for variant, shock in episode_shocks(moves, prep).items():
            shocks.append(shock)
            labels.append((episode, variant, moves))
    if not shocks:
        return result

    pnl = full_pnl(prep.base, revalue(prep.ctx, shocks, max_workers), prep.marks)
    is_swap = [m.type == "varswap" for m in prep.marks]
    for (episode, variant, moves), row in zip(labels, pnl, strict=True):
        total = float(row.sum())
        result.episodes.append(
            EpisodeResult(
                episode=episode,
                variant=variant,
                moves=moves,
                shocked_value=result.base_value + total,
                pnl=total,
                varswap_pnl=float(row[is_swap].sum()),
            )
        )
    if persist:
        _persist(result, Path(cfg.paths.curated))
    return result


def _persist(result: HistoricalReplayResult, curated_root: Path) -> None:
    if not result.episodes:
        return
    rows = [
        {
            "asof_date": result.asof,
            "episode_name": er.episode.name,
            "episode_description": er.episode.description,
            "variant": er.variant,
            "base_value": result.base_value,
            "shocked_value": er.shocked_value,
            "pnl": er.pnl,
            "varswap_pnl": er.varswap_pnl,
        }
        for er in result.episodes
    ]
    table = validate(
        pd.DataFrame(rows), HISTORICAL_REPLAY_SCHEMA, HISTORICAL_REPLAY_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, curated_root / "historical_replay", ["asof_date"])
