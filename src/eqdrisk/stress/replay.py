"""Orchestrates Step 11.1: mark the portfolio once (today's real, unshocked
state), then reprice it under each historical episode's real factor moves and
report the P&L difference. One episode failing to fetch real data (a network
hiccup, a data-provider gap) is reported and skipped, not allowed to kill the
whole report — these are five independent scenarios, not a pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from eqdrisk.config import BaseConfig
from eqdrisk.portfolio.mark import load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.stress.historical_scenarios import EPISODES, HistoricalEpisode, compute_episode_shocks
from eqdrisk.stress.shock import MarketShock


@dataclass
class EpisodeResult:
    episode: HistoricalEpisode
    shocks: dict[str, MarketShock]
    shocked_value: float
    pnl: float


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
        for er in self.episodes:
            lines.append(f"  {er.episode.name} ({er.episode.description}):")
            lines.append(f"    P&L: {er.pnl:+,.2f}  (shocked value {er.shocked_value:,.2f})")
            for u, shock in sorted(er.shocks.items()):
                lines.append(
                    f"      {u}: spot {shock.spot_shock_pct:+.1%}, vol {shock.vol_shock_pct:+.1%}"
                )
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def run_historical_replay(
    cfg: BaseConfig, asof: dt.date, portfolio_path: str
) -> HistoricalReplayResult:
    portfolio = Portfolio.from_yaml(portfolio_path)
    state = load_market_state(cfg, asof, portfolio)
    if state is None:
        result = HistoricalReplayResult(asof=asof)
        result.skipped["_all_"] = "no curated rates available"
        return result

    underlyings = sorted({p.underlying for p in portfolio.positions})
    base_value = mark_with_state(cfg, asof, portfolio, state).total_value()
    result = HistoricalReplayResult(asof=asof, base_value=base_value)

    for episode in EPISODES:
        try:
            shocks = compute_episode_shocks(episode, underlyings)
        except Exception as exc:  # noqa: BLE001 - an external data provider's own
            # failure mode isn't ours to predict; one bad episode must not sink
            # the other four's real results.
            result.skipped[episode.name] = f"could not fetch historical data: {exc}"
            continue
        if not shocks:
            result.skipped[episode.name] = "no real historical data recovered for any underlying"
            continue

        shocked_value = mark_with_state(cfg, asof, portfolio, state, shock=shocks).total_value()
        result.episodes.append(
            EpisodeResult(
                episode=episode,
                shocks=shocks,
                shocked_value=shocked_value,
                pnl=shocked_value - base_value,
            )
        )

    return result
