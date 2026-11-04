"""Historical replay stress scenarios (README 11.1): five named market
episodes, their REAL factor moves pulled from actual historical data, applied
to today's book.

Step 1 established that free-tier data has no deep history for OPTION CHAINS
(no time machine for a live-quotes-only source like yfinance's chain endpoint)
— but that constraint never applied to **underlying prices** or **vol index
levels**, which yfinance/FRED serve for any past date. So each episode's spot
move is the real, per-underlying observed move, and each name's vol move is
its own index's real move (VIX for SPX, VXAPL for AAPL, VXN for NVDA).

There are no calibrated surfaces from 2018-2024, so the index move stands in
for the surface move: `stress/replay.py` maps it onto every grid node through
the 8.3 betas, as VaR does (`risk.var.VolProxy`). The old parallel version
(every node scaled by VIX's ratio) is kept beside it as the naive comparison.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

NEAREST_PRICE_TOLERANCE_DAYS = 5
HISTORICAL_DATA_BUFFER_DAYS = 10


@dataclass
class HistoricalEpisode:
    name: str
    before: dt.date
    after: dt.date
    description: str


EPISODES: list[HistoricalEpisode] = [
    HistoricalEpisode(
        "volmageddon_2018",
        dt.date(2018, 2, 2),
        dt.date(2018, 2, 6),
        "Volmageddon: XIV unwind, VIX spiked intraday from the high-teens to the 30s+",
    ),
    HistoricalEpisode(
        "covid_crash_2020",
        dt.date(2020, 2, 19),
        dt.date(2020, 3, 23),
        "COVID crash: SPX all-time high to bear-market trough in five weeks",
    ),
    HistoricalEpisode(
        "covid_single_day_2020",
        dt.date(2020, 3, 13),
        dt.date(2020, 3, 16),
        "COVID single day: one of the worst single-session SPX drops on record",
    ),
    HistoricalEpisode(
        "rate_shock_2022",
        dt.date(2022, 9, 1),
        dt.date(2022, 9, 30),
        "2022 rate shock: Fed tightening acceleration plus the UK gilt crisis",
    ),
    HistoricalEpisode(
        "yen_carry_unwind_2024",
        dt.date(2024, 8, 1),
        dt.date(2024, 8, 5),
        "Yen carry unwind: a sharp VIX spike out of a very quiet regime",
    ),
]
# First day of factor history the replay needs: the earliest episode, less a
# buffer so its nearest close can fall before the boundary date.
HISTORY_START = min(e.before for e in EPISODES) - dt.timedelta(days=HISTORICAL_DATA_BUFFER_DAYS)


def _nearest_close(series: pd.Series, target: dt.date, label: str) -> float:
    """Nearest available close to `target` (a named market episode's boundary
    date might land on a weekend/holiday) — raises if nothing is within
    `NEAREST_PRICE_TOLERANCE_DAYS`, rather than silently using a stale point."""
    series = series.dropna()
    if series.empty:
        raise ValueError(f"no {label} data available at all")
    diffs = np.array([abs((d - target).days) for d in series.index])
    i = int(diffs.argmin())
    if diffs[i] > NEAREST_PRICE_TOLERANCE_DAYS:
        raise ValueError(f"no {label} price within {NEAREST_PRICE_TOLERANCE_DAYS} days of {target}")
    return float(series.iloc[i])


def episode_moves(
    episode: HistoricalEpisode, levels: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    """Relative move of each column (a spot or a vol index) over the episode,
    from `levels` (daily closes, one column per factor, indexed by date)."""
    moves: dict[str, float] = {}
    for c in columns:
        if c not in levels:
            raise ValueError(f"no {c} data available at all")
        before = _nearest_close(levels[c], episode.before, c)
        after = _nearest_close(levels[c], episode.after, c)
        moves[c] = after / before - 1.0
    return moves
