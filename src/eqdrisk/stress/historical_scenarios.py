"""Historical replay stress scenarios (README 11.1): five named market
episodes, their REAL factor moves pulled from actual historical data, applied
to today's book.

Step 1 established that free-tier data has no deep history for OPTION CHAINS
(no time machine for a live-quotes-only source like yfinance's chain endpoint)
— but that constraint never applied to **underlying prices** or **VIX levels**,
both of which yfinance/FRED serve happily for any past date via their own
historical-OHLC/historical-series endpoints (already used by `io.sources`,
just with an old date range instead of a recent one). So the historical
replay's SPOT shock is the real, per-underlying observed move during each
episode, not a hand-picked approximation.

**A real, honest limitation, not hidden:** the VOL shock is NOT similarly
per-node-real — we have no calibrated historical vol *surfaces* from 2018,
2020, 2022, or 2024 (that would need Step 4's full pipeline to have been
running back then), only the real VIX level before/after. Each episode's vol
shock is therefore a single PARALLEL vol move sized off the real VIX change
(`vol_shock_pct = VIX_after/VIX_before - 1`), applied uniformly across the
surface via `stress.shock.MarketShock` — real skew/term-structure behaviour
during these episodes (which is exactly what made several of them notable)
isn't captured. That gap is precisely what 11.3's conditional-stress
regression is for, and it's deferred pending real accumulated history, same as
8.2/8.3/9/10.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from eqdrisk.io.sources import fetch_underlying_ohlc, fetch_vol_indices
from eqdrisk.stress.shock import MarketShock

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


def _nearest_close(df: pd.DataFrame, target: dt.date, label: str) -> float:
    """Nearest available close to `target` (a named market episode's boundary
    date might land on a weekend/holiday) — raises if nothing is within
    `NEAREST_PRICE_TOLERANCE_DAYS`, rather than silently using a stale point."""
    if df.empty:
        raise ValueError(f"no {label} data available at all")
    diffs = (df["asof_date"] - target).apply(lambda d: abs(d.days))
    idx = diffs.idxmin()
    if diffs.loc[idx] > NEAREST_PRICE_TOLERANCE_DAYS:
        raise ValueError(f"no {label} price within {NEAREST_PRICE_TOLERANCE_DAYS} days of {target}")
    return float(df.loc[idx, "close"])


def compute_episode_shocks(
    episode: HistoricalEpisode, underlyings: list[str]
) -> dict[str, MarketShock]:
    """Real spot move per underlying (from that name's OWN historical OHLC)
    crossed with a single real, VIX-derived parallel vol move (see module
    docstring for why the vol component isn't per-underlying/per-node)."""
    buffer = dt.timedelta(days=HISTORICAL_DATA_BUFFER_DAYS)
    ohlc = fetch_underlying_ohlc(underlyings, episode.before - buffer, episode.after + buffer)
    vix = fetch_vol_indices(episode.before - buffer, episode.after + buffer)
    vix = vix[vix["index"] == "VIX"].rename(columns={"value": "close"})

    vix_before = _nearest_close(vix, episode.before, "VIX")
    vix_after = _nearest_close(vix, episode.after, "VIX")
    vol_shock_pct = vix_after / vix_before - 1.0

    shocks: dict[str, MarketShock] = {}
    for u in underlyings:
        u_df = ohlc[ohlc["underlying"] == u]
        try:
            price_before = _nearest_close(u_df, episode.before, u)
            price_after = _nearest_close(u_df, episode.after, u)
        except ValueError:
            continue
        spot_shock_pct = price_after / price_before - 1.0
        shocks[u] = MarketShock(spot_shock_pct=spot_shock_pct, vol_shock_pct=vol_shock_pct)
    return shocks
