"""Real market inputs for deep-hedging training (`planning/deep_hedging_plan.md`):
loads NVDA's actual calibrated `LocalVolGrid` plus a consistent (spot, r, q) triple
for a chosen tenor `T`, reusing the exact same `load_market_state` machinery Steps
7/11/12 already use — not a separate/simplified data path.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.portfolio.mark import MarketState, _implied_q, load_market_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.vol.local_vol import LocalVolGrid, slice_total_variance

DEFAULT_PORTFOLIO_PATH = "configs/portfolio.yaml"  # already holds real NVDA positions


@dataclass
class HedgingMarketInputs:
    underlying: str
    asof: dt.date
    T: float
    spot: float
    r: float
    q: float
    grid: LocalVolGrid
    surface: pd.DataFrame  # this underlying's calibrated SVI/SSVI pillars

    def atm_implied_vol(self) -> float:
        """The real calibrated ATM (k=0) implied vol at the pillar nearest `T` —
        the constant vol a practitioner Black-Scholes delta hedge would actually
        use, NOT the local-vol surface itself (whose whole point is that it is
        NOT constant across spot/time — that mismatch is exactly what makes the
        BS-delta baseline imperfect, and is the real comparison point for a
        learned hedge trained directly against the true local-vol dynamics)."""
        nearest_idx = (self.surface["T"] - self.T).abs().idxmin()
        nearest = self.surface.loc[nearest_idx]
        w = float(slice_total_variance(nearest, np.array([0.0]))[0])
        return float(np.sqrt(max(w, 1e-12) / nearest["T"]))


def load_hedging_inputs(
    cfg: BaseConfig,
    asof: dt.date,
    T: float,
    underlying: str = "NVDA",
    project_root: Path | None = None,
) -> HedgingMarketInputs | None:
    """Returns `None` if the real curated state (curve/spot/surface/forward/grid)
    isn't available for `underlying` on `asof` — same honest-skip contract as
    `load_market_state` itself, no fallback to synthetic data."""
    project_root = project_root or Path.cwd()
    portfolio = Portfolio.from_yaml(str(project_root / DEFAULT_PORTFOLIO_PATH))
    state: MarketState | None = load_market_state(cfg, asof, portfolio)
    if state is None:
        return None
    if underlying not in state.spot or underlying not in state.forward_curve:
        return None
    if underlying not in state.grids or underlying not in state.surface:
        return None  # no MC-priced position for this underlying that day -> no grid built

    spot = state.spot[underlying]
    forward = state.forward_curve[underlying].forward(T)
    r = state.curve.zero_rate(T)
    q = _implied_q(spot, forward, r, T)
    return HedgingMarketInputs(
        underlying=underlying,
        asof=asof,
        T=T,
        spot=spot,
        r=r,
        q=q,
        grid=state.grids[underlying],
        surface=state.surface[underlying],
    )
