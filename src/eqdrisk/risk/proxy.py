"""Vol proxy regression (README 8.3): each grid node's daily implied-vol change
regressed on the daily change of the name's own published vol index.

Why: VaR (risk/var.py) replays years of past market moves, but calibrated
surfaces exist only for the days the pipeline has run. Each name's vol index
has free daily history going back decades (FRED), so a past day's index move,
mapped onto every grid node through these betas, stands in for the surface
move that day. The regression is fitted on the stored days only, as the 8.2
PCA is: vol-point changes, nodes inside quoted strikes on 80%+ of days,
consecutive trading days.

Indices: SPX on VIX, AAPL on VXAPL (Apple's own 30d implied vol), NVDA on VXN
(Nasdaq-100; NVDA has no index of its own). Each name's own index explains
more than VIX does (measured 2026-11-01: median node R^2 0.30 vs 0.27 for AAPL,
0.30 vs 0.22 for NVDA). Adding 3-month VIX for SPX added at most 0.05 R^2
(daily changes 0.99 correlated with VIX), so one index per name.

The intercept is fitted but not applied in VaR: it is the sample's average
daily drift in vol, not a response to the index. What the index leaves
unexplained (1 - R^2) is the proxy error, measured in dollars in risk/var.py.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eqdrisk.marketdata.calendar import last_n_trading_days
from eqdrisk.vol.pca import MIN_COVERAGE, Node, daily_vol_changes, node_coverage
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

VOL_INDEX_FOR = {"SPX": "VIX", "AAPL": "VXAPL", "NVDA": "VXN"}
# The 8.2 rolling window: fewer changes than this give betas too noisy to use.
MIN_PROXY_CHANGES = 20


@dataclass
class ProxyFit:
    underlying: str
    index: str
    nodes: list[Node]
    coverage: pd.Series
    changes: pd.DataFrame  # rows = change dates, columns = nodes, vol points
    index_change: pd.Series  # same dates, index points
    alpha: np.ndarray  # per node
    beta: np.ndarray  # per node, vol points per index point
    r2: np.ndarray  # per node

    @property
    def residuals(self) -> pd.DataFrame:
        fitted = self.alpha + np.outer(self.index_change.to_numpy(), self.beta)
        return self.changes - fitted

    def beta_field(self) -> np.ndarray:
        return node_field(self.nodes, self.beta)


def node_field(nodes: list[Node], values: np.ndarray) -> np.ndarray:
    """`values` at `nodes` spread onto the full (T_GRID x K_GRID) rectangle: off
    the given nodes, the value of the nearest given k in the same maturity row,
    and for a row with no given node, the nearest row in T. Outside the quoted
    strikes the surface is extrapolation, so its move is held flat from the
    nearest quoted point rather than estimated from fit noise."""
    t_labels = list(T_GRID)
    k_values = np.array(K_GRID)
    given = dict(zip(nodes, np.asarray(values, dtype=float), strict=True))
    rows: dict[int, np.ndarray] = {}
    for i, t in enumerate(t_labels):
        ks = np.array([k for (tl, k) in nodes if tl == t])
        if ks.size == 0:
            continue
        nearest = ks[np.abs(ks[None, :] - k_values[:, None]).argmin(axis=1)]
        rows[i] = np.array([given[(t, float(k))] for k in nearest])
    if not rows:
        raise ValueError("no nodes to build a field from")
    t_years = np.array(list(T_GRID.values()))
    have = np.array(sorted(rows))
    field = np.empty((len(t_labels), len(k_values)))
    for i in range(len(t_labels)):
        field[i] = rows[int(have[np.abs(t_years[have] - t_years[i]).argmin()])]
    return field


def index_changes(dates: pd.Index, index_levels: pd.Series, calendar: str = "NYSE") -> pd.Series:
    """Index change from the previous trading day to each of `dates` (NaN where
    either close is missing). `index_levels` is indexed by date."""
    out = {}
    for d in dates:
        prev = last_n_trading_days(d - dt.timedelta(days=1), 1, calendar)[0]
        out[d] = index_levels.get(d, np.nan) - index_levels.get(prev, np.nan)
    return pd.Series(out, dtype=float)


def fit_proxy(changes: pd.DataFrame, index_change: pd.Series) -> tuple[np.ndarray, ...]:
    """OLS of every column of `changes` on `index_change` with an intercept.
    Returns (alpha, beta, r2), one value per column."""
    x = index_change.to_numpy(dtype=float)
    y = changes.to_numpy(dtype=float)
    xc = x - x.mean()
    beta = (xc @ (y - y.mean(axis=0))) / (xc @ xc)
    alpha = y.mean(axis=0) - beta * x.mean()
    resid = y - alpha - np.outer(x, beta)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (resid**2).sum(axis=0) / ss_tot
    return alpha, beta, r2


def proxy_for_underlying(
    underlying: str,
    surface: pd.DataFrame,
    risk_factors: pd.DataFrame,
    index_levels: pd.Series,
    calendar: str = "NYSE",
) -> ProxyFit | str:
    """Fit on every stored day passed in (the caller limits them to dates up to
    the VaR date, so nothing is fitted on the future). Returns the fit, or the
    reason it could not be made."""
    index = VOL_INDEX_FOR.get(underlying)
    if index is None:
        return f"no vol index mapped for {underlying}"
    coverage = node_coverage(surface)
    nodes = [node for node, share in coverage.items() if share >= MIN_COVERAGE]
    if not nodes:
        return f"no grid node inside quoted strikes on {MIN_COVERAGE:.0%} of days"
    changes = daily_vol_changes(risk_factors, nodes, calendar)
    d_index = index_changes(changes.index, index_levels, calendar).dropna()
    changes = changes.loc[d_index.index]
    if len(changes) < MIN_PROXY_CHANGES:
        return (
            f"{len(changes)} consecutive-day changes with a {index} close, need {MIN_PROXY_CHANGES}"
        )
    alpha, beta, r2 = fit_proxy(changes, d_index)
    return ProxyFit(
        underlying=underlying,
        index=index,
        nodes=nodes,
        coverage=coverage,
        changes=changes,
        index_change=d_index,
        alpha=alpha,
        beta=beta,
        r2=r2,
    )
