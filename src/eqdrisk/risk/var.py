"""Value at Risk and Expected Shortfall by historical simulation (README Step 9).

Each scenario replays one past day's market move on today's book:
- spot: each name's own return that day (yfinance closes);
- vol: the name's vol index (VIX / VXAPL / VXN, FRED) change that day,
  rescaled to today's index level, mapped onto every (k, T) grid node through
  the 8.3 proxy betas (risk/proxy.py), then applied to today's surface;
- rates and dividends held at today's values.

**Why a proxy and not real past surfaces:** the pipeline has stored calibrated
surfaces only since 2026-08-21, not the README's 250-1,000 days. Spot and vol
index history go back years for free. The vol move is therefore the part of each
past day's surface move its index explains: on the stored days, ~60% of SPX
node variance and ~30% of AAPL/NVDA (risk/proxy.py). The rest is measured as the
proxy error, in dollars, by `proxy_error`.

**Two methods on the same scenarios (README 9.2):** full revaluation reprices
every position under the scenario (Monte Carlo exotics price-only, common
random numbers so the base and every scenario share paths); the Taylor method
uses the day's marked Greeks: sticky-moneyness delta and gamma, vega, vanna,
volga, with the vol move read at ATM at each position's own `vol_times`, the
P&L explain's convention. Carry (theta) is left out of both: a scenario is an
instantaneous move, and leaving time out of both keeps the comparison like for
like.

**The variance swap (P006):** the daily mark treats it as a fresh swap at its
own fair strike, so it never shows P&L. In VaR it is struck at the VaR date's
fair strike, and a scenario's P&L is vega_notional x (K_new^2 - K^2) / (2K), K
in vol points (user decision 2026-11-01). VaR is also reported without it.

**Quantiles:** with n scenarios and confidence c, VaR is the k-th worst P&L,
k = floor(n (1 - c)), and ES the average of the k worst (500 days at 99%: the
5th worst and the average of the 5 worst).
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import sources, store
from eqdrisk.io.schemas import (
    PROXY_BETA_REQUIRED_NOT_NULL,
    PROXY_BETA_SCHEMA,
    PROXY_ERROR_REQUIRED_NOT_NULL,
    PROXY_ERROR_SCHEMA,
    VAR_SCENARIO_REQUIRED_NOT_NULL,
    VAR_SCENARIO_SCHEMA,
    VAR_SUMMARY_REQUIRED_NOT_NULL,
    VAR_SUMMARY_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import is_trading_day
from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.mark import (
    MarketState,
    MCSettings,
    PositionMark,
    load_market_state,
    mark_with_state,
)
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.risk.proxy import VOL_INDEX_FOR, ProxyFit, node_field, proxy_for_underlying
from eqdrisk.stress.shock import MarketShock, NodeVolShock
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

VAR_CONFIDENCE = 0.99
ES_CONFIDENCE = 0.975  # FRTB
PRIMARY_WINDOW = 500  # trading days, the regulatory convention (README 9.1)
WINDOWS = (250, 500, 1000)
TEN_DAY = 10
# README 9.3: equity index vol 10 days, single-name vol 20, structured/exotic 60.
LIQUIDITY_HORIZONS = (10, 20, 60)
# Calendar days of history fetched beyond the longest window's trading days.
HISTORY_PADDING_DAYS = 120
EXOTIC_TYPES = ("barrier", "autocall")
# One run per scenario, price only, common random numbers (one seed for the base
# and every scenario) so MC noise largely cancels in scenario P&L. Half the daily
# mark's paths and barrier steps: measured on 10-29 over five seeds, the P008
# autocallable's scenario P&L varied by $0.4-3.8K (sd) on moves of $30-530K, about
# the same as at the daily settings ($0.9-2.2K), at 0.35s a scenario against 0.81s.
VAR_MC_SETTINGS = MCSettings(
    n_paths=32_768, barrier_n_steps=32, n_batches=1, greeks=False, exotic_theta=False
)
# Factor sets moved per scenario. "single_name_vol" moves only AAPL/NVDA vol.
FACTOR_SETS = ("all", "spot", "vol", "single_name_vol")

T_VALUES = np.array(list(T_GRID.values()))
K_VALUES = np.array(K_GRID)


# ---------------------------------------------------------------------------
# History


def load_factor_history(underlyings: Sequence[str], start: dt.date, asof: dt.date) -> pd.DataFrame:
    """Daily closes, one column per underlying and per vol index, indexed by
    date, up to and including `asof`. Vol indices come from FRED; where FRED has
    not yet posted `asof` (it publishes a day late), yfinance's same-day close
    fills it."""
    names = [VOL_INDEX_FOR[u] for u in underlyings if u in VOL_INDEX_FOR]
    end = asof + dt.timedelta(days=1)
    long_columns = ["asof_date", "index", "value"]
    ohlc = sources.fetch_underlying_ohlc(list(underlyings), start, end)
    spot = ohlc.pivot_table(index="asof_date", columns="underlying", values="close")
    vix = sources.fetch_vol_indices(start, asof)
    vix = vix[long_columns] if not vix.empty else pd.DataFrame(columns=long_columns)
    vix = vix[vix["index"].isin(names)]
    missing_today = [
        n for n in names if not ((vix["index"] == n) & (vix["asof_date"] == asof)).any()
    ]
    if missing_today:
        yf_today = sources.fetch_vol_index_closes_yf(missing_today, asof, end)
        if not yf_today.empty:
            yf_today = yf_today.loc[yf_today["asof_date"] == asof, long_columns]
            vix = pd.concat([vix, yf_today], ignore_index=True)
    vix["value"] = vix["value"].astype(float)
    idx = vix.pivot_table(index="asof_date", columns="index", values="value")
    levels = spot.join(idx, how="outer").sort_index()
    levels.index = pd.Index(levels.index, name="asof_date")
    return levels.loc[levels.index <= asof]


def scenario_moves(
    levels: pd.DataFrame, columns: Sequence[str], horizon: int, window: int
) -> pd.DataFrame:
    """Relative change over `horizon` rows (trading days with every column
    present), last `window` of them, indexed by the move's last day.
    Overlapping for horizon > 1 (README 9.1's overlapping 10-day moves)."""
    lv = levels[list(columns)].dropna()
    return (lv / lv.shift(horizon) - 1.0).dropna().tail(window)


# ---------------------------------------------------------------------------
# Scenario shocks


@dataclass
class VolProxy:
    """What turns an index move into a surface move for one underlying."""

    index: str
    index_today: float
    base_iv: np.ndarray  # (T_GRID x K_GRID) implied vol on the VaR date, vol points
    beta: np.ndarray  # same shape, vol points per index point

    def node_vol(self, index_return: float) -> NodeVolShock:
        d_index = index_return * self.index_today  # rescaled to today's level
        return NodeVolShock.from_vol_points(T_VALUES, K_VALUES, self.base_iv, self.beta * d_index)


def build_shocks(
    move: pd.Series,
    proxies: dict[str, VolProxy],
    underlyings: Sequence[str],
    single_names: Sequence[str],
    factors: str,
) -> dict[str, MarketShock]:
    shocks: dict[str, MarketShock] = {}
    for u in underlyings:
        spot_pct = float(move[u]) if factors in ("all", "spot") else 0.0
        moves_vol = factors in ("all", "vol") or (
            factors == "single_name_vol" and u in single_names
        )
        node_vol = None
        if moves_vol and u in proxies:
            node_vol = proxies[u].node_vol(float(move[proxies[u].index]))
        shocks[u] = MarketShock(spot_shock_pct=spot_pct, node_vol=node_vol)
    return shocks


# ---------------------------------------------------------------------------
# Full revaluation


@dataclass
class _RevalContext:
    cfg: BaseConfig
    asof: dt.date
    portfolio: Portfolio
    state: MarketState
    mc: MCSettings
    ids: list[str]


_WORKER: dict[str, _RevalContext] = {}
# A process per core already uses every core; numpy's BLAS threads on top only
# oversubscribe them. Spawned workers read these at import, so they are set
# while the pool starts.
_SINGLE_THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
    os.environ.update(_SINGLE_THREAD_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _init_worker(ctx: _RevalContext) -> None:
    _WORKER["ctx"] = ctx


def _values(ctx: _RevalContext, shock: dict[str, MarketShock] | None) -> np.ndarray:
    """(2, n positions): price, and fair strike for variance swaps (NaN otherwise).
    A position the scenario could not price comes back NaN."""
    res = mark_with_state(ctx.cfg, ctx.asof, ctx.portfolio, ctx.state, shock, ctx.mc)
    by_id = {m.position_id: m for m in res.marks}
    out = np.full((2, len(ctx.ids)), np.nan)
    for i, pid in enumerate(ctx.ids):
        m = by_id.get(pid)
        if m is None:
            continue
        out[0, i] = m.price
        if m.fair_strike is not None:
            out[1, i] = m.fair_strike
    return out


def _values_chunk(shocks: list[dict[str, MarketShock]]) -> list[np.ndarray]:
    ctx = _WORKER["ctx"]
    return [_values(ctx, s) for s in shocks]


def revalue(
    ctx: _RevalContext,
    shocks: list[dict[str, MarketShock]],
    max_workers: int | None = None,
) -> np.ndarray:
    """(n scenarios, 2, n positions), scenarios split across a process pool.
    `max_workers=1` runs in-process (tests)."""
    workers = max_workers if max_workers is not None else os.cpu_count() or 1
    if workers <= 1 or len(shocks) <= 1:
        return np.array([_values(ctx, s) for s in shocks])
    n_chunks = min(len(shocks), workers * 4)
    chunks = [list(c) for c in np.array_split(np.array(shocks, dtype=object), n_chunks)]
    with (
        _single_threaded_children(),
        ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(ctx,)
        ) as executor,
    ):
        parts = list(executor.map(_values_chunk, chunks))
    return np.array([v for part in parts for v in part])


def full_pnl(base: np.ndarray, scen: np.ndarray, marks: list[PositionMark]) -> np.ndarray:
    """(n scenarios, n positions) dollar P&L from `revalue` output."""
    pnl = scen[:, 0, :] - base[0][None, :]
    for i, m in enumerate(marks):
        if m.type == "varswap" and m.status != lifecycle.SETTLED:
            k0, k1 = base[1, i], scen[:, 1, i]
            pnl[:, i] = m.vega * (k1**2 - k0**2) / (2.0 * k0)
    return pnl


# ---------------------------------------------------------------------------
# Taylor (sensitivity-based) P&L


def position_vol_move(m: PositionMark, node_vol: NodeVolShock | None) -> float:
    """Vol-point move where the position's vol exposure sits: ATM at each of its
    `vol_times` (an autocallable's observation dates, else its maturity),
    averaged. The P&L explain measures the day's vol move the same way."""
    if node_vol is None:
        return 0.0
    return float(np.mean([node_vol.dsigma_at(0.0, t) for t in (m.vol_times or [m.T])]))


def taylor_pnl(
    marks: list[PositionMark], shocks: dict[str, MarketShock], spot: dict[str, float]
) -> np.ndarray:
    out = np.zeros(len(marks))
    for i, m in enumerate(marks):
        if m.status == lifecycle.SETTLED:
            continue
        shock = shocks.get(m.underlying, MarketShock())
        d_s = spot[m.underlying] * shock.spot_shock_pct
        d_vol_pts = position_vol_move(m, shock.node_vol)
        if m.type == "equity":
            out[i] = m.delta * d_s
        elif m.type == "varswap":
            k = m.fair_strike or 0.0
            out[i] = m.vega * (d_vol_pts + d_vol_pts**2 / (2.0 * k)) if k > 0 else 0.0
        else:
            d_vol = d_vol_pts / 100.0  # vega etc. are per 1.00 of vol
            delta = m.delta_sm if m.delta_sm is not None else m.delta
            gamma = m.gamma_sm if m.gamma_sm is not None else m.gamma
            out[i] = (
                delta * d_s
                + 0.5 * gamma * d_s**2
                + m.vega * d_vol
                + m.vanna * d_s * d_vol
                + 0.5 * m.volga * d_vol**2
            )
    return out


# ---------------------------------------------------------------------------
# Risk measures


def tail_count(n: int, confidence: float) -> int:
    return max(1, int(np.floor(n * (1.0 - confidence) + 1e-9)))


def var_es(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """(VaR, ES) as positive loss numbers; see the module docstring."""
    pnl = np.sort(np.asarray(pnl, dtype=float))
    k = tail_count(len(pnl), confidence)
    return float(-pnl[k - 1]), float(-pnl[:k].mean())


def liquidity_horizon_es(es_by_horizon: Sequence[float]) -> float:
    """README 9.3: ES = sqrt(ES1^2 + sum_j (ES_j sqrt((LH_j - LH_j-1) / 10))^2),
    ES_j the 10-day ES with only factors of liquidity horizon >= LH_j moving."""
    total = es_by_horizon[0] ** 2
    for j in range(1, len(LIQUIDITY_HORIZONS)):
        scale = (LIQUIDITY_HORIZONS[j] - LIQUIDITY_HORIZONS[j - 1]) / TEN_DAY
        total += es_by_horizon[j] ** 2 * scale
    return float(np.sqrt(total))


def liquidity_horizon_pnl(
    pnl_by_factors: dict[str, np.ndarray], marks: list[PositionMark]
) -> list[np.ndarray]:
    """Book P&L per scenario with only the factors of liquidity horizon >= each
    LH moving. LH 10: everything. LH 20: single-name vol for ordinary positions,
    all vol for the exotics. LH 60: the exotics' vol only."""
    exotic = np.array([m.type in EXOTIC_TYPES for m in marks])
    lh10 = pnl_by_factors["all"].sum(axis=1)
    lh20 = pnl_by_factors["single_name_vol"][:, ~exotic].sum(axis=1) + pnl_by_factors["vol"][
        :, exotic
    ].sum(axis=1)
    lh60 = pnl_by_factors["vol"][:, exotic].sum(axis=1)
    return [lh10, lh20, lh60]


# ---------------------------------------------------------------------------
# Inputs from the store


def _stored(curated_root: Path, table: str, where: str) -> pd.DataFrame:
    root = curated_root / table
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame()
    df = store.query(f"SELECT * FROM t WHERE {where}", views={"t": str(root)}).to_pandas()
    if not df.empty:
        df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    return df


def fit_proxies(
    cfg: BaseConfig, asof: dt.date, underlyings: Sequence[str], levels: pd.DataFrame
) -> tuple[dict[str, ProxyFit], dict[str, str]]:
    """Proxy fits on stored days up to `asof`; reasons for any that fail."""
    curated_root = Path(cfg.paths.curated)
    fits: dict[str, ProxyFit] = {}
    skipped: dict[str, str] = {}
    for u in underlyings:
        where = f"underlying = '{u}' AND asof_date <= DATE '{asof.isoformat()}'"
        surface = _stored(curated_root, "vol_surface", where)
        rf = _stored(curated_root, "risk_factors", where)
        index = VOL_INDEX_FOR.get(u)
        if surface.empty or rf.empty or index is None or index not in levels:
            skipped[u] = "no stored surfaces or vol index history"
            continue
        fit = proxy_for_underlying(u, surface, rf, levels[index].dropna(), cfg.calendar)
        if isinstance(fit, str):
            skipped[u] = fit
        else:
            fits[u] = fit
    return fits, skipped


def base_iv_grid(cfg: BaseConfig, asof: dt.date, underlying: str) -> np.ndarray | None:
    """Implied vol (vol points) on the (T_GRID x K_GRID) rectangle on `asof`."""
    rf = _stored(
        Path(cfg.paths.curated),
        "risk_factors",
        f"underlying = '{underlying}' AND asof_date = DATE '{asof.isoformat()}'",
    )
    if rf.empty:
        return None
    rf = rf.assign(iv=np.sqrt(rf["w"] / rf["T"]) * 100.0)
    grid = rf.pivot_table(index="T_label", columns="k", values="iv")
    grid = grid.reindex(index=list(T_GRID), columns=list(K_GRID))
    if grid.isna().to_numpy().any():
        return None
    return grid.to_numpy()


# ---------------------------------------------------------------------------
# Run


@dataclass
class VaRResult:
    asof: dt.date
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_betas: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def value(self, measure: str, method: str, horizon: int = 1, **filters: object) -> float:
        s = self.summary
        sel = (s["measure"] == measure) & (s["method"] == method) & (s["horizon_days"] == horizon)
        defaults = {
            "window_days": PRIMARY_WINDOW,
            "factors": "all",
            "positions": "book",
            "confidence": VAR_CONFIDENCE if measure == "VaR" else None,
        }
        for key, val in {**defaults, **filters}.items():
            if val is not None:
                sel &= s[key] == val
        rows = s.loc[sel, "value"]
        return float(rows.iloc[0]) if len(rows) else float("nan")

    def render(self) -> str:
        lines = [f"VaR / ES — {self.asof}"]
        if not self.summary.empty:
            s = self.summary
            for method in ("full", "taylor"):
                var = self.value("VaR", method)
                es = self.value("ES", method, confidence=VAR_CONFIDENCE)
                lines.append(
                    f"  1-day 99% ({PRIMARY_WINDOW}d, {method}): VaR ${var:,.0f}, ES ${es:,.0f}"
                )
            ten = self.value("VaR", "full", TEN_DAY)
            if not np.isnan(ten):
                lines.append(
                    f"  10-day 99%: overlapping ${ten:,.0f} vs sqrt(10) x 1-day "
                    f"${self.value('VaR', 'sqrt10_full', TEN_DAY):,.0f}"
                )
            lh = s[s["measure"] == "ES_LH"]
            if not lh.empty:
                lines.append(f"  ES 97.5% liquidity-horizon adjusted: ${lh['value'].iloc[0]:,.0f}")
            rt = s[s["measure"] == "runtime_seconds"]
            for _, r in rt.iterrows():
                lines.append(
                    f"  runtime {r['method']}: {r['value']:.1f}s for {r['n_scenarios']} scenarios"
                )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def _summary_rows(
    rows: list[dict[str, object]],
    pnl: np.ndarray,
    method: str,
    horizon: int,
    window: int,
    factors: str,
    positions: str,
    confidences: Sequence[float] = (VAR_CONFIDENCE,),
    measures: Sequence[str] = ("VaR", "ES"),
) -> None:
    for c in confidences:
        var, es = var_es(pnl, c)
        for measure, value in (("VaR", var), ("ES", es)):
            if measure in measures:
                rows.append(
                    {
                        "measure": measure,
                        "method": method,
                        "horizon_days": horizon,
                        "window_days": window,
                        "confidence": c,
                        "factors": factors,
                        "positions": positions,
                        "value": value,
                        "n_scenarios": len(pnl),
                    }
                )


def _scenario_frame(
    moves: pd.DataFrame,
    full: np.ndarray,
    taylor: np.ndarray,
    marks: list[PositionMark],
    horizon: int,
    factors: str,
) -> pd.DataFrame:
    n, p = full.shape
    return pd.DataFrame(
        {
            "horizon_days": horizon,
            "factors": factors,
            "scenario_date": np.repeat(list(moves.index), p),
            "position_id": [m.position_id for m in marks] * n,
            "underlying": [m.underlying for m in marks] * n,
            "pnl_full": full.ravel(),
            "pnl_taylor": taylor.ravel(),
        }
    )


def proxy_error(
    ctx: _RevalContext,
    base: np.ndarray,
    marks: list[PositionMark],
    fits: dict[str, ProxyFit],
    proxies: dict[str, VolProxy],
    levels: pd.DataFrame,
    max_workers: int | None,
) -> pd.DataFrame:
    """Today's book under each stored day's real spot moves and either the real
    per-node vol change or the index-implied one (betas x that day's index
    change, not rescaled: the same day on both sides). The gap is what the proxy
    misses (README 8.3's proxy error in dollars)."""
    dates = sorted(set.intersection(*(set(f.changes.index) for f in fits.values())))
    spot_ret = levels[list(proxies)].dropna().pct_change()
    dates = [d for d in dates if d in spot_ret.index and spot_ret.loc[d].notna().all()]
    true_shocks, proxy_shocks = [], []
    for d in dates:
        true_s, proxy_s = {}, {}
        for u, fit in fits.items():
            px = proxies[u]
            true_field = node_field(fit.nodes, fit.changes.loc[d].to_numpy())
            proxy_field = px.beta * float(fit.index_change.loc[d])
            spot_pct = float(spot_ret.loc[d, u])
            true_s[u] = MarketShock(
                spot_shock_pct=spot_pct,
                node_vol=NodeVolShock.from_vol_points(T_VALUES, K_VALUES, px.base_iv, true_field),
            )
            proxy_s[u] = MarketShock(
                spot_shock_pct=spot_pct,
                node_vol=NodeVolShock.from_vol_points(T_VALUES, K_VALUES, px.base_iv, proxy_field),
            )
        true_shocks.append(true_s)
        proxy_shocks.append(proxy_s)
    if not dates:
        return pd.DataFrame()
    values = revalue(ctx, true_shocks + proxy_shocks, max_workers)
    pnl = full_pnl(base, values, marks)
    n = len(dates)
    return pd.DataFrame(
        {
            "change_date": np.repeat(dates, len(marks)),
            "position_id": [m.position_id for m in marks] * n,
            "underlying": [m.underlying for m in marks] * n,
            "pnl_true": pnl[:n].ravel(),
            "pnl_proxy": pnl[n:].ravel(),
        }
    )


def run_var(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio_path: str,
    full: bool = True,
    levels: pd.DataFrame | None = None,
    max_workers: int | None = None,
    persist: bool = True,
) -> VaRResult:
    """`full=False` (the backfill of past days) runs only the 1-day 500-day
    window, both methods. `levels` lets a backfill fetch history once."""
    result = VaRResult(asof=asof)
    portfolio = Portfolio.from_yaml(portfolio_path)
    underlyings = sorted(
        {p.underlying for p in portfolio.positions if lifecycle.is_active(p, asof)}
    )
    single_names = [u for u in underlyings if u in cfg.universe.single_names]

    if levels is None:
        n_days = (max(WINDOWS) if full else PRIMARY_WINDOW) + TEN_DAY
        start = asof - dt.timedelta(days=int(n_days * 365 / 252) + HISTORY_PADDING_DAYS)
        levels = load_factor_history(underlyings, start, asof)
    levels = levels.loc[levels.index <= asof]

    fits, skipped = fit_proxies(cfg, asof, underlyings, levels)
    result.skipped.update({f"proxy {u}": r for u, r in skipped.items()})
    if skipped:
        # A name with no vol move would silently understate VaR: refuse instead.
        result.skipped["_all_"] = "no vol proxy for " + ", ".join(sorted(skipped))
        return result
    proxies: dict[str, VolProxy] = {}
    for u, fit in fits.items():
        base_iv = base_iv_grid(cfg, asof, u)
        today = levels[fit.index].get(asof, np.nan)
        if base_iv is None or np.isnan(today):
            result.skipped["_all_"] = f"no risk-factor grid or {fit.index} close for {u} on {asof}"
            return result
        proxies[u] = VolProxy(fit.index, float(today), base_iv, fit.beta_field())
    if asof not in levels.index or levels.loc[asof, underlyings].isna().any():
        result.skipped["_all_"] = f"no spot close for every name on {asof} in the history"
        return result
    state = load_market_state(cfg, asof, portfolio)
    if state is None:
        result.skipped["_all_"] = "no curated rates available"
        return result

    # Greeks for the Taylor method: the day's own mark at the daily settings.
    daily = mark_with_state(cfg, asof, portfolio, state)
    if daily.skipped:
        # A position left out would silently understate VaR: refuse instead.
        result.skipped["_all_"] = "positions not priced: " + ", ".join(sorted(daily.skipped))
        return result
    marks = sorted(daily.marks, key=lambda m: m.position_id)
    ids = [m.position_id for m in marks]
    ctx = _RevalContext(cfg, asof, portfolio, state, VAR_MC_SETTINGS, ids)
    base = _values(ctx, None)
    if np.isnan(base[0]).any():
        result.skipped["_all_"] = "a position could not be priced at the VaR settings"
        return result
    is_varswap = np.array([m.type == "varswap" for m in marks])
    columns = underlyings + [proxies[u].index for u in underlyings]

    summary: list[dict[str, object]] = []
    scenario_frames: list[pd.DataFrame] = []

    def run_set(
        horizon: int, window: int, factors: str
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
        moves = scenario_moves(levels, columns, horizon, window)
        shocks = [
            build_shocks(row, proxies, underlyings, single_names, factors)
            for _, row in moves.iterrows()
        ]
        t0 = time.perf_counter()
        values = revalue(ctx, shocks, max_workers)
        elapsed = time.perf_counter() - t0
        pnl_full = full_pnl(base, values, marks)
        pnl_taylor = np.array([taylor_pnl(marks, s, state.spot) for s in shocks])
        bad = np.isnan(pnl_full).any(axis=1)
        if bad.any():
            result.notes.append(
                f"{int(bad.sum())} of {len(bad)} {horizon}-day '{factors}' scenarios dropped: "
                "a position could not be priced under them"
            )
        keep = ~bad
        scenario_frames.append(
            _scenario_frame(moves[keep], pnl_full[keep], pnl_taylor[keep], marks, horizon, factors)
        )
        return moves[keep], pnl_full[keep], pnl_taylor[keep], elapsed

    # 1-day, all factors: the longest window once, shorter windows as its tail.
    one_window = max(WINDOWS) if full else PRIMARY_WINDOW
    _, f1, t1, elapsed = run_set(1, one_window, "all")
    for window in WINDOWS if full else (PRIMARY_WINDOW,):
        if len(f1) < window:
            result.notes.append(f"only {len(f1)} 1-day scenarios for the {window}-day window")
            continue
        for method, pnl in (("full", f1[-window:]), ("taylor", t1[-window:])):
            confs = (VAR_CONFIDENCE, ES_CONFIDENCE)
            _summary_rows(summary, pnl.sum(axis=1), method, 1, window, "all", "book", confs)
            _summary_rows(
                summary,
                pnl[:, ~is_varswap].sum(axis=1),
                method,
                1,
                window,
                "all",
                "book_ex_varswap",
                confs,
            )
            sqrt10 = np.sqrt(TEN_DAY) * pnl.sum(axis=1)
            _summary_rows(summary, sqrt10, f"sqrt10_{method}", TEN_DAY, window, "all", "book")
    n_primary = min(PRIMARY_WINDOW, len(f1))
    taylor_seconds = _time_taylor(marks, levels, columns, proxies, underlyings, single_names, state)
    summary.append(
        {
            "measure": "runtime_seconds",
            "method": "full",
            "horizon_days": 1,
            "window_days": n_primary,
            "confidence": None,
            "factors": "all",
            "positions": "book",
            "value": elapsed * n_primary / max(len(f1), 1),
            "n_scenarios": n_primary,
        }
    )
    summary.append(
        {
            "measure": "runtime_seconds",
            "method": "taylor",
            "horizon_days": 1,
            "window_days": n_primary,
            "confidence": None,
            "factors": "all",
            "positions": "book",
            "value": taylor_seconds,
            "n_scenarios": n_primary,
        }
    )

    if full:
        ten: dict[str, np.ndarray] = {}
        for factors in FACTOR_SETS:
            _, f10, t10, _ = run_set(TEN_DAY, PRIMARY_WINDOW, factors)
            ten[factors] = f10
            confs = (VAR_CONFIDENCE, ES_CONFIDENCE)
            for method, pnl in (("full", f10), ("taylor", t10)):
                _summary_rows(
                    summary, pnl.sum(axis=1), method, TEN_DAY, len(pnl), factors, "book", confs
                )
                if factors == "all":
                    _summary_rows(
                        summary,
                        pnl[:, ~is_varswap].sum(axis=1),
                        method,
                        TEN_DAY,
                        len(pnl),
                        factors,
                        "book_ex_varswap",
                        confs,
                    )
        if len({len(v) for v in ten.values()}) == 1:
            es_lh = []
            for lh, pnl in zip(LIQUIDITY_HORIZONS, liquidity_horizon_pnl(ten, marks), strict=True):
                _, es = var_es(pnl, ES_CONFIDENCE)
                es_lh.append(es)
                summary.append(
                    {
                        "measure": "ES",
                        "method": "full",
                        "horizon_days": TEN_DAY,
                        "window_days": len(pnl),
                        "confidence": ES_CONFIDENCE,
                        "factors": f"LH{lh}",
                        "positions": "book",
                        "value": es,
                        "n_scenarios": len(pnl),
                    }
                )
            summary.append(
                {
                    "measure": "ES_LH",
                    "method": "full",
                    "horizon_days": TEN_DAY,
                    "window_days": len(ten["all"]),
                    "confidence": ES_CONFIDENCE,
                    "factors": "all",
                    "positions": "book",
                    "value": liquidity_horizon_es(es_lh),
                    "n_scenarios": len(ten["all"]),
                }
            )
        result.proxy_errors = proxy_error(ctx, base, marks, fits, proxies, levels, max_workers)

    result.summary = pd.DataFrame(summary).assign(asof_date=asof)
    result.scenarios = pd.concat(scenario_frames, ignore_index=True).assign(asof_date=asof)
    result.proxy_betas = _beta_frame(fits).assign(asof_date=asof)
    if persist:
        _persist(result, Path(cfg.paths.curated))
    return result


def _time_taylor(
    marks: list[PositionMark],
    levels: pd.DataFrame,
    columns: list[str],
    proxies: dict[str, VolProxy],
    underlyings: list[str],
    single_names: list[str],
    state: MarketState,
) -> float:
    """Wall time of the Taylor method alone over the primary window, shock
    construction included, for the README 9.2 runtime comparison."""
    moves = scenario_moves(levels, columns, 1, PRIMARY_WINDOW)
    t0 = time.perf_counter()
    for _, row in moves.iterrows():
        taylor_pnl(marks, build_shocks(row, proxies, underlyings, single_names, "all"), state.spot)
    return time.perf_counter() - t0


def _beta_frame(fits: dict[str, ProxyFit]) -> pd.DataFrame:
    rows = []
    for u, fit in fits.items():
        for (t_label, k), a, b, r2 in zip(fit.nodes, fit.alpha, fit.beta, fit.r2, strict=True):
            rows.append(
                {
                    "underlying": u,
                    "index": fit.index,
                    "T_label": t_label,
                    "T": T_GRID[t_label],
                    "k": k,
                    "alpha": float(a),
                    "beta": float(b),
                    "r2": float(r2),
                    "n_changes": len(fit.changes),
                    "coverage": float(fit.coverage[(t_label, k)]),
                }
            )
    return pd.DataFrame(rows)


def _persist(result: VaRResult, curated_root: Path) -> None:
    for df, schema, required, name, partition in (
        (
            result.summary,
            VAR_SUMMARY_SCHEMA,
            VAR_SUMMARY_REQUIRED_NOT_NULL,
            "var_summary",
            ["asof_date"],
        ),
        (
            result.scenarios,
            VAR_SCENARIO_SCHEMA,
            VAR_SCENARIO_REQUIRED_NOT_NULL,
            "var_scenarios",
            ["asof_date"],
        ),
        (
            result.proxy_betas,
            PROXY_BETA_SCHEMA,
            PROXY_BETA_REQUIRED_NOT_NULL,
            "proxy_betas",
            ["asof_date", "underlying"],
        ),
        (
            result.proxy_errors.assign(asof_date=result.asof)
            if not result.proxy_errors.empty
            else result.proxy_errors,
            PROXY_ERROR_SCHEMA,
            PROXY_ERROR_REQUIRED_NOT_NULL,
            "proxy_errors",
            ["asof_date"],
        ),
    ):
        if df.empty:
            continue
        store.write_partitioned(validate(df, schema, required), curated_root / name, partition)


def stored_var_dates(cfg: BaseConfig) -> set[dt.date]:
    return set(_stored(Path(cfg.paths.curated), "var_summary", "TRUE").get("asof_date", []))


def run_var_backfill(
    cfg: BaseConfig,
    start: dt.date,
    end: dt.date,
    portfolio_path: str,
    force: bool = False,
    max_workers: int | None = None,
) -> list[VaRResult]:
    """The 1-day 500-day VaR (`full=False`) on every stored day in [start, end]
    with a calibrated surface for each of the book's names, oldest first. Days
    already in `var_summary` are left alone unless `force`: writing a partial
    run over a day's full run would replace it."""
    names = sorted({p.underlying for p in Portfolio.from_yaml(portfolio_path).positions})
    surfaces = _stored(
        Path(cfg.paths.curated),
        "vol_surface",
        f"asof_date BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'",
    )
    if surfaces.empty:
        return []
    per_day = surfaces[surfaces["underlying"].isin(names)].groupby("asof_date")["underlying"]
    dates = sorted(
        d for d, u in per_day if set(u) == set(names) and is_trading_day(d, cfg.calendar)
    )
    if not force:
        done = stored_var_dates(cfg)
        dates = [d for d in dates if d not in done]
    if not dates:
        return []
    n_days = PRIMARY_WINDOW + TEN_DAY
    history_start = dates[0] - dt.timedelta(days=int(n_days * 365 / 252) + HISTORY_PADDING_DAYS)
    levels = load_factor_history(names, history_start, dates[-1])
    return [
        run_var(cfg, d, portfolio_path, full=False, levels=levels, max_workers=max_workers)
        for d in dates
    ]
