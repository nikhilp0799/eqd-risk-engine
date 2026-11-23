"""Quote noise of the Monte Carlo exotics' marks (planning/p008_noise_band_plan.md).

An option quote only pins the price to within its bid-ask spread, so a mark
built from a calibrated surface is uncertain by however much the surface moves
when every quote moves inside its spread. This module measures that directly:

1. move each OK implied vol by u x its half-spread in vol, u ~ U(-1, 1)
   independently per quote (half-spread in vol = (ask - bid) / 2 / vega);
2. refit the name's whole surface with the production `calibrate_underlying`;
3. rebuild the local-vol grid and reprice the name's barrier / autocall
   positions on ONE fixed Monte Carlo seed, so path noise cancels between draws.

The SD of those marks is the position's quote noise for the day. The P&L explain
adds both days' quote noise to the Monte Carlo SE before calling a residual an
alert (pricing/pnl_explain.py `_classify`). The band comes from quotes only and
is never fitted to residuals.

Measured on 2026-11-11/17/20 (30 draws): P008 (NVDA autocallable) $2.1K-4.6K a
day, against a $6.0K residual RMS over 37 explain pairs; the SVI/SSVI choice
flipped in up to 80% of draws, so the model choice sits inside the spread.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import QUOTE_NOISE_REQUIRED_NOT_NULL, QUOTE_NOISE_SCHEMA, validate
from eqdrisk.portfolio import lifecycle
from eqdrisk.portfolio.mark import (
    MarketState,
    MCSettings,
    build_grids_for_state,
    load_market_state,
    mark_with_state,
)
from eqdrisk.portfolio.schema import AutocallPosition, BarrierPosition, Portfolio, Position
from eqdrisk.vol.surface import calibrate_underlying, slice_rows

# 30 draws: the SD is known to about +-13% (user decision, 2026-11-22). Each draw
# is a full refit plus a local-vol grid, about 4 s per name.
QUOTE_NOISE_DRAWS = 30
QUOTE_NOISE_SEED = 20261122
# Floor on a perturbed implied vol, so a wide spread on a cheap wing option
# cannot push it to zero or below.
MIN_PERTURBED_IV = 0.01
# Prices only on the Monte Carlo default seed: one run per draw, no Greeks.
QUOTE_NOISE_MC = MCSettings(greeks=False, exotic_theta=False, n_batches=1)


@dataclass
class QuoteNoiseRow:
    position_id: str
    underlying: str
    sd: float
    n_draws: int
    half_spread_vol_points: float
    model_flip_share: float


@dataclass
class QuoteNoiseResult:
    asof: dt.date
    rows: list[QuoteNoiseRow] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"quote noise {self.asof}: {len(self.rows)} MC position(s) measured"]
        for r in self.rows:
            lines.append(
                f"  {r.position_id} ({r.underlying}): mark SD {r.sd:,.0f} over {r.n_draws} "
                f"refits, median half-spread {r.half_spread_vol_points:.2f} vol pts, "
                f"surface model differs from stored in {100 * r.model_flip_share:.0f}% of draws"
            )
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def half_spread_vol(quotes: pd.DataFrame) -> np.ndarray:
    """Half the bid-ask spread in vol units, per quote. A quote with no usable
    spread or vega takes the median of the others (0 if none has one)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        hs = (quotes["ask"] - quotes["bid"]).to_numpy(float) / 2 / quotes["vega"].to_numpy(float)
    ok = np.isfinite(hs) & (hs >= 0)
    fill = float(np.median(hs[ok])) if ok.any() else 0.0
    return np.where(ok, hs, fill)


def perturb_quotes(quotes: pd.DataFrame, hs: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """`quotes` with each implied vol moved uniformly inside its half-spread and
    total variance recomputed to match."""
    out = quotes.copy()
    iv = out["iv"].to_numpy(float) + rng.uniform(-1.0, 1.0, len(out)) * hs
    out["iv"] = np.maximum(iv, MIN_PERTURBED_IV)
    out["total_variance"] = out["iv"] ** 2 * out["T"]
    return out


def refit_surface(underlying: str, quotes: pd.DataFrame, asof: dt.date) -> pd.DataFrame:
    """The surface `load_market_state` would read had `quotes` been calibrated."""
    groups = {pd.Timestamp(e).date(): df for e, df in quotes.groupby("expiry")}
    slices, _ = calibrate_underlying(underlying, groups)
    return pd.DataFrame(slice_rows(slices, asof))


def surface_model(surface: pd.DataFrame) -> str:
    """'SSVI' when the whole-surface fallback fired, else 'SVI'."""
    return "SSVI" if (surface["model"] == "SSVI").any() else "SVI"


def _load_quotes(curated_root: Path, asof: dt.date, underlying: str) -> pd.DataFrame:
    """OK implied vols with their chain bid and ask (the last snapshot of the day)."""
    day = asof.isoformat()
    return store.query(
        f"""
        WITH c AS (
            SELECT expiry, strike, cp, arg_max(bid, asof_ts) AS bid, arg_max(ask, asof_ts) AS ask
            FROM ch WHERE asof_date = DATE '{day}' AND underlying = '{underlying}'
            GROUP BY expiry, strike, cp
        )
        SELECT iv.*, c.bid, c.ask FROM iv LEFT JOIN c USING (expiry, strike, cp)
        WHERE iv.asof_date = DATE '{day}' AND iv.underlying = '{underlying}' AND iv.reason = 'OK'
        """,
        views={"iv": str(curated_root / "implied_vols"), "ch": str(curated_root / "chains")},
    ).to_pandas()


def _prices(
    cfg: BaseConfig, asof: dt.date, portfolio: Portfolio, state: MarketState
) -> dict[str, float]:
    res = mark_with_state(cfg, asof, portfolio, state, mc_settings=QUOTE_NOISE_MC)
    return {m.position_id: m.price for m in res.marks if m.status != lifecycle.SETTLED}


def measure_underlying(
    cfg: BaseConfig,
    asof: dt.date,
    positions: Portfolio,
    state: MarketState,
    quotes: pd.DataFrame,
    n_draws: int = QUOTE_NOISE_DRAWS,
    seed: int = QUOTE_NOISE_SEED,
) -> list[QuoteNoiseRow]:
    """Quote noise of `positions` (all on one underlying) from `n_draws` refits."""
    u = positions.positions[0].underlying
    hs = half_spread_vol(quotes)
    stored_model = surface_model(state.surface[u])
    rng = np.random.default_rng([seed, asof.toordinal()])
    draws: list[dict[str, float]] = []
    flips = 0
    for _ in range(n_draws):
        surf = refit_surface(u, perturb_quotes(quotes, hs, rng), asof)
        flips += surface_model(surf) != stored_model
        surfaces = {**state.surface, u: surf}
        grids = build_grids_for_state(
            cfg, asof, positions, state.spot, surfaces, state.forward_curve
        )
        drawn = dataclasses.replace(
            state, surface=surfaces, grids={**state.grids, **grids}, w_slices={}
        )
        draws.append(_prices(cfg, asof, positions, drawn))
    ids = set.intersection(*(set(d) for d in draws)) if draws else set()
    return [
        QuoteNoiseRow(
            position_id=pid,
            underlying=u,
            sd=float(np.std([d[pid] for d in draws], ddof=1)),
            n_draws=n_draws,
            half_spread_vol_points=100.0 * float(np.median(hs)),
            model_flip_share=flips / n_draws,
        )
        for pid in sorted(ids)
    ]


def run_quote_noise(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio_path: str,
    n_draws: int = QUOTE_NOISE_DRAWS,
) -> QuoteNoiseResult:
    """Measure and persist the quote noise of every live MC-priced position."""
    result = QuoteNoiseResult(asof=asof)
    portfolio = Portfolio.from_yaml(portfolio_path)
    mc: list[Position] = [
        p
        for p in portfolio.positions
        if isinstance(p, (BarrierPosition, AutocallPosition)) and lifecycle.is_active(p, asof)
    ]
    if not mc:
        return result
    state = load_market_state(cfg, asof, Portfolio(positions=mc))
    if state is None:
        result.skipped["_all_"] = "no curated rates for this date"
        return result
    curated_root = Path(cfg.paths.curated)
    for u in sorted({p.underlying for p in mc}):
        if u not in state.surface or u not in state.grids:
            result.skipped[u] = "no stored surface or local-vol grid"
            continue
        quotes = _load_quotes(curated_root, asof, u)
        if quotes.empty:
            result.skipped[u] = "no OK implied vols"
            continue
        positions = Portfolio(positions=[p for p in mc if p.underlying == u])
        result.rows.extend(measure_underlying(cfg, asof, positions, state, quotes, n_draws))
    _persist(result, curated_root)
    return result


def _persist(result: QuoteNoiseResult, curated_root: Path) -> None:
    if not result.rows:
        return
    df = pd.DataFrame(
        [
            {
                "asof_date": result.asof,
                "position_id": r.position_id,
                "underlying": r.underlying,
                "quote_noise_sd": r.sd,
                "n_draws": r.n_draws,
                "half_spread_vol_points": r.half_spread_vol_points,
                "model_flip_share": r.model_flip_share,
            }
            for r in result.rows
        ]
    )
    table = validate(df, QUOTE_NOISE_SCHEMA, QUOTE_NOISE_REQUIRED_NOT_NULL)
    store.write_partitioned(table, curated_root / "quote_noise", ["asof_date"])


def load_quote_noise(curated_root: Path, day: dt.date) -> dict[str, float]:
    """Stored quote-noise SD by position for `day`; empty when not measured."""
    root = curated_root / "quote_noise"
    if not root.exists() or not any(root.rglob("*.parquet")):
        return {}
    df = store.query(
        f"SELECT position_id, quote_noise_sd FROM qn WHERE asof_date = DATE '{day.isoformat()}'",
        views={"qn": str(root)},
    ).to_pandas()
    return dict(zip(df["position_id"], df["quote_noise_sd"].astype(float), strict=True))
