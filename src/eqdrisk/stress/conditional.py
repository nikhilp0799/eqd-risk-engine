"""Conditional stress (README 11.3): an SPX shock with the rest of the market
moving the way it has in past sell-offs, side by side with the naive version.

**The co-movement.** Each other factor's 10-day log move (AAPL, NVDA, VIX,
VXAPL, VXN) is regressed, with an intercept, on SPX's 10-day log move over the
sell-off periods since 2010-06-01: the worst 10% of SPX 10-day moves. 10-day
moves because a -20% shock is 3-4x the worst single day of recent years and
daily betas extrapolate badly that far (VIX at SPX -20%: +215% from a linear
daily fit, +450% from a log one); 10-day moves since 2010 reach real crash sizes
(worst -23%). Sell-offs only because vol reacts more in them (1-day VIX beta
-8.6 in sell-offs against -4.0 in rallies over the last 1,000 days). Checked on
real episodes (fit on the whole sample): COVID 2020, SPX -34%, VIX +328% actual
against +364% fitted; the 2022 grind, -25%, +102% against +206%; April 2025,
-19%, +243% against +129%. A fair average; the speed of the sell-off matters.

**From index to surface.** A vol index's move is mapped onto every grid node by
the Step 8.3 betas, so skew and term structure come from those: downside nodes
move more than ATM, short maturities more than long. Those betas were fitted on
calm days, and in sell-offs the term structure moves more than they say:
3-month VIX rises 0.75 points per VIX point in 10-day sell-offs against 0.57 in
the betas' 3m/1m ratio. The "conditional_long_end" variant scales every node
beta beyond 1m by that ratio of ratios, to show what the gap is worth.

**The naive version** is 11.1's convention: every name moves like SPX, and every
vol on every surface by VIX's relative move (a parallel relative bump).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    CONDITIONAL_STRESS_MOVE_REQUIRED_NOT_NULL,
    CONDITIONAL_STRESS_MOVE_SCHEMA,
    CONDITIONAL_STRESS_REQUIRED_NOT_NULL,
    CONDITIONAL_STRESS_SCHEMA,
    CRASH_BETA_REQUIRED_NOT_NULL,
    CRASH_BETA_SCHEMA,
    validate,
)
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.risk.proxy import VOL_INDEX_FOR
from eqdrisk.risk.var import (
    T_VALUES,
    VolProxy,
    active_underlyings,
    full_pnl,
    load_factor_history,
    prepare_revaluation,
    revalue,
)
from eqdrisk.stress.shock import MarketShock
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

INDEX_UNDERLYING = "SPX"
SPOT_SHOCKS: tuple[float, ...] = (-0.05, -0.10, -0.20, -0.30)
HORIZON_DAYS = 10
SELL_OFF_QUANTILE = 0.10
HISTORY_START = dt.date(2010, 6, 1)  # the first day all six series have a close
TERM_INDEX = "VXV"  # 3-month SPX vol
VARIANTS = ("conditional", "naive", "conditional_long_end")
# Surface points whose vol move is reported, (T_label, k).
REPORTED_NODES: tuple[tuple[str, float], ...] = (("1m", 0.0), ("1m", -0.1), ("1y", 0.0))


@dataclass
class CrashFit:
    beta: float
    alpha: float
    r2: float
    n: int

    def move(self, spx_log_move: float) -> float:
        """Relative move at an SPX log move."""
        return float(np.expm1(self.beta * spx_log_move + self.alpha))


def log_moves(levels: pd.DataFrame, columns: list[str], horizon: int) -> pd.DataFrame:
    lv = levels[columns].dropna()
    return np.log(lv / lv.shift(horizon)).dropna()


def fit_crash_response(
    levels: pd.DataFrame,
    factors: list[str],
    horizon: int = HORIZON_DAYS,
    quantile: float = SELL_OFF_QUANTILE,
) -> dict[str, CrashFit]:
    """Each factor's log move on SPX's over the worst `quantile` of SPX moves."""
    moves = log_moves(levels, [INDEX_UNDERLYING, *factors], horizon)
    x = moves[INDEX_UNDERLYING]
    sell = x <= x.quantile(quantile)
    fits = {}
    for f in factors:
        y = moves.loc[sell, f]
        beta, alpha = np.polyfit(x[sell], y, 1)
        r2 = float(np.corrcoef(x[sell], y)[0, 1] ** 2)
        fits[f] = CrashFit(float(beta), float(alpha), r2, int(sell.sum()))
    return fits


def crash_term_ratio(
    levels: pd.DataFrame, horizon: int = HORIZON_DAYS, quantile: float = SELL_OFF_QUANTILE
) -> float:
    """3-month VIX change per VIX point, in vol points, over SPX sell-off periods."""
    lv = levels[[INDEX_UNDERLYING, "VIX", TERM_INDEX]].dropna()
    points = (lv[["VIX", TERM_INDEX]] - lv[["VIX", TERM_INDEX]].shift(horizon)).dropna()
    x = np.log(lv[INDEX_UNDERLYING] / lv[INDEX_UNDERLYING].shift(horizon)).loc[points.index]
    sell = x <= x.quantile(quantile)
    return float(np.polyfit(points.loc[sell, "VIX"], points.loc[sell, TERM_INDEX], 1)[0])


def calm_term_ratio(proxy: VolProxy) -> float:
    """The 8.3 betas' 3m over 1m ATM response, the same quantity on calm days."""
    t = list(T_GRID)
    k0 = list(K_GRID).index(0.0)
    return float(proxy.beta[t.index("3m"), k0] / proxy.beta[t.index("1m"), k0])


def scale_long_end(proxy: VolProxy, scale: float) -> VolProxy:
    beta = proxy.beta.copy()
    beta[T_VALUES > T_GRID["1m"]] *= scale
    return VolProxy(proxy.index, proxy.index_today, proxy.base_iv, beta)


def conditional_moves(fits: dict[str, CrashFit], spx_shock: float) -> dict[str, float]:
    moves = {INDEX_UNDERLYING: spx_shock}
    moves.update({f: fit.move(float(np.log1p(spx_shock))) for f, fit in fits.items()})
    return moves


def _node_vol_move(shock: MarketShock, base_iv: np.ndarray, t_label: str, k: float) -> float:
    if shock.node_vol is not None:
        return shock.node_vol.dsigma_at(k, T_GRID[t_label])
    iv = float(base_iv[list(T_GRID).index(t_label), list(K_GRID).index(k)])
    return iv * shock.vol_shock_pct  # a parallel relative bump


@dataclass
class ConditionalStressResult:
    asof: dt.date
    pnl: pd.DataFrame = field(default_factory=pd.DataFrame)  # per shock, variant, position
    moves: pd.DataFrame = field(default_factory=pd.DataFrame)
    betas: pd.DataFrame = field(default_factory=pd.DataFrame)
    term_ratio_crash: float | None = None
    term_ratio_calm: float | None = None
    skipped: dict[str, str] = field(default_factory=dict)

    def book(self, ex_varswap: bool = False) -> pd.DataFrame:
        """Book P&L, rows = SPX shock, columns = variant."""
        df = self.pnl[self.pnl["type"] != "varswap"] if ex_varswap else self.pnl
        return df.pivot_table(
            index="spot_shock_pct", columns="variant", values="pnl", aggfunc="sum"
        ).sort_index(ascending=False)

    def render(self) -> str:
        lines = [f"Conditional stress — {self.asof}"]
        if not self.pnl.empty:
            for label, ex in (("book", False), ("book without the variance swap", True)):
                lines.append(f"  P&L, {label}:")
                table = self.book(ex)
                lines.append("    " + "SPX".rjust(6) + "".join(v.rjust(22) for v in VARIANTS))
                for shock, row in table.iterrows():
                    cells = "".join(f"{row[v]:,.0f}".rjust(22) for v in VARIANTS)
                    lines.append(f"    {shock:+.0%}".rjust(10) + cells)
            lines.append(
                f"  3-month VIX per VIX point: {self.term_ratio_crash:.2f} in sell-offs, "
                f"{self.term_ratio_calm:.2f} in the 8.3 betas"
            )
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def run_conditional_stress(
    cfg: BaseConfig,
    asof: dt.date,
    portfolio_path: str,
    levels: pd.DataFrame | None = None,
    max_workers: int | None = None,
    persist: bool = True,
) -> ConditionalStressResult:
    result = ConditionalStressResult(asof=asof)
    portfolio = Portfolio.from_yaml(portfolio_path)
    if levels is None:
        levels = load_factor_history(
            active_underlyings(portfolio, asof), HISTORY_START, asof, extra_indices=[TERM_INDEX]
        )
    levels = levels.loc[levels.index <= asof]
    prep = prepare_revaluation(cfg, asof, portfolio, levels)
    if isinstance(prep, str):
        result.skipped["_all_"] = prep
        return result
    if INDEX_UNDERLYING not in prep.proxies:
        result.skipped["_all_"] = f"the book has no {INDEX_UNDERLYING} position to shock"
        return result

    others = [u for u in prep.underlyings if u != INDEX_UNDERLYING]
    factors = others + [VOL_INDEX_FOR[u] for u in prep.underlyings]
    fits = fit_crash_response(levels, factors)
    result.term_ratio_crash = crash_term_ratio(levels)
    result.term_ratio_calm = calm_term_ratio(prep.proxies[INDEX_UNDERLYING])
    long_end = {
        u: scale_long_end(p, result.term_ratio_crash / result.term_ratio_calm)
        for u, p in prep.proxies.items()
    }

    shocks: list[dict[str, MarketShock]] = []
    labels: list[tuple[float, str]] = []
    move_rows: list[dict[str, object]] = []
    for spx_shock in SPOT_SHOCKS:
        moves = conditional_moves(fits, spx_shock)
        vix_move = moves[VOL_INDEX_FOR[INDEX_UNDERLYING]]
        variant_shocks = {
            "conditional": {
                u: MarketShock(
                    spot_shock_pct=moves[u],
                    node_vol=prep.proxies[u].node_vol(moves[VOL_INDEX_FOR[u]]),
                )
                for u in prep.underlyings
            },
            "naive": {
                u: MarketShock(spot_shock_pct=spx_shock, vol_shock_pct=vix_move)
                for u in prep.underlyings
            },
            "conditional_long_end": {
                u: MarketShock(
                    spot_shock_pct=moves[u],
                    node_vol=long_end[u].node_vol(moves[VOL_INDEX_FOR[u]]),
                )
                for u in prep.underlyings
            },
        }
        for variant in VARIANTS:
            shock = variant_shocks[variant]
            shocks.append(shock)
            labels.append((spx_shock, variant))
            for u in prep.underlyings:
                move_rows.append(
                    {
                        "spot_shock_pct": spx_shock,
                        "variant": variant,
                        "factor": f"{u} spot",
                        "move": shock[u].spot_shock_pct,
                    }
                )
                index_move = vix_move if variant == "naive" else moves[VOL_INDEX_FOR[u]]
                move_rows.append(
                    {
                        "spot_shock_pct": spx_shock,
                        "variant": variant,
                        "factor": VOL_INDEX_FOR[u],
                        "move": index_move,
                    }
                )
                for t_label, k in REPORTED_NODES:
                    move_rows.append(
                        {
                            "spot_shock_pct": spx_shock,
                            "variant": variant,
                            "factor": f"{u} vol {t_label} k={k:+.1f} (points)",
                            "move": _node_vol_move(shock[u], prep.proxies[u].base_iv, t_label, k),
                        }
                    )

    values = revalue(prep.ctx, shocks, max_workers)
    pnl = full_pnl(prep.base, values, prep.marks)
    rows = []
    for (spx_shock, variant), row in zip(labels, pnl, strict=True):
        for m, value in zip(prep.marks, row, strict=True):
            rows.append(
                {
                    "spot_shock_pct": spx_shock,
                    "variant": variant,
                    "position_id": m.position_id,
                    "underlying": m.underlying,
                    "type": m.type,
                    "pnl": float(value),
                }
            )
    result.pnl = pd.DataFrame(rows)
    result.moves = pd.DataFrame(move_rows)
    result.betas = pd.DataFrame(
        [
            {
                "factor": f,
                "beta": fit.beta,
                "alpha": fit.alpha,
                "r2": fit.r2,
                "n_periods": fit.n,
            }
            for f, fit in fits.items()
        ]
        + [
            {
                "factor": f"{TERM_INDEX} points per VIX point",
                "beta": result.term_ratio_crash,
                "alpha": 0.0,
                "r2": float("nan"),
                "n_periods": next(iter(fits.values())).n,
            }
        ]
    ).assign(horizon_days=HORIZON_DAYS, sell_off_quantile=SELL_OFF_QUANTILE)
    if persist:
        _persist(result, Path(cfg.paths.curated))
    return result


def _persist(result: ConditionalStressResult, curated_root: Path) -> None:
    for df, schema, required, name in (
        (
            result.pnl,
            CONDITIONAL_STRESS_SCHEMA,
            CONDITIONAL_STRESS_REQUIRED_NOT_NULL,
            "conditional_stress",
        ),
        (
            result.moves,
            CONDITIONAL_STRESS_MOVE_SCHEMA,
            CONDITIONAL_STRESS_MOVE_REQUIRED_NOT_NULL,
            "conditional_stress_moves",
        ),
        (result.betas, CRASH_BETA_SCHEMA, CRASH_BETA_REQUIRED_NOT_NULL, "crash_betas"),
    ):
        table = validate(df.assign(asof_date=result.asof), schema, required)
        store.write_partitioned(table, curated_root / name, ["asof_date"])
