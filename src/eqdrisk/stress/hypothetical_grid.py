"""Hypothetical spot x vol stress grid, plus two standalone scenarios (README
11.2): a spot ladder crossed with a parallel-vol ladder, full-repriced at every
combination and rendered as a P&L heatmap, plus a skew-steepening and a
term-structure-inversion scenario reported individually (not part of the
grid — the README frames these as a separate kind of scenario, not another
axis of the same ladder).

**Cost note, locked in the plan doc:** every cell is a FULL portfolio reprice,
and the two MC-priced positions (barrier, autocall) mean a full Monte Carlo
run per cell. 42 grid cells (7 spot x 6 vol) plus 2 standalone scenarios plus
the base case is 45 full reprices — not viable at Step 7's daily-mark MC
settings (~1-2 minutes each). `GRID_MC_SETTINGS` trades more MC noise per cell
for a sweep that finishes in a reasonable time, the same "fewer paths for
frequent/bulk runs, more for a one-off mark" tradeoff Step 7 already
documents for its own daily settings.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.portfolio.mark import MCSettings, load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.stress.shock import MarketShock

# README's own example ladder (11.2).
SPOT_LADDER: tuple[float, ...] = (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10)
VOL_LADDER: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.25, 0.50)

# Engineering defaults for the two standalone scenarios — magnitudes chosen to
# be clearly visible against the grid's own vol ladder, not derived from any
# specific historical episode (that's what Step 11.1 is for).
SKEW_STEEPENING_SHOCK = 0.5
TERM_INVERSION_SHOCK = 0.5

GRID_MC_SETTINGS = MCSettings(n_paths=8_000, barrier_n_steps=32, autocall_n_steps_per_period=4)


@dataclass
class GridCellResult:
    spot_shock_pct: float
    vol_shock_pct: float
    value: float
    pnl: float


@dataclass
class HypotheticalGridResult:
    asof: dt.date
    base_value: float = 0.0
    cells: list[GridCellResult] = field(default_factory=list)
    skew_scenario_pnl: float | None = None
    term_scenario_pnl: float | None = None
    skipped: dict[str, str] = field(default_factory=dict)

    def pnl_matrix(self) -> pd.DataFrame:
        """Rows = spot shock (highest first), columns = vol shock — the P&L heatmap table."""
        rows = sorted({c.spot_shock_pct for c in self.cells}, reverse=True)
        cols = sorted({c.vol_shock_pct for c in self.cells})
        matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
        for c in self.cells:
            matrix.loc[c.spot_shock_pct, c.vol_shock_pct] = c.pnl
        return matrix

    def render(self) -> str:
        lines = [
            f"Hypothetical stress grid — {self.asof}",
            f"  base value: {self.base_value:,.2f}",
        ]
        matrix = self.pnl_matrix()
        if not matrix.empty:
            lines.append("  P&L grid (rows: spot shock, cols: vol shock):")
            header = "spot\\vol".rjust(9) + "".join(f"{c:+.0%}".rjust(12) for c in matrix.columns)
            lines.append("  " + header)
            for spot_shock, row in matrix.iterrows():
                cells = "".join(f"{v:,.0f}".rjust(12) for v in row)
                lines.append(f"  {spot_shock:+.0%}".rjust(11) + cells)
        if self.skew_scenario_pnl is not None:
            lines.append(f"  skew-steepening scenario P&L: {self.skew_scenario_pnl:+,.2f}")
        if self.term_scenario_pnl is not None:
            lines.append(f"  term-structure-inversion scenario P&L: {self.term_scenario_pnl:+,.2f}")
        for name, reason in self.skipped.items():
            lines.append(f"  SKIPPED {name}: {reason}")
        return "\n".join(lines)


def run_hypothetical_grid(
    cfg: BaseConfig, asof: dt.date, portfolio_path: str
) -> HypotheticalGridResult:
    portfolio = Portfolio.from_yaml(portfolio_path)
    state = load_market_state(cfg, asof, portfolio)
    if state is None:
        result = HypotheticalGridResult(asof=asof)
        result.skipped["_all_"] = "no curated rates available"
        return result

    base_value = mark_with_state(
        cfg, asof, portfolio, state, mc_settings=GRID_MC_SETTINGS
    ).total_value()
    result = HypotheticalGridResult(asof=asof, base_value=base_value)

    for spot_shock_pct in SPOT_LADDER:
        for vol_shock_pct in VOL_LADDER:
            shock = MarketShock(spot_shock_pct=spot_shock_pct, vol_shock_pct=vol_shock_pct)
            value = mark_with_state(
                cfg, asof, portfolio, state, shock=shock, mc_settings=GRID_MC_SETTINGS
            ).total_value()
            result.cells.append(
                GridCellResult(spot_shock_pct, vol_shock_pct, value, value - base_value)
            )

    skew_shock = MarketShock(skew_shock=SKEW_STEEPENING_SHOCK)
    skew_value = mark_with_state(
        cfg, asof, portfolio, state, shock=skew_shock, mc_settings=GRID_MC_SETTINGS
    ).total_value()
    result.skew_scenario_pnl = skew_value - base_value

    term_shock = MarketShock(term_shock=TERM_INVERSION_SHOCK)
    term_value = mark_with_state(
        cfg, asof, portfolio, state, shock=term_shock, mc_settings=GRID_MC_SETTINGS
    ).total_value()
    result.term_scenario_pnl = term_value - base_value

    return result
