"""The vol risk-factor grid (README Step 8.1): a FIXED grid in (k, T), evaluated
against each day's moving calibrated surface. "The grid is fixed; the surface
moves" — this is how production risk systems define vol risk factors, and it's
why Step 4's calibration stability matters beyond that day's own fit quality.

Risk factor variable, chosen and documented per the README's own prompt: total
implied variance `w(k,T)` (not log-implied-vol). Reasons: (1) `w` is the
quantity every other module in this codebase already works in natively (SVI/
SSVI parameterise `w` directly, Dupire strips local vol from `w`, the variance
swap replicates `w`) — using the same variable here means no unit conversion
anywhere downstream; (2) `w` is additive in a way vol itself is not (total
variance across independent periods sums; vol doesn't), which is exactly the
property PCA on a *covariance matrix* wants to exploit; log-iv would need an
extra `w = iv^2 * T` round trip for no benefit here.

This module only builds and evaluates the GRID (Step 8.1) — it does not compute
day-over-day changes, PCA, or proxy regressions (Steps 8.2/8.3), which are
deferred pending real multi-day calibrated history (see `planning/decisions.md`,
2026-08-25/28). Once >=2 days of `risk_factors` exist, `Δw(k,T)` is a plain diff
of this table across `asof_date` — no new machinery needed to start computing
it once the data exists.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import RISK_FACTOR_REQUIRED_NOT_NULL, RISK_FACTOR_SCHEMA, validate
from eqdrisk.vol.local_vol import MIN_PILLARS_FOR_LOCAL_VOL, local_variance_at

# README's own example grid (README 8.1).
K_GRID: tuple[float, ...] = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20)
T_GRID: dict[str, float] = {"1m": 1 / 12, "3m": 3 / 12, "6m": 6 / 12, "1y": 1.0, "2y": 2.0}


def evaluate_risk_factor_grid(
    surface_for_underlying: pd.DataFrame,
    k_grid: tuple[float, ...] = K_GRID,
    t_grid: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    """Evaluate `w(k,T)` at every (k, T_label) node for one underlying's one
    day of calibrated surface. Returns None if fewer than
    `MIN_PILLARS_FOR_LOCAL_VOL` expiries are calibrated that day (matching
    `vol.local_vol.local_variance_at`'s own honest-skip condition — there is
    no T-direction information to interpolate the grid's T points from).

    Reuses `local_variance_at` purely for its T-interpolated `w` — the grid is
    defined directly in (k, T) space, so unlike Step 7's per-position pricing,
    no forward curve is needed here at all: SVI/SSVI slices are already
    parameterised in `k = log(K/F)`, forward-agnostic by construction.
    """
    t_grid = t_grid if t_grid is not None else T_GRID
    if len(surface_for_underlying) < MIN_PILLARS_FOR_LOCAL_VOL:
        return None

    rows = []
    for t_label, T in t_grid.items():
        for k in k_grid:
            result = local_variance_at(surface_for_underlying, k, T)
            assert result is not None  # already checked len(surface) >= MIN_PILLARS_FOR_LOCAL_VOL
            rows.append({"k": k, "T": T, "T_label": t_label, "w": result.w})
    return pd.DataFrame(rows)


@dataclass
class RiskFactorGridResult:
    asof: dt.date
    n_underlyings: int = 0
    skipped: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Risk-factor grid — {self.asof}: {self.n_underlyings} underlyings evaluated"]
        for u, reason in self.skipped.items():
            lines.append(f"  {u}: skipped — {reason}")
        return "\n".join(lines)


def run_risk_factor_grid(
    cfg: BaseConfig, asof: dt.date, underlyings: list[str] | None = None
) -> RiskFactorGridResult:
    curated_root = Path(cfg.paths.curated)
    vol_surface_root = curated_root / "vol_surface"
    universe = (
        underlyings if underlyings is not None else cfg.universe.index + cfg.universe.single_names
    )
    result = RiskFactorGridResult(asof=asof)

    if not vol_surface_root.exists() or not any(vol_surface_root.rglob("*.parquet")):
        result.skipped["_all_"] = "no vol_surface data at all"
        return result

    out_rows = []
    for underlying in universe:
        surface = store.query(
            f"SELECT * FROM vs WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"vs": str(vol_surface_root)},
        ).to_pandas()
        if surface.empty:
            result.skipped[underlying] = "no calibrated surface for this date"
            continue

        grid = evaluate_risk_factor_grid(surface)
        if grid is None:
            result.skipped[underlying] = (
                f"fewer than {MIN_PILLARS_FOR_LOCAL_VOL} calibrated expiries"
            )
            continue

        grid["asof_date"] = asof
        grid["underlying"] = underlying
        out_rows.append(grid)
        result.n_underlyings += 1

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
        table = validate(out_df, RISK_FACTOR_SCHEMA, RISK_FACTOR_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "risk_factors", ["asof_date", "underlying"])

    return result
