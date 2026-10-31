"""Vol surface PCA (README Step 8.2): the main shapes in which each name's
implied-vol surface moves from one trading day to the next.

Built on Step 8.1's fixed (k, T) grid (`vol/risk_factors.py`), with two choices
made on the real data (planning/vol_pca_plan.md):

- Changes are in implied vol, vol points, not total variance `w`. `w = sigma^2 T`,
  so the same vol move is ~24x larger in `w` at 2y than at 1m, and PCA on `dw`
  came out with a first component that was almost only the 2y downside wing.
  The stored grid stays in `w`; vol is `sqrt(w / T)`.
- Only nodes inside quoted strikes on at least `MIN_COVERAGE` of stored days go
  in. Outside the quotes the surface is SVI extrapolation and moves with the fit
  parameters, not the market: SPX's +10%/+20% strikes at 1m are never quoted,
  and with them in, SPX's second component sat almost entirely on those nodes.

Changes are taken only between consecutive trading days, so a carried surface
on a non-trading day (2026-08-22, 2026-09-07) or a gap in the history never
shows up as a one-day move.
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
    VOL_PCA_LOADING_REQUIRED_NOT_NULL,
    VOL_PCA_LOADING_SCHEMA,
    VOL_PCA_REQUIRED_NOT_NULL,
    VOL_PCA_SCHEMA,
    VOL_PCA_STABILITY_REQUIRED_NOT_NULL,
    VOL_PCA_STABILITY_SCHEMA,
    validate,
)
from eqdrisk.marketdata.calendar import is_trading_day, last_n_trading_days
from eqdrisk.vol.risk_factors import K_GRID, T_GRID

MIN_COVERAGE = 0.80
# Loadings are re-estimated on every run of this many consecutive changes to
# see whether the shapes hold; also the minimum history for a PCA at all.
ROLLING_WINDOW = 20
N_REPORTED_PCS = 3

Node = tuple[str, float]  # (T_label, k)


def node_coverage(
    surface: pd.DataFrame,
    k_grid: tuple[float, ...] = K_GRID,
    t_grid: dict[str, float] | None = None,
) -> pd.Series:
    """Share of days on which each grid node lies inside the quoted strike range
    ([k_min, k_max]) of both calibrated expiries bracketing its maturity.

    `surface` is the stored `vol_surface` table for one underlying, any number of
    days. A node shorter than the first expiry or longer than the last counts as
    not covered that day.
    """
    t_grid = t_grid if t_grid is not None else T_GRID
    hits: dict[Node, list[bool]] = {(t, k): [] for t in t_grid for k in k_grid}
    for _, day in surface.groupby("asof_date"):
        day = day.sort_values("T")
        for t_label, T in t_grid.items():
            below = day[day["T"] <= T].tail(1)
            above = day[day["T"] >= T].head(1)
            bracket = pd.concat([below, above])
            for k in k_grid:
                inside = (
                    not below.empty
                    and not above.empty
                    and bool(((bracket["k_min"] <= k) & (k <= bracket["k_max"])).all())
                )
                hits[(t_label, k)].append(inside)
    return pd.Series(
        {node: float(np.mean(v)) if v else 0.0 for node, v in hits.items()}, name="coverage"
    )


def daily_vol_changes(
    risk_factors: pd.DataFrame, nodes: list[Node], calendar: str = "NYSE"
) -> pd.DataFrame:
    """Day-over-day implied-vol changes (vol points) at `nodes`, indexed by the
    later date, keeping only pairs of consecutive trading days."""
    rf = risk_factors.assign(iv=np.sqrt(risk_factors["w"] / risk_factors["T"]) * 100.0)
    levels = rf.pivot_table(index="asof_date", columns=["T_label", "k"], values="iv")
    levels = levels.reindex(columns=pd.MultiIndex.from_tuples(nodes, names=["T_label", "k"]))
    levels = levels.loc[[d for d in levels.index if is_trading_day(d, calendar)]].dropna()

    rows, index = [], []
    for prev, cur in zip(levels.index[:-1], levels.index[1:], strict=True):
        if last_n_trading_days(cur - dt.timedelta(days=1), 1, calendar)[0] != prev:
            continue
        rows.append(levels.loc[cur].to_numpy() - levels.loc[prev].to_numpy())
        index.append(cur)
    return pd.DataFrame(rows, index=pd.Index(index, name="asof_date"), columns=levels.columns)


@dataclass
class PCAFit:
    eigenvalues: np.ndarray  # descending, vol points squared
    loadings: np.ndarray  # (n_nodes, n_pcs), column i is PC i+1

    @property
    def explained_share(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()


def fit_pca(changes: np.ndarray) -> PCAFit:
    """Eigendecomposition of the sample covariance of `changes` (rows = days).

    Sign convention: each component's loadings sum to a positive number, so a
    positive PC1 score means vols broadly rose. Signs carry no other meaning.
    """
    centred = changes - changes.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    vectors = vectors[:, order]
    vectors = vectors * np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    return PCAFit(eigenvalues=eigenvalues, loadings=vectors)


def rolling_stability(
    changes: pd.DataFrame,
    full: PCAFit,
    window: int = ROLLING_WINDOW,
    n_pcs: int = N_REPORTED_PCS,
) -> pd.DataFrame:
    """Refit on every run of `window` consecutive changes and compare each
    window's components with the full sample's by |cosine| of the loadings."""
    rows = []
    values = changes.to_numpy()
    for end in range(window, len(values) + 1):
        fit = fit_pca(values[end - window : end])
        for i in range(min(n_pcs, values.shape[1])):
            rows.append(
                {
                    "window_end": changes.index[end - 1],
                    "pc": i + 1,
                    "abs_cosine": float(abs(fit.loadings[:, i] @ full.loadings[:, i])),
                    "explained_share": float(fit.explained_share[i]),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class UnderlyingPCA:
    underlying: str
    changes: pd.DataFrame
    coverage: pd.Series
    fit: PCAFit
    stability: pd.DataFrame

    @property
    def nodes(self) -> list[Node]:
        return list(self.changes.columns)


def analyse_underlying(
    underlying: str,
    surface: pd.DataFrame,
    risk_factors: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE,
    window: int = ROLLING_WINDOW,
    calendar: str = "NYSE",
) -> UnderlyingPCA | str:
    """The PCA for one name, or the reason it was skipped."""
    coverage = node_coverage(surface)
    nodes = [node for node, share in coverage.items() if share >= min_coverage]
    if len(nodes) < 2:
        return f"only {len(nodes)} grid nodes inside quoted strikes on {min_coverage:.0%} of days"
    changes = daily_vol_changes(risk_factors, nodes, calendar)
    if len(changes) < window:
        return f"{len(changes)} consecutive-day changes, need {window}"
    fit = fit_pca(changes.to_numpy())
    return UnderlyingPCA(
        underlying=underlying,
        changes=changes,
        coverage=coverage,
        fit=fit,
        stability=rolling_stability(changes, fit, window),
    )


@dataclass
class VolPCAResult:
    asof: dt.date
    fitted: list[UnderlyingPCA] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Vol surface PCA — history to {self.asof}"]
        for u in self.fitted:
            share = u.fit.explained_share
            top = " / ".join(f"{100 * s:.1f}%" for s in share[:N_REPORTED_PCS])
            lines.append(
                f"  {u.underlying}: {len(u.changes)} daily changes "
                f"({u.changes.index[0]} to {u.changes.index[-1]}), {len(u.nodes)} grid nodes; "
                f"PC1-{N_REPORTED_PCS} {top} (together "
                f"{100 * share[:N_REPORTED_PCS].sum():.1f}%)"
            )
            if not u.stability.empty:
                worst = u.stability.groupby("pc")["abs_cosine"].min()
                lines.append(
                    f"    rolling {ROLLING_WINDOW}-day windows, lowest |cos| with full sample: "
                    + ", ".join(f"PC{pc} {c:.2f}" for pc, c in worst.items())
                )
        for underlying, reason in self.skipped.items():
            lines.append(f"  {underlying}: skipped — {reason}")
        return "\n".join(lines)


def _tables(result: VolPCAResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, loadings, stability = [], [], []
    for u in result.fitted:
        for i, (eig, share) in enumerate(
            zip(u.fit.eigenvalues, u.fit.explained_share, strict=True)
        ):
            summary.append(
                {
                    "underlying": u.underlying,
                    "pc": i + 1,
                    "eigenvalue": float(eig),
                    "explained_share": float(share),
                    "n_changes": len(u.changes),
                    "n_nodes": len(u.nodes),
                    "first_change_date": u.changes.index[0],
                    "last_change_date": u.changes.index[-1],
                }
            )
        for i in range(min(N_REPORTED_PCS, len(u.nodes))):
            for (t_label, k), loading in zip(u.nodes, u.fit.loadings[:, i], strict=True):
                loadings.append(
                    {
                        "underlying": u.underlying,
                        "pc": i + 1,
                        "T_label": t_label,
                        "T": T_GRID[t_label],
                        "k": k,
                        "loading": float(loading),
                        "coverage": float(u.coverage[(t_label, k)]),
                    }
                )
        stability.append(u.stability.assign(underlying=u.underlying))
    return (
        pd.DataFrame(summary).assign(asof_date=result.asof),
        pd.DataFrame(loadings).assign(asof_date=result.asof),
        pd.concat(stability, ignore_index=True).assign(asof_date=result.asof),
    )


def run_vol_pca(
    cfg: BaseConfig, asof: dt.date, underlyings: list[str] | None = None
) -> VolPCAResult:
    """PCA per underlying over every stored day up to and including `asof`."""
    curated_root = Path(cfg.paths.curated)
    universe = (
        underlyings if underlyings is not None else cfg.universe.index + cfg.universe.single_names
    )
    result = VolPCAResult(asof=asof)
    roots = {"vs": curated_root / "vol_surface", "rf": curated_root / "risk_factors"}
    if not all(r.exists() and any(r.rglob("*.parquet")) for r in roots.values()):
        result.skipped["_all_"] = "no vol_surface / risk_factors history"
        return result

    views = {name: str(root) for name, root in roots.items()}
    for underlying in universe:
        where = f"WHERE underlying = '{underlying}' AND asof_date <= DATE '{asof.isoformat()}'"
        surface = store.query(f"SELECT * FROM vs {where}", views=views).to_pandas()
        rf = store.query(f"SELECT * FROM rf {where}", views=views).to_pandas()
        if surface.empty or rf.empty:
            result.skipped[underlying] = "no stored surfaces"
            continue
        for df in (surface, rf):
            df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
        fitted = analyse_underlying(underlying, surface, rf, calendar=cfg.calendar)
        if isinstance(fitted, str):
            result.skipped[underlying] = fitted
        else:
            result.fitted.append(fitted)

    if result.fitted:
        summary, loadings, stability = _tables(result)
        for df, schema, required, name in (
            (summary, VOL_PCA_SCHEMA, VOL_PCA_REQUIRED_NOT_NULL, "vol_pca"),
            (
                loadings,
                VOL_PCA_LOADING_SCHEMA,
                VOL_PCA_LOADING_REQUIRED_NOT_NULL,
                "vol_pca_loadings",
            ),
            (
                stability,
                VOL_PCA_STABILITY_SCHEMA,
                VOL_PCA_STABILITY_REQUIRED_NOT_NULL,
                "vol_pca_stability",
            ),
        ):
            table = validate(df, schema, required)
            store.write_partitioned(table, curated_root / name, ["asof_date", "underlying"])
    return result
