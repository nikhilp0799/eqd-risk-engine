"""Export a fixed set of real curated-table snapshots to flat JSON for the
public showcase site (`showcase/`, `planning/showcase_site_plan.md`).

Reads ONLY already-computed curated Parquet tables — no pricing, calibration,
or MC code runs from here, same "no live computation" rule the Streamlit
dashboard (`app/dashboard.py`) already follows. The featured dates are fixed,
deliberately-chosen real snapshots (see the plan doc), not "whatever is most
recent" — re-running this script against a later curated dataset reproduces
the same JSON as long as those specific dates/tables still exist.

Run from the project root: `python scripts/export_showcase_data.py`.
Writes into `showcase/src/data/*.json`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eqdrisk.io import store
from eqdrisk.pricing.pnl_explain import RESIDUAL_ALERT_THRESHOLD_BP
from eqdrisk.vol.ssvi import SSVIParams
from eqdrisk.vol.svi import SVIParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED = PROJECT_ROOT / "data" / "curated"
OUT_DIR = PROJECT_ROOT / "showcase" / "src" / "data"

DEEP_HEDGE_DATE = dt.date(2026, 9, 22)
VOL_SURFACE_DATE = dt.date(2026, 9, 22)
VOL_SURFACE_UNDERLYING = "SPX"
INCIDENT_DAY0 = dt.date(2026, 9, 2)
INCIDENT_DAY1 = dt.date(2026, 9, 3)

K_GRID = np.linspace(-1.0, 1.0, 61)


def _read(table: str) -> pd.DataFrame:
    root = CURATED / table
    if not root.exists() or not any(root.rglob("*.parquet")):
        raise FileNotFoundError(f"no curated data at {root}")
    return store.query("SELECT * FROM t", views={"t": str(root)}).to_pandas()


def _write(name: str, payload: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    # allow_nan=False: Python otherwise emits a literal `NaN`, which is invalid
    # JSON and only fails later, inside the Next.js build.
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def _reconstruct_smile(row: pd.Series) -> list[float]:
    """Same pure evaluation `app/dashboard.py::reconstruct_curve` uses — the
    (a,b,rho,m,sigma) or (eta,theta) columns are ALREADY-calibrated params
    read straight from the row, not a new fit."""
    T = float(row["T"])
    if row["model"] == "SVI":
        params = SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])
        w = params.total_variance(K_GRID)
    else:
        params = SSVIParams(rho=row["rho"], eta=row["eta"])
        w = params.total_variance(K_GRID, float(row["theta"]))
    return list(np.sqrt(np.clip(w, 1e-12, None) / T))


def export_deep_hedging() -> None:
    df = _read("deep_hedge_results")
    day = df[df["asof_date"] == DEEP_HEDGE_DATE].copy()
    if day.empty:
        raise ValueError(f"no deep_hedge_results for {DEEP_HEDGE_DATE}")
    day["std_reduction_pct"] = 100.0 * (1.0 - day["learned_std"] / day["baseline_std"])
    day["cvar_improvement"] = day["learned_cvar"] - day["baseline_cvar"]
    rows = day.sort_values(["instrument", "loss_type"])[
        [
            "underlying",
            "instrument",
            "loss_type",
            "learned_std",
            "baseline_std",
            "std_reduction_pct",
            "learned_cvar",
            "baseline_cvar",
            "cvar_improvement",
        ]
    ].to_dict(orient="records")
    _write("deep_hedging", {"asof_date": DEEP_HEDGE_DATE, "rows": rows})


def export_vol_surface() -> dict[str, Any]:
    surface_all = _read("vol_surface")
    surface = surface_all[
        (surface_all["underlying"] == VOL_SURFACE_UNDERLYING)
        & (surface_all["asof_date"] == VOL_SURFACE_DATE)
    ].sort_values("T")
    if surface.empty:
        raise ValueError(f"no vol_surface for {VOL_SURFACE_UNDERLYING} on {VOL_SURFACE_DATE}")

    pillars = surface[
        [
            "expiry",
            "T",
            "model",
            "n_points",
            "rmse_vol_points",
            "max_abs_error_vol_points",
            "butterfly_violations",
            "calendar_violated",
        ]
    ].to_dict(orient="records")

    iv_all = _read("implied_vols")
    iv_today = iv_all[
        (iv_all["underlying"] == VOL_SURFACE_UNDERLYING)
        & (iv_all["asof_date"] == VOL_SURFACE_DATE)
        & (iv_all["reason"] == "OK")
    ]

    # Feature a spread-out subset of expiries, not all 16, so the site shows a
    # few clean, legible slices rather than a cramped selector.
    n = len(surface)
    feature_idx = sorted(set(np.linspace(0, n - 1, min(5, n)).round().astype(int)))
    smiles: dict[str, Any] = {}
    for idx in feature_idx:
        row = surface.iloc[idx]
        expiry_key = row["expiry"].isoformat()
        market = iv_today[iv_today["expiry"] == row["expiry"]]
        smiles[expiry_key] = {
            "T": float(row["T"]),
            "model": row["model"],
            "k_grid": list(K_GRID),
            "fit_iv": _reconstruct_smile(row),
            "market_k": list(market["k"]),
            "market_iv": list(market["iv"]),
        }

    # Stickiness-convention deltas (delta_sticky_*) are only ever populated for
    # SVI-fit underlyings (Step 5) — SPX trips the calendar-arbitrage check on
    # every real date so far and always falls back to SSVI (see
    # `vol/surface.py::calibrate_underlying`), which never gets that column.
    # Use the base Black-76 Greeks instead, which SSVI rows DO have.
    greeks_all = _read("greeks")
    greeks_today = greeks_all[
        (greeks_all["underlying"] == VOL_SURFACE_UNDERLYING)
        & (greeks_all["asof_date"] == VOL_SURFACE_DATE)
    ]
    mid_expiry = surface.iloc[len(surface) // 2]["expiry"]
    ladder_df = greeks_today[
        (greeks_today["expiry"] == mid_expiry) & (greeks_today["cp"] == "C")
    ].sort_values("strike")
    greeks_ladder = {
        "expiry": mid_expiry.isoformat(),
        "strike": list(ladder_df["strike"]),
        "delta_spot": list(ladder_df["delta_spot"]),
        "gamma_spot": list(ladder_df["gamma_spot"]),
        "vega": list(ladder_df["vega"]),
        "theta": list(ladder_df["theta"]),
        "price": list(ladder_df["price"]),
    }

    payload = {
        "asof_date": VOL_SURFACE_DATE,
        "underlying": VOL_SURFACE_UNDERLYING,
        "pillars": pillars,
        "smiles": smiles,
        "greeks_ladder": greeks_ladder,
    }
    _write("vol_surface", payload)
    return payload


def export_pnl_explain() -> dict[str, Any]:
    steps_all = _read("pnl_explain")
    day_steps = (
        steps_all[(steps_all["day0"] == INCIDENT_DAY0) & (steps_all["asof_date"] == INCIDENT_DAY1)]
        .set_index("step")
        .reindex(["time", "rates_divs", "spot", "vol"])
    )
    if day_steps["actual_pnl"].isna().all():
        raise ValueError(f"no pnl_explain for {INCIDENT_DAY0} -> {INCIDENT_DAY1}")
    nav = float(day_steps["nav"].iloc[0])
    residual_bp = 10_000.0 * float(day_steps["residual"].sum()) / nav

    by_pos_all = _read("pnl_explain_by_position")
    by_pos = by_pos_all[
        (by_pos_all["day0"] == INCIDENT_DAY0) & (by_pos_all["asof_date"] == INCIDENT_DAY1)
    ].sort_values("residual", key=lambda s: s.abs(), ascending=False)

    payload = {
        "day0": INCIDENT_DAY0,
        "asof_date": INCIDENT_DAY1,
        "nav": nav,
        "residual_bp": residual_bp,
        "residual_alert_threshold_bp": RESIDUAL_ALERT_THRESHOLD_BP,
        "positions": _position_labels(),
        "waterfall": [
            {
                "step": step,
                "actual_pnl": float(row["actual_pnl"]),
                "explained_pnl": float(row["explained_pnl"]),
                "residual": float(row["residual"]),
            }
            for step, row in day_steps.iterrows()
        ],
        "by_position": by_pos[["position_id", "residual"]].to_dict(orient="records"),
        "residual_series": _valid_residual_series(steps_all, by_pos_all),
    }
    _write("pnl_explain", payload)
    return payload


def _valid_residual_series(steps_all: pd.DataFrame, by_pos_all: pd.DataFrame) -> list[dict]:
    """Residual history, restricted to day pairs with real marks on BOTH days.

    `pnl_explain` also holds rows for day pairs where one side's portfolio marks
    were never written (market holidays, missed automation days): some come out
    all-zero, others as large garbage P&L against a zero NAV. Neither is a real
    measurement, so they are dropped rather than charted as "within threshold."
    Older rows predate the `nav` column; for those, NAV is the day's stored
    book value from `portfolio_marks` (matches the stored `nav` exactly on the
    days that have both, e.g. 2026-09-10 and 2026-09-17).
    """
    marks = _read("portfolio_marks").groupby("asof_date")["price"].sum()
    daily = steps_all.groupby(["day0", "asof_date"], as_index=False).agg(
        residual=("residual", "sum"),
        nav=("nav", "max"),
        gross=("actual_pnl", lambda s: s.abs().sum()),
    )
    series = []
    for row in daily.sort_values("asof_date").itertuples():
        if row.day0 not in marks.index or row.asof_date not in marks.index or row.gross == 0:
            continue
        nav = row.nav if pd.notna(row.nav) and row.nav > 0 else float(marks[row.asof_date])
        pos = by_pos_all[
            (by_pos_all["day0"] == row.day0) & (by_pos_all["asof_date"] == row.asof_date)
        ]
        top = pos.loc[pos["residual"].abs().idxmax()] if not pos.empty else None
        series.append(
            {
                "day0": row.day0,
                "asof_date": row.asof_date,
                "residual_bp": 10_000.0 * row.residual / nav,
                "top_position_id": None if top is None else top["position_id"],
                "top_position_share": None if top is None else top["residual"] / row.residual,
            }
        )
    return series


UNDERLYING_NAMES = {"SPX": "S&P 500", "NVDA": "NVIDIA", "AAPL": "Apple"}


def _position_labels() -> dict[str, str]:
    """Plain-language name per position id, from `configs/portfolio.yaml`."""
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "portfolio.yaml").read_text())
    labels = {}
    for p in config["positions"]:
        name = UNDERLYING_NAMES.get(p["underlying"], p["underlying"])
        side = " (short)" if p.get("qty", 1) < 0 else ""
        kind = {
            "vanilla": f"{'call' if p.get('cp') == 'C' else 'put'} option",
            "varswap": "variance swap",
            "barrier": "barrier put",
            "autocall": "autocallable note",
            "equity": "index hedge",
        }[p["type"]]
        labels[p["id"]] = f"{name} {kind}{side}"
    return labels


def export_ai_agent() -> dict[str, Any]:
    ai_all = _read("ai_investigations")
    row = ai_all[(ai_all["day0"] == INCIDENT_DAY0) & (ai_all["asof_date"] == INCIDENT_DAY1)]
    if row.empty:
        raise ValueError(f"no ai_investigations for {INCIDENT_DAY0} -> {INCIDENT_DAY1}")
    row = row.iloc[0]

    flagged_all = _read("ai_flagged_positions")
    flagged = flagged_all[
        (flagged_all["day0"] == INCIDENT_DAY0) & (flagged_all["asof_date"] == INCIDENT_DAY1)
    ]
    changes_all = _read("ai_proposed_changes")
    changes = changes_all[
        (changes_all["day0"] == INCIDENT_DAY0) & (changes_all["asof_date"] == INCIDENT_DAY1)
    ]
    trace_all = _read("ai_investigation_trace")
    trace = trace_all[
        (trace_all["day0"] == INCIDENT_DAY0) & (trace_all["asof_date"] == INCIDENT_DAY1)
    ].sort_values("round")

    payload = {
        "day0": INCIDENT_DAY0,
        "asof_date": INCIDENT_DAY1,
        "model": row["model"],
        "confidence": row["confidence"],
        "summary": row["summary"],
        "root_cause_hypothesis": row["root_cause_hypothesis"],
        "flagged_positions": flagged[["position_id", "reason"]].to_dict(orient="records"),
        "proposed_changes": changes[
            ["parameter", "current_value", "suggested_value", "rationale"]
        ].to_dict(orient="records"),
        "trace": trace[["round", "tool_name", "arguments", "result_summary"]].to_dict(
            orient="records"
        ),
    }
    _write("ai_agent", payload)
    return payload


def main() -> None:
    export_deep_hedging()
    export_vol_surface()
    export_pnl_explain()
    export_ai_agent()


if __name__ == "__main__":
    main()
