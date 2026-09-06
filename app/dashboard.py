"""Streamlit risk dashboard (README Step 14) — six tabs (Surface, Exposure, VaR,
Backtest, Stress, P&L explain) reading ONLY curated artifacts that Steps 1-13's
pipeline already wrote to disk. No pricing, calibration, or MC code is called
from here: the two places that reconstruct a value from stored inputs (the
Surface tab's smooth SVI/SSVI curve, evaluated from stored (a,b,rho,m,sigma) or
(eta,theta) parameters at a k-grid) are pure, deterministic evaluations of
numbers the pipeline already calibrated and wrote — not a new calibration.

Run via `eqdrisk dashboard` (wraps `streamlit run` with the project root as the
working directory, so the relative `configs/*.yaml` paths below resolve the
same way they do for every other CLI command).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.vol.ssvi import SSVIParams
from eqdrisk.vol.svi import SVIParams

CONFIG_PATH = "configs/base.yaml"
VAR_ACCEPTANCE_DAYS = 250  # README's own Step 9 window
PNL_EXPLAIN_ACCEPTANCE_DAYS = 60  # README's own Step 12 acceptance bar


@st.cache_resource
def load_cfg() -> BaseConfig:
    return BaseConfig.from_yaml(CONFIG_PATH)


@st.cache_data(ttl=300)
def read_table(table: str, curated: str) -> pd.DataFrame:
    """Every row ever written to `table`, or an empty frame if it doesn't exist
    yet — callers filter by date/underlying themselves. Cached for 5 minutes
    since curated data only changes once a day (the automated pipeline), not
    on every dashboard interaction."""
    root = Path(curated) / table
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame()
    return store.query("SELECT * FROM t", views={"t": str(root)}).to_pandas()


def n_history_days(cfg: BaseConfig) -> int:
    """Real trading days of calibrated vol-surface history available so far —
    the honest denominator for the VaR/backtest 'pending real history' tabs."""
    df = read_table("vol_surface", cfg.paths.curated)
    if df.empty:
        return 0
    return int(df["asof_date"].nunique())


def reconstruct_curve(row: pd.Series, k_grid: pd.Series | list[float]) -> list[float]:
    """Evaluate the ALREADY-calibrated SVI/SSVI parameters (stored in this
    `vol_surface` row) at `k_grid` — a pure function of stored numbers, not a
    new fit. Returns implied vol, not total variance."""
    import numpy as np

    k = np.asarray(k_grid, dtype=float)
    T = float(row["T"])
    if row["model"] == "SVI":
        params = SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])
        w = params.total_variance(k)
    else:
        params = SSVIParams(rho=row["rho"], eta=row["eta"])
        w = params.total_variance(k, float(row["theta"]))
    return list(np.sqrt(np.clip(w, 1e-12, None) / T))


def render_surface_tab(cfg: BaseConfig) -> None:
    import numpy as np

    universe = cfg.universe.index + cfg.universe.single_names
    underlying = st.selectbox("Underlying", universe, key="surface_underlying")

    surface_all = read_table("vol_surface", cfg.paths.curated)
    if surface_all.empty:
        st.info("No calibrated vol_surface data yet.")
        return
    dates = sorted(surface_all["asof_date"].unique(), reverse=True)
    asof = st.selectbox("As-of date", dates, key="surface_date")

    surface = surface_all[
        (surface_all["underlying"] == underlying) & (surface_all["asof_date"] == asof)
    ].sort_values("T")
    if surface.empty:
        st.warning(f"No calibrated surface for {underlying} on {asof}.")
        return

    st.caption(
        "Calibration quality by expiry — RMSE/max-error in vol points, and the two "
        "no-arbitrage checks Step 4 enforces."
    )
    st.dataframe(
        surface[
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
        ],
        hide_index=True,
    )

    iv_all = read_table("implied_vols", cfg.paths.curated)
    iv_today = (
        iv_all[
            (iv_all["underlying"] == underlying)
            & (iv_all["asof_date"] == asof)
            & (iv_all["reason"] == "OK")
        ]
        if not iv_all.empty
        else pd.DataFrame()
    )

    st.subheader("3D surface (fitted, evaluated from stored calibration parameters)")
    k_grid = np.linspace(-1.0, 1.0, 61)
    surface_matrix = np.array([reconstruct_curve(row, k_grid) for _, row in surface.iterrows()])
    fig3d = go.Figure(
        data=[
            go.Surface(
                x=k_grid,
                y=surface["T"].to_numpy(),
                z=surface_matrix,
                colorscale="Viridis",
                opacity=0.85,
            )
        ]
    )
    if not iv_today.empty:
        fig3d.add_trace(
            go.Scatter3d(
                x=iv_today["k"],
                y=iv_today["T"],
                z=iv_today["iv"],
                mode="markers",
                marker=dict(size=2, color="red"),
                name="market (OK quotes)",
            )
        )
    fig3d.update_layout(
        scene=dict(xaxis_title="log-moneyness k", yaxis_title="T (years)", zaxis_title="IV"),
        height=550,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig3d, width="stretch")

    st.subheader("Single-expiry slice: fit vs. market")
    expiry_choice = st.selectbox("Expiry", surface["expiry"].tolist(), key="surface_expiry")
    row = surface[surface["expiry"] == expiry_choice].iloc[0]
    fit_iv = reconstruct_curve(row, k_grid)
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=k_grid, y=fit_iv, mode="lines", name=f"{row['model']} fit"))
    if not iv_today.empty:
        slice_pts = iv_today[iv_today["expiry"] == expiry_choice]
        if not slice_pts.empty:
            fig2d.add_trace(
                go.Scatter(x=slice_pts["k"], y=slice_pts["iv"], mode="markers", name="market (OK)")
            )
    fig2d.update_layout(
        xaxis_title="log-moneyness k",
        yaxis_title="IV",
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig2d, width="stretch")


def render_exposure_tab(cfg: BaseConfig) -> None:
    marks_all = read_table("portfolio_marks", cfg.paths.curated)
    if marks_all.empty:
        st.info("No portfolio_marks data yet — run `eqdrisk portfolio --date ...`.")
        return

    dates = sorted(marks_all["asof_date"].unique(), reverse=True)
    asof = st.selectbox("As-of date", dates, key="exposure_date")
    latest_date = max(dates)
    if asof != latest_date:
        st.caption(f"Note: {latest_date} is the most recent portfolio mark available.")

    marks = marks_all[marks_all["asof_date"] == asof]
    total_value = marks["price"].sum()
    st.metric("Book value", f"${total_value:,.0f}")

    for by in ["underlying", "expiry_bucket", "moneyness_bucket"]:
        st.subheader(f"Aggregate Greeks by {by}")
        agg = (
            marks.groupby(by)[["price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]]
            .sum()
            .reset_index()
        )
        st.dataframe(agg, hide_index=True)

    st.subheader("Delta ladder across stickiness conventions (Step 5)")
    st.caption(
        "From the per-instrument `greeks` table (Step 5), not the portfolio marks above — "
        "these are vanilla-only, computed at calibration time for a range of strikes, not "
        "the specific positions in the book."
    )
    greeks_all = read_table("greeks", cfg.paths.curated)
    if greeks_all.empty:
        st.info("No greeks data yet.")
        return
    universe = cfg.universe.index + cfg.universe.single_names
    underlying = st.selectbox("Underlying", universe, key="ladder_underlying")
    greeks_today = greeks_all[
        (greeks_all["underlying"] == underlying) & (greeks_all["asof_date"] == asof)
    ]
    if greeks_today.empty:
        st.warning(f"No greeks for {underlying} on {asof}.")
        return
    expiry_choice = st.selectbox(
        "Expiry", sorted(greeks_today["expiry"].unique()), key="ladder_expiry"
    )
    slice_df = greeks_today[greeks_today["expiry"] == expiry_choice].sort_values("strike")
    fig = go.Figure()
    for col, label in [
        ("delta_sticky_strike", "sticky strike"),
        ("delta_sticky_delta", "sticky delta"),
        ("delta_sticky_local_vol", "sticky local vol"),
    ]:
        fig.add_trace(go.Scatter(x=slice_df["strike"], y=slice_df[col], mode="lines", name=label))
    fig.update_layout(
        xaxis_title="strike", yaxis_title="delta", height=400, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, width="stretch")


def render_var_tab(cfg: BaseConfig) -> None:
    days = n_history_days(cfg)
    st.warning(
        "Step 9 (VaR / Expected Shortfall) is not yet built — it needs a "
        f"{VAR_ACCEPTANCE_DAYS}-1000 day real history window (a regulatory-convention "
        "size, not arbitrary) of daily calibrated vol surfaces to compute full-reval and "
        f"Taylor VaR against. Currently available: {days} real trading day(s) of calibrated "
        "history, accumulating one real day at a time via the automated daily pipeline."
    )


def render_backtest_tab(cfg: BaseConfig) -> None:
    days = n_history_days(cfg)
    st.warning(
        "Step 10 (VaR backtesting: Kupiec, Christoffersen, traffic light, PLA) is not yet "
        "built — it needs Step 9's VaR series to exist first, over the same real multi-year "
        f"window. Currently available: {days} real trading day(s) of calibrated history."
    )


def render_stress_tab(cfg: BaseConfig) -> None:
    st.subheader("Hypothetical spot x vol stress grid (Step 11.2)")
    grid_all = read_table("hypothetical_grid", cfg.paths.curated)
    if grid_all.empty:
        st.info("No hypothetical_grid data yet — run `eqdrisk hypotheticalgrid --date ...`.")
    else:
        dates = sorted(grid_all["asof_date"].unique(), reverse=True)
        asof = st.selectbox("As-of date", dates, key="grid_date")
        day_df = grid_all[grid_all["asof_date"] == asof]
        cells = day_df[day_df["scenario"] == "grid"]
        if not cells.empty:
            matrix = cells.pivot(index="spot_shock_pct", columns="vol_shock_pct", values="pnl")
            matrix = matrix.sort_index(ascending=False)
            fig = px.imshow(
                matrix,
                labels=dict(x="vol shock", y="spot shock", color="P&L"),
                x=[f"{c:+.0%}" for c in matrix.columns],
                y=[f"{r:+.0%}" for r in matrix.index],
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                text_auto=",.0f",
            )
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, width="stretch")
        for scenario, label in [
            ("skew_steepening", "Skew-steepening scenario P&L"),
            ("term_inversion", "Term-structure-inversion scenario P&L"),
        ]:
            row = day_df[day_df["scenario"] == scenario]
            if not row.empty:
                st.metric(label, f"${row['pnl'].iloc[0]:+,.0f}")

    st.subheader("Historical replay: five real named episodes (Step 11.1)")
    replay_all = read_table("historical_replay", cfg.paths.curated)
    if replay_all.empty:
        st.info("No historical_replay data yet — run `eqdrisk historicalreplay --date ...`.")
        return
    dates = sorted(replay_all["asof_date"].unique(), reverse=True)
    asof = st.selectbox("As-of date", dates, key="replay_date")
    day_df = replay_all[replay_all["asof_date"] == asof].sort_values("pnl")
    fig = go.Figure(go.Bar(x=day_df["episode_name"], y=day_df["pnl"]))
    fig.update_layout(yaxis_title="P&L ($)", height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        day_df[["episode_name", "episode_description", "base_value", "shocked_value", "pnl"]],
        hide_index=True,
    )


def render_pnl_explain_tab(cfg: BaseConfig) -> None:
    steps_all = read_table("pnl_explain", cfg.paths.curated)
    if steps_all.empty:
        st.info("No pnl_explain data yet — run `eqdrisk explainpnl --day0 ... --day1 ...`.")
        return

    st.subheader("Residual time series")
    totals = steps_all.groupby("asof_date")["residual"].sum().sort_index().reset_index()
    n_days = len(totals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=totals["asof_date"], y=totals["residual"], mode="lines+markers"))
    fig.update_layout(
        yaxis_title="Total residual ($)", height=350, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{n_days} real day-pair(s) of P&L explain available so far. The README's own "
        f"acceptance bar (median residual < 2bp of NAV) needs {PNL_EXPLAIN_ACCEPTANCE_DAYS}+ "
        "days to be a meaningful statistic — not reached yet, shown honestly rather than "
        "computed on too small a sample."
    )

    st.subheader("Waterfall for a selected day pair")
    pairs = steps_all[["day0", "asof_date"]].drop_duplicates().sort_values("asof_date")
    pair_labels = [f"{r.day0} -> {r.asof_date}" for r in pairs.itertuples()]
    choice = st.selectbox("Day pair", pair_labels, index=len(pair_labels) - 1)
    day0_str, day1_str = choice.split(" -> ")
    day0, day1 = dt.date.fromisoformat(day0_str), dt.date.fromisoformat(day1_str)

    day_steps = (
        steps_all[(steps_all["day0"] == day0) & (steps_all["asof_date"] == day1)]
        .set_index("step")
        .reindex(["time", "rates_divs", "spot", "vol"])
    )
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(x=day_steps.index, y=day_steps["actual_pnl"], name="actual"))
    fig_wf.add_trace(go.Bar(x=day_steps.index, y=day_steps["explained_pnl"], name="explained"))
    fig_wf.add_trace(go.Bar(x=day_steps.index, y=day_steps["residual"], name="residual"))
    fig_wf.update_layout(
        barmode="group", yaxis_title="P&L ($)", height=400, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_wf, width="stretch")
    st.dataframe(day_steps.reset_index(), hide_index=True)

    st.subheader("Residual drill-down by position")
    by_pos_all = read_table("pnl_explain_by_position", cfg.paths.curated)
    if by_pos_all.empty:
        st.info("No per-position residual data for this day pair yet.")
        return
    by_pos = by_pos_all[
        (by_pos_all["day0"] == day0) & (by_pos_all["asof_date"] == day1)
    ].sort_values("residual", key=lambda s: s.abs(), ascending=False)
    fig_pos = go.Figure(go.Bar(x=by_pos["position_id"], y=by_pos["residual"]))
    fig_pos.update_layout(yaxis_title="Residual ($)", height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_pos, width="stretch")
    st.dataframe(by_pos, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="EQD Risk Engine", layout="wide")
    st.title("Equity Derivatives Risk Engine")
    st.caption(
        "Every number on this page is read from a stored curated artifact the daily "
        "pipeline already wrote — no pricing or calibration runs from this app."
    )

    cfg = load_cfg()
    tabs = st.tabs(["Surface", "Exposure", "VaR", "Backtest", "Stress", "P&L explain"])
    with tabs[0]:
        render_surface_tab(cfg)
    with tabs[1]:
        render_exposure_tab(cfg)
    with tabs[2]:
        render_var_tab(cfg)
    with tabs[3]:
        render_backtest_tab(cfg)
    with tabs[4]:
        render_stress_tab(cfg)
    with tabs[5]:
        render_pnl_explain_tab(cfg)


main()
