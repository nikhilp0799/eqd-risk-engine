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
from eqdrisk.pricing.pnl_explain import RESIDUAL_ALERT_THRESHOLD_BP
from eqdrisk.vol.local_vol import slice_total_variance

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


def reconstruct_curve(row: pd.Series, k_grid: pd.Series | list[float]) -> list[float]:
    """Evaluate the ALREADY-calibrated SVI/SSVI parameters (stored in this
    `vol_surface` row) at `k_grid` — a pure function of stored numbers, not a
    new fit. Returns implied vol, not total variance."""
    import numpy as np

    k = np.asarray(k_grid, dtype=float)
    T = float(row["T"])
    w = slice_total_variance(row, k)
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
    summary_all = read_table("var_summary", cfg.paths.curated)
    if summary_all.empty:
        st.info("No var_summary data yet — run `eqdrisk var --date ...`.")
        return
    summary_all["asof_date"] = pd.to_datetime(summary_all["asof_date"]).dt.date
    full_days = sorted(
        summary_all.loc[summary_all["horizon_days"] == 10, "asof_date"].unique(), reverse=True
    )
    if not full_days:
        st.info("Only backfilled 1-day VaRs so far; run `eqdrisk var --date ...` for a full one.")
        return
    asof = st.selectbox("VaR date", full_days, key="var_date")
    s = summary_all[summary_all["asof_date"] == asof]

    def pick(
        measure: str, method: str, horizon: int = 1, conf: float = 0.99, **kw: object
    ) -> float:
        sel = (
            (s["measure"] == measure)
            & (s["method"] == method)
            & (s["horizon_days"] == horizon)
            & (s["confidence"].round(4) == conf)
            & (s["window_days"] == kw.get("window", 500))
            & (s["factors"] == kw.get("factors", "all"))
            & (s["positions"] == kw.get("positions", "book"))
        )
        rows = s.loc[sel, "value"]
        return float(rows.iloc[0]) if len(rows) else float("nan")

    st.caption(
        "Historical simulation on today's book: each past day's spot returns and vol-index "
        "moves (VIX / VXAPL / VXN) mapped onto the surface through the Step 8.3 proxy betas. "
        "Losses shown as positive dollars."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1-day 99% VaR, full reval", f"${pick('VaR', 'full'):,.0f}")
    c2.metric("1-day 99% VaR, Taylor", f"${pick('VaR', 'taylor'):,.0f}")
    c3.metric("10-day 99% VaR, overlapping", f"${pick('VaR', 'full', 10):,.0f}")
    c4.metric("10-day 99% VaR, sqrt(10) x 1-day", f"${pick('VaR', 'sqrt10_full', 10):,.0f}")

    st.subheader("1-day 99% VaR through time (500-day window)")
    ts = summary_all[
        (summary_all["measure"] == "VaR")
        & (summary_all["horizon_days"] == 1)
        & (summary_all["window_days"] == 500)
        & (summary_all["confidence"].round(4) == 0.99)
        & (summary_all["factors"] == "all")
        & (summary_all["positions"] == "book")
    ].pivot_table(index="asof_date", columns="method", values="value")
    fig = go.Figure()
    for method in ("full", "taylor"):
        if method in ts:
            fig.add_trace(go.Scatter(x=ts.index, y=ts[method], mode="lines+markers", name=method))
    fig.update_layout(yaxis_title="VaR ($)", height=360, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Window sensitivity, 1-day")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "window (days)": w,
                    "VaR 99% full": pick("VaR", "full", window=w),
                    "VaR 99% Taylor": pick("VaR", "taylor", window=w),
                    "ES 97.5% full": pick("ES", "full", conf=0.975, window=w),
                }
                for w in (250, 500, 1000)
            ]
        ),
        hide_index=True,
    )

    st.subheader("10-day ES 97.5% by factor group and liquidity horizon")
    groups = [("all", "all factors"), ("spot", "spot only"), ("vol", "vol only")]
    groups += [("single_name_vol", "single-name vol only")]
    groups += [("LH10", "LH >= 10 days"), ("LH20", "LH >= 20 days"), ("LH60", "LH >= 60 days")]
    rows = [
        {"factors": label, "ES": pick("ES", "full", 10, 0.975, factors=f)} for f, label in groups
    ]
    lh = s.loc[s["measure"] == "ES_LH", "value"]
    if len(lh):
        rows.append({"factors": "liquidity-horizon adjusted ES", "ES": float(lh.iloc[0])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    scen = read_table("var_scenarios", cfg.paths.curated)
    if not scen.empty:
        scen["asof_date"] = pd.to_datetime(scen["asof_date"]).dt.date
        one = scen[
            (scen["asof_date"] == asof) & (scen["horizon_days"] == 1) & (scen["factors"] == "all")
        ]
        last = sorted(one["scenario_date"].unique())[-500:]
        one = one[one["scenario_date"].isin(last)]
        st.subheader("Full revaluation vs Taylor, per scenario (1-day, book)")
        book = one.groupby("scenario_date")[["pnl_full", "pnl_taylor"]].sum()
        fig = go.Figure(
            go.Scatter(x=book["pnl_full"], y=book["pnl_taylor"], mode="markers", name="scenario")
        )
        lim = float(book.abs().to_numpy().max())
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="equal"))
        fig.update_layout(
            xaxis_title="full revaluation P&L ($)",
            yaxis_title="Taylor P&L ($)",
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        gap = one.assign(gap=one["pnl_taylor"] - one["pnl_full"]).groupby("position_id")
        st.dataframe(
            pd.DataFrame(
                {
                    "RMS P&L, full": gap["pnl_full"].apply(lambda x: (x**2).mean() ** 0.5),
                    "RMS Taylor gap": gap["gap"].apply(lambda x: (x**2).mean() ** 0.5),
                }
            ).reset_index(),
            hide_index=True,
        )

    betas = read_table("proxy_betas", cfg.paths.curated)
    if not betas.empty:
        betas["asof_date"] = pd.to_datetime(betas["asof_date"]).dt.date
        b = betas[betas["asof_date"] == asof]
        st.subheader("Vol proxy (Step 8.3): R^2 per grid node")
        underlying = st.selectbox("Underlying", sorted(b["underlying"].unique()), key="proxy_u")
        bu = b[b["underlying"] == underlying]
        st.caption(
            f"Daily vol-point change regressed on the {bu['index'].iloc[0]} change, "
            f"{int(bu['n_changes'].iloc[0])} stored days; nodes inside quoted strikes only."
        )
        st.dataframe(bu.pivot_table(index="T_label", columns="k", values="r2").round(2))
    errors = read_table("proxy_errors", cfg.paths.curated)
    if not errors.empty:
        errors["asof_date"] = pd.to_datetime(errors["asof_date"]).dt.date
        e = errors[errors["asof_date"] == asof]
        day = e.groupby("change_date")[["pnl_true", "pnl_proxy"]].sum()
        if not day.empty:
            st.subheader("Proxy error in dollars, today's book on each stored day's move")
            day["error"] = day["pnl_proxy"] - day["pnl_true"]
            st.dataframe(day.reset_index())


def render_backtest_tab(cfg: BaseConfig) -> None:
    summary = read_table("var_summary", cfg.paths.curated)
    days = int(summary["asof_date"].nunique()) if not summary.empty else 0
    st.warning(
        "Step 10 (VaR backtesting: Kupiec, Christoffersen, traffic light, PLA) is not yet "
        f"built — it needs about {VAR_ACCEPTANCE_DAYS} days of VaR forecasts against the "
        f"real P&L that followed. Days with a stored VaR so far: {days}."
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
    import numpy as np

    steps_all = read_table("pnl_explain", cfg.paths.curated)
    if steps_all.empty:
        st.info("No pnl_explain data yet — run `eqdrisk explainpnl --day0 ... --day1 ...`.")
        return

    st.subheader("Residual time series (basis points of NAV, with alert threshold)")
    daily = steps_all.groupby("asof_date").agg(residual=("residual", "sum"), nav=("nav", "max"))
    # `nav` is NaN for any day computed before this column existed (older curated
    # partitions genuinely don't have it) — left as NaN, not forced to 0.0, so
    # those days show as a gap in the chart rather than a misleading flat line.
    daily["residual_bp"] = 10_000.0 * daily["residual"] / daily["nav"].replace(0.0, np.nan)
    daily = daily.sort_index().reset_index()
    n_days = len(daily)
    breach_days = daily[daily["residual_bp"].abs() > RESIDUAL_ALERT_THRESHOLD_BP]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["asof_date"], y=daily["residual_bp"], mode="lines+markers", name="residual"
        )
    )
    fig.add_hline(y=RESIDUAL_ALERT_THRESHOLD_BP, line_dash="dash", line_color="red")
    fig.add_hline(y=-RESIDUAL_ALERT_THRESHOLD_BP, line_dash="dash", line_color="red")
    if not breach_days.empty:
        fig.add_trace(
            go.Scatter(
                x=breach_days["asof_date"],
                y=breach_days["residual_bp"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="breach",
            )
        )
    fig.update_layout(
        yaxis_title="Residual (bp of NAV)", height=350, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{n_days} real day-pair(s) of P&L explain available so far. The README's own "
        f"acceptance bar (median residual < 2bp of NAV) needs {PNL_EXPLAIN_ACCEPTANCE_DAYS}+ "
        "days to be a meaningful statistic — not reached yet, shown honestly rather than "
        f"computed on too small a sample. Dashed lines: the {RESIDUAL_ALERT_THRESHOLD_BP:.0f}bp "
        "alert threshold from `pricing/pnl_explain.py` (README 12.3's own suggested number)."
    )
    if not breach_days.empty:
        st.warning(f"{len(breach_days)} day(s) breached the residual alert threshold.")

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
    else:
        by_pos = by_pos_all[
            (by_pos_all["day0"] == day0) & (by_pos_all["asof_date"] == day1)
        ].sort_values("residual", key=lambda s: s.abs(), ascending=False)
        fig_pos = go.Figure(go.Bar(x=by_pos["position_id"], y=by_pos["residual"]))
        fig_pos.update_layout(
            yaxis_title="Residual ($)", height=350, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_pos, width="stretch")
        st.dataframe(by_pos, hide_index=True)

    st.subheader("AI investigation")
    st.caption(
        "AI-GENERATED HYPOTHESIS, NOT VERIFIED — produced by a local open-weight model "
        "(via Ollama, run daily, goes beyond the README's original 16-step plan) reading "
        "this same day pair's real P&L-explain output. Never auto-applied: proposed "
        "changes below are suggestions for a human to review, not actions taken."
    )
    ai_all = read_table("ai_investigations", cfg.paths.curated)
    ai_row = ai_all[(ai_all["day0"] == day0) & (ai_all["asof_date"] == day1)]
    if ai_row.empty:
        st.info("No AI investigation recorded for this day pair yet.")
    else:
        row = ai_row.iloc[0]
        if not bool(row["ai_available"]):
            st.warning("AI unavailable that day (Ollama not reachable) — no investigation run.")
        else:
            st.markdown(f"**Confidence:** {row['confidence']}")
            st.markdown(f"**Summary:** {row['summary']}")
            st.markdown(f"**Root cause hypothesis:** {row['root_cause_hypothesis']}")

            flagged_all = read_table("ai_flagged_positions", cfg.paths.curated)
            flagged = flagged_all[
                (flagged_all["day0"] == day0) & (flagged_all["asof_date"] == day1)
            ]
            if not flagged.empty:
                st.markdown("**Flagged for review:**")
                st.dataframe(flagged[["position_id", "reason"]], hide_index=True)

            changes_all = read_table("ai_proposed_changes", cfg.paths.curated)
            changes = changes_all[
                (changes_all["day0"] == day0) & (changes_all["asof_date"] == day1)
            ]
            if not changes.empty:
                st.markdown("**Proposed changes (not applied):**")
                st.dataframe(
                    changes[["parameter", "current_value", "suggested_value", "rationale"]],
                    hide_index=True,
                )

            trace_all = read_table("ai_investigation_trace", cfg.paths.curated)
            trace = trace_all[
                (trace_all["day0"] == day0) & (trace_all["asof_date"] == day1)
            ].sort_values("round")
            if not trace.empty:
                with st.expander(f"Investigation trace ({len(trace)} real tool call(s))"):
                    st.caption(
                        "What the model actually called mid-investigation (real pricing-engine "
                        "reprices or model-doc reads), with what arguments, and what came back — "
                        "the evidence of genuine tool use, not a claim about it."
                    )
                    st.dataframe(
                        trace[["round", "tool_name", "arguments", "result_summary"]],
                        hide_index=True,
                    )


def render_deep_hedging_tab(cfg: BaseConfig) -> None:
    st.caption(
        "Goes beyond the README's original 16 steps: PyTorch hedging policies trained against "
        "this project's own calibrated local-vol Monte Carlo simulator, compared out-of-sample "
        "against the existing Black-Scholes/MC-Greeks baseline. A separate, additive research "
        "comparison — never wired into the daily pipeline, never touching the pricing engine "
        "itself. Run `eqdrisk deephedge --date ... --instrument ... --loss ...` (or `--all`) to "
        "add a row; this tab only reads what was already trained and persisted."
    )
    results = read_table("deep_hedge_results", cfg.paths.curated)
    if results.empty:
        st.info("No deep-hedging results yet — run `eqdrisk deephedge --date ... --all`.")
        return

    dates = sorted(results["asof_date"].unique(), reverse=True)
    asof = st.selectbox("Asof date", dates)
    day = results[results["asof_date"] == asof].copy()

    day["std_reduction_pct"] = 100.0 * (1.0 - day["learned_std"] / day["baseline_std"])
    day["cvar_improvement_pct"] = (
        100.0 * (day["learned_cvar"] - day["baseline_cvar"]) / day["baseline_cvar"].abs()
    )
    # One row per training seed since Phase 6 (older runs have a single, unseeded
    # row): summarize each combination as the mean and the range across seeds.
    for col in ("seed", "learned_turnover", "baseline_turnover"):
        if col not in day.columns:
            day[col] = pd.NA  # rows written before Phase 6
    summary = (
        day.groupby(["underlying", "instrument", "loss_type"])
        .agg(
            seeds=("seed", lambda s: max(1, s.notna().sum())),
            std_reduction_pct=("std_reduction_pct", "mean"),
            std_reduction_min=("std_reduction_pct", "min"),
            std_reduction_max=("std_reduction_pct", "max"),
            cvar_improvement_pct=("cvar_improvement_pct", "mean"),
            learned_turnover=("learned_turnover", "mean"),
            baseline_turnover=("baseline_turnover", "mean"),
            learned_std=("learned_std", "mean"),
            baseline_std=("baseline_std", "mean"),
            epochs=("epochs", "mean"),
        )
        .reset_index()
        .sort_values(["instrument", "loss_type"])
    )

    st.subheader(f"Learned vs. baseline — {asof}")
    st.dataframe(summary, hide_index=True)
    st.caption(
        "Mean across training seeds, with the min/max range of the std reduction. "
        "std_reduction_pct > 0 means the learned policy has LOWER variance than the baseline; "
        "cvar_improvement_pct > 0 means its CVaR is better (less negative). Turnover is mean "
        "shares traded per path, counted the same way for a learned policy and its baseline. "
        "No single loss objective necessarily wins on every measure, and that is a real finding, "
        "not a bug (see `docs/deep_hedging_ml_framing.md`)."
    )

    combo_label = summary["loss_type"] + " / " + summary["instrument"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=combo_label, y=summary["std_reduction_pct"], name="std reduction %"))
    fig.update_layout(
        yaxis_title="Hedged P&L std reduction vs. baseline (%)",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Out-of-model robustness")
    robust = read_table("deep_hedge_robustness", cfg.paths.curated)
    if robust.empty or not (robust["asof_date"] == asof).any():
        st.info(
            "No robustness results for this date yet — run "
            "`eqdrisk deephedge-robustness --date ...` after `deephedge --all`."
        )
        return
    rday = robust[robust["asof_date"] == asof].copy()
    rday["std_reduction_pct"] = 100.0 * (1.0 - rday["learned_std"] / rday["baseline_std"])
    rday["cvar_improvement_pct"] = (
        100.0 * (rday["learned_cvar"] - rday["baseline_cvar"]) / rday["baseline_cvar"].abs()
    )
    for metric, label in [
        ("std_reduction_pct", "Std reduction vs. baseline (%)"),
        ("cvar_improvement_pct", "CVaR improvement vs. baseline (%)"),
    ]:
        st.markdown(f"**{label}**, mean across seeds, by scenario")
        pivot = (
            rday.groupby(["instrument", "loss_type", "scenario"])[metric]
            .mean()
            .unstack("scenario")
            .reindex(columns=["in_sample", "vol_up_25", "vol_down_25", "jumps", "history"])
            .round(1)
            .reset_index()
        )
        st.dataframe(pivot, hide_index=True)
    history = rday[rday["scenario"] == "history"]
    if not history.empty:
        st.caption(
            "Policies trained on the as-of date's calibrated local-vol model, then scored with no "
            "retraining on stressed simulators and on real price history (rolling windows of each "
            "instrument's life). History windows overlap: effective independent samples per "
            "instrument: "
            + ", ".join(
                f"{inst} {n:.0f}"
                for inst, n in history.groupby("instrument")["n_independent"].first().items()
            )
            + "."
        )


def main() -> None:
    st.set_page_config(page_title="EQD Risk Engine", layout="wide")
    st.title("Equity Derivatives Risk Engine")
    st.caption(
        "Every number on this page is read from a stored curated artifact the daily "
        "pipeline already wrote — no pricing or calibration runs from this app."
    )

    cfg = load_cfg()
    tabs = st.tabs(
        ["Surface", "Exposure", "VaR", "Backtest", "Stress", "P&L explain", "Deep Hedging"]
    )
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
    with tabs[6]:
        render_deep_hedging_tab(cfg)


main()
