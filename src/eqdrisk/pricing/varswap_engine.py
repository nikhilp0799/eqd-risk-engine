"""Variance swap orchestration for one asof date: compute the fair strike for
every calibrated expiry per underlying, off both the real (tradeable) strike
range and a wide (parametric-smile-trusted) range, persist both plus the
truncation error, and cross-check the nearest-30-day fair strike against the
published VIX print (the same comparison Step 4's calibration diagnostics
already made for ATM vol alone — repurposed here for the actual variance-swap
fair strike, which is what VIX itself is designed to replicate).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import VARSWAP_REQUIRED_NOT_NULL, VARSWAP_SCHEMA, validate
from eqdrisk.marketdata.curve import bootstrap_curve
from eqdrisk.pricing.varswap import fair_variance_strike, wide_k_cap

MIN_OBSERVED_STRIKES = 2  # need at least a put and a call side to bound a "narrow" range
VIX_TARGET_T = 30 / 365


@dataclass
class VarSwapRunResult:
    asof: dt.date
    n_priced: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    vix_check: dict | None = None

    def render(self) -> str:
        lines = [f"Variance swaps — {self.asof}"]
        for underlying, n in self.n_priced.items():
            lines.append(f"  {underlying}: {n} expiries priced")
        for underlying, reasons in self.skipped.items():
            lines.append(f"  {underlying}: skipped — {reasons}")
        if self.vix_check:
            lines.append(f"  VIX cross-check: {self.vix_check}")
        return "\n".join(lines)


def run_varswap(
    cfg: BaseConfig, asof: dt.date, underlyings: list[str] | None = None
) -> VarSwapRunResult:
    curated_root = Path(cfg.paths.curated)
    vol_surface_root = curated_root / "vol_surface"
    iv_root = curated_root / "implied_vols"
    forwards_root = curated_root / "forwards"
    curves_root = curated_root / "curves"
    vol_index_root = curated_root / "vol_indices"

    universe = (
        underlyings if underlyings is not None else cfg.universe.index + cfg.universe.single_names
    )
    result = VarSwapRunResult(asof=asof)

    if not vol_surface_root.exists() or not any(vol_surface_root.rglob("*.parquet")):
        result.skipped["_all_"] = ["no vol_surface data at all"]
        return result

    curves_date = store.latest_available_date(curves_root, asof)
    if curves_date is None:
        result.skipped["_all_"] = ["no curated rates available"]
        return result
    rates = store.query(
        f"SELECT * FROM curves WHERE asof_date = DATE '{curves_date.isoformat()}'",
        views={"curves": str(curves_root)},
    ).to_pandas()
    curve = bootstrap_curve(rates)

    out_rows = []
    # underlying -> (T, fair_strike_wide) of the calibrated expiry nearest 30 days
    nearest_by_underlying: dict[str, tuple[float, float]] = {}

    for underlying in universe:
        surface = store.query(
            f"SELECT * FROM vs WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"vs": str(vol_surface_root)},
        ).to_pandas()
        if surface.empty:
            result.skipped[underlying] = ["no calibrated surface for this date"]
            continue

        forwards = store.query(
            f"SELECT * FROM fwd WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"fwd": str(forwards_root)},
        ).to_pandas()
        forward_by_expiry = {row["expiry"]: float(row["forward"]) for _, row in forwards.iterrows()}

        ivs = pd.DataFrame()
        if iv_root.exists() and any(iv_root.rglob("*.parquet")):
            ivs = store.query(
                f"SELECT * FROM iv WHERE asof_date = DATE '{asof.isoformat()}' "
                f"AND underlying = '{underlying}' AND reason = 'OK'",
                views={"iv": str(iv_root)},
            ).to_pandas()

        n_priced = 0
        best_gap = None

        for _, row in surface.iterrows():
            expiry_date = row["expiry"]
            T = float(row["T"])
            forward = forward_by_expiry.get(expiry_date)
            if forward is None:
                continue
            discount_factor = curve.discount_factor(T)

            k_cap = wide_k_cap(row, T)
            fair_wide = fair_variance_strike(row, forward, T, discount_factor, -k_cap, k_cap)
            fair_strike_vol_wide = float(np.sqrt(max(fair_wide, 0.0)) * 100)

            fair_strike_vol_narrow = None
            truncation_error = None
            k_obs_min = k_obs_max = None
            n_obs = 0
            if not ivs.empty:
                slice_ivs = ivs[ivs["expiry"] == expiry_date]
                n_obs = len(slice_ivs)
                if n_obs >= MIN_OBSERVED_STRIKES:
                    k_obs_min = float(slice_ivs["k"].min())
                    k_obs_max = float(slice_ivs["k"].max())
                    fair_narrow = fair_variance_strike(
                        row, forward, T, discount_factor, k_obs_min, k_obs_max
                    )
                    fair_strike_vol_narrow = float(np.sqrt(max(fair_narrow, 0.0)) * 100)
                    truncation_error = fair_strike_vol_wide - fair_strike_vol_narrow

            out_rows.append(
                {
                    "asof_date": asof,
                    "underlying": underlying,
                    "expiry": expiry_date,
                    "T": T,
                    "forward": forward,
                    "discount_factor": discount_factor,
                    "fair_strike_vol_narrow": fair_strike_vol_narrow,
                    "fair_strike_vol_wide": fair_strike_vol_wide,
                    "truncation_error_vol_points": truncation_error,
                    "n_strikes_observed": n_obs,
                    "k_obs_min": k_obs_min,
                    "k_obs_max": k_obs_max,
                    "k_wide_cap": k_cap,
                }
            )
            n_priced += 1

            gap = abs(T - VIX_TARGET_T)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                nearest_by_underlying[underlying] = (T, fair_strike_vol_wide)

        result.n_priced[underlying] = n_priced

    if out_rows:
        table = validate(pd.DataFrame(out_rows), VARSWAP_SCHEMA, VARSWAP_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "varswap", ["asof_date", "underlying"])

    result.vix_check = _vix_cross_check(nearest_by_underlying, vol_index_root, asof)
    return result


def _vix_cross_check(
    nearest_by_underlying: dict[str, tuple[float, float]], vol_index_root: Path, asof: dt.date
) -> dict | None:
    """Compare our own SPX fair variance-swap strike (nearest calibrated expiry to
    30 days) against the published VIX print — VIX is itself defined as this exact
    Carr-Madan replication at the 30-day point, so this is a much more direct
    apples-to-apples check than Step 4's ATM-vol-only comparison."""
    if "SPX" not in nearest_by_underlying:
        return None
    if not vol_index_root.exists() or not any(vol_index_root.rglob("*.parquet")):
        return None

    vix_date = store.latest_available_date(vol_index_root, asof)
    if vix_date is None:
        return None
    vix_df = store.query(
        f"SELECT value FROM vi WHERE asof_date = DATE '{vix_date.isoformat()}' AND index = 'VIX'",
        views={"vi": str(vol_index_root)},
    ).to_pandas()
    if vix_df.empty:
        return None
    vix_value = float(vix_df["value"].iloc[0])

    nearest_T, our_fair_strike = nearest_by_underlying["SPX"]
    return {
        "our_fair_strike_nearest_30d": round(our_fair_strike, 3),
        "nearest_expiry_T": round(nearest_T, 4),
        "vix": vix_value,
        "vix_date": vix_date.isoformat(),
        "abs_diff_vol_points": round(abs(our_fair_strike - vix_value), 3),
        "note": "VIX is exactly this replication at 30d; single-day level comparison only",
    }
