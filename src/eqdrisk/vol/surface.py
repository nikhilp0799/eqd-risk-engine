"""Per-underlying surface calibration orchestration: SVI per slice, butterfly repair,
cross-slice calendar-arbitrage check with SSVI fallback, SABR comparison at the short
end, diagnostics (RMSE, violations, VIX cross-check), and persistence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import VOL_SURFACE_REQUIRED_NOT_NULL, VOL_SURFACE_SCHEMA, validate
from eqdrisk.vol.sabr import DEFAULT_BETA, fit_sabr_slice, sabr_implied_vol
from eqdrisk.vol.ssvi import SSVIParams, fit_ssvi
from eqdrisk.vol.svi import (
    SVIParams,
    count_butterfly_violations,
    fit_svi_slice,
    repair_butterfly_violation,
)

K_GRID_MARGIN = 0.05
K_GRID_POINTS = 200
CALENDAR_ARB_TOLERANCE = 1e-8  # total-variance units; guards against float noise only


@dataclass
class SliceCalibration:
    underlying: str
    expiry: dt.date
    T: float
    model: str
    params: dict
    n_points: int
    rmse_vol_points: float
    max_abs_error_vol_points: float
    max_abs_error_k: float
    butterfly_violations: int
    calendar_violated: bool


@dataclass
class SurfaceCalibrationResult:
    asof: dt.date
    slices: list[SliceCalibration] = field(default_factory=list)
    sabr_comparison: dict[str, dict] = field(default_factory=dict)
    vix_check: dict | None = None
    skipped: dict[str, list[str]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Surface calibration — {self.asof}"]
        by_underlying: dict[str, list[SliceCalibration]] = {}
        for s in self.slices:
            by_underlying.setdefault(s.underlying, []).append(s)
        for underlying, slices in by_underlying.items():
            models = {s.model for s in slices}
            total_butterfly = sum(s.butterfly_violations for s in slices)
            worst_rmse = max((s.rmse_vol_points for s in slices), default=0.0)
            lines.append(
                f"  {underlying}: {len(slices)} slices, model(s)={sorted(models)}, "
                f"worst RMSE={worst_rmse:.3f} vol pts, butterfly violations={total_butterfly}"
            )
            if underlying in self.sabr_comparison:
                cmp = self.sabr_comparison[underlying]
                lines.append(
                    f"    SABR vs SVI (shortest expiry): "
                    f"RMSE_sabr={cmp['rmse_sabr']:.3f}, RMSE_svi={cmp['rmse_svi']:.3f}"
                )
            skipped = self.skipped.get(underlying)
            if skipped:
                lines.append(f"    skipped: {skipped}")
        if self.vix_check:
            lines.append(f"  VIX cross-check: {self.vix_check}")
        return "\n".join(lines)


def _rmse_and_max_error_vol_points(
    iv_fit: np.ndarray, iv_actual: np.ndarray, k: np.ndarray
) -> tuple[float, float, float]:
    diff_vol_points = (iv_fit - iv_actual) * 100.0
    rmse = float(np.sqrt(np.mean(diff_vol_points**2)))
    idx = int(np.argmax(np.abs(diff_vol_points)))
    return rmse, float(abs(diff_vol_points[idx])), float(k[idx])


def _calendar_arbitrage_violated(
    fitted_slices: list[tuple[float, SVIParams, tuple[float, float]]],
) -> bool:
    """Check w(k,T1) <= w(k,T2) for every pair T1<T2, on the overlap of their observed
    k-ranges (checking outside a slice's data support isn't meaningful — extrapolation
    behaviour there isn't what "calendar arbitrage" is diagnosing)."""
    ordered = sorted(fitted_slices, key=lambda x: x[0])
    for i in range(len(ordered) - 1):
        _, p1, range1 = ordered[i]
        for j in range(i + 1, len(ordered)):
            _, p2, range2 = ordered[j]
            lo = max(range1[0], range2[0])
            hi = min(range1[1], range2[1])
            if lo >= hi:
                continue
            grid = np.linspace(lo, hi, 50)
            if np.any(p1.total_variance(grid) > p2.total_variance(grid) + CALENDAR_ARB_TOLERANCE):
                return True
    return False


def calibrate_underlying(
    underlying: str, groups: dict[dt.date, pd.DataFrame]
) -> tuple[list[SliceCalibration], dict | None]:
    svi_fits: dict[dt.date, tuple[SVIParams, pd.DataFrame, float]] = {}
    for expiry, df in groups.items():
        T = float(df["T"].iloc[0])
        k = df["k"].to_numpy()
        w = df["total_variance"].to_numpy()
        weights = df["weight"].to_numpy()
        params = fit_svi_slice(k, w, weights)
        svi_fits[expiry] = (params, df, T)

    calendar_check_input = [
        (T, params, (float(df["k"].min()) - K_GRID_MARGIN, float(df["k"].max()) + K_GRID_MARGIN))
        for params, df, T in svi_fits.values()
    ]
    calendar_violated = _calendar_arbitrage_violated(calendar_check_input)

    ssvi_model: SSVIParams | None = None
    if calendar_violated:
        ssvi_slices = [
            (
                float(params.total_variance(0.0)),
                df["k"].to_numpy(),
                df["total_variance"].to_numpy(),
                df["weight"].to_numpy(),
            )
            for params, df, _ in svi_fits.values()
        ]
        ssvi_model = fit_ssvi(ssvi_slices)

    results = []
    for expiry, (params, df, T) in svi_fits.items():
        k = df["k"].to_numpy()
        w = df["total_variance"].to_numpy()
        iv_actual = np.sqrt(w / T)

        if ssvi_model is not None:
            theta = float(params.total_variance(0.0))
            w_fit = ssvi_model.total_variance(k, theta)
            model_name = "SSVI"
            model_params = {"rho": ssvi_model.rho, "eta": ssvi_model.eta, "theta": theta}
            butterfly_violations = int(
                np.sum(
                    _numeric_durrleman_g(
                        lambda kk, _theta=theta: ssvi_model.total_variance(kk, _theta), k
                    )
                    < 0
                )
            )
        else:
            grid = np.linspace(k.min() - K_GRID_MARGIN, k.max() + K_GRID_MARGIN, K_GRID_POINTS)
            n_violations = count_butterfly_violations(params, grid)
            if n_violations > 0:
                params = repair_butterfly_violation(k, w, df["weight"].to_numpy(), grid, params)
                n_violations = count_butterfly_violations(params, grid)
            w_fit = params.total_variance(k)
            model_name = "SVI"
            model_params = {
                "a": params.a,
                "b": params.b,
                "rho": params.rho,
                "m": params.m,
                "sigma": params.sigma,
            }
            butterfly_violations = n_violations

        iv_fit = np.sqrt(np.clip(w_fit, 1e-12, None) / T)
        rmse, max_err, max_err_k = _rmse_and_max_error_vol_points(iv_fit, iv_actual, k)

        results.append(
            SliceCalibration(
                underlying=underlying,
                expiry=expiry,
                T=T,
                model=model_name,
                params=model_params,
                n_points=len(k),
                rmse_vol_points=rmse,
                max_abs_error_vol_points=max_err,
                max_abs_error_k=max_err_k,
                butterfly_violations=butterfly_violations,
                calendar_violated=calendar_violated,
            )
        )

    sabr_comparison = None
    if results:
        shortest = min(results, key=lambda r: r.T)
        df = svi_fits[shortest.expiry][1]
        strikes_k = df["k"].to_numpy()
        iv_actual = np.sqrt(df["total_variance"].to_numpy() / shortest.T)
        forward_proxy = 1.0  # working directly in k = log(K/F) space, so F=1, K=exp(k)
        strikes = np.exp(strikes_k)
        sabr_params = fit_sabr_slice(
            forward_proxy,
            strikes,
            iv_actual,
            shortest.T,
            df["weight"].to_numpy(),
            beta=DEFAULT_BETA,
        )
        iv_sabr = np.array(
            [sabr_implied_vol(forward_proxy, kk, shortest.T, sabr_params) for kk in strikes]
        )
        rmse_sabr = float(np.sqrt(np.mean((100 * (iv_sabr - iv_actual)) ** 2)))
        sabr_comparison = {
            "expiry": shortest.expiry,
            "rmse_sabr": rmse_sabr,
            "rmse_svi": shortest.rmse_vol_points,
            "sabr_params": {
                "alpha": sabr_params.alpha,
                "beta": sabr_params.beta,
                "rho": sabr_params.rho,
                "nu": sabr_params.nu,
            },
        }

    return results, sabr_comparison


def _numeric_durrleman_g(w_func, k: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Generic finite-difference Durrleman check for any total-variance callable —
    used for SSVI, which doesn't have the same closed-form derivatives as SVI."""
    w = w_func(k)
    wp = (w_func(k + h) - w_func(k - h)) / (2 * h)
    wpp = (w_func(k + h) - 2 * w + w_func(k - h)) / h**2
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def run_surface_calibration(
    cfg: BaseConfig, asof: dt.date, underlyings: list[str] | None = None
) -> SurfaceCalibrationResult:
    curated_root = Path(cfg.paths.curated)
    iv_root = curated_root / "implied_vols"
    vol_index_root = curated_root / "vol_indices"
    universe = (
        underlyings if underlyings is not None else cfg.universe.index + cfg.universe.single_names
    )

    result = SurfaceCalibrationResult(asof=asof)
    out_rows = []

    for underlying in universe:
        if not iv_root.exists() or not any(iv_root.rglob("*.parquet")):
            result.skipped[underlying] = ["no implied_vols data at all"]
            continue
        ivs = store.query(
            f"SELECT * FROM iv WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}' AND reason = 'OK'",
            views={"iv": str(iv_root)},
        ).to_pandas()
        if ivs.empty:
            result.skipped[underlying] = ["no OK-tagged implied vols for this date"]
            continue

        groups = {pd.Timestamp(expiry).date(): df for expiry, df in ivs.groupby("expiry")}
        slices, sabr_comparison = calibrate_underlying(underlying, groups)
        result.slices.extend(slices)
        if sabr_comparison:
            result.sabr_comparison[underlying] = sabr_comparison

        for s in slices:
            out_rows.append(
                {
                    "asof_date": asof,
                    "underlying": s.underlying,
                    "expiry": s.expiry,
                    "T": s.T,
                    "model": s.model,
                    "a": s.params.get("a"),
                    "b": s.params.get("b"),
                    "rho": s.params.get("rho"),
                    "m": s.params.get("m"),
                    "sigma": s.params.get("sigma"),
                    "eta": s.params.get("eta"),
                    "theta": s.params.get("theta"),
                    "n_points": s.n_points,
                    "rmse_vol_points": s.rmse_vol_points,
                    "max_abs_error_vol_points": s.max_abs_error_vol_points,
                    "max_abs_error_k": s.max_abs_error_k,
                    "butterfly_violations": s.butterfly_violations,
                    "calendar_violated": s.calendar_violated,
                }
            )

    if out_rows:
        table = validate(pd.DataFrame(out_rows), VOL_SURFACE_SCHEMA, VOL_SURFACE_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "vol_surface", ["asof_date", "underlying"])

    result.vix_check = _vix_cross_check(result, vol_index_root, asof)
    return result


def _vix_cross_check(
    result: SurfaceCalibrationResult, vol_index_root: Path, asof: dt.date
) -> dict | None:
    """Compare our own SPX 30d ATM vol against the published VIX print for the same date.

    Single-day level comparison only — the README's "correlation > 0.98" target needs a
    time series, which we don't have yet (free-tier, single-snapshot data). Tracked as an
    open item, not silently claimed as satisfied.
    """
    spx_slices = [s for s in result.slices if s.underlying == "SPX"]
    if not spx_slices or not vol_index_root.exists() or not any(vol_index_root.rglob("*.parquet")):
        return None

    target_T = 30 / 365
    nearest = min(spx_slices, key=lambda s: abs(s.T - target_T))
    atm_theta = nearest.params.get("theta")
    if atm_theta is None:
        a, b, rho, m, sigma = (
            nearest.params["a"],
            nearest.params["b"],
            nearest.params["rho"],
            nearest.params["m"],
            nearest.params["sigma"],
        )
        atm_theta = SVIParams(a, b, rho, m, sigma).total_variance(0.0)
    our_atm_vol = float(np.sqrt(atm_theta / nearest.T) * 100)

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

    return {
        "our_atm_vol_nearest_30d": round(our_atm_vol, 3),
        "nearest_expiry_T": round(nearest.T, 4),
        "vix": vix_value,
        "vix_date": vix_date.isoformat(),
        "abs_diff_vol_points": round(abs(our_atm_vol - vix_value), 3),
        "note": (
            "single-day level comparison only; correlation needs a time series we don't have yet"
        ),
    }
