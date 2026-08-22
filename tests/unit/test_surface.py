import datetime as dt

import numpy as np
import pandas as pd

from eqdrisk.vol.ssvi import SSVIParams
from eqdrisk.vol.surface import calibrate_underlying, run_surface_calibration
from eqdrisk.vol.svi import SVIParams


def _slice_df(k: np.ndarray, w: np.ndarray, weight: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"k": k, "total_variance": w, "weight": np.full_like(k, weight)})


def test_calibrate_underlying_uses_svi_when_no_calendar_violation():
    # Two slices with strictly increasing total variance everywhere -> no calendar arb.
    k = np.linspace(-0.3, 0.3, 20)
    p1 = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.03, b=0.12, rho=-0.3, m=0.0, sigma=0.15)
    groups = {
        dt.date(2026, 9, 1): _slice_df(k, p1.total_variance(k)).assign(T=0.1),
        dt.date(2026, 12, 1): _slice_df(k, p2.total_variance(k)).assign(T=0.35),
    }

    results, sabr = calibrate_underlying("TEST", groups)

    assert {r.model for r in results} == {"SVI"}
    assert all(not r.calendar_violated for r in results)
    assert sabr is not None


def test_calibrate_underlying_falls_back_to_ssvi_on_calendar_violation():
    # T2's total variance is LOWER than T1's everywhere -> calendar arbitrage.
    k = np.linspace(-0.3, 0.3, 20)
    p1 = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.15)
    groups = {
        dt.date(2026, 9, 1): _slice_df(k, p1.total_variance(k)).assign(T=0.1),
        dt.date(2026, 12, 1): _slice_df(k, p2.total_variance(k)).assign(T=0.35),
    }

    results, _ = calibrate_underlying("TEST", groups)

    assert {r.model for r in results} == {"SSVI"}
    assert all(r.calendar_violated for r in results)
    # SSVI is arbitrage-free by construction -> should confirm zero butterfly violations too
    assert all(r.butterfly_violations == 0 for r in results)


def test_calibrate_underlying_skips_gracefully_with_one_slice():
    """SABR comparison and calendar check both need >= 1 slice; single-slice input
    should calibrate fine without crashing on comparisons that need 2+ slices."""
    k = np.linspace(-0.3, 0.3, 20)
    p = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    groups = {dt.date(2026, 9, 1): _slice_df(k, p.total_variance(k)).assign(T=0.1)}

    results, sabr = calibrate_underlying("TEST", groups)

    assert len(results) == 1
    assert results[0].model == "SVI"
    assert sabr is not None


def _write_synthetic_iv_store(root, asof, expiries_and_params, underlying="TEST"):
    from eqdrisk.io import store
    from eqdrisk.io.schemas import IMPLIED_VOL_REQUIRED_NOT_NULL, IMPLIED_VOL_SCHEMA, validate

    rows = []
    for expiry, T, params in expiries_and_params:
        k = np.linspace(-0.3, 0.3, 20)
        w = params.total_variance(k)
        iv = np.sqrt(w / T)
        n = len(k)
        rows.append(
            pd.DataFrame(
                {
                    "asof_date": [asof] * n,
                    "underlying": [underlying] * n,
                    "expiry": [expiry] * n,
                    "strike": np.exp(k) * 100.0,
                    "cp": ["C" if kk > 0 else "P" for kk in k],
                    "T": [T] * n,
                    "k": k,
                    "iv": iv,
                    "total_variance": w,
                    "vega": np.ones(n),
                    "weight": np.ones(n),
                    "reason": ["OK"] * n,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    table = validate(df, IMPLIED_VOL_SCHEMA, IMPLIED_VOL_REQUIRED_NOT_NULL)
    store.write_partitioned(table, root / "implied_vols", ["asof_date", "underlying"])


def test_run_surface_calibration_end_to_end(tmp_path):
    from eqdrisk.config import BaseConfig, Paths, Universe

    asof = dt.date(2026, 8, 20)
    p1 = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.03, b=0.12, rho=-0.3, m=0.0, sigma=0.15)
    _write_synthetic_iv_store(
        tmp_path,
        asof,
        [(dt.date(2026, 9, 1), 0.1, p1), (dt.date(2026, 12, 1), 0.35, p2)],
    )

    cfg = BaseConfig(
        run_date=asof,
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_surface_calibration(cfg, asof, underlyings=["TEST"])

    assert len(result.slices) == 2
    assert "TEST" in result.sabr_comparison

    from eqdrisk.io import store

    out = store.query("SELECT * FROM vs", views={"vs": str(tmp_path / "vol_surface")}).to_pandas()
    assert len(out) == 2
    assert set(out["model"]) == {"SVI"}


def test_ssvi_params_type_used_in_calendar_fallback():
    # Sanity: SSVIParams import above is exercised via the fallback path already
    # covered in test_calibrate_underlying_falls_back_to_ssvi_on_calendar_violation.
    assert SSVIParams(rho=0.0, eta=1.0).total_variance(0.0, 0.01) > 0
