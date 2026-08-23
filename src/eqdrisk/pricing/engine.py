"""Vanilla pricing + Greeks orchestration for one asof date (Step 5).

Prices every quality-filtered quote that fed calibration (`implied_vols`,
reason='OK') off the CALIBRATED surface, not the raw quoted IV — consistent with
the rest of the project's vol/price separation: the surface is the model, prices
and Greeks are its outputs. Two other inputs are joined in:

- `forwards` (Step 2) for F per (underlying, expiry).
- The bootstrapped rate curve (`curves` -> `bootstrap_curve`, same as Step 2) for
  discounting, not `forwards.discount_factor_implied` — Step 2 already documented
  that the parity-regression discount factor carries the option-market/financing
  basis (`discount_factor_diff_bp`), not a clean risk-free rate. Pricing wants the
  clean curve rate.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import GREEKS_REQUIRED_NOT_NULL, GREEKS_SCHEMA, validate
from eqdrisk.marketdata.curve import bootstrap_curve
from eqdrisk.pricing.blackscholes import compute_greeks
from eqdrisk.pricing.stickiness import compute_stickiness_deltas
from eqdrisk.vol.ssvi import SSVIParams
from eqdrisk.vol.svi import SVIParams


@dataclass
class PricingResult:
    asof: dt.date
    n_priced: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Pricing — {self.asof}"]
        for underlying, n in self.n_priced.items():
            lines.append(f"  {underlying}: {n} quotes priced")
        for underlying, reasons in self.skipped.items():
            lines.append(f"  {underlying}: skipped — {reasons}")
        return "\n".join(lines)


def _model_sigma_and_svi(
    surface_row: pd.Series, k: np.ndarray, T: float
) -> tuple[np.ndarray, SVIParams | SSVIParams]:
    """Returns model implied vol at `k`, plus the params object used to compute it.

    Sticky-delta/local-vol deltas need dsigma/dk, which only has a closed form for
    SVI (`SVIParams.first_derivative`). SSVI slices (the calendar-arbitrage fallback)
    don't get a stickiness comparison here — the caller checks `isinstance(..., SVIParams)`
    and reports null stickiness deltas for SSVI rows rather than forcing SSVI through an
    SVI shape it wasn't fit with, or silently approximating with a finite difference.
    """
    if surface_row["model"] == "SVI":
        params = SVIParams(
            a=float(surface_row["a"]),
            b=float(surface_row["b"]),
            rho=float(surface_row["rho"]),
            m=float(surface_row["m"]),
            sigma=float(surface_row["sigma"]),
        )
        w = params.total_variance(k)
        return np.sqrt(np.clip(w, 1e-12, None) / T), params
    ssvi = SSVIParams(rho=float(surface_row["rho"]), eta=float(surface_row["eta"]))
    theta = float(surface_row["theta"])
    w = ssvi.total_variance(k, theta)
    return np.sqrt(np.clip(w, 1e-12, None) / T), ssvi


def run_pricing(
    cfg: BaseConfig, asof: dt.date, underlyings: list[str] | None = None
) -> PricingResult:
    curated_root = Path(cfg.paths.curated)
    vol_surface_root = curated_root / "vol_surface"
    iv_root = curated_root / "implied_vols"
    forwards_root = curated_root / "forwards"
    curves_root = curated_root / "curves"
    underlying_root = curated_root / "underlyings"

    universe = (
        underlyings if underlyings is not None else cfg.universe.index + cfg.universe.single_names
    )
    result = PricingResult(asof=asof)

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

    for underlying in universe:
        surface = store.query(
            f"SELECT * FROM vs WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"vs": str(vol_surface_root)},
        ).to_pandas()
        if surface.empty:
            result.skipped[underlying] = ["no calibrated surface for this date"]
            continue

        ivs = store.query(
            f"SELECT * FROM iv WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}' AND reason = 'OK'",
            views={"iv": str(iv_root)},
        ).to_pandas()
        if ivs.empty:
            result.skipped[underlying] = ["no OK-tagged implied vols for this date"]
            continue

        forwards = store.query(
            f"SELECT * FROM fwd WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"fwd": str(forwards_root)},
        ).to_pandas()
        forward_by_expiry = {row["expiry"]: float(row["forward"]) for _, row in forwards.iterrows()}

        spot_df = store.query(
            f"SELECT close FROM u WHERE asof_date = DATE '{asof.isoformat()}' "
            f"AND underlying = '{underlying}'",
            views={"u": str(underlying_root)},
        ).to_pandas()
        if spot_df.empty:
            result.skipped[underlying] = ["no spot price for this date"]
            continue
        spot = float(spot_df["close"].iloc[0])

        surface_by_expiry = {row["expiry"]: row for _, row in surface.iterrows()}
        n_priced = 0

        for expiry, quotes in ivs.groupby("expiry"):
            expiry_date = pd.Timestamp(expiry).date()
            surface_row = surface_by_expiry.get(expiry_date)
            forward = forward_by_expiry.get(expiry_date)
            if surface_row is None or forward is None:
                continue

            T = float(quotes["T"].iloc[0])
            discount_factor = curve.discount_factor(T)
            k = quotes["k"].to_numpy()
            sigma_model, svi_for_stickiness = _model_sigma_and_svi(surface_row, k, T)

            for row_sigma, (_, quote) in zip(sigma_model, quotes.iterrows(), strict=True):
                strike = float(quote["strike"])
                is_call = quote["cp"] == "C"
                greeks = compute_greeks(
                    is_call, forward, strike, T, float(row_sigma), discount_factor, spot
                )
                sticky = None
                if isinstance(svi_for_stickiness, SVIParams):
                    sticky = compute_stickiness_deltas(
                        is_call,
                        forward,
                        strike,
                        T,
                        float(row_sigma),
                        discount_factor,
                        spot,
                        svi_for_stickiness,
                    )
                out_rows.append(
                    {
                        "asof_date": asof,
                        "underlying": underlying,
                        "expiry": expiry_date,
                        "strike": strike,
                        "cp": quote["cp"],
                        "T": T,
                        "forward": forward,
                        "spot": spot,
                        "discount_factor": discount_factor,
                        "model": surface_row["model"],
                        "sigma": float(row_sigma),
                        "price": greeks.price,
                        "delta_spot": greeks.delta_spot,
                        "delta_forward": greeks.delta_forward,
                        "gamma_spot": greeks.gamma_spot,
                        "vega": greeks.vega,
                        "theta": greeks.theta,
                        "rho": greeks.rho,
                        "dividend_rho": greeks.dividend_rho,
                        "vanna_spot": greeks.vanna_spot,
                        "volga": greeks.volga,
                        "delta_sticky_strike": None if sticky is None else sticky.sticky_strike,
                        "delta_sticky_delta": None if sticky is None else sticky.sticky_delta,
                        "delta_sticky_local_vol": (
                            None if sticky is None else sticky.sticky_local_vol
                        ),
                    }
                )
                n_priced += 1

        result.n_priced[underlying] = n_priced

    if out_rows:
        table = validate(pd.DataFrame(out_rows), GREEKS_SCHEMA, GREEKS_REQUIRED_NOT_NULL)
        store.write_partitioned(table, curated_root / "greeks", ["asof_date", "underlying"])

    return result
