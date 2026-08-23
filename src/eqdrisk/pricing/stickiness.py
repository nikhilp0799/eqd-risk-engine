"""Sticky-strike vs. sticky-delta vs. sticky-local-vol total delta (README 5.3).

"The model delta is not the hedge delta, because implied vol moves with spot":

    Delta_total = dV/dS + dV/dsigma * dsigma/dS

Three conventions for dsigma/dS, all keyed off the calibrated SVI slice's own
first derivative dw/dk (`SVIParams.first_derivative`, already exists and is
unit-tested against finite differences in `test_svi.py` — no new vol
differentiation code needed here, just the chain rule to spot):

- Sticky strike: implied vol at a fixed absolute strike K doesn't move when spot
  moves -> dsigma/dS = 0.
- Sticky delta / moneyness: implied vol at a fixed K/F doesn't move -> the smile
  in k = log(K/F) is unchanged, so as F (hence S) moves, sigma(K) moves along the
  fixed sigma(k) curve. Since dk/dS = -1/S (k = log(K/F), dF/dS = F/S, so
  dk/dF * dF/dS = (-1/F)*(F/S) = -1/S), dsigma/dS = dsigma/dk * (-1/S).
- Sticky local vol: per README, approximated as ~2x the sticky-delta dsigma/dS
  (the local-vol backbone moves roughly twice as fast as the implied-vol
  backbone under a sticky-delta smile) — used as given, not re-derived from
  Dupire here (that's Step 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqdrisk.pricing.blackscholes import delta_spot, vega
from eqdrisk.vol.svi import SVIParams

STICKY_LOCAL_VOL_MULTIPLIER = 2.0


@dataclass
class StickinessDeltas:
    sticky_strike: float
    sticky_delta: float
    sticky_local_vol: float


def dsigma_dk(svi_params: SVIParams, k: float, T: float, sigma: float) -> float:
    """sigma = sqrt(w(k)/T) => dsigma/dk = w'(k) / (2*sigma*T)."""
    return float(svi_params.first_derivative(k)) / (2 * sigma * T)


def compute_stickiness_deltas(
    is_call: bool,
    forward: float,
    strike: float,
    T: float,
    sigma: float,
    discount_factor: float,
    spot: float,
    svi_params: SVIParams,
) -> StickinessDeltas:
    """`svi_params` must be the calibrated slice's own params (k = log(strike/forward)
    is derived internally, consistent with the rest of the vol module's convention)."""
    k = float(np.log(strike / forward))
    base_delta = delta_spot(is_call, forward, strike, T, sigma, discount_factor, spot)
    option_vega = vega(forward, strike, T, sigma, discount_factor)

    d_sigma_d_k = dsigma_dk(svi_params, k, T, sigma)
    sticky_delta_dsigma_ds = -d_sigma_d_k / spot
    sticky_local_vol_dsigma_ds = STICKY_LOCAL_VOL_MULTIPLIER * sticky_delta_dsigma_ds

    return StickinessDeltas(
        sticky_strike=base_delta,
        sticky_delta=base_delta + option_vega * sticky_delta_dsigma_ds,
        sticky_local_vol=base_delta + option_vega * sticky_local_vol_dsigma_ds,
    )
