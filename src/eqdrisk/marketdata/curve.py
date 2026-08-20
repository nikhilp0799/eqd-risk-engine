"""Discount curve bootstrap from SOFR + Treasury CMT — log-linear in discount factor.

Interpolating linearly in log discount factor (equivalently, piecewise-constant
forward rates) is the deliberate choice per README 2.1 — linear-in-zero-rate
produces jagged forwards, which then poison the implied-forward regression and
eventually the Dupire local-vol strip (Step 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Treasury/SOFR tenor labels (as ingested into the `curves` table) -> year fraction pillar.
TENOR_YEARS: dict[str, float] = {
    "SOFR": 1 / 365,
    "1M": 1 / 12,
    "3M": 3 / 12,
    "6M": 6 / 12,
    "1Y": 1.0,
    "2Y": 2.0,
    "5Y": 5.0,
    "10Y": 10.0,
}


@dataclass
class Curve:
    """A bootstrapped discount curve, flat-extrapolated in log(P) beyond its pillars.

    Treasury CMT yields are treated as continuously-compounded zero rates at each
    pillar tenor — a standard simplification for a single-curve build (not a full
    OIS/swap bootstrap), adequate for the forward/discount cross-check this curve
    exists to support.
    """

    pillar_T: np.ndarray
    pillar_log_df: np.ndarray

    def discount_factor(self, T: float) -> float:
        log_df = np.interp(T, self.pillar_T, self.pillar_log_df)
        return float(np.exp(log_df))

    def zero_rate(self, T: float) -> float:
        # Below the shortest pillar, discount_factor() flat-extrapolates in log(P);
        # dividing that near-constant log(P) by a shrinking T blows up the rate, so
        # clamp T to the first pillar rather than report a meaningless short-end rate.
        if T < self.pillar_T[0]:
            T = self.pillar_T[0]
        return float(-np.log(self.discount_factor(T)) / T)


def bootstrap_curve(rates: pd.DataFrame) -> Curve:
    """Bootstrap a `Curve` from a single as-of date's rows of the `curves` table.

    `rates` must have `tenor` and `rate` columns (rate in percent, e.g. 4.00 for 4%).
    """
    df = rates[rates["tenor"].isin(TENOR_YEARS)].copy()
    if df.empty:
        raise ValueError("no recognised tenors in rates input")
    df["T"] = df["tenor"].map(TENOR_YEARS)
    df = df.sort_values("T")
    df["discount_factor"] = np.exp(-(df["rate"] / 100.0) * df["T"])
    return Curve(
        pillar_T=df["T"].to_numpy(dtype=float),
        pillar_log_df=np.log(df["discount_factor"].to_numpy(dtype=float)),
    )
