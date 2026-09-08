"""Shared synthetic fixture for the golden-file regression test and the
`tests/integration` end-to-end test (README Step 16's testing pyramid).

Rather than a real historical date (which would need either a live network call
in CI or a large checked-in real-data fixture), this builds ONE synthetic but
internally consistent scenario from a KNOWN SVI parameterization: pick true
SVI params per expiry, evaluate implied vol at a strike grid, convert to real
Black-76 prices (`pricing.blackscholes.call_price`/`put_price`), and build a raw
option-chain DataFrame from those prices. Because call/put parity is exact in
Black-76, the resulting chain recovers the TRUE forward and discount factor via
Step 2's regression with R^2 = 1.0 (up to floating-point rounding) — a
"fixed date" that is exactly reproducible everywhere, with zero network
dependency, consistent with the rest of this test suite (no test anywhere else
touches yfinance/fredapi/pandas_datareader either).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from eqdrisk.pricing.blackscholes import call_price, put_price
from eqdrisk.vol.svi import SVIParams

ASOF = dt.date(2026, 6, 15)
UNDERLYING = "SPX"
SPOT = 100.0
RATE = 0.03
DIV_YIELD = 0.01

# Two expiries with strictly increasing total variance at every k -> no calendar
# arbitrage, both fit as plain SVI (same params already validated arb-free in
# tests/unit/test_surface.py).
EXPIRIES: tuple[tuple[dt.date, float, SVIParams], ...] = (
    (ASOF + dt.timedelta(days=91), 0.25, SVIParams(a=0.01, b=0.10, rho=-0.3, m=0.0, sigma=0.15)),
    (ASOF + dt.timedelta(days=182), 0.50, SVIParams(a=0.03, b=0.12, rho=-0.3, m=0.0, sigma=0.15)),
)
K_GRID = np.linspace(-0.3, 0.3, 21)


def build_synthetic_chain() -> pd.DataFrame:
    asof_ts = pd.Timestamp(ASOF, tz="America/New_York").replace(hour=16, minute=0, second=0)
    rows = []
    for expiry, T, svi in EXPIRIES:
        forward = SPOT * np.exp((RATE - DIV_YIELD) * T)
        discount_factor = np.exp(-RATE * T)
        for k in K_GRID:
            strike = forward * np.exp(k)
            sigma = float(np.sqrt(max(svi.total_variance(k), 1e-8) / T))
            call_mid = call_price(forward, strike, T, sigma, discount_factor)
            put_mid = put_price(forward, strike, T, sigma, discount_factor)
            for cp, mid in (("C", call_mid), ("P", put_mid)):
                spread = max(mid * 0.01, 0.01)
                rows.append(
                    {
                        "asof_date": ASOF,
                        "asof_ts": asof_ts,
                        "underlying": UNDERLYING,
                        "expiry": expiry,
                        "strike": strike,
                        "cp": cp,
                        "bid": mid - spread / 2,
                        "ask": mid + spread / 2,
                        "bid_size": 10,
                        "ask_size": 10,
                        "volume": 100,
                        "open_interest": 100,
                        "underlying_px": SPOT,
                        "last_trade_ts": asof_ts,
                        "source": "synthetic",
                    }
                )
    return pd.DataFrame(rows)


def build_synthetic_rates() -> pd.DataFrame:
    tenors = ["SOFR", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"]
    return pd.DataFrame(
        {
            "asof_date": [ASOF] * len(tenors),
            "tenor": tenors,
            "rate": [RATE * 100] * len(tenors),
            "source": ["synthetic"] * len(tenors),
        }
    )


def build_synthetic_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": [ASOF],
            "underlying": [UNDERLYING],
            "open": [SPOT],
            "high": [SPOT],
            "low": [SPOT],
            "close": [SPOT],
            "volume": [1_000_000],
        }
    )
