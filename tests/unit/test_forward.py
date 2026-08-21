import datetime as dt

import pandas as pd
import pytest

from eqdrisk.io import store
from eqdrisk.io.schemas import CHAIN_REQUIRED_NOT_NULL, CHAIN_SCHEMA, DIVIDEND_SCHEMA, validate
from eqdrisk.marketdata.forward import (
    ForwardConstructionResult,
    announced_dividend_yield,
    fit_forward,
    implied_dividend_yield,
    run_forward_construction,
)

FRESH_TS = pd.Timestamp("2026-08-20 16:00:00", tz="America/New_York")


def _synthetic_parity_chain(
    forward: float,
    discount_factor: float,
    strikes: list[float],
    asof_ts: pd.Timestamp = FRESH_TS,
    last_trade_ts: pd.Timestamp = FRESH_TS,
    open_interest: int = 100,
) -> pd.DataFrame:
    """Rows satisfying C - P = discount_factor * (forward - K) exactly (zero noise),
    fresh and liquid by default so quality filtering is a no-op unless a test
    deliberately overrides asof_ts/last_trade_ts/open_interest to probe it."""
    rows = []
    put_mid = 20.0  # arbitrary constant baseline; only the C-P difference matters here
    for k in strikes:
        call_mid = put_mid + discount_factor * (forward - k)
        common = {
            "strike": k,
            "asof_ts": asof_ts,
            "last_trade_ts": last_trade_ts,
            "open_interest": open_interest,
        }
        rows.append({**common, "cp": "C", "bid": call_mid - 0.05, "ask": call_mid + 0.05})
        rows.append({**common, "cp": "P", "bid": put_mid - 0.05, "ask": put_mid + 0.05})
    return pd.DataFrame(rows)


def test_fit_forward_recovers_known_parameters():
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.98, strikes=strikes)

    fit = fit_forward(chain, spot=100.0, underlying="TEST", expiry=dt.date(2027, 1, 1), T=0.5)

    assert fit is not None
    assert fit.forward == pytest.approx(101.5, abs=1e-6)
    assert fit.discount_factor_implied == pytest.approx(0.98, abs=1e-6)
    assert fit.r_squared > 0.999999
    assert fit.n_strikes == len(strikes)


def test_fit_forward_returns_none_below_min_strikes():
    strikes = [98.0, 100.0, 102.0]  # below MIN_STRIKES=6
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.98, strikes=strikes)
    assert (
        fit_forward(chain, spot=100.0, underlying="TEST", expiry=dt.date(2027, 1, 1), T=0.5) is None
    )


def test_fit_forward_excludes_crossed_and_extreme_moneyness_rows():
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.98, strikes=strikes)

    # Crossed quote on one strike's call leg — caught by quality.classify_quotes.
    crossed_idx = chain.index[(chain["strike"] == 90.0) & (chain["cp"] == "C")][0]
    chain.loc[crossed_idx, ["bid", "ask"]] = [10.0, 5.0]

    # Extreme moneyness row far outside the (tightened, 30%) band, excluded regardless of validity.
    extreme = _synthetic_parity_chain(forward=101.5, discount_factor=0.98, strikes=[500.0])
    chain = pd.concat([chain, extreme], ignore_index=True)

    fit = fit_forward(chain, spot=100.0, underlying="TEST", expiry=dt.date(2027, 1, 1), T=0.5)
    assert fit is not None
    assert fit.n_strikes == len(strikes) - 1  # one strike dropped for the crossed call leg


def test_fit_forward_excludes_stale_quotes():
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.98, strikes=strikes)

    # Make two strikes' worth of quotes (4 rows) stale — last trade 2 days before capture.
    stale_strikes = chain["strike"].isin([90.0, 92.0])
    chain.loc[stale_strikes, "last_trade_ts"] = FRESH_TS - pd.Timedelta(days=2)

    fit = fit_forward(chain, spot=100.0, underlying="TEST", expiry=dt.date(2027, 1, 1), T=0.5)
    assert fit is not None
    assert fit.n_strikes == len(strikes) - 2


def test_implied_dividend_yield_formula():
    import numpy as np

    forward, discount_factor, spot, t = 101.5, 0.98, 100.0, 0.5
    q = implied_dividend_yield(forward, discount_factor, spot, t)
    assert q == pytest.approx(-np.log(forward * discount_factor / spot) / t)


def test_announced_dividend_yield_none_when_no_history(tmp_path):
    assert (
        announced_dividend_yield(tmp_path / "dividends", "SPX", 100.0, dt.date(2026, 8, 20)) is None
    )


def test_announced_dividend_yield_sums_trailing_12mo(tmp_path):
    base = tmp_path / "dividends"
    df = pd.DataFrame(
        {
            "underlying": ["AAPL"] * 4,
            "ex_date": [
                dt.date(2025, 8, 25),
                dt.date(2025, 11, 10),
                dt.date(2026, 2, 9),
                dt.date(2026, 5, 11),
            ],
            "amount": [0.26, 0.26, 0.26, 0.27],
        }
    )
    table = validate(df, DIVIDEND_SCHEMA, ["underlying", "ex_date", "amount"])
    store.write_partitioned(table, base, ["underlying"])

    q = announced_dividend_yield(base, "AAPL", spot=200.0, asof=dt.date(2026, 8, 20))
    assert q == pytest.approx((0.26 + 0.26 + 0.26 + 0.27) / 200.0)


def _write_synthetic_curated_store(root, asof, expiry):
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    asof_ts = pd.Timestamp(asof, tz="America/New_York").replace(hour=16)
    parity = _synthetic_parity_chain(
        forward=101.5, discount_factor=0.98, strikes=strikes, asof_ts=asof_ts, last_trade_ts=asof_ts
    )
    n = len(parity)
    chain_df = pd.DataFrame(
        {
            "asof_date": [asof] * n,
            "asof_ts": parity["asof_ts"],
            "underlying": ["TEST"] * n,
            "expiry": [expiry] * n,
            "strike": parity["strike"],
            "cp": parity["cp"],
            "bid": parity["bid"],
            "ask": parity["ask"],
            "bid_size": pd.array([None] * n, dtype="Int64"),
            "ask_size": pd.array([None] * n, dtype="Int64"),
            "volume": [10] * n,
            "open_interest": parity["open_interest"],
            "underlying_px": [100.0] * n,
            "last_trade_ts": parity["last_trade_ts"],
            "source": ["test"] * n,
        }
    )
    table = validate(chain_df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, root / "chains", ["asof_date", "underlying"])

    rates_df = pd.DataFrame(
        {
            "asof_date": [asof] * 4,
            "tenor": ["SOFR", "1Y", "2Y", "5Y"],
            "rate": [3.65, 3.99, 4.19, 4.37],
            "source": ["test"] * 4,
        }
    )
    from eqdrisk.io.schemas import CURVE_REQUIRED_NOT_NULL, CURVE_SCHEMA

    rates_table = validate(rates_df, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL)
    store.write_partitioned(rates_table, root / "curves", ["asof_date"])


def test_run_forward_construction_end_to_end(tmp_path):
    from eqdrisk.config import BaseConfig, Paths, Universe

    asof = dt.date(2026, 8, 20)
    expiry = dt.date(2027, 2, 20)  # T ~= 0.5y
    _write_synthetic_curated_store(tmp_path, asof, expiry)

    cfg = BaseConfig(
        run_date=asof,
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_forward_construction(cfg, asof)

    assert isinstance(result, ForwardConstructionResult)
    assert len(result.fits) == 1
    fit = result.fits[0]
    assert fit.underlying == "TEST"
    assert fit.forward == pytest.approx(101.5, abs=1e-6)
    assert fit.discount_factor_implied == pytest.approx(0.98, abs=1e-6)

    forwards_out = store.query(
        "SELECT * FROM fwd", views={"fwd": str(tmp_path / "forwards")}
    ).to_pandas()
    assert len(forwards_out) == 1
    assert forwards_out["discount_factor_curve"].iloc[0] > 0

    curve_out = store.query(
        "SELECT * FROM dc", views={"dc": str(tmp_path / "discount_curves")}
    ).to_pandas()
    assert len(curve_out) == 4
