"""Free-tier provider adapters (yfinance, FRED via pandas_datareader).

No historical option-chain backfill is available from these sources — only
the live chain at fetch time. See `docs/model_documentation.md` limitations
section once written; until a paid historical archive is purchased, VaR
backtesting and historical stress replays cannot be built past scaffolding.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

NY_TZ = "America/New_York"

# Index tickers differ between our universe naming and yfinance's symbol.
UNDERLYING_TICKER_MAP = {"SPX": "^SPX"}

FRED_RATE_SERIES = {
    "SOFR": "SOFR",
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
}


def _yf_ticker(underlying: str) -> str:
    return UNDERLYING_TICKER_MAP.get(underlying, underlying)


def fetch_option_chain(underlying: str, asof_ts: pd.Timestamp) -> pd.DataFrame:
    """Live option chain snapshot for `underlying`, normalised to the chain schema.

    yfinance exposes no per-quote timestamp or bid/ask size, and no canonical
    snap time — every row is stamped with `asof_ts` (our capture time), and
    bid_size/ask_size are always null. Both are documented data-source gaps,
    not bugs.
    """
    ticker = yf.Ticker(_yf_ticker(underlying))
    underlying_px = float(ticker.fast_info["lastPrice"])

    frames = []
    for expiry in ticker.options:
        chain = ticker.option_chain(expiry)
        for cp, leg in (("C", chain.calls), ("P", chain.puts)):
            leg = leg.copy()
            leg["cp"] = cp
            leg["expiry"] = pd.Timestamp(expiry).date()
            frames.append(leg)

    if not frames:
        raw = pd.DataFrame(
            columns=["expiry", "strike", "cp", "bid", "ask", "volume", "openInterest"]
        )
    else:
        raw = pd.concat(frames, ignore_index=True)

    n = len(raw)
    return pd.DataFrame(
        {
            "asof_date": asof_ts.date(),
            "asof_ts": asof_ts,
            "underlying": underlying,
            "expiry": raw["expiry"] if n else pd.Series(dtype="object"),
            "strike": raw["strike"].astype(float) if n else pd.Series(dtype="float64"),
            "cp": raw["cp"] if n else pd.Series(dtype="object"),
            "bid": raw["bid"].astype(float) if n else pd.Series(dtype="float64"),
            "ask": raw["ask"].astype(float) if n else pd.Series(dtype="float64"),
            "bid_size": pd.array([None] * n, dtype="Int64"),
            "ask_size": pd.array([None] * n, dtype="Int64"),
            "volume": raw["volume"].fillna(0).astype("int64") if n else pd.Series(dtype="int64"),
            "open_interest": raw["openInterest"].fillna(0).astype("int64")
            if n
            else pd.Series(dtype="int64"),
            "underlying_px": underlying_px,
            "source": "yfinance",
        }
    )


def fetch_underlying_ohlc(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily OHLCV for each ticker in `tickers` over [start, end)."""
    frames = []
    for tk in tickers:
        hist = yf.Ticker(_yf_ticker(tk)).history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            continue
        hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        hist.columns = ["asof_date", "open", "high", "low", "close", "volume"]
        hist["asof_date"] = pd.to_datetime(hist["asof_date"]).dt.date
        hist["underlying"] = tk
        hist["volume"] = hist["volume"].astype("int64")
        frames.append(hist)
    if not frames:
        return pd.DataFrame(
            columns=["asof_date", "underlying", "open", "high", "low", "close", "volume"]
        )
    return pd.concat(frames, ignore_index=True)[
        ["asof_date", "underlying", "open", "high", "low", "close", "volume"]
    ]


def fetch_rates(start: dt.date, end: dt.date) -> pd.DataFrame:
    """SOFR + Treasury CMT rates from FRED, long-format (asof_date, tenor, rate)."""
    df = web.DataReader(list(FRED_RATE_SERIES.values()), "fred", start=start, end=end)
    df = df.rename(columns={v: k for k, v in FRED_RATE_SERIES.items()})
    long = df.reset_index().melt(id_vars="DATE", var_name="tenor", value_name="rate")
    long = long.rename(columns={"DATE": "asof_date"}).dropna(subset=["rate"])
    long["asof_date"] = pd.to_datetime(long["asof_date"]).dt.date
    long["source"] = "FRED"
    return long.reset_index(drop=True)


def fetch_dividends(underlying: str) -> pd.DataFrame:
    """Announced dividend history for `underlying` (empty for non-dividend-paying names)."""
    div = yf.Ticker(_yf_ticker(underlying)).dividends
    if div.empty:
        return pd.DataFrame(columns=["underlying", "ex_date", "amount"])
    out = div.reset_index()
    out.columns = ["ex_date", "amount"]
    out["underlying"] = underlying
    out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.date
    return out[["underlying", "ex_date", "amount"]]
