"""Free-tier provider adapters (yfinance, FRED via pandas_datareader, Treasury CSV).

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

    yfinance exposes no canonical snap time and no bid/ask size — every row is
    stamped with `asof_ts` (our capture time), and bid_size/ask_size are always
    null. Both are documented data-source gaps, not bugs. It DOES expose a
    per-quote `lastTradeDate`, which we keep as `last_trade_ts` for staleness
    filtering (Step 3's STALE reason code).
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
            columns=[
                "expiry",
                "strike",
                "cp",
                "bid",
                "ask",
                "volume",
                "openInterest",
                "lastTradeDate",
            ]
        )
    else:
        raw = pd.concat(frames, ignore_index=True)

    n = len(raw)
    last_trade_ts = (
        pd.to_datetime(raw["lastTradeDate"], utc=True).dt.tz_convert(NY_TZ)
        if n
        else pd.Series(dtype="datetime64[ns, America/New_York]")
    )
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
            "last_trade_ts": last_trade_ts,
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


# Treasury's own daily par yield CSV, the series FRED's DGS* republish a day or
# two later: equal to the basis point on all 490 points compared (2026-08-03 to
# 11-10), and posted the same evening (11-13's by 16:11 ET), so the daily run no
# longer marks on a 1-2 day old curve (planning/same_day_rates_plan.md).
TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
TREASURY_TENORS = {
    "1M": "1 Mo",
    "3M": "3 Mo",
    "6M": "6 Mo",
    "1Y": "1 Yr",
    "2Y": "2 Yr",
    "5Y": "5 Yr",
    "10Y": "10 Yr",
}
SOFR_CARRIED = "FRED SOFR carried"


def fetch_rates(start: dt.date, end: dt.date) -> pd.DataFrame:
    """SOFR + Treasury par yields, long-format (asof_date, tenor, rate, source):
    FRED, with Treasury's same-day CSV on top (`combine_rates`). Treasury failing
    leaves FRED alone, as before; FRED failing fails the feed, since SOFR comes
    only from FRED and a curve written without it would replace a stored one."""
    fred = fetch_fred_rates(start, end)
    try:
        treasury = fetch_treasury_rates(start, end)
    except Exception:
        return fred
    return combine_rates(fred, treasury)


def fetch_fred_rates(start: dt.date, end: dt.date) -> pd.DataFrame:
    """SOFR + Treasury CMT rates from FRED, long-format (asof_date, tenor, rate)."""
    df = web.DataReader(list(FRED_RATE_SERIES.values()), "fred", start=start, end=end)
    df = df.rename(columns={v: k for k, v in FRED_RATE_SERIES.items()})
    long = df.reset_index().melt(id_vars="DATE", var_name="tenor", value_name="rate")
    long = long.rename(columns={"DATE": "asof_date"}).dropna(subset=["rate"])
    long["asof_date"] = pd.to_datetime(long["asof_date"]).dt.date
    long["source"] = "FRED"
    return long.reset_index(drop=True)


def fetch_treasury_rates(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Treasury par yields over [start, end], same long format and tenors as
    `fetch_fred_rates` (no SOFR: that is the NY Fed's). One CSV per year."""
    frames = [
        pd.read_csv(TREASURY_CSV_URL.format(year=year)) for year in range(start.year, end.year + 1)
    ]
    df = pd.concat(frames, ignore_index=True)
    df["asof_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.date
    df = df[(df["asof_date"] >= start) & (df["asof_date"] <= end)]
    df = df.rename(columns={v: k for k, v in TREASURY_TENORS.items()})
    long = df.melt(id_vars="asof_date", value_vars=list(TREASURY_TENORS), var_name="tenor")
    long = long.rename(columns={"value": "rate"}).dropna(subset=["rate"])
    long["source"] = "Treasury"
    return long.reset_index(drop=True)


def combine_rates(fred: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    """Treasury for every date and tenor it has, FRED for the rest. SOFR is
    published the next morning, so a Treasury date past FRED's last SOFR gets
    that last SOFR, marked `SOFR_CARRIED`; FRED's own value replaces it on a
    later run, when the trailing window is re-fetched."""
    cols = ["asof_date", "tenor", "rate", "source"]
    rows = pd.concat([treasury[cols], fred[cols]], ignore_index=True)
    rows = rows.drop_duplicates(subset=["asof_date", "tenor"], keep="first")
    sofr = fred[fred["tenor"] == "SOFR"].sort_values("asof_date")
    if not sofr.empty:
        last = sofr.iloc[-1]
        later = sorted(d for d in treasury["asof_date"].unique() if d > last["asof_date"])
        carried = pd.DataFrame(
            {"asof_date": later, "tenor": "SOFR", "rate": last["rate"], "source": SOFR_CARRIED}
        )
        rows = pd.concat([rows, carried], ignore_index=True)
    return rows.sort_values(["asof_date", "tenor"]).reset_index(drop=True)


# VIX: SPX 30d. VXAPL: Apple's own 30d implied vol. VXN: Nasdaq-100 30d, the
# nearest free index for NVDA (which has none of its own). The last two drive the
# single-name vol moves in VaR (risk/proxy.py).
# VXV (3-month SPX vol) is the crash-regime term-structure check in conditional
# stress (stress/conditional.py).
FRED_VOL_INDEX_SERIES = {"VIX": "VIXCLS", "VXAPL": "VXAPLCLS", "VXN": "VXNCLS", "VXV": "VXVCLS"}
# FRED posts a close the next day; the daily run needs the same evening's.
YF_VOL_INDEX_TICKERS = {"VIX": "^VIX", "VXAPL": "^VXAPL", "VXN": "^VXN", "VXV": "^VIX3M"}
# VVIX/SKEW have no free FRED series under any name we could find (VVIXCLS, VVIX both 404)
# — documented gap, not pursued further given they're secondary benchmarks per the README.


def fetch_vol_indices(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Vol-index closes from FRED, long-format (asof_date, index, value): VIX is the
    independent calibration benchmark for the 30d ATM vol cross-check (Step 4); all
    three drive VaR's vol moves (risk/proxy.py)."""
    df = web.DataReader(list(FRED_VOL_INDEX_SERIES.values()), "fred", start=start, end=end)
    df = df.rename(columns={v: k for k, v in FRED_VOL_INDEX_SERIES.items()})
    long = df.reset_index().melt(id_vars="DATE", var_name="index", value_name="value")
    long = long.rename(columns={"DATE": "asof_date"}).dropna(subset=["value"])
    long["asof_date"] = pd.to_datetime(long["asof_date"]).dt.date
    long["source"] = "FRED"
    return long.reset_index(drop=True)


def fetch_vol_index_closes_yf(names: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Vol-index closes from yfinance over [start, end), same long format as
    `fetch_vol_indices`. Same-day fallback only: yfinance serves ^VXAPL as the
    latest close alone, not a history."""
    frames = []
    for name in names:
        hist = yf.Ticker(YF_VOL_INDEX_TICKERS[name]).history(start=start, end=end)
        if hist.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "asof_date": pd.to_datetime(hist.index).date,
                    "index": name,
                    "value": hist["Close"].to_numpy(dtype=float),
                    "source": "yfinance",
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["asof_date", "index", "value", "source"])
    return pd.concat(frames, ignore_index=True)


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
