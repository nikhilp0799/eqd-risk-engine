import datetime as dt

import pandas as pd
import pytest

import eqdrisk.io.snapshot as snapshot_module
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import CHAIN_REQUIRED_NOT_NULL, CHAIN_SCHEMA, validate
from eqdrisk.io.snapshot import (
    QCReport,
    _clean_chain,
    _prior_day_count,
    _snap_offset_minutes,
    _with_retries,
    run_snapshot,
)


def test_prior_day_count_none_when_no_history(tmp_path):
    assert _prior_day_count(tmp_path / "chains", "SPX", dt.date(2026, 8, 11)) is None


def test_prior_day_count_finds_latest_prior_partition(tmp_path, make_chain_df):
    base = tmp_path / "chains"
    for asof, n in [(dt.date(2026, 8, 10), 5), (dt.date(2026, 8, 11), 7)]:
        table = validate(make_chain_df(asof, "SPX", n), CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
        store.write_partitioned(table, base, ["asof_date", "underlying"])

    assert _prior_day_count(base, "SPX", dt.date(2026, 8, 12)) == 7
    assert _prior_day_count(base, "AAPL", dt.date(2026, 8, 12)) is None


def test_qc_report_render_includes_deltas_and_null_rates():
    qc = QCReport(asof=dt.date(2026, 8, 11))
    qc.chain_rows["SPX"] = 100
    qc.null_rates["SPX"] = {"bid": 0.01, "ask": 0.0}
    qc.prior_day_row_delta["SPX"] = -3

    rendered = qc.render()
    assert "SPX: 100 quotes" in rendered
    assert "'bid': 0.01" in rendered
    assert "-3 rows" in rendered


def test_clean_chain_drops_null_identity_rows(make_chain_df):
    df = make_chain_df(dt.date(2026, 8, 11), "SPX", 3)
    df.loc[1, "strike"] = None
    clean, rejections = _clean_chain(df)
    assert len(clean) == 2
    assert rejections == {"NULL_IDENTITY": 1}


def test_clean_chain_drops_duplicate_contracts(make_chain_df):
    df = make_chain_df(dt.date(2026, 8, 11), "SPX", 3)
    df.loc[1, ["expiry", "strike", "cp"]] = df.loc[0, ["expiry", "strike", "cp"]].values
    clean, rejections = _clean_chain(df)
    assert len(clean) == 2
    assert rejections == {"DUPLICATE_CONTRACT": 1}


def test_clean_chain_no_rejections_when_clean(make_chain_df):
    df = make_chain_df(dt.date(2026, 8, 11), "SPX", 3)
    clean, rejections = _clean_chain(df)
    assert len(clean) == 3
    assert rejections == {}


def test_snap_offset_zero_at_canonical_time():
    asof = dt.date(2026, 8, 11)
    ts = pd.Timestamp("2026-08-11 16:00:00", tz="America/New_York")
    assert _snap_offset_minutes(ts, asof, dt.time(16, 0, 0)) == 0.0


def test_snap_offset_positive_when_late():
    asof = dt.date(2026, 8, 11)
    ts = pd.Timestamp("2026-08-11 16:30:00", tz="America/New_York")
    assert _snap_offset_minutes(ts, asof, dt.time(16, 0, 0)) == 30.0


# --- resilience (2026-10-04): one failing feed must not abort the whole ingest ---


@pytest.fixture
def no_sleep(monkeypatch):
    delays = []
    monkeypatch.setattr(snapshot_module, "_sleep", delays.append)
    return delays


def test_with_retries_recovers_from_a_transient_failure(no_sleep):
    calls = iter([KeyError("currentTradingPeriod"), TimeoutError("curl 28"), "chain"])

    def flaky():
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    retries: list[str] = []
    assert _with_retries("SPX option chain", flaky, retries) == "chain"
    assert len(retries) == 2
    assert "KeyError" in retries[0] and "TimeoutError" in retries[1]
    assert no_sleep == list(snapshot_module.RETRY_DELAYS_SECONDS)


def test_with_retries_gives_up_after_the_last_attempt(no_sleep):
    def always_fails():
        raise KeyError("currentTradingPeriod")

    retries: list[str] = []
    with pytest.raises(KeyError):
        _with_retries("SPX option chain", always_fails, retries)
    assert len(retries) == len(snapshot_module.RETRY_DELAYS_SECONDS)


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=dt.date(2026, 9, 15),
        universe=Universe(index=["SPX"], single_names=["NVDA", "AAPL"]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path / "curated")),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _fake_sources(monkeypatch, make_chain_df, failing: set[str]):
    def fetch_option_chain(underlying, asof_ts):
        if underlying in failing:
            raise KeyError("currentTradingPeriod")
        return make_chain_df(dt.date(2026, 9, 15), underlying, 4)

    empty = pd.DataFrame()
    monkeypatch.setattr(snapshot_module.sources, "fetch_option_chain", fetch_option_chain)
    monkeypatch.setattr(snapshot_module.sources, "fetch_dividends", lambda u: empty)
    monkeypatch.setattr(snapshot_module.sources, "fetch_underlying_ohlc", lambda *a: empty)
    monkeypatch.setattr(snapshot_module.sources, "fetch_rates", lambda *a: empty)
    monkeypatch.setattr(snapshot_module.sources, "fetch_vol_indices", lambda *a: empty)


def test_one_failing_underlying_no_longer_kills_the_others(
    tmp_path, monkeypatch, make_chain_df, no_sleep
):
    """The 2026-09-15 failure mode: `^SPX` errors, and before the fix NVDA and
    AAPL lost their option data that day too."""
    _fake_sources(monkeypatch, make_chain_df, failing={"SPX"})

    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    assert set(result.qc.chain_rows) == {"NVDA", "AAPL"}
    assert "SPX option chain" in result.qc.failures
    assert "FAILED after retries: SPX option chain" in result.qc.render()
    chains = tmp_path / "curated" / "chains" / "asof_date=2026-09-15"
    assert (chains / "underlying=NVDA").exists() and not (chains / "underlying=SPX").exists()


def test_a_failing_rates_feed_is_reported_not_fatal(tmp_path, monkeypatch, make_chain_df, no_sleep):
    _fake_sources(monkeypatch, make_chain_df, failing=set())

    def rates_down(*args):
        raise TimeoutError("curl: (28) Operation timed out")

    monkeypatch.setattr(snapshot_module.sources, "fetch_rates", rates_down)
    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    assert set(result.qc.chain_rows) == {"SPX", "NVDA", "AAPL"}
    assert "rates" in result.qc.failures
