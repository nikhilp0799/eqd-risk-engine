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
    _frozen_feed,
    _prior_day_count,
    _snap_offset_minutes,
    _with_retries,
    catch_up_chains,
    missing_chains,
    rates_missing,
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


def test_rerun_keeps_existing_snapshots_and_fills_only_the_gaps(
    tmp_path, monkeypatch, make_chain_df, no_sleep
):
    """A re-run after the close used to overwrite the day's real snapshot (SPX
    2026-09-09 and 09-14 were lost that way). It must now keep what exists and
    fetch only what failed the first time."""
    _fake_sources(monkeypatch, make_chain_df, failing={"SPX"})
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))
    nvda = (
        tmp_path
        / "curated"
        / "chains"
        / "asof_date=2026-09-15"
        / "underlying=NVDA"
        / "part-0.parquet"
    )
    before = nvda.read_bytes()

    fetched: list[str] = []

    def fetch_option_chain(underlying, asof_ts):
        fetched.append(underlying)
        return make_chain_df(dt.date(2026, 9, 15), underlying, 7)

    monkeypatch.setattr(snapshot_module.sources, "fetch_option_chain", fetch_option_chain)
    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    assert fetched == ["SPX"]
    assert sorted(result.qc.kept_existing) == ["AAPL", "NVDA"]
    assert "kept existing snapshot" in result.qc.render()
    assert nvda.read_bytes() == before


def test_missing_chains_lists_the_names_a_catch_up_retry_must_fetch(
    tmp_path, monkeypatch, make_chain_df, no_sleep
):
    cfg = _cfg(tmp_path)
    universe = cfg.universe.index + cfg.universe.single_names
    assert missing_chains(cfg, dt.date(2026, 9, 15)) == universe
    _fake_sources(monkeypatch, make_chain_df, failing={"SPX"})
    run_snapshot(cfg, dt.date(2026, 9, 15))
    assert missing_chains(cfg, dt.date(2026, 9, 15)) == ["SPX"]
    assert missing_chains(cfg, dt.date(2026, 9, 16)) == universe


# --- catch-up retry (2026-11-07): wait out an outage, each name until its cutoff ---


@pytest.fixture
def clock(monkeypatch):
    """A fake wall clock starting 16:35 ET on 2026-09-15; every sleep advances it."""
    now = [pd.Timestamp("2026-09-15 16:35", tz="America/New_York")]

    def sleep(seconds):
        now[0] += pd.Timedelta(seconds=seconds)

    monkeypatch.setattr(snapshot_module, "_now", lambda: now[0])
    monkeypatch.setattr(snapshot_module, "_sleep", sleep)
    return now


def _outage(monkeypatch, make_chain_df, clock, down_until: dict[str, str]):
    """Fake sources where each name's chain fetch fails until an ET time ("" = all
    evening); returns the fetch log as (name, HH:MM) pairs."""
    _fake_sources(monkeypatch, make_chain_df, failing=set())
    fetches: list[tuple[str, str]] = []

    def fetch_option_chain(underlying, asof_ts):
        fetches.append((underlying, f"{clock[0]:%H:%M}"))
        until = down_until.get(underlying)
        if until is not None and (until == "" or f"{clock[0]:%H:%M}" < until):
            raise KeyError("currentTradingPeriod")
        return make_chain_df(dt.date(2026, 9, 15), underlying, 4)

    monkeypatch.setattr(snapshot_module.sources, "fetch_option_chain", fetch_option_chain)
    monkeypatch.setattr(snapshot_module.sources, "fetch_rates", lambda *a: _curve("2026-09-15"))
    return fetches


def _curve(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"asof_date": [dt.date.fromisoformat(day)], "tenor": ["1Y"], "rate": [4.0], "source": ["x"]}
    )


def test_catch_up_fetches_a_chain_once_the_outage_ends(tmp_path, monkeypatch, make_chain_df, clock):
    fetches = _outage(monkeypatch, make_chain_df, clock, {"SPX": "16:50"})
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))
    assert missing_chains(_cfg(tmp_path), dt.date(2026, 9, 15)) == ["SPX"]

    before = len(fetches)

    logged: list[str] = []
    assert catch_up_chains(_cfg(tmp_path), dt.date(2026, 9, 15), log=logged.append) == []
    # Only the missing name is re-fetched, 10 minutes apart while it is an index.
    assert {u for u, _ in fetches[before:]} == {"SPX"}
    # The interval runs from the end of a failed try (its own retries take 35 s).
    assert _attempts(logged) == [("16:45", "SPX"), ("16:56", "SPX")]


def _attempts(logged: list[str]) -> list[tuple[str, str]]:
    """(HH:MM, names) for each catch-up attempt, from its 'catch-up' log line."""
    return [
        (line.split()[1].rstrip(":"), line.split("fetching ")[1])
        for line in logged
        if line.startswith("catch-up")
    ]


def test_catch_up_stops_each_name_at_its_own_cutoff(tmp_path, monkeypatch, make_chain_df, clock):
    _outage(monkeypatch, make_chain_df, clock, {"SPX": "", "NVDA": ""})
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    logged: list[str] = []
    still_missing = catch_up_chains(_cfg(tmp_path), dt.date(2026, 9, 15), log=logged.append)
    assert still_missing == ["SPX", "NVDA"]
    attempts = _attempts(logged)
    with_spx = [t for t, names in attempts if "SPX" in names]
    nvda_only = [t for t, names in attempts if names == "NVDA"]
    # SPX had no bids by 17:30 (measured 2026-11-06): its last try is just before 17:00.
    assert len(with_spx) == 3 and with_spx[-1] == "16:59"
    # NVDA quotes stayed unchanged to 21:00: tried every ~30 minutes once SPX is done.
    assert nvda_only[0] == "17:30" and nvda_only[-1] == "20:59"
    assert len(nvda_only) == 8


def test_catch_up_on_a_past_date_fetches_nothing(tmp_path, monkeypatch, make_chain_df, clock):
    fetches = _outage(monkeypatch, make_chain_df, clock, {})
    clock[0] = pd.Timestamp("2026-09-16 16:35", tz="America/New_York")

    assert catch_up_chains(_cfg(tmp_path), dt.date(2026, 9, 15), log=lambda s: None) == [
        "SPX",
        "NVDA",
        "AAPL",
    ]
    assert fetches == []


def _rates_posted_at(monkeypatch, clock, posted: str):
    """Fake rates feed: the 2026-09-15 curve appears from an ET time ("" = never)."""

    def fetch_rates(*a):
        if posted and f"{clock[0]:%H:%M}" >= posted:
            return _curve("2026-09-15")
        return _curve("2026-09-14")

    monkeypatch.setattr(snapshot_module.sources, "fetch_rates", fetch_rates)


def test_catch_up_retries_rates_until_treasury_posts_the_curve(
    tmp_path, monkeypatch, make_chain_df, clock
):
    _outage(monkeypatch, make_chain_df, clock, {})
    _rates_posted_at(monkeypatch, clock, "16:50")
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))
    assert rates_missing(_cfg(tmp_path), dt.date(2026, 9, 15))

    logged: list[str] = []
    assert catch_up_chains(_cfg(tmp_path), dt.date(2026, 9, 15), log=logged.append) == []
    assert _attempts(logged) == [("16:45", "rates"), ("16:55", "rates")]
    assert not rates_missing(_cfg(tmp_path), dt.date(2026, 9, 15))


def test_catch_up_gives_up_on_rates_at_the_index_cutoff(
    tmp_path, monkeypatch, make_chain_df, clock
):
    """A fetch that returns at once without the curve once looped forever with
    the clock standing still, 30 s before the cutoff."""
    _outage(monkeypatch, make_chain_df, clock, {})
    _rates_posted_at(monkeypatch, clock, "")
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    logged: list[str] = []
    catch_up_chains(_cfg(tmp_path), dt.date(2026, 9, 15), log=logged.append)
    assert _attempts(logged) == [("16:45", "rates"), ("16:55", "rates"), ("16:59", "rates")]


def test_no_curve_is_due_on_a_bond_market_holiday(tmp_path):
    # Veterans Day 2026: the NYSE trades, the bond market and Treasury's feed do not.
    assert not rates_missing(_cfg(tmp_path), dt.date(2026, 11, 11))
    assert rates_missing(_cfg(tmp_path), dt.date(2026, 11, 12))


def test_force_replaces_an_existing_snapshot(tmp_path, monkeypatch, make_chain_df, no_sleep):
    _fake_sources(monkeypatch, make_chain_df, failing=set())
    run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))
    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15), force=True)

    assert result.qc.kept_existing == []
    assert set(result.qc.chain_rows) == {"SPX", "NVDA", "AAPL"}


# --- frozen feed (2026-10-31): a chain whose latest trade is long before the close ---


def _with_last_trade(df: pd.DataFrame, hour: int, minute: int) -> pd.DataFrame:
    df = df.copy()
    df["last_trade_ts"] = df["last_trade_ts"].map(lambda t: t.replace(hour=hour, minute=minute))
    return df


def test_frozen_feed_flags_a_chain_that_stopped_before_the_staleness_window(make_chain_df):
    """The 2026-10-30 SPX pull: captured at 16:30, latest trade 14:59."""
    asof = dt.date(2026, 10, 30)
    capture = pd.Timestamp("2026-10-30 16:30", tz="America/New_York")
    frozen = _with_last_trade(make_chain_df(asof, "SPX", 3), 14, 59)
    why = _frozen_feed(frozen, asof, capture, dt.time(16, 0))
    assert why is not None and "14:59" in why

    normal = _with_last_trade(make_chain_df(asof, "SPX", 3), 15, 59)
    assert _frozen_feed(normal, asof, capture, dt.time(16, 0)) is None


def test_frozen_feed_measures_a_pre_market_run_against_the_prior_close(make_chain_df):
    """Before the open, the prior session's last trades are current, not frozen."""
    asof = dt.date(2026, 10, 30)
    capture = pd.Timestamp("2026-10-30 07:00", tz="America/New_York")
    chain = make_chain_df(dt.date(2026, 10, 29), "SPX", 3)
    assert _frozen_feed(chain, asof, capture, dt.time(16, 0)) is None


def test_a_frozen_chain_is_refetched_until_it_is_fresh(
    tmp_path, monkeypatch, make_chain_df, no_sleep
):
    _fake_sources(monkeypatch, make_chain_df, failing=set())
    pulls = iter([14, 14, 16])

    def fetch_option_chain(underlying, asof_ts):
        df = make_chain_df(dt.date(2026, 9, 15), underlying, 4)
        return _with_last_trade(df, next(pulls), 0) if underlying == "SPX" else df

    monkeypatch.setattr(snapshot_module.sources, "fetch_option_chain", fetch_option_chain)
    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    assert set(result.qc.chain_rows) == {"SPX", "NVDA", "AAPL"}
    assert no_sleep == list(snapshot_module.FROZEN_FEED_DELAYS_SECONDS[:2])
    assert sum("feed looks frozen" in r for r in result.qc.retries) == 2


def test_a_chain_still_frozen_after_every_wait_is_left_for_a_rerun(
    tmp_path, monkeypatch, make_chain_df, no_sleep
):
    _fake_sources(monkeypatch, make_chain_df, failing=set())

    def fetch_option_chain(underlying, asof_ts):
        df = make_chain_df(dt.date(2026, 9, 15), underlying, 4)
        return _with_last_trade(df, 14, 59) if underlying == "SPX" else df

    monkeypatch.setattr(snapshot_module.sources, "fetch_option_chain", fetch_option_chain)
    result = run_snapshot(_cfg(tmp_path), dt.date(2026, 9, 15))

    assert set(result.qc.chain_rows) == {"NVDA", "AAPL"}
    assert "FrozenFeedError" in result.qc.failures["SPX option chain"]
    assert no_sleep == list(snapshot_module.FROZEN_FEED_DELAYS_SECONDS)
    day = "asof_date=2026-09-15"
    assert (tmp_path / "raw" / "chains" / day / "underlying=SPX" / "part-0.parquet").exists()
    assert not (tmp_path / "curated" / "chains" / day / "underlying=SPX").exists()
