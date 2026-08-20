import datetime as dt

import pandas as pd

from eqdrisk.io import store
from eqdrisk.io.schemas import CHAIN_REQUIRED_NOT_NULL, CHAIN_SCHEMA, validate
from eqdrisk.io.snapshot import QCReport, _clean_chain, _prior_day_count, _snap_offset_minutes


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
