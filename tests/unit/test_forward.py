import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.io import store
from eqdrisk.io.schemas import CHAIN_REQUIRED_NOT_NULL, CHAIN_SCHEMA, DIVIDEND_SCHEMA, validate
from eqdrisk.marketdata.forward import (
    RELIABLE_DISPERSION_BP,
    ForwardConstructionResult,
    announced_dividend_yield,
    build_forward_curve,
    fit_forward,
    implied_dividend_yield,
    is_reliable_forward,
    reliable_forwards,
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


def test_fit_forward_with_curve_df_recovers_forward_even_when_parity_df_differs():
    # American-style drift: the chain's own parity discount factor (0.95) is far
    # from the curve's (0.98, ~300bp). Each strike still implies the same forward
    # when divided by the DF the chain was built with, so fixing the DF at the
    # chain's value must recover F exactly with zero dispersion.
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.95, strikes=strikes)

    fit = fit_forward(
        chain,
        spot=100.0,
        underlying="TEST",
        expiry=dt.date(2027, 1, 1),
        T=0.5,
        curve_discount_factor=0.95,
    )

    assert fit is not None
    assert fit.method == "curve_df"
    assert fit.forward == pytest.approx(101.5, abs=1e-9)
    assert fit.dispersion_bp == pytest.approx(0.0, abs=1e-6)
    # The two-parameter diagnostics are still populated.
    assert fit.discount_factor_implied == pytest.approx(0.95, abs=1e-6)


def test_fit_forward_with_wrong_curve_df_reports_dispersion_not_a_bad_df():
    # If the DF used is off, the per-strike forwards fan out with strike:
    # F_i = F + (DF_true/DF_used - 1) * (F - K_i). Dispersion grows with the
    # error, and a ~300bp DF error on +-10% strikes stays far below the 100bp
    # reliability limit, so the expiry is kept rather than dropped.
    strikes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    chain = _synthetic_parity_chain(forward=101.5, discount_factor=0.95, strikes=strikes)

    fit = fit_forward(
        chain,
        spot=100.0,
        underlying="TEST",
        expiry=dt.date(2027, 1, 1),
        T=0.5,
        curve_discount_factor=0.98,
    )

    assert fit is not None
    assert fit.forward == pytest.approx(101.5, rel=2e-3)
    assert fit.dispersion_bp is not None
    assert 0.0 < fit.dispersion_bp < RELIABLE_DISPERSION_BP
    assert is_reliable_forward(
        fit.r_squared, -300.0, method=fit.method, dispersion_bp=fit.dispersion_bp
    )
    # The same fit judged by the two-parameter rule would be rejected.
    assert not is_reliable_forward(fit.r_squared, -300.0)


def test_is_reliable_forward_rules():
    assert is_reliable_forward(0.999, 100.0)
    assert not is_reliable_forward(0.99, 0.0)
    assert not is_reliable_forward(0.999, 200.0)
    assert is_reliable_forward(0.5, 500.0, method="curve_df", dispersion_bp=40.0)
    assert not is_reliable_forward(0.999, 0.0, method="curve_df", dispersion_bp=150.0)
    assert not is_reliable_forward(0.999, 0.0, method="curve_df", dispersion_bp=None)


def test_reliable_forwards_uses_stored_flag_and_falls_back_for_legacy_rows():
    rows = pd.DataFrame(
        {
            "T": [0.1, 0.5, 1.0, 1.5, 2.0],
            "r_squared": [0.9999, 0.9999, 0.9999, 0.9999, 0.9999],
            # Legacy rows (reliable missing) are judged on these two columns.
            "discount_factor_diff_bp": [10.0, 300.0, 10.0, 300.0, 300.0],
            "reliable": [True, True, False, None, None],
        }
    )
    kept = reliable_forwards(rows)
    # Stored flags win (row 1 kept despite 300bp, row 2 dropped despite 10bp);
    # legacy rows 3 and 4 fail the two-parameter rule.
    assert kept["T"].tolist() == [0.1, 0.5]

    legacy_only = rows.drop(columns=["reliable"])
    assert reliable_forwards(legacy_only)["T"].tolist() == [0.1, 1.0]
    assert reliable_forwards(rows.iloc[0:0]).empty


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


def test_run_forward_construction_uses_curve_df_for_single_names(tmp_path):
    from eqdrisk.config import BaseConfig, Paths, Universe

    asof = dt.date(2026, 8, 20)
    expiry = dt.date(2027, 2, 20)
    _write_synthetic_curated_store(tmp_path, asof, expiry)

    cfg = BaseConfig(
        run_date=asof,
        universe=Universe(index=[], single_names=["TEST"]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_forward_construction(cfg, asof)

    assert len(result.fits) == 1
    assert result.fits[0].method == "curve_df"
    forwards_out = store.query(
        "SELECT * FROM fwd", views={"fwd": str(tmp_path / "forwards")}
    ).to_pandas()
    row = forwards_out.iloc[0]
    assert row["method"] == "curve_df"
    assert row["dispersion_bp"] < RELIABLE_DISPERSION_BP
    assert bool(row["reliable"])
    # Forward recovered within the small fan-out a curve-vs-chain DF gap causes.
    assert row["forward"] == pytest.approx(101.5, rel=1e-3)


def test_forward_curve_continues_carry_beyond_last_pillar():
    # Pillars on an exact 4%/yr carry curve; beyond the last pillar the curve must
    # keep that carry, not hold the last forward flat (zero carry).
    T = [0.1, 0.5, 0.9, 1.0]
    df = pd.DataFrame({"T": T, "forward": [100.0 * np.exp(0.04 * t) for t in T]})
    curve = build_forward_curve(df)

    assert curve.long_end_carry == pytest.approx(0.04, rel=1e-12)
    assert curve.forward(2.0) == pytest.approx(100.0 * np.exp(0.08), rel=1e-12)
    assert curve.forward(0.7) == pytest.approx(100.0 * np.exp(0.028), rel=1e-12)
    # Flat before the first pillar, unchanged.
    assert curve.forward(0.01) == pytest.approx(100.0 * np.exp(0.004), rel=1e-12)


def test_forward_curve_carry_uses_a_span_not_two_nearly_coincident_pillars():
    # The last two pillars are 0.02y apart with a noisy forward between them;
    # the slope must come from at least LONG_END_CARRY_MIN_SPAN_YEARS back.
    df = pd.DataFrame({"T": [0.25, 0.5, 0.98, 1.0], "forward": [101.0, 102.0, 104.5, 104.0]})
    curve = build_forward_curve(df)
    expected = (np.log(104.0) - np.log(102.0)) / 0.5
    assert curve.long_end_carry == pytest.approx(expected, rel=1e-12)
    assert build_forward_curve(df.iloc[:1]).long_end_carry == 0.0
