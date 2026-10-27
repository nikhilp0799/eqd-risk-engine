import datetime as dt

import pandas as pd

from eqdrisk.marketdata.quality import OK
from eqdrisk.pricing.blackscholes import call_price, put_price
from eqdrisk.vol.implied import (
    EXTREME_K,
    ITM_SIDE,
    NO_ARB_INTRINSIC,
    THIN_SLICE,
    extract_slice_ivs,
    invert_iv,
    is_reliable_forward,
)

FRESH = pd.Timestamp("2026-08-20 16:00:00", tz="America/New_York")
FORWARD = 100.0
DISCOUNT_FACTOR = 0.98
T = 0.5
TRUE_SIGMA = 0.20


def _quote_row(strike: float, cp: str, sigma: float = TRUE_SIGMA, **overrides) -> dict:
    pricer = call_price if cp == "C" else put_price
    price = pricer(FORWARD, strike, T, sigma, DISCOUNT_FACTOR)
    row = {
        "strike": strike,
        "cp": cp,
        "bid": price - 0.02,
        "ask": price + 0.02,
        "asof_ts": FRESH,
        "last_trade_ts": FRESH,
        "open_interest": 100,
    }
    row.update(overrides)
    return row


def _slice_df(strikes_otm_only: list[float]) -> pd.DataFrame:
    """One OTM-correct row per strike: calls above forward, puts below."""
    rows = [_quote_row(k, "C" if k > FORWARD else "P") for k in strikes_otm_only]
    return pd.DataFrame(rows)


def test_extract_slice_ivs_recovers_known_sigma():
    strikes = [80, 85, 90, 95, 100.01, 105, 110, 115, 120]
    df = extract_slice_ivs(
        _slice_df(strikes), spot=FORWARD, forward=FORWARD, discount_factor=DISCOUNT_FACTOR, T=T
    )
    ok = df[df["reason"] == OK]
    assert len(ok) == len(strikes)
    assert ok["iv"].apply(lambda x: abs(x - TRUE_SIGMA) < 1e-6).all()
    assert (ok["total_variance"] > 0).all()
    assert (ok["vega"] > 0).all()
    assert (ok["weight"] > 0).all()


def test_itm_side_excluded():
    # A call struck below the forward (ITM) sitting alongside a correctly-OTM put.
    # Only 2 quotes total, so THIN_SLICE also applies to the put — assert on the
    # specific property under test (ITM exclusion), not survival to final OK.
    df = pd.DataFrame([_quote_row(90, "C"), _quote_row(90, "P")])
    tagged = extract_slice_ivs(
        df, spot=FORWARD, forward=FORWARD, discount_factor=DISCOUNT_FACTOR, T=T
    )
    reasons = dict(zip(tagged["cp"], tagged["reason"], strict=False))
    assert reasons["C"] == ITM_SIDE
    assert reasons["P"] != ITM_SIDE


def test_no_arb_intrinsic_detected():
    # Call priced below its discounted intrinsic value (F - K)*DF.
    strike = 80.0
    intrinsic = DISCOUNT_FACTOR * (FORWARD - strike)
    df = pd.DataFrame(
        [
            {
                "strike": strike,
                "cp": "C",
                "bid": intrinsic * 0.5 - 0.02,
                "ask": intrinsic * 0.5 + 0.02,
                "asof_ts": FRESH,
                "last_trade_ts": FRESH,
                "open_interest": 100,
            }
        ]
    )
    tagged = extract_slice_ivs(
        df, spot=FORWARD, forward=FORWARD, discount_factor=DISCOUNT_FACTOR, T=T
    )
    assert tagged["reason"].iloc[0] == NO_ARB_INTRINSIC


def test_extreme_k_detected_for_far_wing():
    # A deep-OTM, low-vol strike where k exceeds 4*sigma*sqrt(T): k=0.30 vs threshold 0.20.
    # Deliberately contrived (a ~1e-9-scale price with a 30% relative spread, still within
    # the wide-spread tolerance at this moneyness) purely to isolate the EXTREME_K path —
    # in practice WIDE_SPREAD dominates first for genuinely tradeable far-wing quotes,
    # which is itself the honest real-world finding, not a shortcoming of this check.
    wing_T, wing_sigma = 1.0, 0.05
    strike = 135.0
    price = call_price(FORWARD, strike, wing_T, wing_sigma, DISCOUNT_FACTOR)
    df = pd.DataFrame(
        [
            {
                "strike": strike,
                "cp": "C",
                "bid": price * 0.85,
                "ask": price * 1.15,
                "asof_ts": FRESH,
                "last_trade_ts": FRESH,
                "open_interest": 100,
            }
        ]
    )
    tagged = extract_slice_ivs(
        df, spot=FORWARD, forward=FORWARD, discount_factor=DISCOUNT_FACTOR, T=wing_T
    )
    assert tagged["reason"].iloc[0] == EXTREME_K


def test_thin_slice_when_too_few_survivors():
    strikes = [90, 95, 100.01, 105]  # 4 < MIN_SLICE_QUOTES (8)
    df = extract_slice_ivs(
        _slice_df(strikes), spot=FORWARD, forward=FORWARD, discount_factor=DISCOUNT_FACTOR, T=T
    )
    assert (df["reason"] == THIN_SLICE).all()


def test_is_reliable_forward_thresholds():
    assert is_reliable_forward(r_squared=0.9999, discount_factor_diff_bp=8.0) is True
    assert is_reliable_forward(r_squared=0.88, discount_factor_diff_bp=-7584.5) is False
    assert is_reliable_forward(r_squared=0.999, discount_factor_diff_bp=500.0) is False


def test_invert_iv_returns_none_when_not_bracketed():
    # A price wildly above the max possible (sigma=5.0) price is unsolvable.
    huge_price = FORWARD * 100
    assert invert_iv(huge_price, FORWARD, 100.0, T, DISCOUNT_FACTOR, is_call=True) is None


def test_run_iv_extraction_end_to_end(tmp_path):
    from eqdrisk.io import store
    from eqdrisk.io.schemas import (
        CHAIN_REQUIRED_NOT_NULL,
        CHAIN_SCHEMA,
        FORWARD_REQUIRED_NOT_NULL,
        FORWARD_SCHEMA,
        validate,
    )
    from eqdrisk.marketdata.calendar import year_fraction
    from eqdrisk.vol.implied import run_iv_extraction

    asof = dt.date(2026, 8, 20)
    expiry = dt.date(2027, 2, 20)
    # run_iv_extraction independently recomputes T from real calendar dates rather than
    # trusting a stored value — generate prices against that same T (~0.504, not exactly
    # the module-level T=0.5 other tests use) so this test isn't checking a T mismatch.
    actual_T = year_fraction(asof, expiry, "ACT/365F")
    strikes = [80, 85, 90, 95, 100.01, 105, 110, 115, 120]

    def _row_at_actual_T(strike: float) -> dict:
        cp = "C" if strike > FORWARD else "P"
        pricer = call_price if cp == "C" else put_price
        price = pricer(FORWARD, strike, actual_T, TRUE_SIGMA, DISCOUNT_FACTOR)
        return {"strike": strike, "cp": cp, "bid": price - 0.02, "ask": price + 0.02}

    rows = [_row_at_actual_T(k) for k in strikes]
    n = len(rows)
    chain_df = pd.DataFrame(
        {
            "asof_date": [asof] * n,
            "asof_ts": [FRESH] * n,
            "underlying": ["TEST"] * n,
            "expiry": [expiry] * n,
            "strike": [r["strike"] for r in rows],
            "cp": [r["cp"] for r in rows],
            "bid": [r["bid"] for r in rows],
            "ask": [r["ask"] for r in rows],
            "bid_size": pd.array([None] * n, dtype="Int64"),
            "ask_size": pd.array([None] * n, dtype="Int64"),
            "volume": [10] * n,
            "open_interest": [100] * n,
            "underlying_px": [FORWARD] * n,
            "last_trade_ts": [FRESH] * n,
            "source": ["test"] * n,
        }
    )
    chain_table = validate(chain_df, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(chain_table, tmp_path / "chains", ["asof_date", "underlying"])

    fwd_df = pd.DataFrame(
        [
            {
                "asof_date": asof,
                "underlying": "TEST",
                "expiry": expiry,
                "T": actual_T,
                "n_strikes": 9,
                "forward": FORWARD,
                "discount_factor_implied": DISCOUNT_FACTOR,
                "discount_factor_curve": DISCOUNT_FACTOR,
                "discount_factor_diff_bp": 0.0,
                "r_squared": 0.9999,
                "implied_dividend_yield": 0.0,
                "announced_dividend_yield": None,
                "dividend_yield_diff": None,
                "flag_r2": False,
                "flag_discount_factor_bp": False,
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": True,
            }
        ]
    )
    fwd_table = validate(fwd_df, FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL)
    store.write_partitioned(fwd_table, tmp_path / "forwards", ["asof_date", "underlying"])

    from eqdrisk.config import BaseConfig, Paths, Universe

    cfg = BaseConfig(
        run_date=asof,
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )

    result = run_iv_extraction(cfg, asof)
    assert result.rejection_counts.get("TEST", {}) == {}
    assert "TEST" not in result.skipped_expiries

    out = store.query("SELECT * FROM iv", views={"iv": str(tmp_path / "implied_vols")}).to_pandas()
    assert len(out) == n
    ok = out[out["reason"] == "OK"]
    assert len(ok) == n
    assert ok["iv"].apply(lambda x: abs(x - TRUE_SIGMA) < 1e-5).all()


def _prev_ok_rows(asof: dt.date, expiries: list[dt.date], carried_from=None) -> pd.DataFrame:
    from eqdrisk.marketdata.calendar import year_fraction

    rows = []
    for expiry in expiries:
        T_prev = year_fraction(asof, expiry, "ACT/365F")
        for k, cp in [(-0.1, "P"), (0.0, "C"), (0.1, "C")]:
            rows.append(
                {
                    "asof_date": asof,
                    "underlying": "TEST",
                    "expiry": expiry,
                    "strike": 100.0 * float(pd.Series([k]).rpow(2.718281828459045).iloc[0]),
                    "cp": cp,
                    "T": T_prev,
                    "k": k,
                    "iv": 0.30,
                    "total_variance": 0.09 * T_prev,
                    "vega": 1.0,
                    "weight": 10.0,
                    "reason": OK,
                    "carried_from": carried_from,
                }
            )
    return pd.DataFrame(rows)


def test_carry_forward_quotes_fills_only_missing_expiries_in_todays_terms():
    from eqdrisk.marketdata.calendar import year_fraction
    from eqdrisk.vol.implied import STALE_FILL_WEIGHT, carry_forward_quotes

    prev, today = dt.date(2026, 8, 20), dt.date(2026, 8, 21)
    kept, missing, expired = dt.date(2026, 12, 18), dt.date(2027, 6, 17), dt.date(2026, 8, 21)
    prev_ok = _prev_ok_rows(prev, [kept, missing, expired])

    carried = carry_forward_quotes(prev_ok, {kept}, today)

    # Only the expiry with no OK quote today, and not one expiring today.
    assert set(carried["expiry"]) == {missing}
    T_today = year_fraction(today, missing, "ACT/365F")
    assert (carried["asof_date"] == today).all()
    assert carried["T"].tolist() == [T_today] * 3
    # Same log-moneyness and implied vol; total variance in today's T.
    assert carried["k"].tolist() == [-0.1, 0.0, 0.1]
    assert carried["iv"].tolist() == [0.30] * 3
    assert carried["total_variance"].tolist() == [0.09 * T_today] * 3
    assert (carried["weight"] == 10.0 * STALE_FILL_WEIGHT).all()
    assert (carried["carried_from"] == prev).all()
    assert (carried["reason"] == OK).all()


def test_carry_forward_quotes_halves_weight_once_and_stops_after_three_trading_days():
    from eqdrisk.vol.implied import STALE_FILL_WEIGHT, carry_forward_quotes

    observed = dt.date(2026, 8, 20)  # Thursday
    expiry = dt.date(2027, 6, 17)
    # Rows stored on Mon 08-24, already carried (weight already halved).
    already = _prev_ok_rows(dt.date(2026, 8, 24), [expiry], carried_from=observed)
    already["weight"] = 10.0 * STALE_FILL_WEIGHT

    # Tue 08-25 is 3 trading days after 08-20 (21, 24, 25): still carried.
    again = carry_forward_quotes(already, set(), dt.date(2026, 8, 25))
    assert len(again) == 3
    assert (again["weight"] == 10.0 * STALE_FILL_WEIGHT).all()
    assert (again["carried_from"] == observed).all()

    # Wed 08-26 is 4 trading days after: dropped.
    assert carry_forward_quotes(again, set(), dt.date(2026, 8, 26)).empty
    assert carry_forward_quotes(already.iloc[0:0], set(), dt.date(2026, 8, 25)).empty


def test_observed_quotes_excludes_carried_rows_and_accepts_legacy_frames():
    from eqdrisk.vol.implied import observed_quotes

    rows = _prev_ok_rows(dt.date(2026, 8, 20), [dt.date(2027, 6, 17)])
    rows.loc[0, "carried_from"] = dt.date(2026, 8, 19)
    assert len(observed_quotes(rows)) == 2
    assert len(observed_quotes(rows.drop(columns=["carried_from"]))) == 3


def _write_stale_fill_day(tmp_path, asof: dt.date, reliable_by_expiry: dict) -> None:
    """One underlying's chain priced at FORWARD / TRUE_SIGMA for each expiry, and a
    forward row per expiry that is reliable or not as given."""
    from eqdrisk.io import store
    from eqdrisk.io.schemas import (
        CHAIN_REQUIRED_NOT_NULL,
        CHAIN_SCHEMA,
        FORWARD_REQUIRED_NOT_NULL,
        FORWARD_SCHEMA,
        validate,
    )
    from eqdrisk.marketdata.calendar import year_fraction

    ts = pd.Timestamp(asof.isoformat() + " 16:00", tz="America/New_York")
    rows, fwd_rows = [], []
    for expiry, reliable in reliable_by_expiry.items():
        T_day = year_fraction(asof, expiry, "ACT/365F")
        for strike in STALE_FILL_STRIKES:
            cp = "C" if strike > FORWARD else "P"
            pricer = call_price if cp == "C" else put_price
            price = pricer(FORWARD, strike, T_day, TRUE_SIGMA, DISCOUNT_FACTOR)
            rows.append(
                {
                    "asof_date": asof,
                    "asof_ts": ts,
                    "underlying": "TEST",
                    "expiry": expiry,
                    "strike": strike,
                    "cp": cp,
                    "bid": price - 0.02,
                    "ask": price + 0.02,
                    "bid_size": None,
                    "ask_size": None,
                    "volume": 10,
                    "open_interest": 100,
                    "underlying_px": FORWARD,
                    "last_trade_ts": ts,
                    "source": "test",
                }
            )
        fwd_rows.append(
            {
                "asof_date": asof,
                "underlying": "TEST",
                "expiry": expiry,
                "T": T_day,
                "n_strikes": len(STALE_FILL_STRIKES),
                "forward": FORWARD,
                "discount_factor_implied": DISCOUNT_FACTOR,
                "discount_factor_curve": DISCOUNT_FACTOR,
                "discount_factor_diff_bp": 0.0 if reliable else 400.0,
                "r_squared": 0.9999,
                "implied_dividend_yield": 0.0,
                "announced_dividend_yield": None,
                "dividend_yield_diff": None,
                "flag_r2": False,
                "flag_discount_factor_bp": not reliable,
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": reliable,
            }
        )
    chain = pd.DataFrame(rows)
    chain["bid_size"] = pd.array(chain["bid_size"], dtype="Int64")
    chain["ask_size"] = pd.array(chain["ask_size"], dtype="Int64")
    store.write_partitioned(
        validate(chain, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL),
        tmp_path / "chains",
        ["asof_date", "underlying"],
    )
    store.write_partitioned(
        validate(pd.DataFrame(fwd_rows), FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL),
        tmp_path / "forwards",
        ["asof_date", "underlying"],
    )


STALE_FILL_STRIKES = [80, 85, 90, 95, 100.01, 105, 110, 115, 120]


def _stale_fill_cfg(tmp_path):
    from eqdrisk.config import BaseConfig, Paths, Universe

    return BaseConfig(
        run_date=dt.date(2026, 8, 21),
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def test_run_iv_extraction_carries_an_expiry_that_loses_its_forward(tmp_path):
    """Day 1 has a reliable forward for both expiries; on day 2 the longer one loses
    it (the typical long-dated single-name case). Day 2's surface input keeps day
    1's quotes for that expiry, flagged, instead of losing the expiry."""
    from eqdrisk.io import store
    from eqdrisk.marketdata.calendar import year_fraction
    from eqdrisk.vol.implied import NO_RELIABLE_FORWARD, run_iv_extraction

    anchor, expiry = dt.date(2026, 12, 18), dt.date(2027, 2, 20)
    cfg = _stale_fill_cfg(tmp_path)
    day1, day2 = dt.date(2026, 8, 20), dt.date(2026, 8, 21)
    _write_stale_fill_day(tmp_path, day1, {anchor: True, expiry: True})
    _write_stale_fill_day(tmp_path, day2, {anchor: True, expiry: False})
    run_iv_extraction(cfg, day1)
    result = run_iv_extraction(cfg, day2)

    assert result.skipped_expiries["TEST"] == [expiry]
    assert result.carried_expiries["TEST"] == [expiry]
    out = store.query(
        f"SELECT * FROM iv WHERE asof_date = DATE '{day2.isoformat()}'",
        views={"iv": str(tmp_path / "implied_vols")},
    ).to_pandas()
    # Today's own rows are kept (audit trail), plus the carried, flagged OK rows.
    assert (out["reason"] == NO_RELIABLE_FORWARD).sum() == len(STALE_FILL_STRIKES)
    carried = out[out["carried_from"].notna()]
    assert len(carried) == len(STALE_FILL_STRIKES)
    assert (carried["reason"] == OK).all()
    assert (pd.to_datetime(carried["expiry"]).dt.date == expiry).all()
    assert (pd.to_datetime(carried["carried_from"]).dt.date == day1).all()
    assert carried["T"].iloc[0] == year_fraction(day2, expiry, "ACT/365F")


def test_run_iv_extraction_does_not_rebuild_a_day_with_nothing_observed(tmp_path):
    """A day where no expiry has an observed OK quote (a failed or after-hours
    snapshot) is a lost day: nothing is carried into it, so no surface can be
    calibrated from the previous day's quotes alone (SPX on 2026-10-23)."""
    from eqdrisk.io import store
    from eqdrisk.vol.implied import NO_RELIABLE_FORWARD, run_iv_extraction

    anchor, expiry = dt.date(2026, 12, 18), dt.date(2027, 2, 20)
    cfg = _stale_fill_cfg(tmp_path)
    day1, day2 = dt.date(2026, 8, 20), dt.date(2026, 8, 21)
    _write_stale_fill_day(tmp_path, day1, {anchor: True, expiry: True})
    _write_stale_fill_day(tmp_path, day2, {anchor: False, expiry: False})
    run_iv_extraction(cfg, day1)
    result = run_iv_extraction(cfg, day2)

    assert "TEST" not in result.carried_expiries
    out = store.query(
        f"SELECT * FROM iv WHERE asof_date = DATE '{day2.isoformat()}'",
        views={"iv": str(tmp_path / "implied_vols")},
    ).to_pandas()
    assert (out["reason"] == NO_RELIABLE_FORWARD).all()
    assert out["carried_from"].isna().all()


def _write_c7_inputs(tmp_path, asof, rate, chain_expiries, forward_rows, underlying="TEST"):
    """Chains for `chain_expiries` priced at FORWARD / TRUE_SIGMA with the flat
    `rate` curve's discount factor, a flat `rate` rates curve, and the given
    forward rows (expiry -> dict of overrides)."""
    import numpy as np

    from eqdrisk.io import store
    from eqdrisk.io.schemas import (
        CHAIN_REQUIRED_NOT_NULL,
        CHAIN_SCHEMA,
        CURVE_REQUIRED_NOT_NULL,
        CURVE_SCHEMA,
        FORWARD_REQUIRED_NOT_NULL,
        FORWARD_SCHEMA,
        validate,
    )
    from eqdrisk.marketdata.calendar import year_fraction

    ts = pd.Timestamp(asof.isoformat() + " 16:00", tz="America/New_York")
    rows = []
    for expiry in chain_expiries:
        T_e = year_fraction(asof, expiry, "ACT/365F")
        df_e = float(np.exp(-rate * T_e))
        for strike in [70, 75, 80, 85, 90, 95, 100.01, 105, 110, 115, 120, 125, 130]:
            cp = "C" if strike > FORWARD else "P"
            price = (call_price if cp == "C" else put_price)(FORWARD, strike, T_e, TRUE_SIGMA, df_e)
            rows.append(
                {
                    "asof_date": asof,
                    "asof_ts": ts,
                    "underlying": underlying,
                    "expiry": expiry,
                    "strike": strike,
                    "cp": cp,
                    "bid": price - 0.02,
                    "ask": price + 0.02,
                    "volume": 10,
                    "open_interest": 100,
                    "underlying_px": FORWARD,
                    "last_trade_ts": ts,
                    "source": "test",
                }
            )
    chain = pd.DataFrame(rows)
    chain["bid_size"] = pd.array([None] * len(chain), dtype="Int64")
    chain["ask_size"] = pd.array([None] * len(chain), dtype="Int64")
    store.write_partitioned(
        validate(chain, CHAIN_SCHEMA, CHAIN_REQUIRED_NOT_NULL),
        tmp_path / "chains",
        ["asof_date", "underlying"],
    )
    rates = pd.DataFrame(
        {
            "asof_date": [asof] * 4,
            "tenor": ["3M", "1Y", "2Y", "5Y"],
            "rate": [100 * rate] * 4,
            "source": ["test"] * 4,
        }
    )
    store.write_partitioned(
        validate(rates, CURVE_SCHEMA, CURVE_REQUIRED_NOT_NULL), tmp_path / "curves", ["asof_date"]
    )
    fwd = []
    for expiry, overrides in forward_rows.items():
        T_e = year_fraction(asof, expiry, "ACT/365F")
        df_e = float(np.exp(-rate * T_e))
        fwd.append(
            {
                "asof_date": asof,
                "underlying": underlying,
                "expiry": expiry,
                "T": T_e,
                "n_strikes": 9,
                "forward": FORWARD,
                "discount_factor_implied": df_e,
                "discount_factor_curve": df_e,
                "discount_factor_diff_bp": 0.0,
                "r_squared": 0.9999,
                "implied_dividend_yield": 0.0,
                "announced_dividend_yield": None,
                "dividend_yield_diff": None,
                "flag_r2": False,
                "flag_discount_factor_bp": False,
                "method": "parity_2p",
                "dispersion_bp": None,
                "reliable": True,
                **overrides,
            }
        )
    store.write_partitioned(
        validate(pd.DataFrame(fwd), FORWARD_SCHEMA, FORWARD_REQUIRED_NOT_NULL),
        tmp_path / "forwards",
        ["asof_date", "underlying"],
    )


def _c7_cfg(tmp_path, asof, single_name: bool):
    from eqdrisk.config import BaseConfig, Paths, Universe

    return BaseConfig(
        run_date=asof,
        universe=Universe(
            index=[] if single_name else ["TEST"], single_names=["TEST"] if single_name else []
        ),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _ivs(tmp_path):
    from eqdrisk.io import store

    return store.query("SELECT * FROM iv", views={"iv": str(tmp_path / "implied_vols")}).to_pandas()


def test_single_name_expiry_without_a_fit_uses_the_forward_curve(tmp_path):
    """C7: a long single-name expiry with clean OTM quotes but no forward fit
    gets its forward from the day's forward curve (here flat, zero carry, so
    exactly FORWARD) and the rates curve's DF, so today's own quotes are used and
    recover the true vol, flagged forward_source = curve."""
    from eqdrisk.vol.implied import FORWARD_FROM_CURVE, FORWARD_FROM_FIT, run_iv_extraction

    asof, short, long_ = dt.date(2026, 8, 20), dt.date(2026, 12, 18), dt.date(2028, 6, 16)
    _write_c7_inputs(tmp_path, asof, 0.03, [short, long_], {short: {}})

    result = run_iv_extraction(_c7_cfg(tmp_path, asof, single_name=True), asof)

    assert result.curve_forward_expiries["TEST"] == [long_]
    assert "TEST" not in result.skipped_expiries
    out = _ivs(tmp_path)
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
    long_ok = out[(out["expiry"] == long_) & (out["reason"] == OK)]
    assert len(long_ok) >= 8
    assert (long_ok["forward_source"] == FORWARD_FROM_CURVE).all()
    assert (long_ok["iv"] - TRUE_SIGMA).abs().max() < 1e-6
    assert (out.loc[out["expiry"] == short, "forward_source"] == FORWARD_FROM_FIT).all()


def test_index_expiry_without_a_fit_is_still_skipped(tmp_path):
    from eqdrisk.vol.implied import NO_RELIABLE_FORWARD, run_iv_extraction

    asof, short, long_ = dt.date(2026, 8, 20), dt.date(2026, 12, 18), dt.date(2028, 6, 16)
    _write_c7_inputs(tmp_path, asof, 0.03, [short, long_], {short: {}})

    result = run_iv_extraction(_c7_cfg(tmp_path, asof, single_name=False), asof)

    assert result.skipped_expiries["TEST"] == [long_]
    out = _ivs(tmp_path)
    long_rows = out[pd.to_datetime(out["expiry"]).dt.date == long_]
    assert (long_rows["reason"] == NO_RELIABLE_FORWARD).all()
    assert long_rows["forward_source"].isna().all()


def test_curve_df_forward_extracts_ivs_with_the_curve_discount_factor(tmp_path):
    """A curve_df fit was made with the curve's DF; its IVs must use that DF too,
    not the (drifting, for American options) parity-implied one."""
    from eqdrisk.vol.implied import run_iv_extraction

    asof, expiry = dt.date(2026, 8, 20), dt.date(2027, 6, 17)
    _write_c7_inputs(
        tmp_path,
        asof,
        0.03,
        [expiry],
        {expiry: {"method": "curve_df", "dispersion_bp": 5.0, "discount_factor_implied": 0.90}},
    )

    run_iv_extraction(_c7_cfg(tmp_path, asof, single_name=True), asof)

    ok = _ivs(tmp_path).query("reason == 'OK'")
    assert len(ok) >= 8
    assert (ok["iv"] - TRUE_SIGMA).abs().max() < 1e-6


def test_run_iv_extraction_carries_across_a_lost_day(tmp_path):
    """Observed day, lost day, then a day missing one expiry: the missing expiry is
    carried from the observed day (still inside the 3-trading-day limit), not lost
    because the day in between had nothing to carry (SPX, 2026-10-23 -> 10-26)."""
    from eqdrisk.io import store
    from eqdrisk.vol.implied import run_iv_extraction

    anchor, expiry = dt.date(2026, 12, 18), dt.date(2027, 2, 20)
    cfg = _stale_fill_cfg(tmp_path)
    day1, lost, day3 = dt.date(2026, 8, 19), dt.date(2026, 8, 20), dt.date(2026, 8, 21)
    _write_stale_fill_day(tmp_path, day1, {anchor: True, expiry: True})
    _write_stale_fill_day(tmp_path, lost, {anchor: False, expiry: False})
    _write_stale_fill_day(tmp_path, day3, {anchor: True, expiry: False})
    for d in (day1, lost):
        run_iv_extraction(cfg, d)
    result = run_iv_extraction(cfg, day3)

    assert result.carried_expiries["TEST"] == [expiry]
    out = store.query(
        f"SELECT * FROM iv WHERE asof_date = DATE '{day3.isoformat()}'",
        views={"iv": str(tmp_path / "implied_vols")},
    ).to_pandas()
    carried = out[out["carried_from"].notna()]
    assert (pd.to_datetime(carried["carried_from"]).dt.date == day1).all()
