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
