import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.curve import bootstrap_curve


def _rates_df():
    return pd.DataFrame(
        {
            "tenor": ["SOFR", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"],
            "rate": [3.65, 3.78, 3.86, 3.94, 3.99, 4.19, 4.37, 4.71],
        }
    )


def test_discount_factor_matches_pillar_exactly():
    curve = bootstrap_curve(_rates_df())
    # 1Y pillar: P(0,1) = exp(-0.0399 * 1)
    assert curve.discount_factor(1.0) == pytest.approx(np.exp(-0.0399), rel=1e-9)


def test_discount_factor_decreases_with_maturity():
    curve = bootstrap_curve(_rates_df())
    dfs = [curve.discount_factor(T) for T in [0.1, 0.5, 1, 2, 5, 10]]
    assert all(a > b for a, b in zip(dfs, dfs[1:], strict=False))


def test_flat_extrapolation_beyond_last_pillar():
    curve = bootstrap_curve(_rates_df())
    assert curve.discount_factor(10.0) == pytest.approx(curve.discount_factor(15.0), rel=1e-9)


def test_zero_rate_matches_input_rate_at_pillar():
    curve = bootstrap_curve(_rates_df())
    assert curve.zero_rate(1.0) == pytest.approx(0.0399, rel=1e-6)


def test_zero_rate_clamped_below_shortest_pillar():
    curve = bootstrap_curve(_rates_df())
    # Should not blow up for T smaller than the SOFR pillar (~1/365).
    assert curve.zero_rate(0.0001) == pytest.approx(curve.zero_rate(1 / 365), rel=1e-6)


def test_bootstrap_raises_on_no_recognised_tenors():
    with pytest.raises(ValueError, match="no recognised tenors"):
        bootstrap_curve(pd.DataFrame({"tenor": ["BOGUS"], "rate": [1.0]}))
