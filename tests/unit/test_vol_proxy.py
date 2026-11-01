import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eqdrisk.marketdata.calendar import last_n_trading_days
from eqdrisk.risk.proxy import (
    MIN_PROXY_CHANGES,
    fit_proxy,
    index_changes,
    node_field,
    proxy_for_underlying,
)
from eqdrisk.vol.risk_factors import K_GRID, T_GRID


def _history(n_days: int, seed: int = 0):
    """`n_days` consecutive trading days where every node's implied vol is
    20 + beta(T) x (index - first index), beta falling with maturity: each
    daily vol change is exactly beta x the index change."""
    days = last_n_trading_days(dt.date(2026, 10, 30), n_days)
    rng = np.random.default_rng(seed)
    index = pd.Series(18.0 + np.cumsum(rng.normal(0.0, 1.0, n_days)), index=days)
    beta = {t: 0.6 / (1 + 4 * T) for t, T in T_GRID.items()}
    surface, rf = [], []
    for d in days:
        for T in (0.05, 3.0):  # brackets every grid maturity, quotes span all k
            surface.append({"asof_date": d, "T": T, "k_min": -0.5, "k_max": 0.5})
        for t_label, T in T_GRID.items():
            iv = 20.0 + beta[t_label] * (index[d] - index.iloc[0])
            for k in K_GRID:
                rf.append(
                    {"asof_date": d, "T_label": t_label, "T": T, "k": k, "w": (iv / 100) ** 2 * T}
                )
    return pd.DataFrame(surface), pd.DataFrame(rf), index, beta


def test_fit_proxy_recovers_a_known_line_and_r2():
    x = pd.Series([1.0, -2.0, 0.5, 3.0, -1.0])
    changes = pd.DataFrame({"a": 0.1 + 0.5 * x, "b": -0.2 + 2.0 * x})
    alpha, beta, r2 = fit_proxy(changes, x)
    np.testing.assert_allclose(alpha, [0.1, -0.2], atol=1e-12)
    np.testing.assert_allclose(beta, [0.5, 2.0], atol=1e-12)
    np.testing.assert_allclose(r2, [1.0, 1.0], atol=1e-12)

    noisy = changes.assign(a=changes["a"] + [0.3, -0.3, 0.2, -0.1, 0.4])
    _, _, r2_noisy = fit_proxy(noisy, x)
    assert 0.0 < r2_noisy[0] < 1.0


def test_index_changes_use_the_previous_trading_day():
    fri, mon, tue = dt.date(2026, 10, 23), dt.date(2026, 10, 26), dt.date(2026, 10, 27)
    levels = pd.Series({fri: 15.0, mon: 17.0})
    out = index_changes(pd.Index([mon, tue]), levels)
    assert out[mon] == pytest.approx(2.0)  # across the weekend
    assert np.isnan(out[tue])  # no Tuesday close


def test_node_field_fills_from_the_nearest_k_then_the_nearest_maturity():
    nodes = [("1m", -0.1), ("1m", 0.0), ("6m", 0.0)]
    field = node_field(nodes, np.array([1.0, 2.0, 5.0]))
    t = list(T_GRID)
    k = list(K_GRID)
    one_m = field[t.index("1m")]
    assert one_m[k.index(-0.3)] == 1.0  # beyond the lowest given k: held flat
    assert one_m[k.index(0.2)] == 2.0
    assert np.all(field[t.index("3m")] == one_m)  # 3m nearer 1m than 6m
    assert np.all(field[t.index("2y")] == 5.0)  # nearest row with a node is 6m


def test_proxy_for_underlying_recovers_betas_on_noiseless_history():
    surface, rf, index, beta = _history(MIN_PROXY_CHANGES + 5)
    fit = proxy_for_underlying("SPX", surface, rf, index)
    assert not isinstance(fit, str)
    assert fit.index == "VIX"
    assert len(fit.nodes) == len(K_GRID) * len(T_GRID)
    expected = np.array([beta[t] for t, _ in fit.nodes])
    np.testing.assert_allclose(fit.beta, expected, atol=1e-6)
    np.testing.assert_allclose(fit.r2, 1.0, atol=1e-9)
    assert np.abs(fit.residuals.to_numpy()).max() < 1e-6
    assert fit.beta_field().shape == (len(T_GRID), len(K_GRID))


def test_proxy_for_underlying_refuses_too_short_a_history():
    surface, rf, index, _ = _history(MIN_PROXY_CHANGES - 5)
    reason = proxy_for_underlying("SPX", surface, rf, index)
    assert isinstance(reason, str) and "need" in reason


def test_proxy_for_underlying_needs_a_mapped_index():
    surface, rf, index, _ = _history(MIN_PROXY_CHANGES + 5)
    assert isinstance(proxy_for_underlying("JPM", surface, rf, index), str)
