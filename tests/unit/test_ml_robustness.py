import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch
from test_ml_deep_hedge import SPOT, _flat_inputs
from test_ml_run import ASOF, UNDERLYING, _cfg, _seed_real_data, _tiny_train_cfg

import eqdrisk.ml.robustness as robustness_module
import eqdrisk.ml.run as run_module
from eqdrisk.io import store
from eqdrisk.ml.evaluate import compare_vanilla_on_paths, evaluate_vanilla_hedge
from eqdrisk.ml.hedge_model import HedgeNet
from eqdrisk.ml.registry import load_model, save_model
from eqdrisk.ml.robustness import (
    SCENARIOS,
    history_paths,
    jump_paths,
    load_history_closes,
    run_robustness,
    scaled_vol_paths,
)
from eqdrisk.ml.run import INSTRUMENTS, LOSS_TYPES, run_deep_hedge
from eqdrisk.ml.simulate import simulate_training_paths

# --- registry -------------------------------------------------------------------


def test_saved_model_round_trips_exactly(tmp_path):
    torch.manual_seed(0)
    net = HedgeNet(hidden=8)
    save_model(net, tmp_path, ASOF, "vanilla", "cost", 2)
    loaded = load_model(tmp_path, ASOF, "vanilla", "cost", 2)

    x = torch.tensor([[0.1, 0.5], [-0.2, 0.9]], dtype=torch.float64)
    assert loaded is not None
    with torch.no_grad():
        assert torch.equal(net(x), loaded(x))


def test_saved_model_keeps_its_position_limit(tmp_path):
    torch.manual_seed(0)
    save_model(HedgeNet(hidden=8, limit=1.5), tmp_path, ASOF, "barrier", "cvar", 0)
    loaded = load_model(tmp_path, ASOF, "barrier", "cvar", 0)
    assert loaded is not None and loaded.limit == 1.5


def test_model_saved_before_position_limits_loads_unbounded(tmp_path):
    torch.manual_seed(0)
    net = HedgeNet(hidden=8)
    path = save_model(net, tmp_path, ASOF, "vanilla", "cost", 0)
    torch.save({"hidden": 8, "state_dict": net.state_dict()}, path)  # pre-limit format
    loaded = load_model(tmp_path, ASOF, "vanilla", "cost", 0)
    assert loaded is not None and loaded.limit is None


def test_load_model_returns_none_when_never_saved(tmp_path):
    assert load_model(tmp_path, ASOF, "vanilla", "cost", 0) is None


# --- scenario generators ----------------------------------------------------------


def _log_ret_std(sim):
    return float(np.diff(np.log(sim.paths.numpy()), axis=1).std())


def test_scaled_vol_with_factor_one_is_the_unshocked_simulation():
    inputs = _flat_inputs()
    base = simulate_training_paths(inputs, 256, 8, seed=11)
    same = scaled_vol_paths(inputs, 1.0, 256, 8, seed=11)
    assert torch.equal(base.paths, same.paths)


def test_scaled_vol_scales_realized_volatility():
    inputs = _flat_inputs()
    base = _log_ret_std(simulate_training_paths(inputs, 2048, 8, seed=11))
    up = _log_ret_std(scaled_vol_paths(inputs, 1.25, 2048, 8, seed=11))
    assert up / base == pytest.approx(1.25, rel=0.02)


def test_jump_paths_with_zero_intensity_are_the_unshocked_simulation():
    inputs = _flat_inputs()
    base = simulate_training_paths(inputs, 256, 8, seed=11)
    same = jump_paths(inputs, 256, 8, seed=11, intensity=0.0)
    assert torch.allclose(base.paths, same.paths, rtol=1e-12)


def test_jump_paths_are_compensated_but_add_a_crash_tail():
    """Compensation keeps the expected terminal price where the diffusion put it;
    the jumps themselves must still fatten the downside."""
    inputs = _flat_inputs(T=1.0)
    base = simulate_training_paths(inputs, 8192, 16, seed=5)
    shocked = jump_paths(inputs, 8192, 16, seed=5, intensity=2.0, mean=-0.15, std=0.05)

    assert torch.allclose(shocked.paths[:, 0], base.paths[:, 0])
    base_t, shocked_t = base.paths[:, -1].numpy(), shocked.paths[:, -1].numpy()
    assert shocked_t.mean() == pytest.approx(base_t.mean(), rel=0.02)
    assert np.quantile(shocked_t, 0.01) < np.quantile(base_t, 0.01)


def test_history_paths_samples_real_closes_at_the_training_grid():
    closes = np.exp(0.001 * np.arange(1000))  # +0.1% log return every day
    T = 0.25  # 63 trading days
    windows = history_paths(closes, spot=SPOT, T=T, n_steps=4, stride=10)

    paths = windows.paths.paths.numpy()
    offsets = np.rint(windows.paths.t_grid / T * 63).astype(int)
    assert np.allclose(paths[:, 0], SPOT)
    assert np.allclose(paths[0], SPOT * np.exp(0.001 * offsets))
    assert windows.n_windows == len(range(0, 1000 - 63, 10))
    assert windows.n_independent == pytest.approx((1000 / 252) / T)


def test_history_paths_rejects_unadjusted_splits():
    closes = np.r_[np.full(300, 100.0), np.full(300, 10.0)]  # a 10:1 split left in
    with pytest.raises(ValueError, match="unadjusted"):
        history_paths(closes, spot=SPOT, T=0.25, n_steps=4)


def test_load_history_closes_caches_so_reruns_replay_identical_history(tmp_path):
    calls = []

    def fake_fetch(tickers, start, end):
        calls.append(tickers)
        days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(5)]
        return pd.DataFrame({"asof_date": days[::-1], "close": [5.0, 4.0, 3.0, 2.0, 1.0]})

    args = ("NVDA", dt.date(2006, 1, 1), dt.date(2026, 1, 1), tmp_path)
    first = load_history_closes(*args, fetch=fake_fetch)
    second = load_history_closes(*args, fetch=fake_fetch)
    assert len(calls) == 1
    assert np.array_equal(first, [1.0, 2.0, 3.0, 4.0, 5.0])  # sorted by date
    assert np.array_equal(first, second)


# --- shared scoring ---------------------------------------------------------------


def test_compare_on_paths_is_exactly_what_evaluate_reports():
    """The Phase 7 refactor must not change how anything is scored."""
    inputs = _flat_inputs()
    torch.manual_seed(0)
    net = HedgeNet(hidden=4)
    via_evaluate = evaluate_vanilla_hedge(inputs, net, SPOT, True, 64, 4, 5.0, 999)
    sim = simulate_training_paths(inputs, 64, 4, 999)
    via_compare = compare_vanilla_on_paths(inputs, net, SPOT, True, sim, 5.0)
    assert via_compare == via_evaluate


# --- end to end -------------------------------------------------------------------


def _fake_history(tickers, start, end):
    rng = np.random.default_rng(0)
    n = 700  # long enough for the 1-year autocallable window
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    closes = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    return pd.DataFrame({"asof_date": days, "close": closes})


def test_run_robustness_scores_every_saved_policy_on_every_scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "_train_cfg", _tiny_train_cfg)
    monkeypatch.setattr(robustness_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    cfg = _cfg(tmp_path)
    for instrument in INSTRUMENTS:
        for loss in LOSS_TYPES:
            run_deep_hedge(
                cfg,
                ASOF,
                instrument,
                loss,
                underlying=UNDERLYING,
                n_paths_eval=32,
                project_root=tmp_path,
                seeds=(0,),
            )

    df = run_robustness(
        cfg,
        ASOF,
        seeds=(0,),
        n_paths=32,
        project_root=tmp_path,
        fetch=_fake_history,
        underlying=UNDERLYING,
    )

    assert df is not None
    assert len(df) == len(INSTRUMENTS) * len(LOSS_TYPES) * len(SCENARIOS)
    assert set(df["scenario"]) == set(SCENARIOS)
    assert df.loc[df["scenario"] == "history", "n_independent"].notna().all()
    assert df.loc[df["scenario"] != "history", "n_independent"].isna().all()

    stored = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_robustness")}
    ).to_pandas()
    assert len(stored) == len(df)

    # Regression check: the in-sample scenario reproduces the stored training-run
    # results exactly, because it is the same held-out seed and the same scorer.
    trained = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "deep_hedge_results")}
    ).to_pandas()
    in_sample = df[df["scenario"] == "in_sample"]
    merged = in_sample.merge(trained, on=["instrument", "loss_type", "seed"], suffixes=("", "_t"))
    assert len(merged) == len(in_sample)
    assert np.allclose(merged["learned_std"], merged["learned_std_t"], rtol=1e-12)
    assert np.allclose(merged["baseline_std"], merged["baseline_std_t"], rtol=1e-12)


def test_run_robustness_refuses_policies_that_were_never_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(robustness_module, "_train_cfg", _tiny_train_cfg)
    _seed_real_data(tmp_path)
    with pytest.raises(FileNotFoundError, match="deephedge --all"):
        run_robustness(
            _cfg(tmp_path),
            ASOF,
            seeds=(0,),
            n_paths=32,
            project_root=tmp_path,
            fetch=_fake_history,
            underlying=UNDERLYING,
        )
