import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.ml.baseline import (
    autocallable_static_delta_hedge_pnl,
    barrier_static_delta_hedge_pnl,
    bs_delta_hedge_pnl,
)
from eqdrisk.ml.evaluate import (
    evaluate_autocall_hedge,
    evaluate_barrier_hedge,
    evaluate_vanilla_hedge,
)
from eqdrisk.ml.hedge_model import (
    HedgeNet,
    rollout_autocallable_hedged_pnl,
    rollout_hedged_pnl,
)
from eqdrisk.ml.losses import cost_adjusted_loss, cvar_loss, variance_loss
from eqdrisk.ml.market import HedgingMarketInputs, load_hedging_inputs
from eqdrisk.ml.payoffs import (
    autocallable_payoff_and_alive_schedule,
    down_and_in_put_payoff,
    vanilla_payoff,
)
from eqdrisk.ml.simulate import simulate_training_paths
from eqdrisk.ml.train import (
    TrainConfig,
    train_autocall_hedge,
    train_barrier_hedge,
    train_vanilla_hedge,
    train_vanilla_hedge_general,
)
from eqdrisk.pricing.autocallable import AutocallableSpec, autocallable_payoff
from eqdrisk.vol.local_vol import LocalVolGrid

UNDERLYING = "TEST"
ASOF = dt.date(2026, 8, 20)
FLAT_SIGMA = 0.30
SPOT = 100.0
T = 0.25


def _flat_vol_grid(sigma: float = FLAT_SIGMA) -> LocalVolGrid:
    s_grid = np.linspace(1.0, 500.0, 20)
    t_grid = np.linspace(0.0, 2.0, 10)
    sigma_loc = np.full((len(t_grid), len(s_grid)), sigma)
    return LocalVolGrid(s_grid=s_grid, t_grid=t_grid, sigma_loc=sigma_loc, n_floored=0)


def _flat_surface(sigma: float = FLAT_SIGMA, T: float = T) -> pd.DataFrame:
    """A single SVI pillar with a=sigma^2*T, b=rho=m=0 -> w(k)=a for all k, i.e.
    a flat smile at `sigma` — matches `_flat_vol_grid`'s constant local vol."""
    return pd.DataFrame(
        [
            {
                "T": T,
                "model": "SVI",
                "a": sigma**2 * T,
                "b": 0.0,
                "rho": 0.0,
                "m": 0.0,
                "sigma": 0.10,
            }
        ]
    )


def _flat_inputs(sigma: float = FLAT_SIGMA, T: float = T) -> HedgingMarketInputs:
    return HedgingMarketInputs(
        underlying=UNDERLYING,
        asof=ASOF,
        T=T,
        spot=SPOT,
        r=0.03,
        q=0.01,
        grid=_flat_vol_grid(sigma),
        surface=_flat_surface(sigma, T),
    )


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


# --- HedgingMarketInputs.atm_implied_vol ------------------------------------


def test_atm_implied_vol_recovers_flat_sigma():
    inputs = _flat_inputs(sigma=0.25)
    assert inputs.atm_implied_vol() == pytest.approx(0.25, rel=1e-6)


# --- load_hedging_inputs -----------------------------------------------------


def test_load_hedging_inputs_returns_none_when_no_curated_data(tmp_path):
    cfg = _cfg(tmp_path)
    assert load_hedging_inputs(cfg, ASOF, T, underlying=UNDERLYING) is None


# --- simulate_training_paths --------------------------------------------------


def test_simulate_training_paths_shape_and_start():
    inputs = _flat_inputs()
    sim = simulate_training_paths(inputs, n_paths=64, n_steps=8, seed=0)
    assert sim.paths.shape[1] == 9  # n_steps rounded to a power of two + 1, >= 9
    assert torch.allclose(sim.paths[:, 0], torch.tensor(SPOT, dtype=sim.paths.dtype))
    assert torch.all(torch.isfinite(sim.paths))
    assert not sim.paths.requires_grad


# --- vanilla_payoff -----------------------------------------------------------


def test_vanilla_payoff_call_and_put():
    paths = torch.tensor([[100.0, 110.0], [100.0, 90.0]], dtype=torch.float64)
    call = vanilla_payoff(paths, strike=100.0, is_call=True)
    put = vanilla_payoff(paths, strike=100.0, is_call=False)
    assert call.tolist() == pytest.approx([10.0, 0.0])
    assert put.tolist() == pytest.approx([0.0, 10.0])


# --- rollout_hedged_pnl --------------------------------------------------------


class _ZeroNet(torch.nn.Module):
    """Always outputs zero holdings — a network that never hedges. With zero
    transaction costs this must reduce hedged P&L to exactly -payoff, a real
    correctness check on the rollout's accounting, not just "it runs"."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=features.dtype)


def test_rollout_with_never_hedging_net_equals_negative_payoff():
    paths = torch.tensor([[100.0, 105.0, 110.0], [100.0, 95.0, 90.0]], dtype=torch.float64)
    t_grid = np.array([0.0, 0.5, 1.0])
    hedged_pnl = rollout_hedged_pnl(
        paths,
        t_grid,
        _ZeroNet(),
        strike=100.0,
        T=1.0,
        cost_bps=0.0,
        payoff_fn=lambda p: vanilla_payoff(p, strike=100.0, is_call=True),
    )
    expected = -vanilla_payoff(paths, strike=100.0, is_call=True)
    assert hedged_pnl.tolist() == pytest.approx(expected.tolist())


def test_rollout_gradients_flow_to_network_parameters():
    """The whole point of the PyTorch rollout: loss.backward() must populate
    real (non-None, non-zero) gradients on the network's parameters."""
    inputs = _flat_inputs()
    sim = simulate_training_paths(inputs, n_paths=32, n_steps=4, seed=0)
    net = HedgeNet(hidden=4)
    hedged_pnl = rollout_hedged_pnl(
        sim.paths,
        sim.t_grid,
        net,
        strike=SPOT,
        T=inputs.T,
        cost_bps=5.0,
        payoff_fn=lambda p: vanilla_payoff(p, strike=SPOT, is_call=True),
    )
    loss = variance_loss(hedged_pnl)
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads if g is not None)


# --- bs_delta_hedge_pnl --------------------------------------------------------


def test_bs_delta_hedge_variance_shrinks_with_more_rebalancing():
    """More frequent discrete rebalancing must reduce hedge error variance —
    the defining convergence property of delta hedging, not assumed."""
    inputs = _flat_inputs()
    sigma = inputs.atm_implied_vol()

    def hedge_std(n_steps: int) -> float:
        sim = simulate_training_paths(inputs, n_paths=4000, n_steps=n_steps, seed=7)
        pnl = bs_delta_hedge_pnl(
            sim.paths.numpy(),
            sim.t_grid,
            strike=SPOT,
            is_call=True,
            T=inputs.T,
            r=inputs.r,
            q=inputs.q,
            sigma=sigma,
            cost_bps=0.0,
        )
        return float(np.std(pnl))

    assert hedge_std(64) < hedge_std(4)


# --- end-to-end train + evaluate (fast smoke test) -----------------------------


def test_train_and_evaluate_vanilla_hedge_runs_end_to_end():
    inputs = _flat_inputs()
    train_cfg = TrainConfig(n_paths_train=64, n_steps=4, epochs=3, hidden=4, cost_bps=5.0)
    result = train_vanilla_hedge(inputs, strike=SPOT, is_call=True, cfg=train_cfg)

    assert len(result.loss_history) == 3
    assert all(np.isfinite(loss_val) for loss_val in result.loss_history)

    comparison = evaluate_vanilla_hedge(
        inputs,
        result.net,
        strike=SPOT,
        is_call=True,
        n_paths_eval=64,
        n_steps=4,
        cost_bps=5.0,
        seed_eval=999,
    )
    assert np.isfinite(comparison.learned.std)
    assert np.isfinite(comparison.baseline.std)
    assert comparison.n_eval_paths > 0


# --- Phase 2: cvar_loss / cost_adjusted_loss -----------------------------------


def test_cvar_loss_only_reflects_the_worst_tail():
    hedged_pnl = torch.tensor([10.0, 5.0, 0.0, -5.0, -100.0], dtype=torch.float64)
    # alpha=0.8 -> worst 20% -> exactly the single worst observation (-100).
    loss = cvar_loss(hedged_pnl, alpha=0.8)
    assert loss.item() == pytest.approx(100.0)


def test_cost_adjusted_loss_increases_with_turnover():
    hedged_pnl = torch.zeros(10, dtype=torch.float64)
    low_turnover = torch.full((10,), 1.0, dtype=torch.float64)
    high_turnover = torch.full((10,), 100.0, dtype=torch.float64)
    assert cost_adjusted_loss(hedged_pnl, high_turnover) > cost_adjusted_loss(
        hedged_pnl, low_turnover
    )


def test_train_vanilla_hedge_general_runs_for_every_loss_type():
    inputs = _flat_inputs()
    train_cfg = TrainConfig(n_paths_train=64, n_steps=4, epochs=3, hidden=4, cost_bps=5.0)
    for loss_type in ("variance", "cvar", "cost"):
        result = train_vanilla_hedge_general(
            inputs, strike=SPOT, is_call=True, loss_type=loss_type, cfg=train_cfg
        )
        assert len(result.loss_history) == 3
        assert all(np.isfinite(loss_val) for loss_val in result.loss_history)


# --- Phase 3: autocallable payoff/alive-schedule -------------------------------


def _autocall_spec(obs_times=None) -> AutocallableSpec:
    return AutocallableSpec(
        notional=1_000_000.0,
        autocall_barrier=1.00,
        coupon_barrier=0.75,
        put_barrier=0.65,
        coupon_rate=0.0225,
        obs_times=obs_times if obs_times is not None else np.array([0.25, 0.5, 0.75, 1.0]),
    )


def test_autocallable_torch_payoff_matches_real_numpy_payoff_undiscounted():
    """The differentiable torch payoff must exactly match this project's real,
    already-tested `pricing/autocallable.py::autocallable_payoff` (same rules,
    r=0 to strip that function's discounting for a fair undiscounted compare)."""
    spec = _autocall_spec()
    rng = np.random.default_rng(0)
    obs_levels_np = 100.0 * np.exp(rng.normal(0, 0.2, size=(500, 4)).cumsum(axis=1))

    expected = autocallable_payoff(obs_levels_np, spec, initial_level=100.0, r=0.0)
    actual, _ = autocallable_payoff_and_alive_schedule(
        torch.from_numpy(obs_levels_np), spec, initial_level=100.0
    )
    assert actual.numpy() == pytest.approx(expected, rel=1e-9)


def test_alive_schedule_marks_early_autocall_correctly():
    spec = _autocall_spec()
    # Path 0 autocalls immediately at obs 0 (level >= 100). Path 1 never autocalls.
    obs_levels = torch.tensor([[110.0, 110.0, 110.0, 110.0], [50.0, 50.0, 50.0, 50.0]])
    _, alive_schedule = autocallable_payoff_and_alive_schedule(
        obs_levels, spec, initial_level=100.0
    )
    assert alive_schedule[0].tolist() == [True, False, False, False]
    assert alive_schedule[1].tolist() == [True, True, True, True]


# --- Phase 3: rollout_autocallable_hedged_pnl ----------------------------------


def test_rollout_autocallable_never_hedging_equals_negative_payoff():
    spec = _autocall_spec()
    obs_levels = torch.tensor(
        [[105.0, 110.0, 90.0, 95.0], [80.0, 70.0, 60.0, 50.0]], dtype=torch.float64
    )
    hedged_pnl = rollout_autocallable_hedged_pnl(
        obs_levels, spec.obs_times, _ZeroNet(), spec, initial_level=100.0, T=1.0, cost_bps=0.0
    )
    expected, _ = autocallable_payoff_and_alive_schedule(obs_levels, spec, initial_level=100.0)
    assert hedged_pnl.tolist() == pytest.approx((-expected).tolist())


def test_rollout_autocallable_gradients_flow_to_network_parameters():
    spec = _autocall_spec()
    inputs = _flat_inputs(T=1.0)
    sim = simulate_training_paths(inputs, n_paths=32, n_steps=4, seed=0)
    obs_levels = sim.paths[:, 1:]
    net = HedgeNet(hidden=4)
    hedged_pnl = rollout_autocallable_hedged_pnl(
        obs_levels, spec.obs_times, net, spec, initial_level=inputs.spot, T=inputs.T, cost_bps=5.0
    )
    variance_loss(hedged_pnl).backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads if g is not None)


# --- Phase 3: autocallable_static_delta_hedge_pnl ------------------------------


def test_autocallable_static_hedge_zero_delta_equals_negative_payoff():
    spec = _autocall_spec()
    obs_levels = np.array([[105.0, 110.0, 90.0, 95.0], [80.0, 70.0, 60.0, 50.0]])
    baseline_pnl = autocallable_static_delta_hedge_pnl(
        obs_levels, spec, initial_level=100.0, static_delta_shares=0.0, cost_bps=0.0
    )
    expected, _ = autocallable_payoff_and_alive_schedule(
        torch.from_numpy(obs_levels), spec, initial_level=100.0
    )
    assert baseline_pnl == pytest.approx((-expected).numpy())


# --- Phase 3: end-to-end train + evaluate (fast smoke test) -------------------


def test_train_and_evaluate_autocall_hedge_runs_end_to_end():
    inputs = _flat_inputs(T=1.0)
    spec = AutocallableSpec(
        notional=1_000_000.0,
        autocall_barrier=1.00,
        coupon_barrier=0.75,
        put_barrier=0.65,
        coupon_rate=0.0225,
        obs_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    train_cfg = TrainConfig(n_paths_train=64, n_steps=4, epochs=3, hidden=4, cost_bps=5.0)
    result = train_autocall_hedge(inputs, spec, loss_type="variance", cfg=train_cfg)

    assert len(result.loss_history) == 3
    assert all(np.isfinite(loss_val) for loss_val in result.loss_history)

    comparison = evaluate_autocall_hedge(
        inputs,
        spec,
        result.net,
        n_paths_eval=64,
        cost_bps=5.0,
        seed_eval=999,
        greeks_n_paths=64,
        greeks_n_steps_per_period=2,
    )
    assert np.isfinite(comparison.learned.std)
    assert np.isfinite(comparison.baseline.std)
    assert comparison.n_eval_paths > 0


# --- Phase 5: down_and_in_put_payoff -------------------------------------------


def test_down_and_in_put_payoff_worthless_when_never_breached():
    paths = torch.tensor([[100.0, 105.0, 95.0, 90.0]], dtype=torch.float64)  # never <= 80
    payoff = down_and_in_put_payoff(paths, strike=100.0, barrier=80.0)
    assert payoff.tolist() == pytest.approx([0.0])


def test_down_and_in_put_payoff_activates_on_intraperiod_breach_not_just_terminal():
    """A path that dips below the barrier mid-path then recovers above the
    strike by expiry must still pay the (now zero) intrinsic put value only if
    ITM at expiry — but the knock-in must be recognized even though the FINAL
    level never breached the barrier itself."""
    paths = torch.tensor([[100.0, 70.0, 95.0]], dtype=torch.float64)  # dips to 70, recovers to 95
    payoff = down_and_in_put_payoff(paths, strike=100.0, barrier=80.0)
    assert payoff.tolist() == pytest.approx([5.0])  # knocked in, ITM put: 100 - 95


def test_down_and_in_put_payoff_matches_real_pricer_convention_zero_cost_sanity():
    """Deep and shallow ITM/OTM sanity: a path staying far above both strike
    and barrier is worthless; deeply breaching and finishing far ITM pays the
    full intrinsic value."""
    paths = torch.tensor([[100.0, 110.0, 120.0], [100.0, 60.0, 50.0]], dtype=torch.float64)
    payoff = down_and_in_put_payoff(paths, strike=100.0, barrier=80.0)
    assert payoff.tolist() == pytest.approx([0.0, 50.0])


# --- Phase 5: rollout (reuses rollout_hedged_pnl directly) --------------------


def test_barrier_rollout_never_hedging_equals_negative_payoff():
    paths = torch.tensor([[100.0, 70.0, 90.0], [100.0, 105.0, 110.0]], dtype=torch.float64)
    t_grid = np.array([0.0, 0.5, 1.0])
    hedged_pnl = rollout_hedged_pnl(
        paths,
        t_grid,
        _ZeroNet(),
        strike=100.0,
        T=1.0,
        cost_bps=0.0,
        payoff_fn=lambda p: down_and_in_put_payoff(p, strike=100.0, barrier=80.0),
    )
    expected = -down_and_in_put_payoff(paths, strike=100.0, barrier=80.0)
    assert hedged_pnl.tolist() == pytest.approx(expected.tolist())


# --- Phase 5: barrier_static_delta_hedge_pnl -----------------------------------


def test_barrier_static_hedge_zero_delta_equals_negative_payoff():
    paths = np.array([[100.0, 70.0, 90.0], [100.0, 105.0, 110.0]])
    baseline_pnl = barrier_static_delta_hedge_pnl(
        paths, strike=100.0, barrier=80.0, static_delta_shares=0.0, cost_bps=0.0
    )
    expected = -down_and_in_put_payoff(torch.from_numpy(paths), strike=100.0, barrier=80.0).numpy()
    assert baseline_pnl == pytest.approx(expected)


# --- Phase 5: end-to-end train + evaluate (fast smoke test) -------------------


def test_train_and_evaluate_barrier_hedge_runs_end_to_end():
    inputs = _flat_inputs(T=1.0)
    strike = SPOT
    barrier = SPOT * 0.8
    train_cfg = TrainConfig(n_paths_train=64, n_steps=8, epochs=3, hidden=4, cost_bps=5.0)
    result = train_barrier_hedge(inputs, strike, barrier, loss_type="variance", cfg=train_cfg)

    assert len(result.loss_history) == 3
    assert all(np.isfinite(loss_val) for loss_val in result.loss_history)

    comparison = evaluate_barrier_hedge(
        inputs,
        result.net,
        strike,
        barrier,
        n_paths_eval=64,
        n_steps=8,
        cost_bps=5.0,
        seed_eval=999,
        greeks_n_paths=64,
        greeks_n_steps=8,
    )
    assert np.isfinite(comparison.learned.std)
    assert np.isfinite(comparison.baseline.std)
    assert comparison.n_eval_paths > 0
