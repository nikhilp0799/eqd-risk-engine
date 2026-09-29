# Deep hedging: the ML problem, defined

How the deep-hedging module (`src/eqdrisk/ml/`) is framed as a machine-learning problem, and
how it is evaluated offline. Everything here reflects the code as built (`hedge_model.py`,
`losses.py`, `train.py`, `evaluate.py`, `run.py`).

---

## 1. What kind of ML problem is it

It is **not** regression, classification or recommendation. There are no labels: nothing tells
the model "the right hedge at this moment was 0.43".

**ML name:** a **sequential decision-making problem under uncertainty**, also called
**stochastic optimal control**. It sits in the **reinforcement-learning family**. More
precisely, it is **policy optimization with a differentiable simulator**: direct policy search
using pathwise gradients, trained by backpropagating through the whole sequence of decisions.
In the literature it is called **deep hedging** (Buehler, Gonon, Teichmann, Wood, 2019).

**Simple:** like learning to drive in a simulator. Nobody shows the "correct" steering angle;
the driver only finds out at the end of each run how badly it went, and adjusts until runs are
consistently smooth.

**Business:** a **hedging-optimization problem**. The desk sold a product and must trade the
underlying over its life to offset the risk. The question is how many shares to hold at each
point in time so the final result is as stable as possible and bad outcomes stay small, after
trading costs.

**Why it is not quite standard RL:** in standard RL the agent only sees a reward and must
estimate gradients by trial and error. Here the entire P&L calculation is ordinary arithmetic
inside PyTorch, so the exact gradient of the final outcome with respect to every network weight
is available. That makes training more stable and data-efficient. The agent's trades also do not
move the market (price paths are fixed inputs): a non-reactive environment.

---

## 2. The problem defined in ML terms

| ML concept | In this build |
|---|---|
| **Episode** | One simulated market path, from today to the product's maturity |
| **Time steps** | 32 rebalances (plain option), 64 (barrier option), one per quarterly observation date (autocallable note) |
| **State (features)** | 2 features: log-moneyness `log(S_t / K)` and normalized time to maturity `tau_t / T` |
| **Action** | Hedge ratio `delta_t`: shares held (for the note, a fraction of notional) |
| **Policy** | `delta_t = f_theta(state_t)`, one network shared across all time steps |
| **Model** | MLP 2 -> 64 -> 64 -> 1, float64. ReLU on the two hidden layers; the output layer is linear (no activation) so the hedge can be negative (short the underlying) and is not capped. Unbounded, so no hard limit on hedge size |
| **Environment** | Local-volatility Monte Carlo calibrated to that day's real market, `dS = (r - q) S dt + sigma_loc(S, t) S dW`, Sobol-sampled |
| **Outcome per episode** | Hedged P&L `X_theta` = trading gains - transaction costs (5bp per trade) - payoff owed at maturity |
| **Objective** | Minimize a risk measure `rho(X_theta)` over `theta` (three choices, below) |
| **Training data** | 8,000 simulated paths (seed 1), fixed, full-batch |
| **Optimizer** | Adam, learning rate 2e-3, 300 epochs |
| **Test data** | 8,000 different paths (seed 999), never seen in training |
| **Benchmark** | Black-Scholes delta hedge (plain option); a Monte Carlo delta set once at inception (note, barrier option) |

The hedged P&L:

```
X_theta = sum_t delta_t (S_{t+1} - S_t)  -  sum_t c * |delta_t - delta_{t-1}| * S_t  -  Payoff(S_path)
          [gains from hedging]              [trading costs]                            [what is owed]
```

**Three loss functions** (`losses.py`):

| Name on the site | ML loss | Meaning |
|---|---|---|
| Stability | `Var(X_theta)` | Make every scenario's outcome as similar as possible |
| Tail protection | `CVaR_95(-X_theta)`: mean of the worst 5% of losses, via `torch.topk` | Care only about the worst outcomes |
| Cost-aware | `Var(X_theta) + 0.1 * E[turnover]` | Stability, with heavy trading penalized |

In one sentence: **empirical risk minimization of a risk measure of terminal hedged P&L, over a
time-shared neural policy, trained on simulated paths from a calibrated market model.**

---

## 3. Offline evaluation metrics

All measured on the held-out 8,000 paths, for the learned policy and the benchmark on the same
paths:

| Metric | Definition | Better | Shown on the site as |
|---|---|---|---|
| **Std of hedged P&L** | Spread of outcomes across scenarios (hedging error) | Lower | P&L swings |
| **Std reduction vs benchmark** | `1 - std_learned / std_benchmark` | Higher | Reduction in P&L swings |
| **CVaR_95 (expected shortfall)** | Mean of the worst 5% of outcomes | Less negative | Worst-case loss |
| **CVaR improvement vs benchmark** | `(CVaR_learned - CVaR_benchmark) / abs(CVaR_benchmark)` | Higher | Improvement in worst-case loss |
| **Mean hedged P&L** | Average outcome | See note | Stored, not shown |
| **Training loss curve** | Loss per epoch (`loss_history`) | Converging | Diagnostic only |

Comparing learned vs benchmark on identical held-out paths is the offline equivalent of an A/B
test against the incumbent model.

**Why mean P&L is not a headline metric:** the P&L excludes the premium received for selling the
product, so the mean is roughly minus the product's price for every strategy and says little
about hedge quality. Spread and tail are what matter.

---

## 4. What the evaluation does not do yet

1. **No distribution shift.** Test paths are new but come from the same simulator as training.
   There is no test on real historical price paths or under a different market model. This is
   the largest gap.
2. **One seed, no confidence intervals.** Each result is a single training run, so small
   differences (for example 1.8% or 2.3%) cannot yet be separated from noise.
3. **Turnover is not reported at evaluation**, even though the Cost-aware objective optimizes it.
4. **Weak benchmark for the exotics.** It is set once and never rebalanced, which flatters the
   results on the note and the barrier option.
5. **Dying ReLU, measured.** Units that output 0 on every held-out state (2026-09-22 real data,
   variance loss), out of 64 per layer:

   | Model | Dead at init (L1 / L2) | Dead after training (L1 / L2) |
   |---|---|---|
   | Plain option (NVDA) | 14 / 15 | 15 / 41 |
   | Barrier option (SPX) | 12 / 16 | 13 / 24 |
   | Autocallable note (NVDA) | 12 / 17 | 12 / 35 |

   About 20-25% are dead before training (2 narrow-range inputs leave many randomly initialized
   units never active); training then kills 38-64% of the second layer.

   **Tested 2026-09-25: it does not cost hedge quality.** ReLU vs LeakyReLU(0.01) vs SiLU,
   everything else fixed (variance loss, same held-out paths, seed 999), 3 seeds each (seed
   changes both initial weights and training paths). Held-out hedged-P&L std, mean with
   [min, max] across seeds:

   | Model | Benchmark | ReLU | LeakyReLU | SiLU |
   |---|---|---|---|---|
   | Barrier option | 507.0 | 141.9 [139.3, 143.6] | 142.6 [140.0, 145.2] | 268.2 [266.0, 272.3] |
   | Autocallable note | 641.2K | 431.4K [429.7K, 433.8K] | 431.6K [431.0K, 432.1K] | 436.4K [435.5K, 437.1K] |
   | Plain option | 2.948 | 2.989 [2.957, 3.032] | 3.029 [3.002, 3.076] | 6.224 [6.216, 6.235] |

   ReLU and LeakyReLU are indistinguishable within seed noise, despite ReLU's dead units. SiLU
   is clearly worse under the ReLU-tuned learning rate (2e-3) and 300 epochs; it was not tuned
   separately, so this rules out SiLU at current settings only. Decision: keep ReLU.

   The same run gives the first seed-variance estimate (partly addressing gap 2): the barrier
   (about 72% std reduction) and autocallable (about 33%) results are stable across seeds. The
   plain-option Stability result is not a win: all 3 seeds are 0.3-2.8% worse than the
   benchmark (mean -1.4%); the single published run's +0.4% was a favourable draw.
6. **Under-trained, and a no-op objective (checks run 2026-09-28).** All three objectives, 3
   seeds each, same held-out paths. Std reduction vs benchmark, mean [min, max]:

   | Model | Stability | Tail protection | Cost-aware |
   |---|---|---|---|
   | Barrier option | +72.0% [71.7, 72.5] | +57.5% [54.9, 61.6] | +72.0% [71.7, 72.5] |
   | Autocallable note | +32.7% [32.3, 33.0] | +0.9% [-1.3, 2.3] | +32.7% [32.3, 33.0] |
   | Plain option | -1.4% [-2.8, -0.3] | -11.0% [-12.4, -9.3] | -1.4% [-2.8, -0.3] |

   - **The autocallable trade-off is real.** Tail protection improves CVaR by +2.3% on every
     seed [2.31, 2.32]; Stability worsens it on every seed [-3.7, -3.4].
   - **Cost-aware is identical to Stability** (same results and same held-out turnover, for
     example 11,674 vs 11,674 for the note). The penalty `0.1 * E[turnover]` is negligible
     against the variance term (note: variance about 1.8e11, penalty about 1e3), so it changes
     nothing. It needs a scale-aware weight (for example relative to the benchmark variance).
   - **No overfitting:** held-out objective within about 1-5% of the training objective.
   - **Not converged at 300 epochs:** plain option and barrier losses still fall 5-8% over the
     last 50 epochs. At 1,000 epochs (one seed) the plain option's held-out std drops from 2.98
     to 2.84, beating the benchmark (2.948) by 3.6%; the earlier "not a win" result was a
     training-budget artifact. SiLU at 1,000 epochs improves from 6.22 to 3.34, still behind and
     still not converged.
   - **Sanity check passed:** the learned plain-option hedge ratio tracks the Black-Scholes delta
     closely (mean absolute gap 0.01-0.03 shares between 85% and 115% of strike, at 90%, 50% and
     10% of life remaining), deviating only in the sparse far wings.
7. **Two features only.** No volatility, path history or current holding in the state. For the
   barrier option, "has the barrier been hit yet" is missing, and it matters.
