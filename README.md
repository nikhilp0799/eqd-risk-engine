# eqd-risk-engine

**An end-to-end equity derivatives risk analytics engine.**

Ingests raw option chains, calibrates arbitrage-free volatility surfaces, prices vanillas and structured products, computes VaR / Expected Shortfall two independent ways, runs conditional and reverse stress tests, and attributes daily P&L to Greeks with residual monitoring.

Built to mirror the daily workflow of an equity derivatives risk quant: *the number moved — why?*

---

## Current build status (updated 2026-09-02)

**This README doubles as the original build plan (kept intentionally — it explains *why* each
step matters and what "done" looks like), but a lot of it is no longer just a plan.** Steps 1–7,
8.1, 11.1, 11.2, 12, and 13 are built, tested, and verified against real live market data. Each
step section below is tagged with its actual status.

| Status | Steps |
|---|---|
| **Done** | 0–7 (data → curves → IV → calibration → pricing/Greeks → exotics → portfolio), 8.1 (risk-factor grid), 11.1 (historical replay stress), 11.2 (hypothetical stress grid), 12 (daily P&L explain), 13 (incident report) |
| **Partial** | 8 (8.2 PCA / 8.3 proxy modelling need real multi-day history), 11 (11.3 conditional stress / 11.4 reverse stress need the same) |
| **Not started** | 9 (VaR) and 10 (backtesting) — blocked on the same real-history dependency as above; 14 (dashboard), 15 (model doc), 16 (engineering polish) — no data dependency, just not built yet |

**Why some steps are deferred rather than skipped:** several of the acceptance criteria below (PCA
on real vol-surface changes, a 250–1000 day VaR window, a conditional-stress beta estimated from
history) need a real, *accumulating* history of daily calibrated vol surfaces — something a
free-tier, single-snapshot-per-day pipeline builds up one real trading day at a time, not
retroactively. That daily data pull is now fully automated (`scripts/daily_ingest.sh` on a
`launchd` schedule — see `docs/AUTOMATION.md`) rather than run by hand, so this history keeps
accumulating unattended.

**Real findings so far, not just passing tests** — the project's own standard is honest numbers,
not tuned-to-look-clean ones:
- Black-76 price matches QuantLib to 1e-8; several Greeks match to 1e-6.
- A genuine SPX calendar-arbitrage violation, found on real data, correctly triggered the SSVI
  fallback (Step 4).
- The local-vol Monte Carlo engine reprices calibrated vanillas within a fraction of a standard
  error; the Brownian-bridge barrier correction cut a >25-SE naive-discretisation bias down to
  ~0.1 SE at the same coarse step count (Step 6).
- A live historical-replay stress test against five real episodes (Volmageddon, the COVID crash,
  the 2022 rate shock, the 2024 yen carry unwind) using REAL pulled spot/VIX moves — the COVID
  scenario alone showed an 82%-of-book loss on the current 9-position portfolio (Step 11.1).
- Two real bugs were caught by testing a claim rather than assuming it: a wall-clock staleness bug
  that could zero out all quote coverage depending on time of day (Step 4), and a documented
  "exact" stress-shock math claim that turned out to be off by up to ~40% once actually tested
  (Step 11).
- The project's first real day-over-day P&L explain run found a $178k unexplained residual, 99.6%
  of it traced to one position (an NVDA autocallable) whose vega-only Greek set couldn't see its
  true vol sensitivity near the note's barriers — exactly the gap the README had predicted for
  this instrument type back in Step 6. Written up as a real incident report and partially fixed
  (Step 12/13); see `docs/incidents/2026-09-01_p008_autocallable_vega_residual.md` for the honest
  before/after (the fix helps but doesn't fully close the gap, and the report explains why).

---

## Table of contents

- [Why this project](#why-this-project)
- [Job-requirement coverage map](#job-requirement-coverage-map)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Step 0 — Environment setup](#step-0--environment-setup)
- [Step 1 — Data layer](#step-1--data-layer)
- [Step 2 — Curves, forwards, implied dividends](#step-2--curves-forwards-implied-dividends)
- [Step 3 — Implied vol extraction and quality filtering](#step-3--implied-vol-extraction-and-quality-filtering)
- [Step 4 — Volatility surface calibration (SVI / SSVI / SABR)](#step-4--volatility-surface-calibration-svi--ssvi--sabr)
- [Step 5 — Vanilla pricing and sensitivities](#step-5--vanilla-pricing-and-sensitivities)
- [Step 6 — Exotics: local vol, variance swaps, barriers, autocallables](#step-6--exotics-local-vol-variance-swaps-barriers-autocallables)
- [Step 7 — Portfolio definition](#step-7--portfolio-definition)
- [Step 8 — Risk factors, PCA, and proxy modelling](#step-8--risk-factors-pca-and-proxy-modelling)
- [Step 9 — VaR and Expected Shortfall](#step-9--var-and-expected-shortfall)
- [Step 10 — Backtesting the VaR model](#step-10--backtesting-the-var-model)
- [Step 11 — Stress testing and reverse stress testing](#step-11--stress-testing-and-reverse-stress-testing)
- [Step 12 — Daily P&L explain](#step-12--daily-pl-explain)
- [Step 13 — The incident report](#step-13--the-incident-report)
- [Step 14 — Dashboard](#step-14--dashboard)
- [Step 15 — Model documentation](#step-15--model-documentation)
- [Step 16 — Engineering polish](#step-16--engineering-polish)
- [Timelines](#timelines)
- [Acceptance criteria](#acceptance-criteria)
- [Interview preparation](#interview-preparation)
- [References](#references)

---

## Why this project

Most quant portfolio projects stop at "I implemented Black-Scholes and fitted a smile." That demonstrates coursework. A risk desk hires for something narrower and more specific:

1. Can you produce a **number that survives scrutiny** — with the data cleaning, the arbitrage constraints, and the validation evidence behind it?
2. When the number **moves unexpectedly**, can you decompose it and find the cause?
3. Can you **write it down** so a validator, an auditor, or a trader can follow the argument?

This project is organised around those three questions. The P&L explain module (Step 12) and the incident report (Step 13) are the parts that most distinguish it from a pricing exercise — they are the day job.

---

## Job-requirement coverage map

| JD requirement | Where it lives | Artifact that proves it |
|---|---|---|
| Volatility surface calibration | Step 4 | `calibration_report.html`, RMSE time series |
| Vanilla option pricing & risk analytics | Step 5 | `pricing/` module + unit tests vs. QuantLib |
| Value-at-Risk calculations | Steps 9–10 | VaR time series, backtest report |
| Scenario analysis & stress testing | Step 11 | Stress grid, conditional stress, reverse stress |
| Sensitivity & exposure analysis | Steps 5, 12 | Greek ladders, exposure dashboard |
| Proxy modelling | Step 8 | Proxy error quantification section |
| Time series construction | Steps 1–2 | Data pipeline + rejection reports |
| Volatility modelling | Steps 4, 8 | SVI/SSVI/SABR comparison, vol factor PCA |
| Risk factor analysis | Step 8 | PCA loadings, stability analysis |
| Investigating production issues | Step 13 | Incident report |
| Documentation for validation/governance | Step 15 | 12–15 page model doc |
| Structured products (variance swaps, barriers, autocallables) | Step 6 | Exotic pricers + convergence studies |
| Regulatory / capital frameworks | Steps 9–11 | FRTB-style ES, PLA test, Basel traffic light |
| Large datasets, market data, production systems | Steps 1, 16 | Parquet/DuckDB store, CLI, CI |
| Strong Python | Everywhere | Typed, tested, documented, `ruff`-clean |

---

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │           RAW DATA SOURCES           │
                     │  chains · underlyings · rates · divs │
                     └──────────────────┬───────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  1. INGEST         snapshot → validate → Parquet (partitioned) │
   │  2. CURVES         discount curve · implied forward · impl div │
   │  3. IV + FILTERS   invert prices → IV · quality gates → reasons│
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  4. CALIBRATION    SVI per slice → no-arb repair → SSVI fallbk │
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
        ┌───────────────────────────┴────────────────────────────┐
        ▼                                                        ▼
┌───────────────────────────┐                    ┌───────────────────────────┐
│ 5. VANILLA PRICING        │                    │ 6. EXOTIC PRICING         │
│    BS on forwards         │                    │    Dupire local vol → MC  │
│    Greeks (1st + 2nd ord) │                    │    var swap replication   │
│    sticky-strike/delta    │                    │    barrier · autocallable │
└─────────────┬─────────────┘                    └─────────────┬─────────────┘
              └──────────────────────┬───────────────────────-─┘
                                     ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  7. PORTFOLIO      positions · mark-to-model · exposure agg    │
   │  8. RISK FACTORS   PCA on total variance · single-name proxies │
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
   ┌──────────────┬─────────────────┴──────────────┬───────────────┐
   ▼              ▼                                ▼               ▼
┌────────┐  ┌──────────────┐              ┌──────────────┐  ┌────────────┐
│9. VaR  │  │10. BACKTEST  │              │11. STRESS    │  │12. P&L     │
│  ES    │  │  Kupiec      │              │  historical  │  │  EXPLAIN   │
│full rvl│  │  Christoff.  │              │  conditional │  │  residual  │
│vs Taylr│  │  PLA / TL    │              │  reverse     │  │  monitoring│
└────────┘  └──────────────┘              └──────────────┘  └────────────┘
                                    │
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  DELIVERABLES   dashboard · model doc · backtest report · IR   │
   └────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
eqd-risk-engine/
├── README.md
├── pyproject.toml
├── Makefile
├── .github/workflows/ci.yml
├── configs/
│   ├── base.yaml                  # paths, universe, run params
│   ├── portfolio.yaml             # positions
│   ├── calibration.yaml           # SVI bounds, weights, tolerances
│   ├── var.yaml                   # horizons, confidence, window
│   └── stress.yaml                # scenario definitions
├── data/
│   ├── raw/                       # immutable snapshots (never edited)
│   ├── curated/                   # partitioned parquet
│   └── reference/                 # holidays, corp actions, index members
├── src/eqdrisk/
│   ├── io/
│   │   ├── sources.py             # provider adapters
│   │   ├── snapshot.py            # daily job
│   │   └── store.py               # parquet/duckdb read-write
│   ├── marketdata/
│   │   ├── calendar.py            # trading days, year fractions
│   │   ├── curve.py               # discount curve bootstrap
│   │   ├── forward.py             # implied forward + dividend
│   │   └── quality.py             # filters + reason codes
│   ├── vol/
│   │   ├── implied.py             # IV inversion (Jäckel-style)
│   │   ├── svi.py                 # raw SVI, Durrleman, calendar checks
│   │   ├── ssvi.py                # arbitrage-free global fallback
│   │   ├── sabr.py                # Hagan + short-end comparison
│   │   ├── surface.py             # Surface object, interpolation
│   │   └── localvol.py            # Dupire from parameterised surface
│   ├── pricing/
│   │   ├── blackscholes.py        # price + analytic Greeks
│   │   ├── montecarlo.py          # LV paths, Brownian bridge
│   │   ├── varswap.py             # Carr-Madan replication
│   │   ├── barrier.py
│   │   └── autocall.py
│   ├── portfolio/
│   │   ├── positions.py
│   │   └── valuation.py           # full reval, aggregation
│   ├── risk/
│   │   ├── factors.py             # PCA, factor returns
│   │   ├── proxy.py               # single-name proxy model
│   │   ├── var.py                 # HS, full reval + Taylor
│   │   ├── es.py                  # FRTB-style ES, liquidity horizons
│   │   ├── backtest.py            # Kupiec, Christoffersen, TL, PLA
│   │   ├── stress.py              # historical, hypothetical, conditional
│   │   └── reverse.py             # min-Mahalanobis loss shock
│   ├── attribution/
│   │   └── explain.py             # Greek P&L decomposition
│   ├── reporting/
│   │   ├── dashboard.py           # streamlit app
│   │   └── templates/
│   └── cli.py                     # typer entrypoint
├── tests/
│   ├── unit/
│   ├── property/                  # hypothesis-based invariants
│   └── regression/                # golden-file outputs
├── notebooks/                     # exploration only, never the deliverable
└── docs/
    ├── model_documentation.md
    ├── var_backtest_report.md
    └── incident_2024-03-12.md
```

**Rule:** notebooks are scratch. Everything reproducible lives in `src/` and is invoked from the CLI. A reviewer should be able to run one command and regenerate every number in your documentation.

---

## Step 0 — Environment setup

**Status: done.** Full skeleton, CI, pre-commit, pydantic config schema, CLI scaffolding — all green from the start.

### 0.1 Toolchain

```bash
# Python 3.11+ recommended
curl -LsSf https://astral.sh/uv/install.sh | sh   # or use conda/poetry
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

`pyproject.toml` core dependencies:

```toml
[project]
name = "eqdrisk"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26", "scipy>=1.11", "pandas>=2.1",
    "pyarrow>=14", "duckdb>=0.9",
    "numba>=0.59",              # MC inner loops
    "pydantic>=2.5",            # config + schema validation
    "typer>=0.9", "rich>=13",   # CLI
    "pyyaml", "matplotlib", "plotly", "streamlit",
    "yfinance", "pandas-datareader", "fredapi",
    "statsmodels",              # regressions, diagnostics
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "hypothesis", "ruff", "mypy", "QuantLib"]
```

`QuantLib` is a **dev** dependency deliberately: you use it to validate your own pricers in tests, never in the production path. Saying "I benchmarked against QuantLib and matched to 1e-8" in your model doc is worth far more than using it as your engine.

### 0.2 Repository hygiene from day one

```bash
git init
pre-commit install     # ruff format, ruff check, mypy on src/
make test              # must pass before every commit
```

Set up CI on the first day, not the last. A green badge on a quant repo signals engineering discipline that most candidates lack.

### 0.3 Config-driven design

Every run parameter goes in YAML, validated by a pydantic model. No magic numbers in code.

```yaml
# configs/base.yaml
run_date: "2026-08-11"
universe:
  index: ["SPX"]
  single_names: ["AAPL", "NVDA", "JPM", "XLE"]
paths:
  raw: "data/raw"
  curated: "data/curated"
calendar: "NYSE"
daycount: "ACT/365F"
```

**Acceptance for Step 0:** `make test` passes on an empty test suite, CI runs, `eqdrisk --help` prints.

---

## Step 1 — Data layer

**Status: done, verified live.** Real ingestion for SPX/AAPL/NVDA/JPM/XLE, schema-enforced, reason-coded row rejection, QC reports with prior-day diffs.

### 1.1 Sources

| Data | Source | Cost | Notes |
|---|---|---|---|
| Option chains (history) | HistoricalData.net / OptionsDX archives | $0–$99 | **Buy 2017→present.** Covers Feb-2018, Mar-2020, Aug-2024 |
| Option chains (live snap) | `yfinance`, ThetaData free tier | $0 | Current chain only — cannot backfill |
| Underlying OHLC | `yfinance`, Stooq | $0 | Need ≥ 3y for VaR window |
| Rates | FRED: `SOFR`, `DGS1MO`…`DGS10` | $0 | via `fredapi` |
| Dividends | `yfinance` + implied from parity | $0 | Use *both*, compare |
| Vol indices | CBOE / FRED: `VIXCLS`, `VVIX`, `SKEW` | $0 | Independent calibration benchmark |
| Corp actions | `yfinance` splits | $0 | Prevents broken strike series |

> **The single most important decision in this project:** buy a historical chain archive. Without it, your VaR backtest has too few observations to be statistically meaningful, and you cannot run historical stress replays. ~$99 converts this from a toy into a credible piece of work.

### 1.2 Storage schema

Partitioned Parquet, queried through DuckDB:

```
data/curated/
├── chains/asof_date=2026-08-11/underlying=SPX/part-0.parquet
├── underlyings/year=2026/part-0.parquet
├── curves/asof_date=2026-08-11/part-0.parquet
└── surfaces/asof_date=2026-08-11/underlying=SPX/part-0.parquet
```

Chain schema (enforce with pydantic/pyarrow, fail loudly on violation):

| column | type | note |
|---|---|---|
| `asof_date` | date32 | snapshot date |
| `asof_ts` | timestamp[ns, ET] | **exact** snap time |
| `underlying` | string | |
| `expiry` | date32 | |
| `strike` | float64 | |
| `cp` | string | `C` / `P` |
| `bid`, `ask` | float64 | |
| `bid_size`, `ask_size` | int64 | for quote weighting |
| `volume`, `open_interest` | int64 | |
| `underlying_px` | float64 | snapped at `asof_ts` |
| `source` | string | provenance |

### 1.3 The ingest job

```python
# src/eqdrisk/io/snapshot.py
def run_snapshot(cfg: BaseConfig, asof: date) -> SnapshotResult:
    """Pull → validate schema → write raw → write curated → emit QC report."""
```

Behaviours that matter:

- **Raw is immutable.** Write the untouched vendor payload to `data/raw/` first. Every downstream transform is reproducible from raw. If you later find a filter bug you can replay history.
- **Idempotent.** Re-running for the same date overwrites cleanly, never appends duplicates.
- **Emits a QC report** every run: row counts, null rates, per-reason rejection counts, and a diff against the prior day.

### 1.4 Timestamp alignment — the trap

SPX options settle 4:15pm ET; the index prints 4:00pm; Treasury yields post ~4:30pm. If you naively join on `date`, you build a forward error into every surface, and it shows up later as unexplained P&L residual.

Define one canonical snap time in config (`16:00:00 ET`), pull every input as close to it as possible, and **record the actual timestamp of each input**. Your QC report should flag any input more than N minutes from the canonical snap.

**Acceptance for Step 1:**
- One command ingests a date end-to-end and writes curated Parquet.
- Schema violations raise, not warn.
- QC report renders with rejection counts by reason.
- A DuckDB query over ≥ 200 dates returns in < 2 seconds.

---

## Step 2 — Curves, forwards, implied dividends

**Status: done, verified live** (refined further by Step 3's root-cause fix below). Curve bootstrap + put-call-parity forward regression + implied dividend, live on all 5 underlyings.

### 2.1 Discount curve

Bootstrap from SOFR + Treasury CMT. Interpolate **linearly in log discount factor** (equivalently, piecewise-constant forward rates) — never linearly in zero rate, which produces jagged forwards.

$$ P(0,T) = \exp\left(-\int_0^T f(u)\,du\right), \qquad \log P \text{ linear in } T $$

### 2.2 Implied forward from put-call parity

This is the piece that separates a real implementation from a textbook one. For European options (SPX is European — use it), parity gives:

$$ C(K) - P(K) = P(0,T)\,\bigl(F - K\bigr) $$

So for a single expiry, regress $C(K) - P(K)$ against $K$ across your liquid strikes:

$$ C(K)-P(K) = \underbrace{P(0,T)F}_{\alpha} - \underbrace{P(0,T)}_{\beta}\,K $$

- Slope $\beta$ recovers the discount factor → **cross-check against your bootstrapped curve.**
- $F = \alpha / \beta$ gives the implied forward.
- Weight the regression by quote quality (tight spreads, near-ATM strikes get more weight).
- Report $R^2$; a low $R^2$ means bad quotes or a mis-snapped underlying — that's a QC signal.

### 2.3 Implied dividend

$$ q_{\text{impl}} = -\frac{1}{T}\log\!\left(\frac{F \cdot P(0,T)}{S_0}\right) $$

Build the implied dividend term structure daily. Compare against announced dividends from `yfinance`. Two outputs worth putting in the doc:

1. A chart of implied vs. announced dividend yield through time.
2. The single-name cases where they diverge (borrow cost, hard-to-borrow names) — and an explanation.

**Acceptance for Step 2:**
- Parity-implied discount factor within 5bp of bootstrapped curve for liquid expiries.
- Forward term structure is smooth and monotone in the absence of large discrete dividends.
- Regression $R^2 > 0.999$ for SPX liquid expiries; flag anything below.

---

## Step 3 — Implied vol extraction and quality filtering

**Status: done, verified live.** The quote-quality filter chain (ZERO_BID/CROSSED/STALE/LOW_OI/WIDE_SPREAD/...) fixed Step 2's noise at the root; Black-76 IV inversion produces a genuinely correct-looking SPX skew on real data.

### 3.1 Inversion

Work in **undiscounted forward (Black-76) space** throughout. Define normalised moneyness:

$$ k = \log(K/F), \qquad w = \sigma_{\text{impl}}^2 T \quad \text{(total implied variance)} $$

Invert with a robust root-finder. Brent on $[10^{-4}, 5.0]$ works; a rational-approximation initial guess (Jäckel's "Let's Be Rational" approach) converges in 2 iterations and is worth implementing for speed if you're inverting millions of quotes.

Always invert **OTM** options — calls for $k>0$, puts for $k<0$ — since ITM options are dominated by intrinsic value and the vega is too small for stable inversion.

### 3.2 The filter chain (with reason codes)

Do **not** silently drop rows. Every rejection gets a code and a count:

| Code | Rule | Rationale |
|---|---|---|
| `ZERO_BID` | `bid <= 0` | No two-sided market |
| `CROSSED` | `bid > ask` | Bad data |
| `WIDE_SPREAD` | `(ask-bid)/mid > τ(moneyness)` | Unreliable mid |
| `STALE` | quote ts older than N min | Not a current market |
| `NO_ARB_INTRINSIC` | price below intrinsic bound | Violates static arb |
| `ITM_SIDE` | ITM option for that $k$ | Use OTM only |
| `EXTREME_K` | \|k\| > 4σ√T | Extrapolation region |
| `LOW_OI` | OI below threshold | Illiquid |
| `IV_SOLVE_FAIL` | root-finder no convergence | Numerically unstable |
| `THIN_SLICE` | < 8 surviving quotes in expiry | Cannot calibrate |

Make the spread threshold $\tau$ a **function of moneyness** — a 20% relative spread is normal for a 10-delta wing and pathological ATM. A flat threshold either kills your wings or admits garbage.

### 3.3 Weighting

Downstream calibration weights each quote by:

$$ w_i \propto \frac{\text{vega}_i}{\text{spread}_i} $$

Vega-weighting means you fit in price space where price is sensitive to vol; spread-weighting means you trust tight markets more. This is what makes the wings behave.

**Acceptance for Step 3:**
- Rejection report renders per date, per underlying, per reason.
- Surviving quote count per expiry plotted through time — sudden drops indicate data issues.
- Round-trip test: price → IV → price recovers within 1e-10.

---

## Step 4 — Volatility surface calibration (SVI / SSVI / SABR)

**Status: done, verified live.** SVI + Durrleman butterfly repair + SSVI calendar-arbitrage fallback + SABR comparison, all exercised on real data — including a genuine SPX calendar-arbitrage violation correctly triggering the SSVI fallback.

### 4.1 Raw SVI

Per expiry slice, fit total implied variance:

$$ w(k) = a + b\left\{\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right\} $$

Parameter meaning — know this cold for interview:

| param | controls | constraint |
|---|---|---|
| $a$ | overall variance level | $a + b\sigma\sqrt{1-\rho^2} \ge 0$ |
| $b$ | wing slopes (angle between asymptotes) | $b \ge 0$ |
| $\rho$ | skew / rotation | $\|\rho\| < 1$ |
| $m$ | horizontal shift of the smile minimum | free |
| $\sigma$ | ATM curvature (smoothness of the kink) | $\sigma > 0$ |

**Objective:**

$$ \min_{a,b,\rho,m,\sigma} \sum_i w_i \left(w_{\text{model}}(k_i) - w_{\text{mkt}}(k_i)\right)^2 $$

**Calibration strategy that actually converges:**
1. Reduce dimensionality — for fixed $(m,\sigma)$ the problem is a *quasi-explicit* linear least squares in $(a, b\rho, b)$ (Zeliade's method). Solve the inner problem exactly, optimise only over $(m,\sigma)$ with Nelder-Mead.
2. Warm-start from the previous day's parameters. Day-over-day parameter jumps are then themselves a QC signal.
3. Multi-start on the first day of a series.

### 4.2 No-arbitrage constraints — the part that matters

**Butterfly (static, within slice).** Durrleman's condition: define

$$ g(k) = \left(1 - \frac{k w'(k)}{2w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} $$

Require $g(k) \ge 0$ for all $k$ on your grid. Equivalent statement: the risk-neutral density $p(k) \ge 0$ everywhere.

**Calendar spread (across slices).** Total implied variance must be non-decreasing in maturity at fixed log-moneyness:

$$ w(k, T_1) \le w(k, T_2) \quad \forall\, T_1 < T_2 $$

**Implementation:** after each slice fit, evaluate both conditions on a dense grid. Log any violation with location and magnitude. Then repair:

- Butterfly violation → re-fit with the constraint imposed via penalty or SLSQP.
- Calendar violation → fall back to **SSVI**, which is arbitrage-free by construction:

$$ w(k,\theta_t) = \frac{\theta_t}{2}\left\{1 + \rho\varphi(\theta_t)k + \sqrt{\bigl(\varphi(\theta_t)k+\rho\bigr)^2 + 1 - \rho^2}\right\} $$

with $\theta_t$ the ATM total variance term structure and $\varphi$ a power-law or Heston-like function. Gatheral–Jacquier give the exact conditions on $\varphi$ for absence of both butterfly and calendar arbitrage.

### 4.3 SABR comparison

Fit Hagan SABR (with the arbitrage-free PDE variant, or Obłój's corrected formula) to the short end. Then write up **where each parameterisation breaks**:

- SABR: excellent for a single slice with few strikes, no natural term-structure coupling, the classic Hagan expansion admits negative densities for low strikes / long maturities.
- SVI: excellent per-slice fit and asymptotically consistent with Lee's moment formula, but per-slice fitting can violate calendar arbitrage.
- SSVI: arbitrage-free globally, fewer degrees of freedom, worse pointwise fit.

That comparison paragraph is a strong interview answer.

### 4.4 Calibration diagnostics (persist these daily)

- RMSE **in vol points**, overall and by moneyness bucket (wings vs. ATM).
- Max absolute error and its location.
- Parameter time series with day-over-day change — jumps flag data problems.
- Arbitrage violation counts and magnitudes.
- **Independent benchmark:** your SVI-implied 30-day ATM variance vs. published VIX. Should track within ~0.5 vol points; a persistent gap means a bug in your forward or your interpolation.

**Acceptance for Step 4:**
- 0 butterfly violations, 0 calendar violations on final surfaces.
- SPX RMSE < 0.3 vol points ATM, < 1.0 in wings.
- 30d ATM vol vs. VIX correlation > 0.98.
- Calibration of a full SPX surface completes in < 5 seconds.

---

## Step 5 — Vanilla pricing and sensitivities

**Status: done, verified live.** Full Greek set (incl. vanna/volga) validated against QuantLib (price to 1e-8) and central finite differences; sticky-strike/delta/local-vol conventions computed live (the ≥60-day historical comparison chart is deferred pending accumulated history — see the status table above).

### 5.1 Pricing

Black-76 on forwards:

$$ C = P(0,T)\left[F\,N(d_1) - K\,N(d_2)\right], \quad d_{1,2} = \frac{\log(F/K) \pm \tfrac{1}{2}\sigma^2T}{\sigma\sqrt{T}} $$

Implement analytically, and **test against QuantLib to 1e-8** in a unit test.

### 5.2 Greeks

Compute analytically where possible, and validate each against a central finite difference:

| Greek | Definition | Why risk cares |
|---|---|---|
| Delta | $\partial V/\partial S$ | Directional exposure, hedge ratio |
| Gamma | $\partial^2 V/\partial S^2$ | Hedge slippage, convexity P&L |
| Vega | $\partial V/\partial\sigma$ | Vol exposure |
| Theta | $\partial V/\partial t$ | Carry |
| Rho / dividend rho | $\partial V/\partial r$, $\partial V/\partial q$ | Curve and div risk |
| **Vanna** | $\partial^2 V/\partial S\partial\sigma$ | Skew P&L — critical for a book with skew |
| **Volga** | $\partial^2 V/\partial\sigma^2$ | Vol-of-vol P&L, wing risk |

Vanna and volga are non-negotiable here: without them your P&L explain residual will be large and you will have nothing to say about why.

### 5.3 The sticky-strike vs. sticky-delta study

**This is the highest-value differentiator in the whole project.** The model delta is not the hedge delta, because implied vol moves with spot.

$$ \Delta_{\text{total}} = \frac{\partial V}{\partial S} + \frac{\partial V}{\partial \sigma}\cdot\frac{\partial \sigma}{\partial S} $$

Three conventions:

| Convention | Assumption | $\partial\sigma/\partial S$ |
|---|---|---|
| **Sticky strike** | vol at fixed $K$ unchanged | 0 |
| **Sticky delta / moneyness** | vol at fixed $K/F$ unchanged | $-\dfrac{\partial\sigma}{\partial k}\cdot\dfrac{1}{S}$ |
| **Sticky local vol** | local vol surface unchanged | $\approx 2\times$ sticky-delta backbone |

Compute all three for the book daily. Then compute what each would have predicted vs. realised P&L over a quarter, and show which regime the market was actually in. That is a genuinely desk-relevant result and it will absolutely generate a follow-up question in interview.

**Acceptance for Step 5:**
- All Greeks match central differences to < 1e-5 relative.
- Price matches QuantLib to < 1e-8.
- Delta comparison chart across three conventions over ≥ 60 days.

---

## Step 6 — Exotics: local vol, variance swaps, barriers, autocallables

**Status: done, verified live — all five sub-parts (6.1–6.5).** Dupire local vol + numba/Sobol/Brownian-bridge Monte Carlo, variance swap replication, barrier closed-form + bridge-corrected MC, and a CRN-Greeks autocallable — each validated against an independent check (QuantLib/VIX/parity identity/synthetic scale-invariance), not just internally consistent.

### 6.1 Dupire local volatility

Strip local vol from the *parameterised* surface (not from raw quotes — differentiating noisy data twice is hopeless). In total-variance / log-moneyness form:

$$ \sigma_{\text{loc}}^2(k,T) = \frac{\partial_T w}{1 - \frac{k}{w}\partial_k w + \frac{1}{4}\left(-\frac14 - \frac1w + \frac{k^2}{w^2}\right)(\partial_k w)^2 + \frac12 \partial^2_{kk} w} $$

The denominator is exactly Durrleman's $g(k)$ — **which is why Step 4's arbitrage enforcement is a prerequisite, not a nicety.** If the surface admits butterfly arbitrage, local variance goes negative and your MC blows up. Say this out loud in interview; it demonstrates you understand why the constraints exist.

Since SVI gives $w$, $\partial_k w$, $\partial^2_{kk}w$ analytically, only $\partial_T w$ needs numerical treatment (finite difference across calibrated slices with monotone interpolation in $T$).

### 6.2 Monte Carlo engine

- Log-spot Euler scheme with the local vol surface, `numba`-jitted.
- **Variance reduction:** antithetic variates + Sobol quasi-random sequences with Brownian bridge construction.
- **Validation:** MC-price a vanilla and confirm it reproduces the calibrated market price to within MC error. If your LV MC doesn't reprice vanillas, your Dupire stripping is wrong. This is the single most important test in the module.
- Report standard errors and a convergence study (price vs. path count, log-log slope ≈ −1/2).

### 6.3 Variance swap

Fair strike by Carr–Madan static replication:

$$ K_{\text{var}}^2 = \frac{2}{T}\left[\int_0^{F} \frac{P(K)}{K^2}dK + \int_{F}^{\infty}\frac{C(K)}{K^2}dK\right] \cdot \frac{1}{P(0,T)} $$

Practical points to write up:
- Discretisation: your strike grid is finite. Quantify the **replication error from truncating the wings** — this is a real risk-management issue, not an academic one.
- Compare your computed fair strike against VIX²-style calculation for the 30-day point.
- Explain the jump/gap risk that makes the replication imperfect in practice.

### 6.4 Barrier option

Down-and-in put. Two engines:
- Closed-form (Merton/Reiner-Rubinstein) under constant vol — used only as a benchmark.
- LV Monte Carlo with **Brownian-bridge barrier correction** to remove discretisation bias.

Show the bias: naive discrete monitoring systematically underprices knock-ins. Plot price vs. time-step count with and without the bridge correction. That plot is a strong artifact.

### 6.5 Autocallable

Structure: quarterly observation dates; if $S_{T_i} \ge$ autocall barrier, redeem early with coupon; memory coupon if $S_{T_i} \ge$ coupon barrier; at maturity, if $S_T <$ put barrier, investor is short a down-and-in put.

Risk characteristics to document (this is why the JD names them):
- **Delta flips sign** near the autocall barrier as maturity approaches.
- **Gamma is discontinuous** at observation dates.
- Short vega near barrier, and large **vanna/volga** — genuinely skew-sensitive.
- Path dependence means Greeks must come from MC with common random numbers (fix the seed across bumps, or your bump-and-reval Greeks are pure noise).

**Acceptance for Step 6:**
- LV MC reprices calibrated vanillas within 2 standard errors across the strike grid.
- Local variance strictly positive across the entire grid.
- Barrier bridge correction demonstrably removes discretisation bias.
- Autocall Greeks stable under CRN, with documented standard errors.

---

## Step 7 — Portfolio definition

**Status: done, verified live.** The README's own 9-position example portfolio marks end-to-end with zero skips on real market data, aggregated by underlying/expiry-bucket/moneyness-bucket.

Keep it small enough to reason about, diverse enough to hit every risk type.

```yaml
# configs/portfolio.yaml
positions:
  - {id: P001, type: vanilla, underlying: SPX, cp: P, strike: 5200, expiry: 2026-12-18, qty: -150}
  - {id: P002, type: vanilla, underlying: SPX, cp: C, strike: 6000, expiry: 2026-12-18, qty: 100}
  - {id: P003, type: vanilla, underlying: SPX, cp: P, strike: 4800, expiry: 2027-06-17, qty: 200}
  - {id: P004, type: vanilla, underlying: AAPL, cp: C, strike: 240,  expiry: 2026-10-16, qty: -500}
  - {id: P005, type: vanilla, underlying: NVDA, cp: P, strike: 150,  expiry: 2026-09-18, qty: 300}
  - {id: P006, type: varswap, underlying: SPX, expiry: 2026-12-18, vega_notional: 250000}
  - {id: P007, type: barrier, underlying: SPX, sub: down_and_in_put,
     strike: 5500, barrier: 4500, expiry: 2027-06-17, qty: -50}
  - {id: P008, type: autocall, underlying: NVDA, notional: 5000000,
     autocall_barrier: 1.00, coupon_barrier: 0.75, put_barrier: 0.65,
     coupon: 0.0225, obs: quarterly, expiry: 2028-08-15}
  - {id: P009, type: equity, underlying: SPX, qty: -80}     # delta hedge
```

Design intent — be able to explain each:
- A **risk reversal** in SPX (P001/P002) creates skew exposure.
- **Long downside** in a second expiry (P003) creates calendar/term-structure exposure.
- **Single-name positions** (P004/P005) force the proxy modelling problem.
- **Variance swap** (P006) gives pure convexity/vol exposure and tests replication.
- **Short down-and-in put** (P007) creates barrier gap risk.
- **Autocallable** (P008) creates path dependence, discontinuous gamma, and the classic short-vol/short-skew dealer position.
- **Delta hedge** (P009) means residual risk is second-order — which makes the P&L explain interesting rather than trivially delta-dominated.

**Acceptance for Step 7:** portfolio marks to model daily; aggregate Greeks reported by underlying, by expiry bucket, and by moneyness bucket.

---

## Step 8 — Risk factors, PCA, and proxy modelling

**Status: partially built.** 8.1 (the fixed risk-factor grid) is built and verified live. 8.2 (PCA) and 8.3 (proxy modelling) are deferred — both need a real, accumulating time series of daily Δw(k,T), which a single day's snapshot can't provide; see the status table above.

### 8.1 Building the risk-factor grid

Fixed grid in $(k, T)$ — e.g. $k \in \{-0.30, -0.20, -0.10, 0, 0.10, 0.20\}$, $T \in \{1\text{m}, 3\text{m}, 6\text{m}, 1\text{y}, 2\text{y}\}$. Every day, evaluate the calibrated surface on this grid. **The grid is fixed; the surface moves.** This is how production risk systems define vol risk factors, and it's why calibration must be stable.

Risk factor returns: daily changes in total implied variance $\Delta w(k,T)$ (or log-changes in implied vol — document your choice and why).

### 8.2 PCA

Run PCA on the covariance matrix of $\Delta w$. Expect:

| PC | Typical variance explained | Interpretation |
|---|---|---|
| PC1 | 75–90% | **Level** — parallel vol shift |
| PC2 | 5–15% | **Skew** — steepening/flattening |
| PC3 | 2–6% | **Term structure** — front vs. back |
| PC4+ | residual | curvature, noise |

Deliverables:
- Scree plot and cumulative variance explained.
- Loading heatmaps over the $(k,T)$ grid.
- **Loading stability:** re-estimate on rolling windows and plot how the loadings drift. Factor instability is a real model risk and flagging it shows maturity.

### 8.3 Proxy modelling — the JD-specific piece

Single-name surfaces are thin: few strikes, few expiries, wide spreads, sometimes no quotes at all for a given day/tenor. Production systems **proxy** these to index factors.

Model:

$$ \Delta w^{(n)}_{j} = \alpha_j + \beta_j \,\Delta w^{(\text{SPX})}_{j} + \gamma_j\, \Delta \text{(idiosyncratic factor)} + \varepsilon_j $$

for each grid node $j$, where the idiosyncratic factor might be the name's own ATM vol or a sector proxy.

What to report — the error analysis is the whole point:
- $R^2$ per node: high near ATM, degrading in the wings and long tenors.
- **Proxy error distribution**: the residual $\varepsilon$ is the model risk you're taking by proxying.
- **VaR impact**: compute portfolio VaR using true single-name surfaces vs. proxied surfaces. The difference *is* the cost of the proxy, expressed in the unit risk managers care about. Almost no candidate does this.
- Idiosyncratic event failure: proxies break on earnings. Show it — plot proxy residuals around an earnings date and note that event-driven vol requires separate treatment.

**Acceptance for Step 8:**
- PC1–PC3 explain > 90% of surface variance.
- Loadings are economically interpretable and plotted.
- Proxy $R^2$ reported per node with a documented degradation pattern.
- Proxy VaR error quantified in dollars.

---

## Step 9 — VaR and Expected Shortfall

**Status: not started.** Needs a 250–1000 day historical window the project's now-automated daily pull is building toward — see the status table above.

### 9.1 Historical simulation

Window: 500 business days (regulatory convention), also run 250 and 1000 for sensitivity.

For each historical day $s$ in the window:
1. Extract the factor moves: $\Delta S/S$, $\Delta w(k,T)$ for every grid node, $\Delta r$, $\Delta q$.
2. Apply them to **today's** market state → shocked market.
3. Revalue the portfolio.
4. Record hypothetical P&L $L_s$.

$$ \text{VaR}_\alpha = -\text{Quantile}_{1-\alpha}\{L_s\}, \qquad \text{ES}_\alpha = -\mathbb{E}\left[L_s \mid L_s \le -\text{VaR}_\alpha\right] $$

Report 1-day 99%, 10-day 99% (both scaled-by-$\sqrt{10}$ and overlapping-10-day, and **discuss why they differ** — square-root scaling assumes i.i.d. returns, which vol data emphatically violates).

### 9.2 The two-method comparison — the core intellectual content

Run VaR **two ways** on the same scenarios:

**(a) Full revaluation.** Reprice every instrument under every scenario. Accurate, slow.

**(b) Sensitivity-based (Taylor).**

$$ \Delta V \approx \Delta\cdot\delta S + \tfrac12\Gamma\,(\delta S)^2 + \mathcal{V}\cdot\delta\sigma + \text{Vanna}\cdot\delta S\,\delta\sigma + \tfrac12\text{Volga}\,(\delta\sigma)^2 + \Theta\,\delta t $$

Then analyse the gap:
- Plot full-reval VaR vs. Taylor VaR through time.
- Show where Taylor breaks down: large moves, barrier proximity, autocall observation dates, short-dated high-gamma positions.
- Quantify the runtime difference (this is why banks use sensitivity-based VaR in production and full-reval for validation).

**This comparison is the daily argument on a real risk desk.** Being able to say "our sensitivity VaR understated by 8% on high-gamma days, here's the chart" is exactly the conversation the role involves.

### 9.3 FRTB-flavoured Expected Shortfall

- ES at **97.5%** (the FRTB standard) rather than 99% VaR.
- **Liquidity horizons**: 10 days for equity index vol, 20 days for single-name vol, 60 days for structured/exotic exposures. Compute the horizon-scaled ES:

$$ \text{ES} = \sqrt{\left(\text{ES}^{(1)}\right)^2 + \sum_{j\ge2}\left(\text{ES}^{(j)}\sqrt{\tfrac{LH_j - LH_{j-1}}{10}}\right)^2} $$

- Decompose ES contribution by risk factor group (spot, vol level, skew, term).

**Acceptance for Step 9:**
- VaR time series over ≥ 250 days from both methods.
- Full-reval vs. Taylor gap quantified and explained.
- ES decomposition by factor group.
- Full-reval VaR for the whole book runs in < 60 seconds.

---

## Step 10 — Backtesting the VaR model

**Status: not started.** Depends entirely on Step 9's output.

A VaR number without a backtest is an opinion. This section is what makes it a model.

### 10.1 Kupiec unconditional coverage (POF test)

$H_0$: exception rate $= 1-\alpha$. With $x$ exceptions in $n$ days:

$$ LR_{uc} = -2\log\left[\frac{(1-p)^{n-x}p^x}{(1-\hat p)^{n-x}\hat p^x}\right] \sim \chi^2_1, \quad \hat p = x/n $$

### 10.2 Christoffersen independence test

Exceptions must not cluster. Build the transition matrix of exception/no-exception and test:

$$ LR_{ind} \sim \chi^2_1, \qquad LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2 $$

Clustered exceptions mean the model doesn't react to changing volatility — a real, common failure of equally-weighted historical simulation.

### 10.3 Basel traffic light

Over 250 days at 99%: 0–4 exceptions **green**, 5–9 **yellow** (capital multiplier increases), 10+ **red**. Report yours and interpret it honestly. If you're in yellow, explain *why* — that's more impressive than a suspiciously clean green.

### 10.4 P&L attribution test (FRTB PLA)

Compare **risk-theoretical P&L** (from your risk model's factors) against **hypothetical P&L** (full reval, no new trades):

- **Spearman rank correlation** — must exceed 0.80 for green.
- **Kolmogorov–Smirnov statistic** on the two distributions — must be below 0.09 for green.

This test directly links Step 9's two-method comparison to Step 12's P&L explain, and it is a genuinely current regulatory topic that will land well in interview.

### 10.5 Honest reporting

Write the backtest report the way a validator would want to read it: state the test, state the result, state the pass/fail, and **discuss the failures**. A report that finds problems with your own model and explains them is far stronger than one that claims perfection.

**Acceptance for Step 10:** `docs/var_backtest_report.md` contains all four tests with results, charts, and interpretation — including at least one honest discussion of where the model underperforms.

---

## Step 11 — Stress testing and reverse stress testing

**Status: partially built.** 11.1 (historical replay, using real pulled spot/VIX moves from five named episodes) and 11.2 (the hypothetical spot×vol grid) are built and verified live. 11.3 (conditional stress) and 11.4 (reverse stress) are deferred — both need a real historical factor covariance/beta estimate; see the status table above.

Three tiers, increasing in sophistication.

### 11.1 Historical replays

Apply actual factor moves from named episodes to today's book:

| Scenario | Period | What it tests |
|---|---|---|
| Volmageddon | 2018-02-05/06 | Short-vol convexity, VIX spike |
| COVID crash | 2020-02-20 → 2020-03-23 | Sustained crash + vol explosion + skew |
| COVID single day | 2020-03-16 | Extreme one-day gap |
| Rate shock | 2022-09 | Rates + equity correlation |
| Yen carry unwind | 2024-08-05 | Sharp vol spike from a quiet regime |

### 11.2 Hypothetical grids

Spot × vol ladder: spot $\in \{-30\%, -20\%, -10\%, -5\%, 0, +5\%, +10\%\}$ crossed with parallel vol $\in \{-20\%, -10\%, 0, +10\%, +25\%, +50\%\}$ relative. Render as a heatmap. Add separate skew-steepening and term-structure-inversion scenarios.

### 11.3 Conditional stress — the sophisticated version

A naive parallel vol bump alongside a −20% spot move is **unrealistic**: in a crash, vol rises *and* skew steepens *and* the term structure inverts. Model the co-movement:

$$ \Delta w_j = \alpha_j + \beta_j\, r_{\text{spot}} + \varepsilon_j $$

Estimate $\beta_j$ per grid node from history (optionally conditioning on the crash regime only — the beta in a selloff is not the beta in a rally, and saying so demonstrates real understanding). Then a spot shock generates a **consistent, empirically-grounded surface response**.

Show side by side: naive parallel bump vs. conditional stress, and the difference in portfolio loss. For a book with short downside puts, the difference will be large — that's the finding.

### 11.4 Reverse stress testing

Instead of "what happens if X?", ask **"what X causes a $5m loss, and how plausible is it?"**

Solve:

$$ \min_{\mathbf{x}} \; \mathbf{x}^\top \Sigma^{-1}\mathbf{x} \quad \text{s.t.} \quad L(\mathbf{x}) \le -\$5\text{m} $$

The objective is squared Mahalanobis distance — the plausibility of the shock under the historical factor covariance. Solve with SLSQP over the reduced PCA factor space (3–5 factors keeps it tractable).

Report:
- The shock vector, expressed in interpretable terms ("spot −14%, PC1 +2.1σ, PC2 +1.4σ").
- Its Mahalanobis distance, and the implied probability under a normal approximation.
- **Which factor it loads on** — that tells the risk manager where the book is actually vulnerable, which is often not where they assumed.

**Acceptance for Step 11:**
- All three tiers implemented and reported.
- Conditional vs. naive stress difference quantified.
- Reverse stress returns an interpretable, plausibility-ranked shock.

---

## Step 12 — Daily P&L explain

**Status: done, verified live.** `pricing/pnl_explain.py` builds the time -> rates/divs -> spot ->
vol waterfall from two real `MarketState`s; verified on the first real consecutive trading-day pair
the automated pipeline produced (2026-08-31 -> 2026-09-01), where it found a genuine $178k
residual, 99.6% traced to one position — see Step 13. The 60+ day acceptance bar (median residual
< 2bp) is still pending real history; every automated day adds one more real point toward it.

**This is the module that makes the project read as risk rather than pricing.**

### 12.1 Decomposition

$$
\begin{aligned}
\text{P\&L}_{\text{explained}} = \;& \underbrace{\Delta \cdot \delta S}_{\text{delta}} + \underbrace{\tfrac12\Gamma(\delta S)^2}_{\text{gamma}} + \underbrace{\mathcal{V}\cdot\delta\sigma_{\text{ATM}}}_{\text{vega}} \\
&+ \underbrace{\text{Vanna}\cdot\delta S\,\delta\sigma}_{\text{skew P\&L}} + \underbrace{\tfrac12\text{Volga}(\delta\sigma)^2}_{\text{vol-of-vol}} \\
&+ \underbrace{\Theta\,\delta t}_{\text{time decay}} + \underbrace{\rho\,\delta r + \rho_q\,\delta q}_{\text{carry}}
\end{aligned}
$$

$$ \text{Residual} = \text{P\&L}_{\text{actual (full reval)}} - \text{P\&L}_{\text{explained}} $$

### 12.2 Order matters

P&L explain is path-dependent in the order you apply moves. Fix a convention and document it: **time → rates/divs → spot → vol**, applying each move to the state after the previous. Report the residual after each step so you can see which move introduces the unexplained portion. Stating that you know the ordering convention matters is itself a signal of practical experience.

### 12.3 Residual monitoring

- Plot residual as a **time series in basis points of portfolio value**.
- Set a threshold (e.g. 5bp of NAV) and generate an alert when breached.
- Maintain a **residual attribution**: which position, which underlying, which risk factor.
- Expect residuals to be largest on: high-gamma days, autocall observation dates, barrier proximity, ex-dividend dates, and expiry rolls. Confirming those expectations empirically is a real result.

### 12.4 Why it matters

If your residual is small and stable, your risk representation is adequate. If it spikes, either your Greeks are incomplete (missing vanna/volga), your surface moved in a way your factors don't capture, or you have a data problem. **Each of those is a genuine production issue** — which leads directly to the next step.

**Acceptance for Step 12:**
- Daily explain runs over ≥ 60 days with a residual time series.
- Median residual < 2bp of NAV.
- At least three residual spikes investigated and root-caused.

---

## Step 13 — The incident report

**Status: done, verified live.** Step 12's first real day pair handed this a genuine incident (see
`docs/incidents/2026-09-01_p008_autocallable_vega_residual.md`): 99.6% of a $178k residual traced
to the NVDA autocallable's vega-only Greek set. Fixed by adding vanna/volga to the autocallable and
barrier Monte Carlo pricers (same architectural gap in both) and re-verified live — the fix is
real and directionally correct (vol-step residual fell ~13%) but honestly does not fully close the
gap, which the report itself explains.

Take the worst residual day from Step 12 and write it up as a production incident, in the format a bank actually uses.

```markdown
# Incident: P&L Explain Residual Breach — 2024-03-12

## Summary
Unexplained P&L residual reached 41bp of NAV (threshold: 5bp) on 2024-03-12,
concentrated in the SPX March expiry bucket.

## Detection
Automated residual monitor (threshold 5bp) triggered at 17:04 ET.

## Impact
$412k unexplained on a $100m book. No trading decisions were taken on the
affected numbers; risk report was reissued at 18:20 ET.

## Investigation
1. Residual decomposed by position → 94% attributable to SPX Mar-2024 options.
2. Greeks reviewed → normal magnitudes, no calculation anomaly.
3. Surface diff vs. prior day → Mar-2024 slice showed a 1.8 vol-point parallel
   shift not present in adjacent expiries.
4. Forward reconstruction → implied forward for Mar-2024 jumped 0.4%, while
   Apr-2024 and beyond were unchanged.
5. Root cause identified: SPX ex-dividend on 2024-03-15 was not reflected in the
   dividend input for the Mar expiry, so the implied forward used a stale
   dividend, biasing the entire slice's implied vols.

## Root cause
The dividend ingestion job sourced announced dividends with a T+1 lag. For
expiries within 3 business days of an ex-date, the forward used a dividend
that had already gone ex.

## Remediation
- Short term: manual override applied; numbers reissued.
- Permanent: forward construction switched to parity-implied dividends
  (Step 2.3) as primary, with announced dividends retained as a
  reconciliation check.
- Control added: QC alert when parity-implied and announced dividend yields
  diverge by more than 25bp.

## Validation
Backfilled 90 days with the corrected forward. Median residual fell from
3.1bp to 1.4bp; the Mar-2024 spike no longer reproduces.

## Lessons
Dividend timing near expiry is a systematic failure mode, not a one-off.
Any input with an announcement lag should be cross-checked against a
market-implied equivalent.
```

**This single document may do more for your candidacy than any other artifact.** The JD explicitly lists *"help investigate and resolve production issues related to risk calculations, data quality, model behavior."* This is a demonstration, not a claim.

---

## Step 14 — Dashboard

**Status: not started.**

Streamlit, six tabs:

1. **Surface** — 3D surface, slice fits with market points overlaid, RMSE by bucket, arbitrage check status.
2. **Exposure** — Greeks by underlying / expiry / moneyness bucket; delta ladder across the three stickiness conventions.
3. **VaR** — time series, full-reval vs. Taylor, exception markers, ES by factor group.
4. **Backtest** — Kupiec / Christoffersen / traffic light / PLA results with exception timeline.
5. **Stress** — spot×vol heatmap, historical replay bars, conditional vs. naive comparison, reverse stress shock display.
6. **P&L explain** — waterfall chart for the selected day, residual time series with threshold band, drill-down by position.

Design constraint: **every number on the dashboard must be traceable to a stored artifact.** No live computation in the UI layer — the dashboard reads what the pipeline wrote. That's how production risk systems are built and it's a good habit to demonstrate.

---

## Step 15 — Model documentation

**Status: not started.**

12–15 pages, written in the style of an SR 11-7 model development document. Structure:

1. **Purpose and scope** — what the model does, what it does not do, intended use.
2. **Product coverage** — instruments, with payoff descriptions.
3. **Methodology**
   - Market data construction, forward and dividend derivation
   - Surface parameterisation, calibration objective, arbitrage constraints
   - Pricing models per product with rationale for each choice
   - Risk factor definition and proxy methodology
   - VaR/ES methodology
4. **Assumptions and limitations** — be exhaustive and honest. Examples: local vol misprices forward-start/cliquet risk; historical simulation assumes the future resembles the window; proxy models fail on idiosyncratic events; square-root time scaling assumes i.i.d.
5. **Data** — sources, quality controls, rejection criteria, known gaps.
6. **Implementation** — architecture, numerical methods, performance.
7. **Testing and validation evidence** — unit tests, benchmark comparisons (QuantLib, VIX), convergence studies, backtest results.
8. **Model risk assessment** — what could go wrong, materiality, mitigants, monitoring.
9. **Ongoing monitoring plan** — the thresholds and alerts you'd run in production.

**The limitations section is what a validator reads first.** Candidates who write "the model works well" get discounted. Candidates who write "local volatility reproduces vanilla prices exactly but underprices forward-skew-dependent payoffs; for the autocallable this means our vega is directionally right but the vega convexity is understated relative to a stochastic-local-vol model" get hired.

---

## Step 16 — Engineering polish

**Status: partially built.** CI (ruff/mypy/pytest) is green throughout and property-based tests (`hypothesis`) are already in use — the fuller testing-pyramid/performance/reproducibility checklist below isn't formally completed yet.

### Testing pyramid

| Layer | What | Examples |
|---|---|---|
| **Unit** | Pure functions | BS price vs. QuantLib; Greeks vs. finite difference; SVI at known params |
| **Property** (`hypothesis`) | Invariants that must always hold | Put-call parity for any $(F,K,T,\sigma)$; monotonicity of price in vol; $\sigma_{\text{loc}}^2 > 0$; VaR ≤ ES |
| **Regression** | Golden files | Full pipeline on a fixed date reproduces stored outputs bit-for-bit |
| **Integration** | End-to-end | `eqdrisk run --date X` completes and writes all artifacts |

Property-based tests are unusual in quant portfolios and signal real software maturity.

### CLI

```bash
eqdrisk ingest      --date 2026-08-11
eqdrisk calibrate   --date 2026-08-11 --underlying SPX
eqdrisk price       --date 2026-08-11 --portfolio configs/portfolio.yaml
eqdrisk var         --date 2026-08-11 --method both --confidence 0.99
eqdrisk backtest    --start 2025-01-01 --end 2026-08-11
eqdrisk stress      --date 2026-08-11 --scenarios configs/stress.yaml
eqdrisk reverse     --date 2026-08-11 --loss-target -5000000
eqdrisk explain     --date 2026-08-11
eqdrisk run         --date 2026-08-11        # full daily pipeline
eqdrisk dashboard
```

### Performance targets

| Operation | Target |
|---|---|
| Full SPX surface calibration | < 5 s |
| Book full revaluation | < 3 s |
| 500-scenario full-reval VaR | < 60 s |
| Autocall MC (100k paths) | < 10 s |
| Full daily pipeline | < 5 min |

### Reproducibility

- Every run writes a manifest: git SHA, config hash, input data hashes, timestamps, library versions.
- Fixed seeds for all MC, recorded in the manifest.
- `make reproduce DATE=2026-08-11` regenerates everything from raw.

---

## Timelines

### Six weeks (recommended)

| Week | Focus | Exit criterion |
|---|---|---|
| 1 | Steps 0–2: env, ingest, curves, forwards | One command ingests a date and produces a clean curve + forward term structure |
| 2 | Steps 3–4: IV, filters, SVI, arbitrage | Zero-arbitrage SPX surface, RMSE < 0.3 vol pts, VIX cross-check passes |
| 3 | Steps 5–7: pricing, Greeks, exotics, portfolio | Book marks; LV MC reprices vanillas; all Greeks validated |
| 4 | Steps 8–10: factors, proxy, VaR, backtest | VaR time series + full backtest report |
| 5 | Step 11: stress and reverse stress | All three stress tiers + reverse stress solver |
| 6 | Steps 12–16: explain, incident, dashboard, doc | Complete documentation set, CI green |

### One weekend (minimum viable, still tells the story)

Saturday morning: ingest one SPX chain, build curve, imply forward.
Saturday afternoon: IV inversion + filters + SVI fit with Durrleman check.
Sunday morning: vanilla book, Greeks including vanna/volga, historical-sim VaR.
Sunday afternoon: one spot×vol stress grid, Kupiec backtest, README documenting what's stubbed.

Then be explicit in the README about what's built vs. designed. "Implemented: X. Designed but not yet built: Y, with the interface specified in `src/.../`" is honest and still demonstrates the thinking.

### Prioritisation if time runs short

Cut in this order (last is most expendable):
1. Keep: SVI + arbitrage constraints, VaR full-reval vs. Taylor, P&L explain, backtest.
2. Then: conditional stress, proxy model, incident report.
3. Then: autocallable, reverse stress, dashboard.
4. First to cut: SABR comparison, barrier bridge correction, FRTB liquidity horizons.

---

## Acceptance criteria

Final checklist before you call it done and put it on a CV:

**Correctness**
- [ ] Vanilla prices match QuantLib to 1e-8
- [ ] All Greeks match central differences to 1e-5 relative
- [ ] Put-call parity holds to machine precision in property tests
- [ ] LV Monte Carlo reprices calibrated vanillas within 2 SE
- [ ] Local variance strictly positive across the full grid
- [ ] Zero butterfly and calendar arbitrage violations on final surfaces

**Risk analytics**
- [ ] VaR time series ≥ 250 days, both methods, gap analysed
- [ ] Kupiec, Christoffersen, traffic light, and PLA all reported
- [ ] ES decomposed by factor group
- [ ] Three tiers of stress implemented, conditional vs. naive quantified
- [ ] Reverse stress returns an interpretable shock with a plausibility measure
- [ ] P&L residual median < 2bp NAV, with ≥ 3 spikes root-caused

**Validation evidence**
- [ ] 30d ATM vol tracks VIX with correlation > 0.98
- [ ] Parity-implied discount factor within 5bp of bootstrapped curve
- [ ] MC convergence study with log-log slope ≈ −0.5
- [ ] Proxy error quantified in VaR dollars

**Engineering**
- [ ] Test coverage > 80% on `src/`
- [ ] CI green, `ruff` and `mypy` clean
- [ ] Full pipeline reproducible from raw via one command
- [ ] Run manifest with git SHA and data hashes

**Communication**
- [ ] Model documentation complete, limitations section substantive
- [ ] VaR backtest report with honest failure discussion
- [ ] Incident report written
- [ ] README (this file) accurate to what's actually built

---

## Interview preparation

Expect the conversation to go deep on whichever piece you sound most confident about. Prepare these five thoroughly.

**1. "Why does your VaR differ between full revaluation and the sensitivity approach?"**
Taylor expansion truncates at second order. It breaks down for large moves, near barriers, at autocall observation dates, and for short-dated high-gamma positions where the payoff is strongly non-quadratic. Quote your own numbers: the average gap, the worst day, and what was in the book that day.

**2. "Which delta do you hedge with?"**
Neither sticky-strike nor sticky-delta is universally right; the market regime determines the empirical backbone. Present your quarter-long study, note that index skew typically implies a backbone between the two, and that local vol implies roughly twice the sticky-delta adjustment.

**3. "Why do you enforce no-arbitrage on the surface?"**
Beyond principle: the Dupire denominator *is* Durrleman's butterfly condition. Arbitrage in the surface means negative local variance means your Monte Carlo is undefined. The constraint is a numerical prerequisite, not a purity concern.

**4. "How do you handle single names with thin surfaces?"**
Proxy to index factors with a name-specific beta and idiosyncratic term. Then immediately volunteer the limitation: proxies fail around earnings, and you quantified the proxy error in VaR dollars — give the number.

**5. "Walk me through a day your risk numbers looked wrong."**
Tell the incident report as a story. Detection, decomposition, hypothesis, elimination, root cause, fix, validation. This is the question the whole project was built to answer.

**Things to volunteer that reward you:**
- The VIX cross-check as an independent calibration validation.
- That your Basel traffic light is yellow and why (if it is) — honesty about model performance reads as maturity.
- The residual ordering convention in P&L explain.
- The FRTB PLA test — it's current, and it connects your VaR method to your attribution method.

**Things to be careful about:**
- Don't overclaim on the exotics. If your autocall Greeks have wide MC error bars, say so.
- Don't say local vol is "the" model for autocallables — a desk would use stochastic-local-vol; know the difference and why LV understates forward skew.
- Don't present historical simulation as unconditionally superior; know its weakness (it reacts slowly to regime change, which is exactly what Christoffersen's test detects).

---

## References

**Volatility surfaces**
- Gatheral, *The Volatility Surface* (2006) — the foundational text
- Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014) — SVI, SSVI, arbitrage conditions
- Zeliade Systems, "Quasi-Explicit Calibration of Gatheral's SVI Model" (2009) — the calibration technique that actually converges
- Hagan et al., "Managing Smile Risk" (2002) — SABR
- Lee, "The Moment Formula for Implied Volatility at Extreme Strikes" (2004) — wing asymptotics

**Pricing and Greeks**
- Andersen & Piterbarg, *Interest Rate Modeling* Vol. 1 — numerical methods, broadly applicable
- Carr & Madan, "Towards a Theory of Volatility Trading" (1998) — variance swap replication
- Jäckel, "Let's Be Rational" (2015) — fast IV inversion
- Bergomi, *Stochastic Volatility Modeling* (2016) — skew dynamics, sticky rules, forward skew

**Risk management**
- Jorion, *Value at Risk*, 3rd ed. — the standard reference
- Christoffersen, "Evaluating Interval Forecasts" (1998) — independence test
- Kupiec, "Techniques for Verifying the Accuracy of Risk Measurement Models" (1995)
- BCBS d457, *Minimum Capital Requirements for Market Risk* (FRTB) — ES, liquidity horizons, PLA test
- Federal Reserve SR 11-7, *Guidance on Model Risk Management* — the model documentation standard

---

## Licence and disclaimer

MIT. This is a research and demonstration project. It is not investment advice, has not been validated by any independent function, and must not be used for actual risk management or trading decisions.
