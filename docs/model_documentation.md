# EQD Risk Engine — Model Documentation

**Document type:** model development document (SR 11-7 style)
**Model owner:** Nikhil Pandey
**Scope of this document:** Steps 1-14 of the engine as built and verified against real live
market data through 2026-09-05. Steps 8.2/8.3 (PCA/proxy risk factors), 9/10 (VaR and backtesting),
and 11.3/11.4 (conditional and reverse stress) are explicitly out of scope for this version — they
are not yet built, for reasons given in Section 4.

---

## 1. Purpose and scope

This model prices and risk-manages a small, fixed book of equity derivative positions (vanilla
options, variance swaps, barrier options, and a single-name autocallable note) across five
underlyings (SPX, AAPL, NVDA, JPM, XLE). It performs five functions:

1. Builds discount curves, forward curves, and implied dividend yields from real market data.
2. Extracts and quality-filters implied volatility from real option chains, then calibrates an
   arbitrage-checked (SVI/SSVI) volatility surface per underlying per day.
3. Prices the book's instruments off that surface (closed-form where available, Monte Carlo
   otherwise) and computes a full Greek set per position.
4. Decomposes day-over-day portfolio P&L into a Greeks-based waterfall and reports the residual —
   the model's own self-check on how complete its risk representation is.
5. Runs hypothetical and historical stress scenarios against the book.

**What this model does not do**, by design, not oversight: it does not compute VaR or Expected
Shortfall (Section 4), it does not backtest a VaR series, it does not run PCA-based or
econometric-proxy risk factors, and it does not support conditional or reverse stress testing. All
four are blocked on the same root cause — insufficient real historical depth — and are discussed
honestly in Section 4 rather than approximated with a token or vacuous version.

**Intended use:** this is a demonstration/portfolio project built to exercise the same design
decisions a real risk-desk quant would face, not a production trading system. It has no live
trading connectivity, uses free-tier market data (Section 5), and its "book" is a fixed, static
9-position configuration (`configs/portfolio.yaml`), not a live position feed.

---

## 2. Product coverage

| Type | Positions | Payoff |
|---|---|---|
| Vanilla option | P001-P005 | Standard European call/put, Black-76 forward-space pricing. |
| Variance swap | P006 | Pays the difference between realized variance and a fair strike fixed at inception; here always marked as a *fresh* swap struck at today's own fair strike, so its model P&L is exactly zero by construction (Section 3). |
| Barrier option | P007 | Down-and-in put: activates a vanilla put only if the underlying trades below a barrier at any point up to expiry (continuously monitored, not discretely). |
| Autocallable note | P008 | Quarterly-observation reverse convertible: early redemption at par-plus-coupon if the underlying is at/above an autocall barrier on an observation date; a "memory" coupon if between a lower coupon barrier and the autocall barrier; otherwise no payment and the memory counter increments. At maturity, if never autocalled, the investor is short a European (maturity-only) down-and-in put struck at inception. |
| Equity (delta-one) | P009 | Underlying stock/index position, no optionality. |

The book (`configs/portfolio.yaml`) has 9 positions across SPX, AAPL, and NVDA. JPM and XLE are
configured in the universe but currently have zero real single-name option coverage (Section 5) and
so cannot be assigned positions yet.

---

## 3. Methodology

### 3.1 Market data construction

Discount curves are bootstrapped from real SOFR/Treasury tenor points (`marketdata/curve.py`).
Forward curves (`marketdata/forward.py`) are **not** the textbook closed-form
$F(0,T) = S_0 e^{(r-q)T}$ — dividends are not perfectly observable from a free-tier feed, so
$F(0,T)$ is instead solved via a put-call-parity regression against the day's own option chain,
jointly fitting an implied forward and an implied continuous dividend yield per expiry, with the
regression's own $R^2$ and basis-point fit error stored as first-class quality metrics
(`FORWARD_SCHEMA`). A stricter "is this forward good enough to compute moneyness with" gate
(`is_reliable_forward`, $R^2 \geq 0.995$, $|\text{bp diff}| \leq 150$) is deliberately more
permissive than the acceptance bar used to flag a forward as suspicious in the first place — the
two questions ("is this fit good" and "is this fit good *enough to use*") are answered separately
and documented as such, not conflated.

**Consequence for downstream steps:** because the forward is a jointly-fitted regression output,
not a closed-form function of $(S_0, r, q)$, it cannot be decomposed *after the fact* into a rate
piece, a dividend piece, and a spot piece. This directly shapes the P&L explain waterfall (3.5) and
is documented as a limitation in Section 4, not hidden.

### 3.2 Surface parameterization, calibration objective, and arbitrage constraints

Each expiry's raw implied-vol slice (Section 5) is fit to raw SVI:

$$w(k) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2+\sigma^2}\right)$$

using Zeliade's quasi-explicit method: for fixed $(m,\sigma)$ the problem is linear in $(a, b\rho,
b)$, solved by weighted least squares, with only the 2-D $(m,\sigma)$ outer problem solved by
Nelder-Mead. **Documented simplification:** the inner linear solve is unconstrained OLS followed by
a projection onto the feasible region ($b \geq 0$, $|\rho| < 1$, non-negative ATM variance), not a
constrained QP solved directly as in the original Zeliade paper — adequate for the sparse,
already-quality-filtered slices this model fits, flagged as needing revisiting for larger or
noisier slices.

Every per-expiry SVI fit is checked against Durrleman's butterfly condition $g(k) \geq 0$ (no
static butterfly arbitrage) and, jointly across expiries, against the calendar-arbitrage condition
that total variance $w(k,T)$ must be non-decreasing in $T$ at fixed $k$. **If the calendar
condition is violated, the entire day's surface for that underlying falls back to SSVI**
(Gatheral-Jacquier power-law, $\phi(\theta) = \eta\theta^{-1/2}$, a single shared $(\rho,\eta)$
across all expiries with the sufficient no-arbitrage condition $\eta(1+|\rho|) \leq 2$ enforced as
a hard constraint) — arbitrage-free *by construction* rather than checked after the fact. This is
not a hypothetical fallback path: it fired for real on SPX on 2026-09-04 (all 14 expiries,
confirmed via direct query of the persisted `vol_surface` table), while AAPL and NVDA's SVI fits on
the same day needed no fallback. **A less obvious tuning issue was also caught and fixed by
testing, not assumed correct:** the constrained SLSQP solver used for one intermediate refit
approach was incompatible with a clipped inner linear solve (see project decision log,
2026-08-21) — the two were interacting to silently ignore the constraint in some cases before the
mismatch was found and fixed.

### 3.3 Pricing models per product, with rationale

| Product | Model | Rationale |
|---|---|---|
| Vanilla option | Closed-form Black-76 | Exact, fast, matches the exchange-quoted convention this book is calibrated against; validated to $10^{-8}$ against QuantLib (Section 7). |
| Variance swap | Carr-Madan static replication off the calibrated smile (log-contract decomposition into an OTM put/call strip) | Model-independent within the no-arbitrage assumptions of the replication itself; avoids needing a separate stochastic-vol model just to price a variance product. |
| Barrier option | Reiner-Rubinstein closed form (flat-vol assumption) for validation, but the position is actually marked off a **local-vol Monte Carlo** engine with a Brownian-bridge continuity correction | The closed form assumes constant vol, which this project's own calibrated surface contradicts; MC under the *calibrated* local-vol surface is the position's actual mark, with the closed form used only as an independent correctness check under its own (flat-vol) assumptions. |
| Autocallable note | Local-vol Monte Carlo, common-random-number (CRN) bump-and-reval Greeks | No useful closed form exists for this path-dependent, multi-barrier payoff; CRN is not optional here — independently reseeded bump-and-reval Greeks on a path-dependent payoff are pure noise, not an approximation with a known error, because the discrete autocall/coupon/put decisions themselves change under different random draws. |

**Local volatility** (Dupire) is stripped from the *calibrated* SVI/SSVI surface, not the raw noisy
market points, using a smoothing-spline (with PCHIP fallback) $T$-derivative to avoid amplifying
calibration noise into the local-vol surface (a documented, tested fix — the naive finite-difference
$\partial_T w$ was measurably noisy on real data before this was in place). Wing extrapolation is
capped at Step 3's own extreme-log-moneyness quality threshold, not extrapolated indefinitely.

Monte Carlo paths use Sobol quasi-random sequences with a classic power-of-two Brownian bridge path
construction (Glasserman), numba-JIT-compiled for the inner Euler step. Barrier positions use a
Brownian-bridge continuity correction for the discretely-simulated barrier crossing probability,
since naive discrete monitoring measurably underprices a knock-in (confirmed: >25 standard errors
of bias at a coarse 16-step monitoring grid on real parameters, cut to ~0.1 SE at the *same* step
count once the correction is applied — see Section 7).

### 3.4 Risk factor definition (Step 8.1)

The only risk-factor representation currently built is a **fixed vol grid**: total implied variance
$w(k,T)$ evaluated at a fixed $(k,T)$ grid from each day's calibrated surface, requiring no forward
curve at all (a deliberately simpler, more robust representation than a moneyness grid that would
depend on that day's own forward). PCA-based risk factors (8.2) and econometric proxy models for
sparse names (8.3) are specified in the original plan but not built — see Section 4.

### 3.5 P&L explain methodology

Given two real, already-observed market states (day 0 and day 1), the book's P&L is decomposed
into a **fixed-order** Greeks-based waterfall: **time → rates/divs → spot → vol**. Four
intermediate market states are constructed by mixing fields from the two real loaded states in
that order; at each step, the *actual* P&L (a full reprice, exact) is compared against an
*explained* P&L (that step's Greeks, evaluated at the state at the *start* of the step, times that
step's own driving-variable change), and the residual (actual minus explained) is reported per
step and per position.

**A structural, not incidental, property of this design:** because rates/divs (3.1) can only move
the discount curve — not the forward, which is a jointly-fitted quantity that embeds that day's own
carry — the "rates/divs" step's explained P&L is rho-only, and any real carry P&L shows up inside
the *spot* step instead. Because Greeks are evaluated at the *start* of each step rather than at a
single global base state, cross-terms like vanna (spot-vol interaction) are implicitly absorbed
into whichever step comes *later* in the fixed order — the vol step's vega is evaluated *after* the
spot step has already moved the underlying, so it already reflects part of what a separate vanna
term would otherwise capture. This is a real, demonstrated consequence of the ordering convention,
not an approximation error to be minimized (Section 4 discusses where it still falls short).

There is no VaR methodology to document (Section 4).

---

## 4. Assumptions and limitations

This section is written to be read first, per the model's own operating principle: a limitations
section that says "the model works well" is worth less than one that says exactly where and why it
does not.

**Local volatility structurally misprices forward-skew-dependent payoffs, and this has been
measured, not just asserted.** The clearest evidence in this entire project is the incident
documented in `docs/incidents/2026-09-01_p008_autocallable_vega_residual.md`: on the first real
day-over-day P&L explain run, 99.6% of a $178k residual (on a ~$4.16M book) traced to the NVDA
autocallable, whose vega-only Greek set could not see how much its effective vol sensitivity
changes near the autocall/coupon/put barriers. Adding vanna and volga (via bump-and-reval Monte
Carlo under CRN) reduced that step's residual by only about 13% — the vega+volga explained P&L
moved from -$220,644 to -$195,329 against an actual of -$28,751, still off by roughly 7x. **The
honest conclusion, stated plainly in that report, is that a smooth second-order Taylor expansion
around one ATM vol scalar cannot capture a barrier-laden payoff's true vol sensitivity, no matter
which named Greeks are included** — the barriers create genuine kinks in price-vs-vol space. A
proper fix needs either a full-reprice ("what-if this step's actual vol surface change") term for
MC-priced positions in the waterfall, or accepting that these positions' vol-step residual is a
structurally different quantity from vanilla positions' and should be thresholded separately. This
is not fixed in the current build.

**The hypothetical-stress vol-shock model is a documented approximation, not an exact
recalibration.** Applying a parallel vol shock to an already-built local-vol grid via $\sigma_{loc}
\to \sigma_{loc}\sqrt{C}$ was originally believed to be exact (the reasoning: SVI's $w(k)$ is linear
in $(a,b)$, so uniformly rescaling should preserve Durrleman's $g(k)$ shape). Writing a
proof-by-construction test falsified this: $g$ contains an additive constant that does not scale
with $w$, so the shortcut is a bounded approximation whose error grows with shock size — measured
at ~7% at a 10%-relative vol shock, ~18% at 25%, and ~40% at the stress ladder's most extreme 50%
shock. The stress grid and historical replay results in this project use this approximation
throughout; a full re-strip of local vol under each shocked surface would be exact but is not
currently built (it would multiply the already-expensive MC cost of the 7×6 grid by a further
recalibration step per cell).

**No VaR, Expected Shortfall, or backtesting exists in this build.** This is a scope decision, not
an oversight: the acceptance criteria for these steps (a 250-1000 day real-history window, a
regulatory-convention size) require real, accumulating daily calibrated-surface history that a
free-tier, single-snapshot-per-day pipeline can only build up one real trading day at a time — as
of this document, roughly 5-9 real trading days exist. Building a token or vacuous VaR against
insufficient history was explicitly rejected in favor of building Step 11's history-independent
stress-testing parts instead (a documented order deviation from the original step sequence). The
same root cause blocks 8.2 (PCA risk factors), 8.3 (proxy modelling for sparse names), and 11.3/11.4
(conditional and reverse stress, which need a real historical beta/covariance estimate).

**No automated residual monitor exists.** The P&L-explain residual — this model's own primary
self-diagnostic — is currently reviewed manually (via the CLI or the Streamlit dashboard), not via
an automated threshold alert. This is flagged explicitly as an open item in the incident report
above; the daily automated pipeline (Section 6) computes the residual every trading day but does
not yet page anyone when it spikes.

**Two persistent, real data gaps, not bugs:** JPM and XLE have shown zero calibrated vol-surface
coverage on every real trading day tested so far (confirmed again via direct query as of
2026-09-04: 0 rows for either underlying across the full history, vs. 36/62/89 rows for
AAPL/NVDA/SPX respectively) — a genuine single-name strike-sparsity and staleness issue on
free-tier option-chain data, not a code defect, and not currently mitigated (that mitigation is
exactly what the unbuilt Step 8.3 proxy-modelling step would address). Separately, no paid
historical option-chain archive exists, so this model's entire notion of "history" is whatever the
automated daily pipeline (Section 6) has itself accumulated since ingestion started, not a
backfilled dataset.

**The forward curve cannot be decomposed into rate/dividend/spot contributions after the fact**
(Section 3.1), which is why the P&L-explain rates/divs step only captures a discount-curve effect —
real carry P&L is folded into the spot step, not cleanly separated. This is disclosed in the
waterfall's own design (3.5), not silently absorbed.

**Sequential P&L-explain ordering absorbs cross-terms, by design, but this means the waterfall's
individual step attributions are convention-dependent, not unique.** A different move order (say,
vol before spot) would attribute a different fraction of any given day's vanna effect to each step.
The total (sum across all steps) is order-independent and telescopes exactly to the true full-reval
P&L difference — confirmed to $10^{-9}$ relative on synthetic data — but the *per-step* breakdown is
a reporting convention, stated as such in the module's own docstring, not a unique decomposition.

**Wing extrapolation and truncation are capped, not eliminated.** The variance-swap Carr-Madan
replication truncates its integration at Step 3's own extreme-log-moneyness quality threshold, and
the resulting truncation error was measured (not assumed) to **not have a fixed sign** on real SPX
data — it is a real, bounded source of imprecision in the fair-strike calculation, disclosed in the
varswap output's own persisted `truncation_error_vol_points` column rather than hidden.

**MC noise is a designed cost/precision tradeoff, not an oversight.** Daily marks use higher path
counts (`MC_N_PATHS = 50,000`) than the 42-cell hypothetical stress grid (`GRID_MC_SETTINGS`, 8,000
paths) — a deliberate "more precision for a once-a-day mark, more speed for a bulk sweep" choice,
not a hidden inconsistency; both settings are named constants, not magic numbers buried in the
pricing calls.

**Two real wall-clock-dependent staleness bugs were found and fixed during this project, and a
third-order concern is now documented rather than assumed solved.** Quote staleness must be judged
against the correct "as-of" reference time depending on whether the pipeline runs mid-session,
after close, or before the market opens; both an after-close case (Step 4) and a separate
pre-market case (Step 8) were found live, zeroing out real quote coverage before being fixed. The
fix (`marketdata.quality.staleness_reference_ts`, a three-regime function) is tested, but any future
change to when the automated pipeline runs (currently 16:30 ET) should re-verify this logic rather
than assume it is timing-invariant.

**This is a single, static, 9-position book**, not a live position feed, and the model has no
connectivity to any trading or booking system. Its purpose is to demonstrate the analytics, not to
operate as a production risk system.

---

## 5. Data

**Sources (all free-tier, no paid historical archive):** `yfinance` for underlying prices and
option chains, FRED (via `pandas-datareader`/`fredapi`) for SOFR/Treasury rate points and the VIX
index, `pandas-market-calendars` (NYSE calendar) for trading-day logic. No vendor-provided implied
vols or Greeks are used — every implied vol in this system is inverted in-house from raw
bid/ask/last-trade quotes (Section 3.3 inputs).

**Quality control and rejection criteria:** every raw option quote is assigned exactly one reason
code before being used in calibration (`vol/implied.py`): `OK` (used), or one of `WIDE_SPREAD`,
`CROSSED`, `STALE`, `ZERO_BID`, `LOW_OI`, `IV_SOLVE_FAIL`, `ITM_SIDE`, `NO_ARB_INTRINSIC`,
`NO_RELIABLE_FORWARD`, `THIN_SLICE` (rejected, with a specific, auditable reason — never silently
dropped). On SPX for 2026-09-04, 1,162 quotes were accepted as `OK` across 14 expiries; the
rejected population is retained in the same `implied_vols` table with its reason code, not deleted,
so rejection rates are themselves auditable.

**Known gaps:** JPM and XLE persistently show zero usable single-name coverage (Section 4). No
historical option-chain backfill exists — the model's real history starts from whenever the
automated daily pipeline (Section 6) began running, and grows by exactly one real trading day at a
time.

---

## 6. Implementation

**Architecture** (`src/eqdrisk/`): `io/` (schema-enforced Parquet read/write over DuckDB — every
curated table has an explicit `pyarrow.Schema` and a not-null identity-column check, failing loudly
on structural violations rather than writing malformed data); `marketdata/` (curves, forwards,
quality filtering, calendar); `vol/` (implied-vol inversion, SVI/SSVI/SABR calibration, local vol,
risk factors); `pricing/` (Black-76, Monte Carlo engine, variance swap, barrier, autocallable, P&L
explain); `portfolio/` (schema and marking orchestration); `stress/` (shocks, historical replay,
hypothetical grid); `cli.py` (Typer entrypoint); `app/dashboard.py` (Streamlit, reads only stored
artifacts, no live computation).

**Storage:** hive-partitioned Parquet on local disk, queried through embedded DuckDB — not a
database server, a local analytical query engine over files. Every curated table is
schema-validated on write (`io/schemas.py`).

**Numerical methods:** Nelder-Mead for SVI's 2-D outer calibration problem (with a closed-form
linear inner solve); SLSQP for SSVI's constrained fit; Brent's method for implied-vol inversion;
Sobol quasi-random sequences with Brownian-bridge path construction for Monte Carlo, numba-JIT-
compiled for the inner simulation loop; common random numbers for every bump-and-reval Greek
calculation on a path-dependent payoff.

**Automation:** the full daily pipeline (`ingest → curves → iv → calibrate (per underlying) →
price → varswap → riskfactors → portfolio → explainpnl`) runs unattended via a macOS `launchd`
agent (`scripts/daily_ingest.sh`, Mon-Fri 16:30 local) plus a `pmset` auto-wake schedule so it runs
even with the laptop's lid closed — chosen over cron specifically because launchd catches up on a
missed run after sleep, where cron silently skips it. `docs/AUTOMATION.md` has the full writeup.

**Performance, measured (Step 16), not just estimated — including one self-correction.** Real SPX
full-surface calibration on 2026-09-04's data took **0.22s** — comfortably under the README's < 5s
target. A single autocallable Monte Carlo price at 100,000 paths took **0.86s** (JIT warm) —
comfortably under the < 10s target for a single price. Full book revaluation (all 9 positions, real
2026-09-04 data) originally took **126.28s**, roughly 42x over the < 3s target, and a full
`make reproduce` production run took **404.4s** against the < 5-minute target.

**The first explanation written here for those two misses was wrong, and finding that out is the
more useful result.** It blamed Step 13's vanna/volga fix (9 MC reprices per position instead of 4)
without ever profiling the actual call stack — an unverified claim, which is exactly the failure
mode this document's own Section 4 warns readers to watch for elsewhere in the model. Profiling
(`cProfile`) corrected this: of the 126.28s, **111.7s (89%) was spent in `load_market_state`**, not
MC pricing (`mark_with_state`, covering both MC-priced positions' full Greek sets, took only
11.3s). The real cause: `vol/local_vol.py::build_local_vol_grid` calls `local_variance_at` once per
(strike, time) grid point (~12,000 times across both underlyings), and each call independently
re-fits three GCV-optimized smoothing splines from scratch — expensive, and entirely unrelated to
Step 13 (this cost has existed since Step 6.1).

**Fixed with a safe, zero-modeling-risk change:** each grid row's spline fits are a pure function of
their own inputs with no shared state, so `build_local_vol_grid` now computes rows across a process
pool instead of sequentially — same math, same values (all 1,713 tests still pass), just
parallelized. Result: `load_market_state` dropped from 111.7s to 27.5s (~4x on an 8-core machine),
book reval from 126.28s to **38.6s**, and the full `make reproduce` pipeline from 404.4s to
**287.4s** — newly meeting the <5-minute target, though book reval alone still misses its <3s
target by a wide margin (~13x over, down from ~42x) — a real, disclosed, not-fully-closed gap. A
deeper fix would need to touch the local-vol stripping algorithm itself (e.g. avoiding redundant
GCV smoothing-parameter searches across nearby strikes), which carries real modeling-behavior risk
and was deliberately left out of this round's scope.

---

## 7. Testing and validation evidence

| Check | Result |
|---|---|
| Black-76 price vs. QuantLib | Matches to $10^{-8}$; delta/gamma/vega/vanna/dividend_rho to $10^{-6}$ |
| rho, volga conventions vs. QuantLib | Two Greeks confirmed (by hand derivation) to use a different, documented convention, not a bug |
| Put-call parity, monotonicity in vol, IV round-trip | Property-based tests (`hypothesis`), not just fixed examples |
| SVI/SSVI arbitrage checks | A genuine real SPX calendar-arbitrage violation (2026-09-04, all 14 expiries) correctly triggered the SSVI fallback |
| Local-vol MC reprice of calibrated vanillas | Within a fraction of a standard error; grid-resolution convergence quantified, not assumed |
| Barrier discretization bias | Naive discrete monitoring: >25 SE biased low at a coarse 16-step grid; Brownian-bridge correction: ~0.1 SE at the *same* step count |
| Barrier in+out = vanilla parity | Model-independent identity, validated across strike/barrier configurations |
| Autocallable CRN scale-invariance | Under flat vol, delta/gamma (and, after Step 13, vanna) come out exactly 0.0 — a deterministic proof point, not a statistical one |
| Variance swap fair strike vs. VIX-style cross-check | Real SPX gap shrank from 3.775 vol points (naive ATM-vol comparison) to 1.258 vol points (proper fair-strike comparison) |
| Historical replay (5 real named episodes) | COVID crash: -$3.40M, ~82% of book value, using real pulled spot/VIX data |
| Hypothetical stress grid | Base case exactly $0; worst cell -$1.44M at spot -30%/vol +50% |
| P&L explain telescoping | Total actual P&L across all 4 steps matches true full-reval P&L to $10^{-9}$ relative |
| P&L explain on equity/varswap positions | Exactly zero residual in every step (linear payoff for equity; marked to 0 by construction for varswap) — an exact, not approximate, design validation |
| P&L explain, real multi-day | 4 real consecutive day-pairs (2026-08-31 through 2026-09-04): residuals of +$151.7k, -$252.4k, +$349.0k, -$170.7k — genuinely oscillating in sign, not a cherry-picked single "good" day |
| Full test suite | 1,710 unit/property tests passing; ruff and mypy clean throughout |

Several real defects were caught during development by writing a test for a claim rather than
trusting the algebra or a single manual check — a pattern this document treats as evidence of
process quality, not just a list of bugs: a dict-key type mismatch that silently miscategorized
100% of expiries (Step 3); two separate wall-clock staleness bugs (Section 4); a false "exact" math
claim in the stress-shock model (Section 4); a `ZeroDivisionError` in the P&L-explain vol step for
zero-expiry equity positions; and a cross-platform (Linux CI vs. macOS local) floating-point
exact-equality test failure in the new vanna Greek, caught by CI and fixed with a tolerance rather
than loosened to hide the platform difference.

---

## 8. Model risk assessment

| Risk | Materiality | Mitigant / current state |
|---|---|---|
| MC-priced positions' vol Greeks understate true sensitivity near barriers | **High** — directly measured: 99.5% of a real day's P&L residual on this specific book | Vanna/volga added (Step 13), reduces but does not close the gap; not further mitigated in this build |
| Vol-shock stress approximation error | **Medium**, grows with shock size (~7-40% measured) | Documented and bounded; not eliminated; matters most for the largest stress-ladder shocks |
| No automated residual monitoring | **Medium** — a real spike could go unnoticed between manual reviews | Manual review currently in place; a real threshold check is a named, not-yet-built follow-up |
| Two single names (JPM, XLE) have no real coverage | **Low** for this book (no positions assigned there), but blocks any future position in those names | Not currently mitigated; Step 8.3 (proxy modelling) would address this if built |
| No VaR/backtesting | **High** for any real risk-limit use case, **not applicable** to this project's actual scope | Explicitly out of scope pending real multi-year history; not approximated with a token version |
| Single free-tier data vendor, no failover | **Medium** — a vendor outage or schema change could silently degrade quality-filtered coverage | Reason-code-based rejection makes degraded coverage auditable (not silent), but there is no secondary data source |
| Reverse convertible-style payoff concentration (P008 alone is a $5M notional autocallable) | **High** relative to book size (~$4.16M NAV includes this position's mark) | Directly responsible for the largest documented residual in this project; see Section 4's first item |

---

## 9. Ongoing monitoring plan

The following would be the production monitoring plan if this were operated as a live system, most
of it not yet implemented (marked accordingly):

1. **Daily calibration quality** — RMSE and max-abs-error in vol points per expiry, butterfly and
   calendar-arbitrage violation counts. *Implemented*: computed and persisted every day
   (`vol_surface` table), visible in the dashboard's Surface tab.
2. **Daily P&L-explain residual** — in dollars and as a fraction of NAV, by step and by position.
   *Implemented as a computation and a dashboard view*; *not implemented* as an automated alert
   (Section 4, Section 8).
3. **Quote rejection-rate monitoring** — a sudden jump in any single reason code's share of a
   day's quotes would indicate a data-quality regression before it reaches calibration.
   *Not implemented as an alert*; the underlying data (per-quote reason codes) already exists.
4. **VaR exception tracking (Kupiec/Christoffersen/traffic-light/PLA)** — *not applicable*, since no
   VaR series exists yet.
5. **Real-history depth tracking** — a simple count of real trading days accumulated so far,
   since several steps' eventual activation (8.2/8.3/11.3/11.4 at tens-to-hundreds of days, 9/10 at
   250-1,000) is gated purely on this growing. *Implemented*: the dashboard's VaR/Backtest tabs
   already report this count honestly rather than hiding behind a generic "not available" message.
6. **Automation health** — whether the daily `launchd` job actually ran and succeeded.
   *Implemented*: per-day log files (`logs/daily_ingest_*.log`), checked manually; no automated
   failure alert exists yet.
