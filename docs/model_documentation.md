# EQD Risk Engine — Model Documentation

**Document type:** model development document (SR 11-7 style)
**Model owner:** Nikhil Pandey
**Scope of this document:** Steps 1-14 of the engine as built and verified against real live
market data through 2026-09-05, plus Step 8.2 (vol surface PCA, added 2026-10-31 on the stored
history to 2026-10-30), Step 8.3 (vol proxy regression) and Step 9 (VaR and Expected Shortfall),
both added 2026-11-01. Steps 10 (backtesting) and 11.3/11.4 (conditional and reverse stress) are
out of scope for this version — they are not yet built, for reasons given in Section 4.

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
6. Computes VaR and Expected Shortfall by historical simulation, by full revaluation and by a
   Greeks (Taylor) approximation, with each name's vol moves proxied from its published vol index
   (3.6).

**What this model does not do**, by design, not oversight: it does not backtest its VaR series, and
it does not support conditional or reverse stress testing. Both are blocked on the same root cause —
insufficient real historical depth — and are discussed honestly in Section 4 rather than
approximated with a token or vacuous version.

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

The book (`configs/portfolio.yaml`) holds 9 live positions across SPX, AAPL, and NVDA. Two of the
original options have expired and settled (P005 on 2026-09-18, P004 on 2026-10-16) and were rolled
into the same trade at the next quarterly expiry (P010, P011; same side and size, strike at the old
option's moneyness on the first stored day, nearest listed strike). JPM and XLE are configured in
the universe but currently have zero real single-name option coverage (Section 5) and so cannot be
assigned positions yet.

**Position lifecycle (`portfolio/lifecycle.py`, since 2026-10-20).** Each position's status on a
date is recomputed from stored daily closes, so it is reproducible from data: vanilla options settle
in cash at the expiry close; the down-and-in barrier knocks in once a daily close since monitoring
start is at or below the barrier (daily closes approximate continuous monitoring) and is then priced
as the vanilla put it has become; the autocallable's past observations (fixed on the first close on
or after each date) decide redemption, coupons paid in cash, or coupon memory carried forward. A
settled position keeps its settled value with zero Greeks, so it adds no risk and no stress P&L, and
the P&L explain books its last move as settlement, not residual. Positions within a week of expiry
or an autocall observation are listed in the daily output. The autocallable is priced against its
own initial fixing (issued 2026-08-15, NVDA $225.16). Before this fix it was priced against each
day's spot, which re-issued the note at the money every day: its delta read about 2,700 NVDA shares
on 2026-10-19 instead of about 13,400. Variance swaps are not settled, because P006 is marked as a
fresh swap at its own fair strike each day and has no traded strike to settle against.

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

Single names (NVDA, AAPL, JPM, XLE) list American options, whose put-call parity drifts from the
European relation with maturity, so their forwards are fitted with the discount factor fixed at the
rates curve's and judged by how tightly the strikes agree (dispersion at most 100bp of the forward)
rather than by the parity-implied discount factor; SPX keeps the two-parameter fit. The same
reliability rule decides both which expiries the vol surface uses and which forwards the pricing
forward curve is built from. A single-name expiry with clean out-of-the-money quotes but too few
strikes with both legs clean for a fit takes its forward from that day's forward curve (flagged
`implied_vols.forward_source = "curve"`). Beyond the last reliable expiry the forward curve
continues at the carry measured over its last 0.25y, not flat.

For SPX the forward fit counts a quote if its contract traded at any point in that day's session;
the vol stage, and the single names' fits, keep the 30-minute last-trade rule. Long-dated index
options rarely trade in the last half hour, and the fit needs the call and the put at the same
strike, so under the 30-minute rule the longest reliable SPX forward had a median of 0.48y across
33 stored days (0.12y on the worst day), and the surface's long end existed only through stale
fill. With the session rule it is 2.3y (1.18y worst), with no added scatter in the fitted carry.
Contracts with no trade that day stay excluded: every junk quote found (for example a call last
traded months earlier, $337 away from parity on a $4.80 spread) was on such a contract.

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
| Autocallable note | Local-vol Monte Carlo on a time grid containing every observation date exactly; common-random-number (CRN) bump-and-reval Greeks with 5% spot / 3 vol-point bumps and a one-day-revaluation theta; a standard error from 16 independently scrambled sub-runs | No useful closed form exists for this path-dependent, multi-barrier payoff; CRN is not optional here — independently reseeded bump-and-reval Greeks on a path-dependent payoff are pure noise, not an approximation with a known error, because the discrete autocall/coupon/put decisions themselves change under different random draws. |

**Local volatility** (Dupire) is stripped from the *calibrated* SVI/SSVI surface, not the raw noisy
market points, using a smoothing-spline (with PCHIP fallback) $T$-derivative to avoid amplifying
calibration noise into the local-vol surface (a documented, tested fix — the naive finite-difference
$\partial_T w$ was measurably noisy on real data before this was in place). The spline's smoothness
is fixed (`T_SMOOTHING_LAM`) rather than chosen by cross-validation per point, which was found to
create day-to-day local-vol noise of 1.7-2.4 vol points at 2 years (Section 4). Beyond the last
calibrated expiry each strike keeps its last implied vol (total variance keeps growing at the last
expiry's average rate). Wing extrapolation is capped at Step 3's own extreme-log-moneyness quality
threshold, not extrapolated indefinitely. The Monte Carlo exotics' grid is built only from expiries
at least a month out (`LOCAL_VOL_MIN_PILLAR_T`): NVDA's weekly expiries are fitted with 5-15
vol-point errors, and through the grid's first rows their day-to-day swings moved the autocallable
by $30-50K on days when implied vols barely moved. Below the first kept expiry local vol is flat at
that expiry's average variance, so the market's variance to it is kept and only the vol shape
inside the first month is given up. This also removes a degenerate first grid row: at very short
maturities the fixed-strength spline of total variance dipped below zero on 25 of 36 stored days.
Vanillas and variance swaps price off the full surface.

**SVI wings past the quotes.** Every use of a fitted slice (vanilla marks, local vol, the
dashboard and site smiles) reads it through one function, which stores each slice's quoted
log-moneyness range (`k_min`, `k_max`) and checks the fitted wing. A raw-SVI wing is kept as fitted
only if its kink lies inside the quotes and its slopes respect Lee's moment bound,
$b(1+|\rho|) \le 2$. Otherwise, past the last quote, total variance continues in a straight line
from the edge, with the edge slope clipped to $[0, 2]$ (it never slopes back down). Inside the
quotes nothing changes, and the fitted parameters are identical. The cause was AAPL weeklies, quoted
only to about $k=-0.10$: on 2026-10-09 the 10-23 slice put its kink outside the quotes with a wing
slope of 17.7 (657% vol at $k=-0.34$), and the term-structure spline carried it to 426% vol for
P004, a short call 34% in the money a week from expiry. That overvalued it by $14.4K, with a delta
of -404 instead of -500. Out of sample (dropping each slice's outermost quotes and refitting, 2,887
slices), the guarded wing missed the dropped quotes by a median 2.30 vol points (mean 4.04, 2.2%
off by more than 20 points), against 2.32 (mean 6.95, 3.7%) for the raw wing. A flat vol past the
quotes (median 9.6) and forcing Lee's bound into the fit (mean 17.8) were measured and rejected.

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
depend on that day's own forward). Each name's grid moves are also modelled on its own published
vol index (8.3, part of 3.6).

**PCA of daily vol moves (Step 8.2, `vol/pca.py`).** Run per name over all stored days, on daily
changes in implied vol (vol points, $\sigma = \sqrt{w/T}$) between consecutive trading days only,
at grid nodes inside the quoted strikes of both bracketing expiries on at least 80% of days. Vol
points rather than $\Delta w$, because $\Delta w$ scales with $T$ and put almost all of PC1 on the 2y
row; quoted nodes only, because extrapolated nodes move with the SVI fit, not the market, and with
them SPX's PC2 sat on never-quoted upside strikes. On the history to 2026-10-30 (29-32 changes per
name) PC1-3 explain 99.4% (SPX, 13 nodes), 92.5% (AAPL, 15) and 95.5% (NVDA, 28). PC1 is a
same-sign level move, 3-4 times larger at 1m than at 1y, and stable across every 20-day window
(|cos| >= 0.94). PC2 and PC3 are not: PC2's lowest window match is 0.61-0.72 for SPX and AAPL, and
AAPL's PC3 and PC4 swap between windows. These components are a measured description of how the
surfaces move; VaR's vol model is the index regression in 3.6, whose betas have PC1's shape.

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

**Smile dynamics: the spot step uses sticky-moneyness Greeks (since 2026-10-21).** The spot step's
intermediate state rebuilds the vol surface around the new spot and forward from the same smile in
log-moneyness, so vol moves with moneyness (sticky moneyness). The conventional Greeks bump spot
with vol held fixed by strike (sticky strike). On a skewed underlying the two value a spot move
differently, and that difference used to land in the residual. Every option mark now carries both
pairs: `delta`/`gamma` (sticky strike, the conventional Black-Scholes figures for vanillas) and
`delta_sm`/`gamma_sm` (sticky moneyness: for vanillas a closed-form reprice at +-1% spot with the
forward and log-moneyness moving together; for the Monte Carlo exotics two extra repricings with the
local-vol grid's spot axis scaled by the bump). The explain uses the sticky-moneyness pair. A
revaluation ladder (repricing on a grid of spot and vol moves) was measured as the alternative and
rejected: on P008 over 23 day pairs it gave 21.0bp mean spot-step residual with a sticky-strike grid
and 7.3bp with a sticky-moneyness grid, against 7.1bp for sticky-moneyness Greeks, so the gain is
the convention, not the ladder.

### 3.6 VaR and Expected Shortfall (Steps 8.3 and 9, `risk/proxy.py`, `risk/var.py`)

**Scenarios.** Historical simulation: each scenario is one past trading day (or overlapping 10-day
period), applied instantaneously to the VaR date's market. Primary window 500 trading days, also
250 and 1,000. Spot: each name's own relative change. Rates and dividends: held at the VaR date's
values.

**Vol moves through a proxy (8.3).** Calibrated surfaces exist only for the days the pipeline has
run, not for the window. Each name's daily vol-point change at each grid node is regressed on the
daily change in its own vol index over the stored days up to the VaR date (no look-ahead): SPX on
VIX, AAPL on VXAPL, NVDA on VXN (Nasdaq-100; NVDA has no index). Nodes and changes follow 8.2's
rules (inside quoted strikes on 80% of days, consecutive trading days, vol points); at least 20
changes are required, otherwise VaR is not computed. In a scenario the index's relative change is
rescaled to the VaR date's level, multiplied by each node's beta, and added to that date's implied
vol at the node; between nodes the move is interpolated, beyond them held flat from the nearest
quoted node. The intercept is not applied (it is the sample's average daily drift). The median node
R² is 0.65 (SPX), 0.37 (AAPL) and 0.31 (NVDA) on the history to 2026-10-29, highest at short-dated
ATM and falling into the wings and long maturities.

**Applying the move.** As a total-variance multiplier $((\sigma + \Delta\sigma)/\sigma)^2$ per
node (vol floored at 1 point), through the same shock path as the stress tests: vanillas and the
variance swap re-read the shocked surface; the Monte Carlo exotics rescale their local-vol grid by
the multiplier (the approximation in Section 4, small for moves of a few vol points) and, under a
spot move, re-anchor it at the shocked spot (sticky moneyness, as the vanillas and the P&L explain).

**Two methods.** Full revaluation reprices every position; Monte Carlo exotics price-only at
32,768 paths, one seed for the base and every scenario. The Taylor method uses the day's marked
sticky-moneyness delta and gamma, vega, vanna and volga, with the vol move read ATM at each
position's `vol_times` (the P&L explain's convention). Carry is excluded from both. The variance
swap is marked daily as a fresh swap with zero P&L; in VaR it is struck at the VaR date's fair
strike, scenario P&L $N_{vega}(K_{new}^2 - K^2)/(2K)$.

**Measures.** With $n$ scenarios and confidence $c$, VaR is the $k$-th worst P&L,
$k = \lfloor n(1-c) \rfloor$, and ES the average of the $k$ worst. 10-day figures from overlapping
10-day moves, and as $\sqrt{10}$ times 1-day for comparison. ES at 97.5% on 10-day moves by factor
set (spot only, vol only, single-name vol only) and by the README's liquidity-horizon cascade (10
days for everything, 20 for single-name vol, 60 for the exotics' vol).

**Results (2026-10-29 book, 500 days).** 1-day 99% VaR $361,884 full, $269,142 Taylor; without the
variance swap $231,507 and $234,798. The Taylor shortfall is the swap: its Taylor P&L is 74% of its
full P&L on average, because ATM vega misses the larger downside move of the smile. 10-day 99% VaR
$752,244 from overlapping moves against $1,144,377 by $\sqrt{10}$ scaling. 10-day 97.5% ES
$711,649, $736,937 after the liquidity-horizon cascade. 500 full revaluations in 48.7 s.

---

## 4. Assumptions and limitations

This section is written to be read first, per the model's own operating principle: a limitations
section that says "the model works well" is worth less than one that says exactly where and why it
does not.

**The daily P&L explain still leaves a material residual on the Monte Carlo exotics, and its
history shows how easily a residual is misdiagnosed.** The clearest evidence in this project is the
incident documented in `docs/incidents/2026-09-01_p008_autocallable_vega_residual.md`: on the first
real day-over-day explain run, a $178k residual (on a ~$4.16M book) was almost all in P008, the
NVDA autocallable. It was first blamed on the note's vega-only Greek set; adding vanna and volga
appeared to help by about 13%. Both conclusions came from single Monte Carlo runs and did not
survive testing on six weeks of history. The residual actually came from (1) Monte Carlo noise in
the note's bump-and-reval Greeks (a single-run spread of about +-$116k on a big-move day; gamma,
vanna and volga essentially noise), (2) long-dated forwards and vol-surface coverage changing day
to day, (3) extrapolation past the last expiry that assumed zero future vol and carry,
(4) local-vol smoothing re-chosen at every grid point every day, (5) observation dates
simulated up to 5.5 days off, and (6) barriers measured against each day's spot instead of the
note's initial fixing (Section 2, position lifecycle), (7) Greeks that held vol fixed by strike
while the explain moved it with moneyness (Section 3.5), and (8) weekly expiries whose fitting
noise moved the note's local-vol model (Section 3.3). Each was measured and fixed (follow-up section
of the incident report). On the 16 comparable day pairs the mean absolute residual fell from 105bp
to 11bp of NAV. Fitting SPX forwards from any trade in the session (Section 3.1) then removed the
SPX options' artificial residuals: P003 and P007 fell from $758 and $334 to $87 and $39 a day on
average. Guarding SVI wings past the last quote (Section 3.3) then removed the near-expiry AAPL
call's: P004's summed absolute residual over 27 pairs fell from $37.6K to $2.0K. On the 16
comparable pairs the residual is now 9.5bp; over all 28 stored pairs it is 8.6bp (median 6.8bp,
worst 25bp), and the sum of each position's absolute residual is 10.4bp (1.9bp without P008).

What remains: the explain's vol step is still one vega times the ATM move at the position's
exposure dates. A parallel vol ladder, and vega by maturity bucket plus a skew sensitivity, were
measured and did not help, so they were not built. The largest remaining day (2026-09-28, 25bp) is
almost all P008; it was 9bp before the wing guard only because P004's mispricing offset it. Residuals carry a
Monte Carlo standard error, so a day is flagged only when the gap exceeds the 5bp threshold by more
than its noise.

**Several input-stability choices are judgment calls, tuned on this five-name universe.** The
local-vol term-structure smoothing strength is fixed (`T_SMOOTHING_LAM = 0.01`), chosen by
measuring local-vol Monte Carlo against Black-76 on all stored days. The exotics' local-vol grid
ignores expiries inside a month (`LOCAL_VOL_MIN_PILLAR_T`); 2 weeks and 18 days gave most of the
same gain, and longer cutoffs gained more partly by ignoring the short end, so the value is a
judgment that gives up the vol shape inside the first month. A long single-name expiry with
clean quotes but no forward fit of its own takes its forward from that day's forward curve, so its
implied vols inherit any error in the curve's long-end carry (roughly 1-2 vol points near the money
per 2% forward error at 2 years). An expiry that loses its quotes keeps the previous day's quotes,
at half weight, for up to 3 trading days, but only on a day the underlying has observed quotes of
its own: a day with none (a failed or after-hours snapshot) gets no surface rather than one rebuilt
from the previous day. All three are flagged in the stored data
(`implied_vols.forward_source`, `implied_vols.carried_from`, `vol_surface.n_carried`) and should be
revisited if the universe changes. The SPX forward fit's "traded during the session" window is the
same kind of choice: a 3-trading-day window reached further (3.2y) with slightly more scatter.

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

**VaR's vol moves are what each name's vol index explains, not the surface's full move.** The proxy
(3.6) leaves 35% (SPX) to roughly 65-70% (AAPL, NVDA) of node-level vol variance unexplained on the
stored days, and moves level, skew and term structure together by construction, so the ES
decomposition cannot split vol into them. Measured in dollars: replaying today's book on the 29
stored days with their real per-node vol changes and with the index-implied ones, the proxy misses
$79K a day RMS against a typical daily swing of $163K, nearly all on the SPX variance swap. The
proxy's behaviour on earnings days is untested: the only earnings move in the stored history
(NVDA, 2026-08-26) fell in a week with no stored surfaces. The betas rest on about 30 changes per
name and move as days are added.

**Other VaR assumptions.** Rates and dividends do not move in scenarios. The exotics' vol shock uses
the local-vol rescaling approximation (above), small for the moves of a few vol points typical
here but not zero. The 500-day window holds the April 2025 sell-off and rebound, which dominates
the tail: a 1,000-day window gives a lower 1-day VaR ($343K against $362K).

**No backtest exists.** Step 10 needs about 250 days of daily VaR against the P&L that followed;
the daily VaR record started on 2026-10-29. Conditional and reverse stress (11.3/11.4) need a
historical factor covariance estimate and are not built.

**The residual monitor is automated but simple.** Every daily run checks the total and
per-position residual against 5bp of NAV and logs ALERT lines (or INCONCLUSIVE when Monte Carlo
noise is wider than the threshold); a local AI agent then drafts an investigation. Nobody is paged:
alerts are read in the daily log and the dashboard.

**Two persistent, real data gaps, not bugs:** JPM and XLE have shown zero calibrated vol-surface
coverage on every real trading day tested so far (confirmed again via direct query as of
2026-09-04: 0 rows for either underlying across the full history, vs. 36/62/89 rows for
AAPL/NVDA/SPX respectively) — a genuine single-name strike-sparsity and staleness issue on
free-tier option-chain data, not a code defect, and not currently mitigated (that mitigation is
a vol-index proxy as in 3.6 could cover them, but is not set up for names without positions). Separately, no paid
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

**This is a single 9-live-position book, maintained by hand in a config file** (expiries settled
and rolled by the lifecycle rules above), not a live position feed, and the model has no
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

**Fix #1, safe and zero-modeling-risk:** each grid row's spline fits are a pure function of their
own inputs with no shared state, so `build_local_vol_grid` computes rows across a process pool
instead of sequentially — same math, same values, just parallelized. Result: `load_market_state`
111.7s -> 27.5s (~4x on an 8-core machine), book reval 126.28s -> 38.6s, full pipeline 404.4s ->
287.4s (newly meets the <5min target).

**Fix #2, a real algorithm change, researched and validated rather than assumed safe:** researched
what production Dupire-formula implementations actually do — analytic derivatives from the
calibrated smile's own closed form wherever possible — and found this module was independently
smoothing THREE quantities (`w`, `dk_w`, `dkk_w`) across time per grid point, when Dupire's formula
wants derivatives of ONE smooth total-variance surface. An initial idea (reuse one GCV-selected
smoothing parameter across nearby strikes) was tried, measured, and abandoned first: a from-scratch
reimplementation avoiding scipy's private internals turned out ~42x slower per reference point than
scipy's own private path, netting far less benefit than hoped. The fix that stuck: smooth only `w`
across T (one spline fit per grid point instead of three) and derive `dk_w`/`dkk_w` via finite
differences across neighboring grid points already being computed for the same row — validated,
not assumed, against the unchanged per-point reference across every real calibrated expiry for
SPX/AAPL/NVDA: for SPX the new method is consistently as accurate or *more* accurate at every
expiry (one 21-day expiry improved from 62 to 16.5 standard errors vs. the closed-form benchmark);
AAPL/NVDA are essentially unchanged. **Result: `load_market_state` 27.5s -> 12.1s (~2.3x further,
~9.2x combined from the original baseline), book reval 38.6s -> 23.4s.** Still misses the <3s
target (~7.8x over, down from ~42x originally) — a real, disclosed, not-fully-closed gap; the
full-pipeline number above reflects fix #1 only and was not re-measured after fix #2 (no real
trading day was available to re-run `make reproduce` when this was written), reported honestly as
partial rather than assumed.

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
| VaR, equity position | Full revaluation and Taylor P&L both equal quantity x spot x return to 1e-12 in every scenario |
| VaR measures | Property tests: ES >= VaR, VaR is one of the scenario losses, higher confidence never lowers VaR |
| VaR price-only Monte Carlo | Same price as the Greeks run to 1e-12 (same paths); seed-to-seed sd of P008 scenario P&L $0.4-3.8K on $30-530K moves |
| Proxy regression | Recovers known betas with R² = 1 on noiseless synthetic history |
| P&L explain telescoping | Total actual P&L across all 4 steps matches true full-reval P&L to $10^{-9}$ relative |
| P&L explain on equity/varswap positions | Exactly zero residual in every step (linear payoff for equity; marked to 0 by construction for varswap) — an exact, not approximate, design validation |
| P&L explain, real multi-day | 4 real consecutive day-pairs (2026-08-31 through 2026-09-04): residuals of +$151.7k, -$252.4k, +$349.0k, -$170.7k — genuinely oscillating in sign, not a cherry-picked single "good" day |
| Full test suite | 1,949 unit/property/integration tests passing (2026-11-01); ruff and mypy clean throughout |

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
| P&L-explain residual | **Medium** — 8.6bp of NAV mean absolute residual over 28 real day pairs (median 6.8bp, worst 25bp), still above the 5bp threshold on average; almost all of what remains is P008 | Eight input and model causes found and fixed (Section 4), plus the SPX long-end forwards and the SVI wing guard; bucketed vega and skew measured and rejected |
| Vol-shock stress approximation error | **Medium**, grows with shock size (~7-40% measured) | Documented and bounded; not eliminated; matters most for the largest stress-ladder shocks |
| Residual monitoring alerts are logged, not paged | **Low** — daily alerts are visible in the log and dashboard but nobody is notified | Automated 5bp check with a noise band since 2026-09-10 / 2026-10; paging not built |
| Two single names (JPM, XLE) have no real coverage | **Low** for this book (no positions assigned there), but blocks any future position in those names | Not currently mitigated; the 8.3 index proxy could cover them if given an index |
| VaR vol moves are an index proxy; no backtest yet | **High** for any real risk-limit use case: single-name proxy R² ~0.3, VaR record a few days long | Proxy error measured in dollars daily (`proxy_errors`); full revaluation alongside Taylor; backtest pending ~250 days of VaR |
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
4. **VaR exception tracking (Kupiec/Christoffersen/traffic-light/PLA)** — *not implemented*: the
   daily VaR series exists since 2026-10-29 (`var_summary`), the tests need ~250 days of it.
5. **Real-history depth tracking** — a simple count of real trading days accumulated so far,
   since several steps' eventual activation (11.3/11.4 at tens-to-hundreds of days, 10 at ~250
   days of VaR) is gated purely on this growing. *Implemented*: the dashboard's Backtest tab
   reports the count of stored VaR days rather than a generic "not available" message.
6. **Automation health** — whether the daily `launchd` job actually ran and succeeded.
   *Implemented*: per-day log files (`logs/daily_ingest_*.log`), checked manually; no automated
   failure alert exists yet.
