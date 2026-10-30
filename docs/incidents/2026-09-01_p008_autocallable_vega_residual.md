# Incident: P&L Explain Residual Breach — P008 (NVDA autocallable), 2026-08-31 -> 2026-09-01

> **Correction (2026-10-14, updated 2026-10-27).** The root cause below (missing vanna and volga)
> was wrong. A follow-up investigation found the residual came from Monte Carlo noise in the note's
> Greeks, unstable long-dated market inputs, local-vol smoothing noise, mistimed observation dates,
> and weekly expiries moving the note's local-vol model; adding vanna and volga added noise rather
> than information. On the same day pair the residual is now -$4,488 (-10.7bp of NAV, 2 SE +-5bp),
> down from +$178,461. The original report is kept below
> unchanged as the record of what was believed at the time; see
> [Follow-up investigation](#follow-up-investigation-2026-10-05-to-2026-10-14) at the end.

## Summary

The first real consecutive-trading-day run of Step 12's daily P&L explain (day-over-day residual
decomposition) found an unexplained residual of +$178,461 on a ~$4.16M book (~4.3% of NAV, far
above any sane single-digit-bp threshold). 99.6% of it (+$177,775) traced to one position: P008,
a 2-year NVDA autocallable note. Root cause: the position's Monte Carlo Greek set (delta, gamma,
vega only) does not include vanna or volga, so the P&L explain waterfall's vol step badly
overstated the position's true sensitivity to the day's implied-vol move. Fixed by extending the
autocallable (and, for the same reason, barrier) Monte Carlo pricers to also compute vanna and
volga via bump-and-reval under common random numbers.

## Detection

No automated residual monitor exists in this project yet (see Lessons). This was caught by manual
review of Step 12's `explainpnl` output for the first real day pair the automated data pipeline
produced.

## Impact

$178,461 unexplained on a $4.16M book (~43bp of NAV). No trading decisions were taken on the
affected numbers — this is a research/demonstration project, not a live trading desk — but the
size of the residual would fail any reasonable P&L-explain acceptance threshold (the README's own
target for this step is a median residual under 2bp once enough history accumulates).

## Investigation

1. Residual decomposed by position (`by_position_residual` in `PnLExplainResult`) -> 99.6%
   attributable to P008 alone; every other position's residual was under $500.
2. Residual decomposed by waterfall step for P008 -> almost entirely in the `vol` step: explained
   -$220,644 (a large predicted loss from vega) vs. actual -$28,744 (a much smaller real move).
   Vega alone overstated the true vol sensitivity by roughly 7-8x.
3. Reviewed the Greek set actually available for this position type. `PositionMark` (the
   portfolio-marking data model) has always had `vanna` and `volga` fields, but
   `autocallable_greeks` (`src/eqdrisk/pricing/autocallable.py`) only ever populated
   delta/gamma/vega — a limitation documented since Step 7 as a known scope gap for
   Monte-Carlo-priced (path-dependent) instruments, not a new discovery.
4. This is exactly the failure mode the README's own Step 6 discussion predicted for
   autocallables ("genuinely skew-sensitive, large vanna/volga") and that Step 12's own docstring
   predicted in general ("if your residual spikes, your Greeks are incomplete"). The documented
   gap showed up, on the first real day it was ever tested against, exactly where predicted.

## Root cause

`autocallable_greeks` and `down_and_in_put_greeks` (the barrier equivalent, sharing the identical
architecture and the identical gap) compute delta/gamma/vega via bump-and-reval Monte Carlo under
common random numbers, but never computed the cross (vanna) or second-order vol (volga) terms —
each would have needed additional full MC repricing passes that Step 6.4/6.5 did not build at the
time. For a payoff with hard barriers (autocall, coupon, put), the position's effective vol
exposure itself changes sharply with both spot level and vol level near those barriers — exactly
what vanna and volga are for. A vega-only linear-in-vol approximation cannot see this, so the
Greeks-based "explained" P&L for the vol step used a Greek set that was structurally too thin for
this instrument type.

## Remediation

- `autocallable_greeks` and `down_and_in_put_greeks` now also return `vanna` and `volga`, computed
  under the same CRN seed as the existing delta/gamma/vega: volga from a central second difference
  in vol (free once vega is computed as a central difference too, rather than the original forward
  difference), vanna from the four cross-bumped (spot, vol) corner prices. Total MC cost per
  position rose from 4 full reprices to 9 (delta/gamma/vega/volga's five single-variable prices,
  plus vanna's four cross corners) — acceptable for a once-a-day batch mark, not something that
  needs to be fast.
- `portfolio/mark.py`'s `_mark_barrier` and `_mark_autocall` now pass these through into
  `PositionMark.vanna` / `.volga` (fields that already existed on the dataclass and were already
  reported in the aggregate risk output, but silently read as 0.0 for every MC-priced position
  until now — a second, quieter instance of the same gap, in the aggregate Greek report rather
  than P&L explain).
- Barrier options were fixed alongside autocallables even though no barrier position was the
  actual incident today (P007's residual was $419, unremarkable) — the architectural gap is
  identical, and it would be dishonest to fix only the instrument that happened to get caught.

## Validation

Re-ran the same real day pair (2026-08-31 -> 2026-09-01) after the fix:

| step | actual | explained (before) | explained (after) | residual (before) | residual (after) |
|---|---|---|---|---|---|
| vol | -28,751 | -220,644 | -195,329 | +191,900 | +166,578 |
| TOTAL | -59,655 | -236,690 | -211,360 | +178,461 | +151,705 |

The vol step's explained P&L moved from -$220,644 to -$195,329 — a real ~11% reduction in how much
vega+volga overstates the move, narrowing that step's own residual by about 13%. P008's share of
the total residual is still 99.5% ($151,021 of $151,705).

**Note on the totals:** the actual P&L figures shifted slightly between the "before" and "after"
runs (e.g. `rates_divs` actual went from exactly $0.00 to -$1,400) for a reason unrelated to this
fix: the "before" run was executed on 2026-09-01 itself, before that day's own discount curve had
finished being ingested by the automated pipeline (16:30 ET), so `load_market_state` fell back to
2026-08-31's curve for both days, making the `rates_divs` step trivially zero. The "after" run,
executed a day later, picked up 2026-09-01's own real curve (confirmed via
`data/curated/discount_curves/asof_date=2026-09-01`, several bp different from 2026-08-31 across
every tenor) — a genuine improvement in data completeness, not an artifact of the code change. The
`vol` step comparison above is unaffected by this, since both the "before" and "after" runs use a
single, internally consistent curve on both sides of that step.

**Honest conclusion:** the fix is real and directionally correct, but it does not close the gap.
A $151K residual concentrated in one position is still far outside any acceptable threshold. The
deeper issue is that a barrier-laden payoff's vol sensitivity is not well approximated by a
second-order Taylor expansion around a single ATM vol scalar at all — the autocall/coupon/put
barriers create genuine kinks in price-vs-vol space that no smooth polynomial in one variable can
capture, only a name for it (vanna, volga) rather than a full fix. Closing this properly would need
either a full-reprice ("what-if this step's actual vol surface change") term for MC-priced
positions in the waterfall, rather than a Greeks-based approximation, or accepting that MC
positions' vol-step residual is a structurally different quantity than vanilla positions' and
should be reported (and thresholded) separately.

## Lessons

1. A documented scope limitation is not the same as an acceptable one once real data actually
   exercises it — this is the first time this project's daily pipeline produced two real,
   consecutive, fully-calibrated days, and the very first real P&L explain run found the exact gap
   the README had predicted years (in project time) earlier.
2. Fixing "the" incident (autocallable) without also fixing the structurally identical gap in an
   adjacent instrument type (barrier) would have been a superficial fix — the underlying lesson is
   about the MC-Greeks architecture, not about NVDA or P008 specifically.
3. No automated residual monitor exists yet; this was caught by a human reading `explainpnl`
   output. A genuine next step (not yet built) would be a real threshold check, e.g. flagging
   any day where a single position's residual exceeds some fraction of book NAV, appended to the
   daily automation pipeline (`scripts/daily_ingest.sh`) rather than a hypothetical description of
   one.
4. Comparing two runs of the same historical day-pair analysis, executed on different calendar
   days, is not automatically apples-to-apples if the target period's own data was still being
   ingested at the time of the first run — worth remembering for any future day-pair replay done
   same-day rather than the day after.

## Follow-up investigation (2026-10-05 to 2026-10-14)

Six weeks of daily history made it possible to test the original diagnosis instead of reasoning
about it. It did not hold up. Seven separate causes were found; each was measured on the full
stored history before being fixed, and the whole history was rebuilt from the stored raw option
chains after each fix. Planning record: `planning/pnl_explain_noise_plan.md` and
`planning/input_stability_plan.md` (local, not in the repo).

### What was actually wrong

1. **Monte Carlo noise in P008's Greeks.** Repeating the 2026-09-03 explain with 10 different
   random seeds showed a single-run spread of about +-$116K on that day, almost all in the vol
   step. The note's Greeks came from 1% spot and 1 vol-point bumps of a payoff with hard barriers:
   day-0 gamma varied by ~105% across seeds, vanna by 250-5,000%, volga by 330-2,300%. Adding vanna
   and volga (the original remediation) therefore added noise, and the "~13% reduction" measured
   above was a single-seed result. Fixes: a Monte Carlo standard error on every residual (16
   independently scrambled quasi-random sub-runs), an alert only when the residual exceeds the
   threshold by more than its noise, and better-conditioned Greeks for Monte Carlo exotics (5%
   spot / 3 vol-point bumps, theta from a one-day revaluation). Vanna is deliberately not used:
   the vol step reprices after the spot move, so a vanna term would double-count.
2. **Long-dated inputs flickering day to day.** NVDA's vol surface ended anywhere between 0.24y
   and 1.29y depending on the day (P008 matures at 1.88y), so the note was priced on an
   extrapolated long end that changed for no market reason. Causes: put-call-parity forwards for
   American single-name options failed a discount-factor test at long maturities; the pricing
   forward curve used forwards the surface had rejected; extrapolation past the last expiry
   assumed zero future volatility and zero carry; and most long expiries had too few strikes
   with both legs tradeable to fit a forward at all. Fixes: single-name forwards fitted with the
   rates curve's discount factor, one reliability rule for surface and pricing, flat implied vol
   and continued carry past the last expiry, yesterday's quotes carried for up to 3 trading days
   when an expiry disappears, and forwards from the day's forward curve for single-name expiries
   with clean quotes but no fit. NVDA's surface now reaches 2.2-2.3y on every stored day.
3. **Local-vol smoothing noise.** The smoothing spline behind the Dupire local-vol surface chose
   its smoothness automatically (GCV) at every grid point, every day: from 3.7e-6 to 19 across a
   single day's grid. Pricing 1.88y vanillas through the local-vol Monte Carlo, the error against
   Black-76 on the implied surface moved 1.7-2.4 vol points a day (2-3.5x the market's own moves)
   and reversed the next day. A fixed smoothness cut that to 0.26-0.36 vol points.
4. **Observation dates off by up to 5.5 days.** The autocallable was simulated on an evenly spaced
   grid with each observation snapped to the nearest point, which is only exact when observations
   are evenly spaced from today. For a seasoned note the error drifted daily and jumped by about
   10 days on some nights (P008's theta read $14,658 instead of about $1,000 on 2026-10-09). The
   simulation grid now contains every observation date exactly.
5. **Lost data.** SPX chains for 2026-09-09 and 2026-09-14 had been overwritten by after-hours
   re-runs of the pipeline. Ingest now keeps an existing snapshot unless forced.
6. **The note was re-issued every day (found 2026-10-20).** The autocallable's barriers were
   measured against each day's spot instead of the note's own initial fixing, so the model priced
   a fresh at-the-money note every day, hiding most of its real NVDA exposure (delta about 2,700
   shares instead of about 13,400 on 2026-10-19). Now priced against its 2026-08-14 fixing
   (NVDA $225.16), together with a position lifecycle (expiry settlement, barrier knock-in,
   autocall observations). This changed P008's value and risk far more than this day's residual.
7. **Greeks and the explain assumed different smile dynamics (found 2026-10-21).** The explain's
   spot step rebuilds the vol surface around the new spot from the same smile in log-moneyness
   (vol moves with moneyness), but the Greeks bumped spot with vol fixed by strike. On NVDA's
   skew the two value a 3% move differently. Measured on P008 over 23 day pairs: a revaluation
   ladder did not help (spot-step residual 19.3bp with Greeks, 21.0bp with a ladder); Greeks
   computed with the vol surface moving with spot cut it to 7.1bp. Every option now reports
   both deltas, and the explain uses the sticky-moneyness pair.
8. **Weekly options moved the note's local-vol model (found 2026-10-24).** With causes 1-7 fixed,
   the vol step was the remaining gap, and vega by maturity bucket plus a skew sensitivity did not
   help (25.8bp and 23.8bp against 23.0bp for P008). An exact attribution of the vol step by
   maturity band (day 0's surface with day 1's expiries swapped in band by band, full reprice) put
   almost all of the three largest days on expiries under 18 days: -80 of -107bp on 2026-09-01,
   +93 of +72bp on 09-04, -70 of -115bp on 09-08. NVDA's weekly expiries are fitted with 5-15
   vol-point errors, and their swings flowed into the first rows of the local-vol grid, moving a
   note whose first observation was two months away by $30-50K. Separately, the fixed-strength
   spline of total variance dipped below zero at very short maturities on 25 of 36 stored days,
   leaving the grid's first row degenerate. The exotics' local-vol grid is now built from expiries
   at least a month out (flat local vol below the first kept expiry, at its average variance);
   vanillas keep every expiry. Measured on all 24 day pairs before adopting it, P008's mean
   residual fell from 25.1bp to 12.9 / 12.0 / 9.2bp with a 2-week / 18-day / 1-month cutoff, so the
   gain does not depend on the exact value; 1 month was chosen (longer cutoffs gained more, partly
   by ignoring the short end, which was not trusted). P007, the SPX barrier, was unaffected.

### Numbers

This day pair (2026-08-31 -> 2026-09-01):

| Stage | Total residual |
|---|---|
| Original report (single run, 1%/1pt Greeks) | +$178,461 (about 429bp of NAV) |
| After the vanna/volga change | +$151,705 |
| Noise-aware residual and better Greeks | -$43,927 (-104bp) |
| Causes 1-7 fixed | -$39,201 (-94bp, 2 SE +-7bp): spot step +$3.2K, vol step -$40.9K |
| Cause 8 fixed | -$4,085 (-9.7bp, 2 SE +-5bp): spot step +$2.5K, vol step -$5.1K |
| SPX forwards fitted from the whole session | -$4,488 (-10.7bp, 2 SE +-5bp): P008 unchanged at -$4,463; the SPX options' small residuals that partly offset it moved |
| SVI wings guarded past the last quote (current) | **-$4,834 (-11.5bp, 2 SE +-4.7bp)**: P008 -$4,835, its local-vol grid now built from guarded wings |

On the 16 day pairs explained both before and after the input and model fixes, the mean absolute
total residual fell from 104.9bp to 9.5bp. Across all 28 stored day pairs it is 8.6bp, with a
median day of 6.8bp; the worst is 25bp (2026-09-28), almost all P008. The previous worst day
(2026-10-09, 38bp) was an AAPL call a week from expiry valued at 426% vol off an SVI wing that shot
upward past the last quote; guarding those wings (model documentation, Section 3.3) took that day
to 4.2bp. An
earlier version of this report also said P008's typical daily price move fell from $54.6K to $9.4K;
that figure was an artefact of cause 6 (a note re-struck at the money every day has little delta)
and is withdrawn: priced correctly, the note's value moves about $38K on a typical day, as a $5M
note with about 13,400 shares of NVDA delta should.

### What remains, honestly

This day pair is now about two standard errors from the 5bp threshold, and on full revaluation the
day's vol-surface change cost the book $9.2K, not the $44.8K it showed before cause 8: most of what
looked like a vol move was the weekly expiries moving the model. Across history the largest
remaining day (2026-09-28, 25bp) is almost all P008. The 1-month cutoff gives up the vol shape inside the first month for the
Monte Carlo exotics; a note within a month of an observation date is priced with flat local vol up
to the first kept expiry.

### Corrected lessons

1. Test a diagnosis against noise before acting on it: one run of a Monte Carlo explain is one
   draw, and this one was a draw from a distribution about +-$100K wide.
2. A residual can be produced by the inputs and the model as easily as by missing Greeks. Most of
   this one came from our own calibration (forwards, extrapolation, smoothing) and simulation
   (observation dates) changing day to day, not from the market.
3. The Impact section above understates the size by a factor of ten ("~43bp"); $178K on a
   $4.16M book is about 429bp.
4. When Greeks cannot explain a move, attribute the move exactly before adding more Greeks.
   Vega by maturity and a skew sensitivity were measured to add nothing (cause 8); repricing with
   one maturity band of the surface changed at a time found the cause in one run.
