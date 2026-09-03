# Incident: P&L Explain Residual Breach — P008 (NVDA autocallable), 2026-08-31 -> 2026-09-01

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
