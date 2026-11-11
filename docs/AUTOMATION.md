# Daily data-pull automation

The whole pipeline (`ingest` → `curves` → `iv` → `calibrate` per underlying → `price` → `varswap` →
`riskfactors` → `portfolio` → `explainpnl`) used to be run by hand — which is exactly why real
calibrated history stopped accumulating for 4+ days at a stretch (see `planning/decisions.md`,
2026-08-25/28). This is now automated via a macOS `launchd` agent so it runs whether or not anyone
remembers to trigger it. `portfolio` and `explainpnl` (day-over-day, against the previous trading
day via `last_n_trading_days`) were added on 2026-09-05 alongside Step 14's dashboard — without
them, `portfolio_marks` and `pnl_explain` went stale even though everything upstream kept
accumulating (a real gap the dashboard's Exposure and P&L-explain tabs would otherwise have shown
as stale data). A real residual-alert threshold check was added on 2026-09-10 (Step 13's incident
report had flagged the lack of one): `explainpnl`'s own output now includes `ALERT:` lines whenever
the total or any single position's residual exceeds 5bp of NAV (README 12.3's own suggested
number), and `daily_ingest.sh` greps for that marker after each run, logging a distinct
`RESIDUAL ALERT` line if found — without changing the script's exit code, since a breach is a real
finding to review, not a crash.

**Expiries and observations are applied automatically (since 2026-10-20).** `eqdrisk portfolio`
settles expired options at their expiry close, applies barrier knock-ins and autocall observations
from stored closes (`portfolio/lifecycle.py`), and prints `UPCOMING:` lines for any expiry or
autocall observation in the next 7 days. If a due fixing is missing from the stored closes, the
position is skipped with a reason rather than settled on a guess. Rolls are added by hand to
`configs/portfolio.yaml` with an `active_from` date.

**A frozen feed is re-fetched, not stored (since 2026-10-31).** On 2026-10-30 the 16:30 pull got
an SPX chain whose latest trade was 14:59 ET (16:15 on a normal day), so every quote failed the
30-minute staleness filter and the day had no SPX surface or P&L explain. A weekend re-pull could
not rebuild it: it had Friday's trades but only 984 of 17,337 quotes still carried a bid. `ingest`
now checks each chain's latest trade against the same staleness reference `iv` uses; if no quote
could pass, it logs a retry and re-fetches after 1, 5 and 10 minutes. If the feed is still frozen
the raw payload is kept but the curated chain is not written, so a later re-run the same evening
(without `--force`) fetches that name again.

**VaR runs last, every day (since 2026-11-01).** After the AI investigation, `eqdrisk var` computes
the day's VaR and ES (README Step 9, `risk/var.py`): about 3,000 scenario revaluations across a
process pool, a few minutes. It needs the day's SPX, AAPL and NVDA surfaces and at least 20 stored
consecutive-day changes per name for the vol proxy, and refuses (logged as `SKIPPED _all_`) rather
than computing a VaR that leaves a name's vol risk out. It fetches spot history from yfinance and
vol-index history from FRED; FRED posts a close the next day, so the same evening's VIX, VXAPL and
VXN close comes from yfinance. Each day's stored VaR is the record the Step 10 backtest will test
against the P&L that followed. `eqdrisk varbackfill --start X --end Y` fills days without one (1-day
VaR only) and leaves days that have one alone.

**Re-running a past day is safe for the option chains (since 2026-10-14).** `eqdrisk ingest` keeps
an existing chain snapshot for a date and fetches only the underlyings that are missing; pass
`--force` to replace one on purpose. Before this, an after-hours re-run of the pipeline overwrote
the real 16:30 SPX snapshots of 2026-09-09 and 2026-09-14 with stale after-hours quotes, and the
originals could not be recovered (both days now mark only the non-SPX positions).
`eqdrisk missingchains --date X` prints the universe names with no chain snapshot for that day
(2026-11-04 prints `AAPL NVDA`, both lost to a Yahoo outage), so you can see what a re-run would
fetch before running it.

**A missing chain is retried until its cutoff (since 2026-11-07).** Yahoo outages lost chains on
2026-10-23 (SPX, AAPL), 10-30 (SPX) and 11-04 (AAPL, NVDA), each lasting longer than the ingest's
own retries. After `ingest`, `eqdrisk catchup` re-runs the ingest for the missing names only: SPX
every 10 minutes with a last try just before 17:00, single names until 21:00 (every 30 minutes
once SPX is done). The rest of the pipeline waits for it, so an outage day finishes late but
complete. The cutoffs (`chain_catch_up` in `configs/base.yaml`) come from sampling chains after
the close on 2026-11-06 and scoring them against the 16:30 snapshot: SPX had 3,206 quotes passing
the filters at 17:00 (3,240 at 16:30, mids about 1% apart) and 1 at 17:30, with the bids pulled;
AAPL and NVDA quotes were unchanged to 21:00. The wait checks the wall clock every 30 seconds, so
a Mac that sleeps through a cutoff skips that name rather than fetching it late. A name still
missing at the end is logged as `*** CHAINS MISSING ***`. On a past date every cutoff has passed,
so `catchup` fetches nothing.

## What's set up

- **`scripts/daily_ingest.sh`** — runs the full daily pipeline for "today," skipping outright (not
  silently corrupting anything) on non-trading days via `marketdata.calendar.is_trading_day`.
  Logs to `logs/daily_ingest_YYYY-MM-DD.log` (gitignored — these are local run records, not
  project deliverables).
- **`~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist`** — a launchd agent that runs the script
  Mon-Fri at 16:30 local time (30 minutes after the 16:00 ET close, matching `canonical_snap_time`
  in `configs/base.yaml`). Loaded via `launchctl load ~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist`.
  `snap_tolerance_minutes` is 45 so this planned 30-minute lag does not raise the capture-time
  alert every day (it did at 15); the alert still fires for genuinely late runs.
  launchd's own behavior: if the Mac is asleep at 16:30, it runs the job as soon as the machine
  wakes up next (not silently skipped, unlike plain cron) — but it still won't run if the Mac was
  fully shut down the whole time.

## The one manual step: auto-wake

To have the job actually fire near 16:30 even with the lid closed (not just "whenever you next open
it"), run this once yourself (needs `sudo`, so not something to run on your behalf without you
present):

```
sudo pmset repeat wakeorpoweron MTWRF 16:25:00
```

This tells macOS to wake the machine (from sleep; won't power on from a full shutdown) at 16:25
Monday-Friday, five minutes before the launchd job fires. **Note:** `pmset repeat` only supports one
repeating schedule at a time — running this replaces any other repeating wake/sleep schedule already
set on this machine. Check first with `pmset -g sched` if you're not sure whether one already exists.

**With the lid closed on battery this is not enough (found 2026-11-10).** The wake fired at
16:25:00 and macOS went back to sleep 6 seconds later; with the lid closed it only allows short
maintenance wakes (2-20 seconds every 10-17 minutes, `pmset -g log`). The run started in one at
16:42 and moved only in them: one SPX request "timed out after 622,623 ms" against a 30-second
timeout, the ingest ended past SPX's 17:00 cutoff, and the day had no SPX. `caffeinate` cannot
hold a closed laptop on battery either. Keep the lid open, or the Mac on power, from 16:25 to
17:00 on trading days. The run log now says when this happens: `*** ASLEEP: the Mac slept 2h31m
of the 2h43m since the run started ... ***`, from the gap between the wall clock and the monotonic
clock, which stops during sleep (`src/eqdrisk/io/runclock.py`). A snapshot from GitHub's servers,
which do not sleep, is being tested (`.github/workflows/cloud_snapshot.yml`, run by hand for now).

## Checking on it

- `launchctl list | grep eqdrisk` — confirms the agent is loaded (exit status `0` in the listing
  means the last run succeeded, or it just hasn't run yet).
- `tail logs/daily_ingest_$(date +%Y-%m-%d).log` — the day's run log.
- `cat logs/launchd_stderr.log` — launchd-level failures (e.g. the script itself failing to start),
  separate from the pipeline's own per-step logging inside the daily log.

## Undoing it

```
launchctl unload ~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist
rm ~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist
sudo pmset repeat cancel   # only if you want to remove the auto-wake schedule too
```
