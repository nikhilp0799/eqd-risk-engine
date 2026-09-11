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

## What's set up

- **`scripts/daily_ingest.sh`** — runs the full daily pipeline for "today," skipping outright (not
  silently corrupting anything) on non-trading days via `marketdata.calendar.is_trading_day`.
  Logs to `logs/daily_ingest_YYYY-MM-DD.log` (gitignored — these are local run records, not
  project deliverables).
- **`~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist`** — a launchd agent that runs the script
  Mon-Fri at 16:30 local time (30 minutes after the 16:00 ET close, matching `canonical_snap_time`
  in `configs/base.yaml`). Loaded via `launchctl load ~/Library/LaunchAgents/com.eqdrisk.daily-ingest.plist`.
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
