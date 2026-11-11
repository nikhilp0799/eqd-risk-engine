"""How long the Mac slept during a daily run (planning/sleep_safe_snapshot_plan.md).

On 2026-11-10 the run started at 16:42 in a few-second maintenance wake of a
sleeping Mac (lid closed, on battery) and moved only in such wakes: one SPX
chain request "timed out after 622,623 ms" against a 30 s timeout, the ingest
ended past SPX's 17:00 catch-up cutoff, and the log said only "past the retry
cutoff". The cause shows up as the gap between the wall clock and
`time.monotonic()`: on macOS the latter is mach_absolute_time, system-wide and
stopped while asleep, so a mark taken at the start of the script can be
compared from any later process.
"""

from __future__ import annotations

import time

# A minute of drift is not sleep worth reporting.
REPORT_FROM_SECONDS = 60.0


def mark() -> str:
    """Wall clock and monotonic clock now, as the script stores them."""
    return f"{time.time():.3f} {time.monotonic():.3f}"


def slept_seconds(since: str, wall: float | None = None, mono: float | None = None) -> float | None:
    """Seconds asleep since `since` (a `mark()`); None if the monotonic clock went
    backwards, i.e. the Mac restarted in between."""
    wall0, mono0 = (float(x) for x in since.split())
    wall = time.time() if wall is None else wall
    mono = time.monotonic() if mono is None else mono
    # A restart sets the monotonic clock back to near zero; the second's slack
    # covers `mark()` rounding a reading up.
    if mono < mono0 - 1.0:
        return None
    return max(0.0, (wall - wall0) - (mono - mono0))


def sleep_report(since: str, wall: float | None = None, mono: float | None = None) -> str | None:
    """A log line when the Mac slept at least `REPORT_FROM_SECONDS` since `since`."""
    slept = slept_seconds(since, wall, mono)
    if slept is None:
        return "the Mac restarted during this run: time asleep unknown"
    if slept < REPORT_FROM_SECONDS:
        return None
    wall = time.time() if wall is None else wall
    elapsed = wall - float(since.split()[0])
    return (
        f"the Mac slept {_duration(slept)} of the {_duration(elapsed)} since the run started: "
        "fetches made in its short wakes time out and miss their cutoffs (lid open or on power "
        "from 16:25 to 17:00 avoids this)"
    )


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"
