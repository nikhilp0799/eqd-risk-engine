from eqdrisk.io.runclock import mark, sleep_report, slept_seconds

START = "1000.000 500.000"


def test_the_2026_11_10_run_reports_its_sleep():
    """16:42:25 -> 19:25:51 on the wall clock, about 12 minutes awake."""
    line = sleep_report(START, wall=1000.0 + 9_806, mono=500.0 + 720)
    assert line is not None
    assert line.startswith("the Mac slept 2h31m of the 2h43m since the run started")


def test_an_awake_run_reports_nothing():
    assert sleep_report(START, wall=1000.0 + 600, mono=500.0 + 590) is None
    assert slept_seconds(START, wall=1000.0 + 600, mono=500.0 + 590) == 10.0


def test_a_restart_in_between_is_said_not_guessed():
    assert slept_seconds(START, wall=1000.0 + 600, mono=20.0) is None
    assert "restarted" in (sleep_report(START, wall=1000.0 + 600, mono=20.0) or "")


def test_mark_round_trips():
    assert slept_seconds(mark()) is not None
