import datetime as dt
import json
from unittest.mock import patch

import pandas as pd
import pytest

from eqdrisk.agent import benchmark
from eqdrisk.agent.benchmark import (
    ALERT,
    CLEAN,
    INCONCLUSIVE,
    ToolCache,
    answer_key,
    data_turn,
    load_items,
    load_results,
    model_key,
    run_benchmark,
    score,
    summarize,
)
from eqdrisk.agent.factcheck import FactCheckResult
from eqdrisk.agent.investigate import (
    PROMPT_VERSION,
    AIInvestigationResult,
    ToolCallTrace,
    _build_messages,
    closing_line,
)
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    AI_INVESTIGATION_REQUIRED_NOT_NULL,
    AI_INVESTIGATION_SCHEMA,
    validate,
)
from eqdrisk.pricing.pnl_explain import PnLExplainResult, StepResult

DAY0 = dt.date(2026, 11, 6)
DAY1 = dt.date(2026, 11, 9)


def _explain(breaches=(), inconclusive=()) -> PnLExplainResult:
    """Shaped like 2026-11-09: P008 carries the residual, spread over steps."""
    return PnLExplainResult(
        day0=DAY0,
        day1=DAY1,
        steps=[
            StepResult(step="time", actual_pnl=3_741.50, explained_pnl=2_378.52),
            StepResult(step="spot", actual_pnl=-92_293.84, explained_pnl=-110_580.42),
            StepResult(step="vol", actual_pnl=-43_334.01, explained_pnl=-30_194.58),
        ],
        by_position_residual={"P008": 6_979.95, "P003": -392.14, "P006": 0.0},
        by_position_residual_se={"P008": 1_293.26},
        total_residual_se=1_294.80,
        nav=3_654_285.66,
        breaches=list(breaches),
        inconclusive=list(inconclusive),
    )


def _user_turn(result: PnLExplainResult) -> str:
    """The exact user turn the daily run builds (and stores in its prompt)."""
    return _build_messages(result, "  no history")[1]["content"]


def _answer(summary: str, root: str = "", flagged=(), trace=(), status="verified"):
    inv = AIInvestigationResult(
        day0=DAY0,
        day1=DAY1,
        model="m",
        ai_available=True,
        summary=summary,
        root_cause_hypothesis=root,
        flagged_positions=list(flagged),
        trace=list(trace),
    )
    inv.factcheck = FactCheckResult(n_numbers=1, unsupported=[] if status == "verified" else ["x"])
    return inv


def test_answer_key_reads_the_explain_the_model_was_given():
    key = answer_key(_user_turn(_explain(breaches=["total residual exceeds"])))
    assert key.status == ALERT
    assert key.top_position == "P008"
    assert key.total_residual == pytest.approx(6_510.13, abs=0.01)
    assert key.total_residual_bp == pytest.approx(17.8)
    assert key.position_residual_bp["P008"] == pytest.approx(19.1)
    assert not key.noise_relevant  # 17.8bp against a 7.1bp band


def test_answer_key_status_follows_the_explain_lines():
    assert answer_key(_user_turn(_explain())).status == CLEAN
    inconclusive = answer_key(_user_turn(_explain(inconclusive=["P008 band too wide"])))
    assert inconclusive.status == INCONCLUSIVE and inconclusive.noise_relevant


def test_a_step_residual_reported_as_the_book_total_is_not_complete():
    """The real 2026-11-09 qwen2.5:7b answer: it passed the fact-check, because
    +1,362.98 is a residual in the data, but that is the time step's, not the
    book's (+6,510.13), so the day's key figure was never stated."""
    key = answer_key(_user_turn(_explain(breaches=["total"])))
    reprice = ToolCallTrace(1, "what_if_reprice", json.dumps({"position_id": "P008"}), "{}")
    wrong = _answer(
        "a large residual of +1362.98 for the book, with P008 contributing +6,979.95",
        flagged=[("P008", "largest residual")],
        trace=[reprice],
    )
    scores = score(wrong, key)
    assert scores["grounded"] and scores["status_ok"] and scores["top_named"]
    assert scores["called_tool"]
    assert scores["complete"] is False

    right = _answer("the book's residual is +17.8bp of NAV, P008 +6,979.95", flagged=[("P008", "")])
    assert score(right, key)["complete"] is True


def test_scores_that_do_not_apply_to_the_day_are_none():
    clean = answer_key(_user_turn(_explain()))
    flagged_anyway = score(_answer("residual 6,510.13", flagged=[("P008", "")]), clean)
    assert flagged_anyway["status_ok"] is False
    assert flagged_anyway["top_named"] is None  # nothing to find on a clean day
    assert flagged_anyway["called_tool"] is None

    inconclusive = answer_key(_user_turn(_explain(inconclusive=["P008"])))
    scores = score(_answer("P008 residual 6,979.95 is within Monte Carlo noise"), inconclusive)
    assert scores["status_ok"] is None and scores["noise_ok"] is True
    assert score(_answer("P008 residual 6,979.95"), inconclusive)["noise_ok"] is False


def test_an_answer_that_is_not_json_gets_no_credit_for_its_content():
    """Unparsed means no flags; on a clean day that would otherwise count as the
    right call. Calling the tool is behaviour, so it still counts."""
    clean = answer_key(_user_turn(_explain()))
    unparsed = _answer("P008 residual 6,510.13, nothing to flag")
    assert unparsed.factcheck is not None
    unparsed.factcheck.parsed = False
    scores = score(unparsed, clean)
    assert scores["status_ok"] is False and scores["complete"] is False
    assert scores["top_named"] is None

    alert = answer_key(_user_turn(_explain(breaches=["total"])))
    reprice = ToolCallTrace(1, "what_if_reprice", json.dumps({"position_id": "P008"}), "{}")
    unparsed.trace = [reprice]
    scores = score(unparsed, alert)
    assert scores["top_named"] is False and scores["called_tool"] is True


def test_no_answer_scores_nothing():
    key = answer_key(_user_turn(_explain(breaches=["total"])))
    down = AIInvestigationResult(day0=DAY0, day1=DAY1, model="m", ai_available=False)
    assert set(score(down, key).values()) == {None}


def test_tool_cache_reprices_once_and_never_keeps_an_error(tmp_path):
    cache = ToolCache(tmp_path / "cache.json")
    calls = []

    def fake_tool(name, arguments, *rest):
        calls.append(arguments)
        if arguments["position_id"] == "BAD":
            return {"error": "market state missing"}
        return {"price_delta": -1.0}

    with patch.object(benchmark.tools, "execute_tool_call", side_effect=fake_tool):
        execute = cache.executor(None, "p.yaml", DAY1, tmp_path)
        for _ in range(2):
            assert execute("what_if_reprice", {"position_id": "P008"}) == {"price_delta": -1.0}
            assert "error" in execute("what_if_reprice", {"position_id": "BAD"})
    assert calls.count({"position_id": "P008"}) == 1
    assert calls.count({"position_id": "BAD"}) == 2
    # Kept across runs: a new cache on the same file does not reprice again.
    assert len(ToolCache(tmp_path / "cache.json").entries) == 1


def test_model_key_is_safe_as_a_directory_name():
    assert model_key("qwen2.5:7b") == "qwen2.5-7b"
    assert model_key("llama3.1:8b-instruct") == "llama3.1-8b-instruct"


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=DAY0,
        universe=Universe(index=["SPX"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path / "curated")),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _store_investigation(cfg, day0, day1, prompt):
    row = {
        "asof_date": day1,
        "day0": day0,
        "model": "qwen2.5:7b",
        "ai_available": True,
        **dict.fromkeys(
            ("summary", "root_cause_hypothesis", "confidence", "raw_response", "factcheck_status")
        ),
        "prompt": prompt,
        "factcheck_n_numbers": None,
        "factcheck_unsupported": None,
        "factcheck_mislabelled": None,
        "factcheck_retried": False,
    }
    table = validate(
        pd.DataFrame([row]), AI_INVESTIGATION_SCHEMA, AI_INVESTIGATION_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, f"{cfg.paths.curated}/ai_investigations", ["asof_date"])


def test_run_benchmark_scores_each_day_and_a_rerun_replaces_it(tmp_path):
    cfg = _cfg(tmp_path)
    system = "SYSTEM PROMPT OF THE DAY\n\n"
    _store_investigation(cfg, DAY0, DAY1, system + _user_turn(_explain(breaches=["total"])))
    _store_investigation(
        cfg, dt.date(2026, 9, 1), dt.date(2026, 9, 2), system + _user_turn(_explain())
    )
    answer = {
        "role": "assistant",
        "content": json.dumps(
            {
                "summary": "total residual +17.8bp of NAV, P008 +6,979.95",
                "root_cause_hypothesis": "P008",
                "confidence": "low",
                "flagged_positions": [{"position_id": "P008", "reason": "largest"}],
                "proposed_changes": [],
            }
        ),
    }
    with patch.object(benchmark.ollama_client, "chat", return_value=answer) as chat:
        for _ in range(2):
            df = run_benchmark(cfg, ["qwen2.5:7b"], 2, "p.yaml", log=lambda s: None)

    assert len(df) == 4
    stored = load_results(cfg)
    assert len(stored) == 4  # the rerun replaced each (model, day), not appended
    by_day = stored.set_index(["asof_date", "sample"]).sort_index()
    assert set(stored["split"]) == {"dev", "test"}
    alert_day = stored[stored["day_status"] == ALERT]
    assert alert_day["complete"].all() and alert_day["status_ok"].all()
    clean_day = stored[stored["day_status"] == CLEAN]
    assert not clean_day["status_ok"].any()  # flagged P008 on a clean day
    # Today's system prompt, not the stored one, and the sample's own seed.
    first = chat.call_args_list[0]
    assert first.args[0][0]["content"] != "SYSTEM PROMPT OF THE DAY"
    assert first.args[0][1]["content"].startswith("TODAY'S P&L EXPLAIN")
    assert {c.kwargs["options"]["seed"] for c in chat.call_args_list} == {0, 1}
    assert len(by_day) == 4

    assert set(stored["prompt_version"]) == {PROMPT_VERSION}

    report = summarize(stored).set_index("split")
    assert list(report.index) == ["dev", "test"]
    assert report.loc["test", "complete"] == "100% of 2"
    # The status score is shown per kind of day: always flagging is right on
    # every alert day and wrong on every clean one.
    assert report.loc["test", "alert_flagged"] == "100% of 2"
    assert report.loc["dev", "clean_left_clean"] == "0% of 2"


def test_a_stored_day_is_run_with_todays_closing_line(tmp_path):
    """Days stored before the clean-day line (prompt v1) end without it; an
    alert day already ends with the reprice line. The benchmark cuts each back
    to its data and adds the line the daily run would send now, once."""
    cfg = _cfg(tmp_path)
    alert_turn = _user_turn(_explain(breaches=["total"]))
    v1_clean_turn = data_turn(_user_turn(_explain()))
    assert "clean day" not in v1_clean_turn
    _store_investigation(cfg, DAY0, DAY1, "SYSTEM\n\n" + alert_turn)
    _store_investigation(
        cfg, dt.date(2026, 9, 1), dt.date(2026, 9, 2), "SYSTEM\n\n" + v1_clean_turn
    )

    clean, alert = load_items(cfg)
    assert alert.user_turn() == alert_turn
    assert alert.user_turn().count("Before answering") == 1
    assert clean.user_turn() == _user_turn(_explain())
    assert clean.user_turn().endswith(closing_line(alert=False, clean=True, worst_position=None))
