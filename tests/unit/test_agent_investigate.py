import datetime as dt
from unittest.mock import patch

import pandas as pd

from eqdrisk.agent.investigate import (
    ProposedChange,
    ToolCallTrace,
    _build_messages,
    _historical_trend,
    _parse_response,
    _run_agentic_loop,
    run_daily_investigation,
)
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import PNL_EXPLAIN_REQUIRED_NOT_NULL, PNL_EXPLAIN_SCHEMA, validate
from eqdrisk.pricing.pnl_explain import PnLExplainResult, StepResult

DAY0 = dt.date(2026, 8, 20)
DAY1 = dt.date(2026, 8, 21)
PORTFOLIO_PATH = "configs/portfolio.yaml"  # never actually read unless a tool call uses it


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=DAY0,
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _pnl_result() -> PnLExplainResult:
    return PnLExplainResult(
        day0=DAY0,
        day1=DAY1,
        steps=[StepResult(step="vol", actual_pnl=1000.0, explained_pnl=800.0)],
        by_position_residual={"P1": 200.0},
        nav=1_000_000.0,
    )


def _final_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool_call_message(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


# --- _parse_response ---------------------------------------------------


def test_parse_response_accepts_plain_json():
    parsed, err = _parse_response('{"summary": "all clean", "confidence": "high"}')
    assert err is None
    assert parsed["summary"] == "all clean"


def test_parse_response_strips_markdown_code_fences():
    raw = '```json\n{"summary": "fenced"}\n```'
    parsed, err = _parse_response(raw)
    assert err is None
    assert parsed["summary"] == "fenced"


def test_parse_response_reports_honest_error_on_invalid_json():
    parsed, err = _parse_response("not json at all")
    assert parsed == {}
    assert err is not None
    assert "JSON parse failed" in err


def test_parse_response_rejects_non_object_json():
    parsed, err = _parse_response("[1, 2, 3]")
    assert parsed == {}
    assert err is not None
    assert "expected a JSON object" in err


# --- _historical_trend ----------------------------------------------------


def test_historical_trend_reports_no_history_when_table_absent(tmp_path):
    trend = _historical_trend(tmp_path, DAY1, 5)
    assert trend == "No prior history available."


def test_historical_trend_summarizes_real_persisted_residuals(tmp_path):
    rows = []
    for i, asof in enumerate([dt.date(2026, 8, 19), dt.date(2026, 8, 20), DAY1]):
        rows.append(
            {
                "asof_date": asof,
                "day0": asof - dt.timedelta(days=1),
                "step": "vol",
                "actual_pnl": 100.0,
                "explained_pnl": 100.0 - i,
                "residual": float(i),
                "nav": 1_000_000.0,
            }
        )
    table = validate(pd.DataFrame(rows), PNL_EXPLAIN_SCHEMA, PNL_EXPLAIN_REQUIRED_NOT_NULL)
    store.write_partitioned(table, tmp_path / "pnl_explain", ["asof_date"])

    trend = _historical_trend(tmp_path, DAY1, 5)
    assert "No prior history" not in trend
    assert str(DAY1) in trend
    assert "bp of NAV" in trend


# --- _build_messages ---------------------------------------------------------


def test_build_messages_includes_real_numbers_and_tool_instructions():
    result = _pnl_result()
    messages = _build_messages(result, "no history")
    system, user = messages[0]["content"], messages[1]["content"]
    assert "what_if_reprice" in system
    assert "read_model_doc_section" in system
    assert "JSON object" in system
    assert "flagged_positions" in system
    assert "1,000,000.00" in user


def test_build_messages_adds_no_directive_when_clean():
    result = _pnl_result()  # no breaches set -> a clean day
    messages = _build_messages(result, "no history")
    user = messages[1]["content"]
    assert "Before answering, call what_if_reprice" not in user


def test_build_messages_directs_worst_position_when_breached():
    """The recency-weighted end-of-user-message directive (not just the system
    prompt) is what actually gets a small local model to call the tool on a
    real breach day — confirmed empirically against the live model, not
    assumed. Names the REAL worst position from `by_position_residual`, not a
    generic instruction."""
    result = PnLExplainResult(
        day0=DAY0,
        day1=DAY1,
        steps=[StepResult(step="vol", actual_pnl=1000.0, explained_pnl=800.0)],
        by_position_residual={"P1": 50.0, "P2": -900.0, "P3": 10.0},
        nav=1_000_000.0,
        breaches=["total residual +900.0bp of NAV exceeds the 5bp threshold"],
    )
    messages = _build_messages(result, "no history")
    user = messages[1]["content"]
    assert "Before answering, call what_if_reprice on P2" in user


# --- _run_agentic_loop -----------------------------------------------------


def test_agentic_loop_returns_immediately_when_no_tool_call(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with patch(
        "eqdrisk.agent.investigate.ollama_client.chat",
        return_value=_final_message('{"summary": "clean day"}'),
    ) as mock_chat:
        raw, trace = _run_agentic_loop(cfg, result, "no history", PORTFOLIO_PATH, tmp_path)

    assert raw == '{"summary": "clean day"}'
    assert trace == []
    assert mock_chat.call_count == 1


def test_agentic_loop_executes_a_real_tool_call_then_answers(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "model_documentation.md").write_text(
        "## 4. Assumptions and limitations\nknown vega-only gap\n## 5. Data\nmore"
    )
    responses = [
        _tool_call_message("read_model_doc_section", {"section": 4}),
        _final_message('{"summary": "checked limitations"}'),
    ]
    with patch("eqdrisk.agent.investigate.ollama_client.chat", side_effect=responses) as mock_chat:
        raw, trace = _run_agentic_loop(cfg, result, "no history", PORTFOLIO_PATH, tmp_path)

    assert raw == '{"summary": "checked limitations"}'
    assert mock_chat.call_count == 2
    assert len(trace) == 1
    assert trace[0].tool_name == "read_model_doc_section"
    assert "vega-only gap" in trace[0].result_summary


def test_agentic_loop_handles_unknown_tool_gracefully(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    responses = [
        _tool_call_message("not_a_real_tool", {}),
        _final_message('{"summary": "recovered"}'),
    ]
    with patch("eqdrisk.agent.investigate.ollama_client.chat", side_effect=responses):
        raw, trace = _run_agentic_loop(cfg, result, "no history", PORTFOLIO_PATH, tmp_path)

    assert raw == '{"summary": "recovered"}'
    assert "unknown tool" in trace[0].result_summary


def test_agentic_loop_forces_final_answer_after_max_rounds(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    responses = [
        _tool_call_message("read_model_doc_section", {"section": 99}),
        _tool_call_message("read_model_doc_section", {"section": 99}),
        _tool_call_message("read_model_doc_section", {"section": 99}),
        _final_message('{"summary": "forced final answer"}'),
    ]
    with patch("eqdrisk.agent.investigate.ollama_client.chat", side_effect=responses) as mock_chat:
        raw, trace = _run_agentic_loop(cfg, result, "no history", PORTFOLIO_PATH, tmp_path)

    assert raw == '{"summary": "forced final answer"}'
    assert len(trace) == 3
    # The 4th (forced) call must have been made with tools=None.
    assert mock_chat.call_count == 4
    _, kwargs = mock_chat.call_args_list[3]
    assert kwargs.get("tools") is None


def test_agentic_loop_returns_none_when_chat_unreachable(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with patch("eqdrisk.agent.investigate.ollama_client.chat", return_value=None):
        raw, trace = _run_agentic_loop(cfg, result, "no history", PORTFOLIO_PATH, tmp_path)

    assert raw is None
    assert trace == []


# --- run_daily_investigation ------------------------------------------------


def test_run_daily_investigation_degrades_honestly_when_ollama_unavailable(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with patch("eqdrisk.agent.ollama_client.is_available", return_value=False):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert investigation.ai_available is False
    assert "AI unavailable" in investigation.render()
    # Persisted even when unavailable, so the daily record is never silently missing.
    df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigations")}
    ).to_pandas()
    assert len(df) == 1
    assert bool(df.iloc[0]["ai_available"]) is False


def test_run_daily_investigation_degrades_honestly_when_chat_returns_none(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch("eqdrisk.agent.investigate.ollama_client.chat", return_value=None),
    ):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert investigation.ai_available is False


def test_run_daily_investigation_parses_well_formed_response(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    fake_response = (
        '{"summary": "vol move explains most of the day", '
        '"root_cause_hypothesis": "vega move on P1", '
        '"confidence": "medium", '
        '"flagged_positions": [{"position_id": "P1", "reason": "residual above trend"}], '
        '"proposed_changes": []}'
    )
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch(
            "eqdrisk.agent.investigate.ollama_client.chat",
            return_value=_final_message(fake_response),
        ),
    ):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert investigation.ai_available is True
    assert investigation.parse_error is None
    assert investigation.confidence == "medium"
    assert investigation.flagged_positions == [("P1", "residual above trend")]
    assert investigation.proposed_changes == []

    flagged_df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_flagged_positions")}
    ).to_pandas()
    assert len(flagged_df) == 1
    assert flagged_df.iloc[0]["position_id"] == "P1"

    md_path = tmp_path / "logs" / "ai_investigations" / f"{DAY1.isoformat()}.md"
    assert md_path.exists()
    assert "NOT VERIFIED" in md_path.read_text()


def test_run_daily_investigation_falls_back_honestly_on_unparseable_response(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch(
            "eqdrisk.agent.investigate.ollama_client.chat",
            return_value=_final_message("the model rambled, not json"),
        ),
    ):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert investigation.ai_available is True
    assert investigation.parse_error is not None
    assert "the model rambled" in investigation.summary
    assert "structured parsing failed" in investigation.render()


def test_run_daily_investigation_persists_proposed_changes(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    fake_response = (
        '{"summary": "s", "root_cause_hypothesis": "r", "confidence": "low", '
        '"flagged_positions": [], '
        '"proposed_changes": [{"parameter": "RESIDUAL_ALERT_THRESHOLD_BP", '
        '"current_value": "5", "suggested_value": "8", "rationale": "chronic small breaches"}]}'
    )
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch(
            "eqdrisk.agent.investigate.ollama_client.chat",
            return_value=_final_message(fake_response),
        ),
    ):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert investigation.proposed_changes == [
        ProposedChange(
            parameter="RESIDUAL_ALERT_THRESHOLD_BP",
            current_value="5",
            suggested_value="8",
            rationale="chronic small breaches",
        )
    ]
    changes_df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_proposed_changes")}
    ).to_pandas()
    assert len(changes_df) == 1
    assert changes_df.iloc[0]["parameter"] == "RESIDUAL_ALERT_THRESHOLD_BP"


def test_run_daily_investigation_persists_and_renders_tool_call_trace(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "model_documentation.md").write_text(
        "## 4. Assumptions and limitations\nknown vega-only gap\n## 5. Data\nmore"
    )
    responses = [
        _tool_call_message("read_model_doc_section", {"section": 4}),
        _final_message('{"summary": "checked docs", "confidence": "high"}'),
    ]
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch("eqdrisk.agent.investigate.ollama_client.chat", side_effect=responses),
    ):
        investigation = run_daily_investigation(cfg, result, PORTFOLIO_PATH, project_root=tmp_path)

    assert len(investigation.trace) == 1
    assert investigation.trace[0] == ToolCallTrace(
        round=1,
        tool_name="read_model_doc_section",
        arguments='{"section": 4}',
        result_summary=investigation.trace[0].result_summary,
    )
    assert "investigation trace" in investigation.render()

    trace_df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigation_trace")}
    ).to_pandas()
    assert len(trace_df) == 1
    assert trace_df.iloc[0]["tool_name"] == "read_model_doc_section"

    md_text = (tmp_path / "logs" / "ai_investigations" / f"{DAY1.isoformat()}.md").read_text()
    assert "investigation trace" in md_text
