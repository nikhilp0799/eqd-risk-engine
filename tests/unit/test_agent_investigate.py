import datetime as dt
from unittest.mock import patch

import pandas as pd

from eqdrisk.agent.investigate import (
    ProposedChange,
    _build_prompt,
    _historical_trend,
    _limitations_excerpt,
    _parse_response,
    run_daily_investigation,
)
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.io.schemas import PNL_EXPLAIN_REQUIRED_NOT_NULL, PNL_EXPLAIN_SCHEMA, validate
from eqdrisk.pricing.pnl_explain import PnLExplainResult, StepResult

DAY0 = dt.date(2026, 8, 20)
DAY1 = dt.date(2026, 8, 21)


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


# --- _limitations_excerpt ------------------------------------------------


def test_limitations_excerpt_reads_real_section_from_repo_doc():
    import eqdrisk

    project_root = list(eqdrisk.__path__)[0]
    # src/eqdrisk -> project root is two levels up (src/eqdrisk/../..)
    from pathlib import Path

    root = Path(project_root).parent.parent
    excerpt = _limitations_excerpt(root)
    assert "vega" in excerpt.lower() or "limitation" in excerpt.lower()


def test_limitations_excerpt_falls_back_when_doc_missing(tmp_path):
    excerpt = _limitations_excerpt(tmp_path)
    assert "vega-only" in excerpt


def test_limitations_excerpt_falls_back_when_section_not_found(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "model_documentation.md").write_text("# Just a title\nNo sections here.")
    excerpt = _limitations_excerpt(tmp_path)
    assert "vega-only" in excerpt


def test_limitations_excerpt_truncates_to_max_chars(tmp_path):
    (tmp_path / "docs").mkdir()
    body = "x" * 5000
    (tmp_path / "docs" / "model_documentation.md").write_text(
        f"## 4. Assumptions and limitations\n{body}\n## 5. Data\nmore"
    )
    excerpt = _limitations_excerpt(tmp_path)
    assert len(excerpt) <= 2500


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


# --- _build_prompt ---------------------------------------------------------


def test_build_prompt_includes_real_numbers_and_asks_for_json():
    result = _pnl_result()
    prompt = _build_prompt(result, "no history", "no known limitations")
    assert "1,000,000.00" in prompt or "1000000" in prompt.replace(",", "")
    assert "JSON object" in prompt
    assert "flagged_positions" in prompt
    assert "proposed_changes" in prompt


# --- run_daily_investigation ------------------------------------------------


def test_run_daily_investigation_degrades_honestly_when_ollama_unavailable(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with patch("eqdrisk.agent.ollama_client.is_available", return_value=False):
        investigation = run_daily_investigation(cfg, result, project_root=tmp_path)

    assert investigation.ai_available is False
    assert "AI unavailable" in investigation.render()
    # Persisted even when unavailable, so the daily record is never silently missing.
    df = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigations")}
    ).to_pandas()
    assert len(df) == 1
    assert bool(df.iloc[0]["ai_available"]) is False


def test_run_daily_investigation_degrades_honestly_when_generate_returns_none(tmp_path):
    cfg = _cfg(tmp_path)
    result = _pnl_result()
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch("eqdrisk.agent.ollama_client.generate", return_value=None),
    ):
        investigation = run_daily_investigation(cfg, result, project_root=tmp_path)

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
        patch("eqdrisk.agent.ollama_client.generate", return_value=fake_response),
    ):
        investigation = run_daily_investigation(cfg, result, project_root=tmp_path)

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
        patch("eqdrisk.agent.ollama_client.generate", return_value="the model rambled, not json"),
    ):
        investigation = run_daily_investigation(cfg, result, project_root=tmp_path)

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
        patch("eqdrisk.agent.ollama_client.generate", return_value=fake_response),
    ):
        investigation = run_daily_investigation(cfg, result, project_root=tmp_path)

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
