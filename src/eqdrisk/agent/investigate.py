"""Daily AI investigation agent — goes beyond the README's original 16-step plan,
at the user's own request to make the project "independent, self-reliant using
AI." Runs every day (not just on a residual breach), reads that day's REAL
already-computed P&L-explain output, and asks a small local open-weight model
(via `agent/ollama_client.py` — free, no API key, nothing leaves this machine)
to write a plain-English summary, a root-cause hypothesis, positions to flag
for human review, and PROPOSED (never auto-applied) config/threshold changes.

**Genuinely agentic, not a single-shot call** (`planning/ai_agent_tool_use_plan.md`):
the model gets up to `MAX_TOOL_ROUNDS` real tool-call round-trips
(`agent/tools.py` — a real pricing-engine reprice under a hypothetical shock, and
reading any section of this project's own model documentation) before it must
give its final answer. It decides for itself whether a day needs deeper
investigation or not — a real (bounded) decision loop, not narrated autonomy.

**Grounding, not hallucination:** the model is handed only real numbers this
project already computed (`pricing.pnl_explain`'s output, the persisted historical
residual trend) plus real tools that return real computed results — never asked
to invent its own numbers. **Every output is labeled as an unverified AI
hypothesis**, in the persisted markdown, the CLI render, and the dashboard —
matching this project's own "measure, don't assume" standard rather than
exempting the one part of the system an LLM produces. The full tool-call trace
(what was called, with what arguments, and what came back) is persisted and
shown alongside the final answer, not hidden — the actual evidence of autonomy
for an outside reviewer, not a claim about it.

**What this deliberately does not do:** never auto-applies a proposed config
change (always a written suggestion for a human to review), never recalibrates
or touches model parameters, never sends any project data anywhere except
`localhost:11434`. The reprice tool is read-only from the book's perspective —
it prices a hypothetical shock, it never writes to the real portfolio.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eqdrisk.agent import ollama_client, tools
from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    AI_FLAGGED_POSITION_REQUIRED_NOT_NULL,
    AI_FLAGGED_POSITION_SCHEMA,
    AI_INVESTIGATION_REQUIRED_NOT_NULL,
    AI_INVESTIGATION_SCHEMA,
    AI_INVESTIGATION_TRACE_REQUIRED_NOT_NULL,
    AI_INVESTIGATION_TRACE_SCHEMA,
    AI_PROPOSED_CHANGE_REQUIRED_NOT_NULL,
    AI_PROPOSED_CHANGE_SCHEMA,
    validate,
)
from eqdrisk.pricing.pnl_explain import PnLExplainResult

HISTORY_LOOKBACK_DAYS = 5
MAX_TOOL_ROUNDS = 3


@dataclass
class ProposedChange:
    parameter: str
    current_value: str
    suggested_value: str
    rationale: str


@dataclass
class ToolCallTrace:
    round: int
    tool_name: str
    arguments: str  # JSON-encoded
    result_summary: str  # JSON-encoded, truncated


@dataclass
class AIInvestigationResult:
    day0: dt.date
    day1: dt.date
    model: str
    ai_available: bool
    summary: str = ""
    root_cause_hypothesis: str = ""
    confidence: str = ""
    flagged_positions: list[tuple[str, str]] = field(default_factory=list)  # (position_id, reason)
    proposed_changes: list[ProposedChange] = field(default_factory=list)
    trace: list[ToolCallTrace] = field(default_factory=list)
    raw_response: str = ""
    parse_error: str | None = None

    def _trace_lines(self) -> list[str]:
        return [
            f"    round {t.round}: {t.tool_name}({t.arguments}) -> {t.result_summary}"
            for t in self.trace
        ]

    def render(self) -> str:
        lines = [
            "AI INVESTIGATION — AI-GENERATED HYPOTHESIS, NOT VERIFIED, FOR HUMAN REVIEW",
            f"  {self.day0} -> {self.day1}  (model: {self.model})",
        ]
        if not self.ai_available:
            lines.append("  AI unavailable today (Ollama not reachable) — no investigation run.")
            if self.trace:
                lines.append("  partial investigation trace before failure:")
                lines.extend(self._trace_lines())
            return "\n".join(lines)
        if self.parse_error:
            lines.append(f"  NOTE: structured parsing failed ({self.parse_error}); raw text below.")
            lines.append(f"  {self.summary}")
            if self.trace:
                lines.append("  investigation trace:")
                lines.extend(self._trace_lines())
            return "\n".join(lines)
        lines.append(f"  confidence: {self.confidence}")
        lines.append(f"  summary: {self.summary}")
        lines.append(f"  root cause hypothesis: {self.root_cause_hypothesis}")
        if self.flagged_positions:
            lines.append("  flagged for review:")
            for pid, reason in self.flagged_positions:
                lines.append(f"    {pid}: {reason}")
        if self.proposed_changes:
            lines.append("  proposed changes (not applied):")
            for c in self.proposed_changes:
                lines.append(
                    f"    {c.parameter}: {c.current_value} -> {c.suggested_value} ({c.rationale})"
                )
        if self.trace:
            lines.append("  investigation trace:")
            lines.extend(self._trace_lines())
        return "\n".join(lines)


def _historical_trend(curated_root: Path, day1: dt.date, n_days: int) -> str:
    """Real total residual (bp of NAV) for the last `n_days` real day-pairs on or
    before `day1`, from the persisted `pnl_explain` table — gives the model real
    recent context instead of judging the day in isolation."""
    table_root = curated_root / "pnl_explain"
    if not table_root.exists() or not any(table_root.rglob("*.parquet")):
        return "No prior history available."
    df = store.query("SELECT * FROM t", views={"t": str(table_root)}).to_pandas()
    daily = df.groupby(["day0", "asof_date"]).agg(residual=("residual", "sum"), nav=("nav", "max"))
    daily = daily.reset_index().sort_values("asof_date")
    daily = daily[daily["asof_date"] <= day1].tail(n_days)
    if daily.empty:
        return "No prior history available."
    lines = []
    for _, r in daily.iterrows():
        bp = 0.0 if not r["nav"] or pd.isna(r["nav"]) else 10_000.0 * r["residual"] / r["nav"]
        lines.append(f"  {r['day0']} -> {r['asof_date']}: total residual {bp:+.1f}bp of NAV")
    return "\n".join(lines)


_FINAL_JSON_INSTRUCTIONS = """Respond with ONLY a JSON object (no markdown fences, no other \
text) with these exact keys:
{
  "summary": "one paragraph, plain English, what happened today",
  "root_cause_hypothesis": "your best hypothesis for the largest residual/breach today, \
referencing specific positions/steps from the data above (or from a tool call you made), or \
'no unusual residual today' if clean",
  "confidence": "low, medium, or high",
  "flagged_positions": [{"position_id": "...", "reason": "..."}],
  "proposed_changes": [{"parameter": "...", "current_value": "...", "suggested_value": "...", \
"rationale": "..."}]
}
If nothing is unusual today, flagged_positions and proposed_changes should be empty lists."""


def _build_messages(result: PnLExplainResult, history: str) -> list[dict[str, Any]]:
    system = (
        "You are a risk-desk assistant investigating one day's real P&L-explain output for an "
        "equity derivatives book. Use ONLY real numbers — from the data given below, or from "
        "tool calls you make — never invent figures. You have two real tools available: "
        "`what_if_reprice` (a real pricing-engine call to check a position's actual sensitivity "
        "under a hypothetical spot/vol shock) and `read_model_doc_section` (read this project's "
        "own model documentation). If the data below is clean and unremarkable, answer directly, "
        "no tool calls needed. But if any position has a large or unusual residual, you MUST call "
        "`what_if_reprice` on that position before you are allowed to answer — do not answer "
        "directly first. This verifies your hypothesis with a real number instead of a guess. "
        "You have at most "
        f"{MAX_TOOL_ROUNDS} rounds of tool calls before you must give your final answer.\n\n"
        f"{_FINAL_JSON_INSTRUCTIONS}"
    )
    user = f"""TODAY'S P&L EXPLAIN ({result.day0} -> {result.day1}, NAV={result.nav:,.2f}):
{result.render()}

RECENT HISTORY (last {HISTORY_LOOKBACK_DAYS} real day-pairs):
{history}
"""
    if result.breaches and result.by_position_residual:
        residuals = result.by_position_residual
        worst_position = max(residuals, key=lambda p: abs(residuals[p]))
        user += (
            f"\nBefore answering, call what_if_reprice on {worst_position}, today's largest-"
            "residual position, to verify your hypothesis with a real number.\n"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_response(raw: str) -> tuple[dict, str | None]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return {}, f"expected a JSON object, got {type(parsed).__name__}"
        return parsed, None
    except json.JSONDecodeError as exc:
        return {}, f"JSON parse failed: {exc}"


def _run_agentic_loop(
    cfg: BaseConfig,
    result: PnLExplainResult,
    history: str,
    portfolio_path: str,
    project_root: Path,
) -> tuple[str | None, list[ToolCallTrace]]:
    """Up to `MAX_TOOL_ROUNDS` real tool-call round-trips, then a forced
    tools-off final answer if the model still wants to call something.
    Returns `(final_raw_content, trace)` — `final_raw_content` is `None` if
    the Ollama chat call itself failed (treated as AI-unavailable by the
    caller), distinct from a malformed-but-present final answer."""
    messages = _build_messages(result, history)
    trace: list[ToolCallTrace] = []

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        message = ollama_client.chat(messages, tools=tools.TOOL_SPECS)
        if message is None:
            return None, trace
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or None), trace

        messages.append(
            {"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls}
        )
        for call in tool_calls:
            fn = call.get("function", {})
            name = str(fn.get("name", ""))
            arguments = fn.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_result = tools.execute_tool_call(
                name, arguments, cfg, portfolio_path, result.day1, project_root
            )
            trace.append(
                ToolCallTrace(
                    round=round_num,
                    tool_name=name,
                    arguments=json.dumps(arguments),
                    result_summary=json.dumps(tool_result)[:1000],
                )
            )
            messages.append({"role": "tool", "content": json.dumps(tool_result)})

    messages.append(
        {
            "role": "user",
            "content": f"No more tool calls are allowed. {_FINAL_JSON_INSTRUCTIONS}",
        }
    )
    message = ollama_client.chat(messages, tools=None)
    if message is None:
        return None, trace
    return (message.get("content") or None), trace


def run_daily_investigation(
    cfg: BaseConfig,
    result: PnLExplainResult,
    portfolio_path: str,
    project_root: Path | None = None,
) -> AIInvestigationResult:
    project_root = project_root or Path.cwd()
    curated_root = Path(cfg.paths.curated)

    if not ollama_client.is_available():
        investigation = AIInvestigationResult(
            day0=result.day0, day1=result.day1, model=ollama_client.OLLAMA_MODEL, ai_available=False
        )
        _persist(investigation, curated_root)
        return investigation

    history = _historical_trend(curated_root, result.day1, HISTORY_LOOKBACK_DAYS)
    raw, trace = _run_agentic_loop(cfg, result, history, portfolio_path, project_root)

    if raw is None:
        investigation = AIInvestigationResult(
            day0=result.day0,
            day1=result.day1,
            model=ollama_client.OLLAMA_MODEL,
            ai_available=False,
            trace=trace,
        )
        _persist(investigation, curated_root)
        return investigation

    parsed, parse_error = _parse_response(raw)
    investigation = AIInvestigationResult(
        day0=result.day0,
        day1=result.day1,
        model=ollama_client.OLLAMA_MODEL,
        ai_available=True,
        raw_response=raw,
        parse_error=parse_error,
        trace=trace,
    )
    if parse_error:
        investigation.summary = raw[:2000]
    else:
        investigation.summary = str(parsed.get("summary", ""))
        investigation.root_cause_hypothesis = str(parsed.get("root_cause_hypothesis", ""))
        investigation.confidence = str(parsed.get("confidence", ""))
        investigation.flagged_positions = [
            (str(p.get("position_id", "")), str(p.get("reason", "")))
            for p in parsed.get("flagged_positions", [])
            if isinstance(p, dict)
        ]
        investigation.proposed_changes = [
            ProposedChange(
                parameter=str(c.get("parameter", "")),
                current_value=str(c.get("current_value", "")),
                suggested_value=str(c.get("suggested_value", "")),
                rationale=str(c.get("rationale", "")),
            )
            for c in parsed.get("proposed_changes", [])
            if isinstance(c, dict)
        ]

    _persist(investigation, curated_root)
    _write_markdown(investigation, project_root)
    return investigation


def _persist(investigation: AIInvestigationResult, curated_root: Path) -> None:
    row = {
        "asof_date": investigation.day1,
        "day0": investigation.day0,
        "model": investigation.model,
        "ai_available": investigation.ai_available,
        "summary": investigation.summary,
        "root_cause_hypothesis": investigation.root_cause_hypothesis,
        "confidence": investigation.confidence,
        "raw_response": investigation.raw_response,
    }
    table = validate(
        pd.DataFrame([row]), AI_INVESTIGATION_SCHEMA, AI_INVESTIGATION_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, curated_root / "ai_investigations", ["asof_date"])

    if investigation.flagged_positions:
        flagged_df = pd.DataFrame(
            [
                {
                    "asof_date": investigation.day1,
                    "day0": investigation.day0,
                    "position_id": pid,
                    "reason": reason,
                }
                for pid, reason in investigation.flagged_positions
            ]
        )
        flagged_table = validate(
            flagged_df, AI_FLAGGED_POSITION_SCHEMA, AI_FLAGGED_POSITION_REQUIRED_NOT_NULL
        )
        store.write_partitioned(flagged_table, curated_root / "ai_flagged_positions", ["asof_date"])

    if investigation.proposed_changes:
        changes_df = pd.DataFrame(
            [
                {
                    "asof_date": investigation.day1,
                    "day0": investigation.day0,
                    "parameter": c.parameter,
                    "current_value": c.current_value,
                    "suggested_value": c.suggested_value,
                    "rationale": c.rationale,
                }
                for c in investigation.proposed_changes
            ]
        )
        changes_table = validate(
            changes_df, AI_PROPOSED_CHANGE_SCHEMA, AI_PROPOSED_CHANGE_REQUIRED_NOT_NULL
        )
        store.write_partitioned(changes_table, curated_root / "ai_proposed_changes", ["asof_date"])

    if investigation.trace:
        trace_df = pd.DataFrame(
            [
                {
                    "asof_date": investigation.day1,
                    "day0": investigation.day0,
                    "round": t.round,
                    "tool_name": t.tool_name,
                    "arguments": t.arguments,
                    "result_summary": t.result_summary,
                }
                for t in investigation.trace
            ]
        )
        trace_table = validate(
            trace_df, AI_INVESTIGATION_TRACE_SCHEMA, AI_INVESTIGATION_TRACE_REQUIRED_NOT_NULL
        )
        store.write_partitioned(trace_table, curated_root / "ai_investigation_trace", ["asof_date"])


def _write_markdown(investigation: AIInvestigationResult, project_root: Path) -> None:
    out_dir = project_root / "logs" / "ai_investigations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{investigation.day1.isoformat()}.md").write_text(
        f"# AI-GENERATED HYPOTHESIS — NOT VERIFIED, FOR HUMAN REVIEW\n\n"
        f"Generated by a local open-weight model ({investigation.model}, via Ollama), not a human "
        f"analyst or a frontier AI system. Treat everything below as a hypothesis to check, not a "
        f"conclusion.\n\n{investigation.render()}\n"
    )
