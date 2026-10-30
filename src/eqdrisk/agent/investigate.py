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

**Fact-checked, not trusted (2026-10-19, `agent/factcheck.py`):** every number in
the answer is checked, with its unit, against the exact prompt and tool results the
model saw. One corrective retry names any unsupported numbers (or says the answer
was not valid JSON); if the problem survives it, the answer is kept and marked, and
an answer that is still not valid JSON is checked on its raw text and marked
`unparsed`. The prompt is stored with the investigation, so any figure can be traced
to what the model was given. Since 2026-10-30 a number must also be called what it is:
a what-if result written up as "the actual price change" is marked
`mislabelled_numbers`.

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

from eqdrisk.agent import factcheck, ollama_client, tools
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
    # Provenance and fact-check (planning/ai_factcheck_plan.md): the exact text
    # the model was given, and whether every number it wrote is in that text.
    prompt: str = ""
    factcheck: factcheck.FactCheckResult | None = None
    factcheck_retried: bool = False

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
            if self.factcheck is not None:
                lines.append(f"  {self.factcheck.render()}")
            lines.append(f"  {self.summary}")
            if self.trace:
                lines.append("  investigation trace:")
                lines.extend(self._trace_lines())
            return "\n".join(lines)
        if self.factcheck is not None:
            retried = " (after one corrective retry)" if self.factcheck_retried else ""
            lines.append(f"  {self.factcheck.render()}{retried}")
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
) -> tuple[str | None, list[ToolCallTrace], list[dict[str, Any]]]:
    """Up to `MAX_TOOL_ROUNDS` real tool-call round-trips, then a forced
    tools-off final answer if the model still wants to call something.
    Returns `(final_raw_content, trace)` — `final_raw_content` is `None` if
    the Ollama chat call itself failed (treated as AI-unavailable by the
    caller), distinct from a malformed-but-present final answer. Also returns
    the conversation so far, for the fact-check and its corrective retry."""
    messages = _build_messages(result, history)
    trace: list[ToolCallTrace] = []

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        message = ollama_client.chat(messages, tools=tools.TOOL_SPECS)
        if message is None:
            return None, trace, messages
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or None), trace, messages

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
        return None, trace, messages
    return (message.get("content") or None), trace, messages


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
    raw, trace, messages = _run_agentic_loop(cfg, result, history, portfolio_path, project_root)
    prompt = "\n\n".join(m["content"] for m in messages[:2])

    if raw is None:
        investigation = AIInvestigationResult(
            day0=result.day0,
            day1=result.day1,
            model=ollama_client.OLLAMA_MODEL,
            ai_available=False,
            trace=trace,
            prompt=prompt,
        )
        _persist(investigation, curated_root)
        return investigation

    investigation = AIInvestigationResult(
        day0=result.day0,
        day1=result.day1,
        model=ollama_client.OLLAMA_MODEL,
        ai_available=True,
        trace=trace,
        prompt=prompt,
    )
    _fill_from_raw(investigation, raw)
    context = _factcheck_context(messages)
    shares = _residual_shares(result)
    labelled = _labelled_sources(messages, trace, shares)
    checked = _check(investigation, context, shares, labelled)
    # One corrective retry in total, for whatever went wrong: invalid JSON, numbers
    # not in the data, or numbers called the wrong quantity. A retry that is not
    # valid JSON is discarded.
    if investigation.parse_error or checked.unsupported or checked.mislabelled:
        retry_raw = _corrective_retry(messages, raw, investigation.parse_error, checked)
        if retry_raw is not None:
            retry = AIInvestigationResult(
                day0=result.day0, day1=result.day1, model=investigation.model, ai_available=True
            )
            _fill_from_raw(retry, retry_raw)
            if not retry.parse_error:
                _fill_from_raw(investigation, retry_raw)
                investigation.factcheck_retried = True
                checked = _check(investigation, context, shares, labelled)
    investigation.factcheck = checked

    _persist(investigation, curated_root)
    _write_markdown(investigation, project_root)
    return investigation


def _fill_from_raw(investigation: AIInvestigationResult, raw: str) -> None:
    parsed, parse_error = _parse_response(raw)
    investigation.raw_response = raw
    investigation.parse_error = parse_error
    if parse_error:
        investigation.summary = raw[:2000]
        return
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


def _answer_texts(investigation: AIInvestigationResult) -> list[str]:
    texts = [investigation.summary, investigation.root_cause_hypothesis]
    texts += [reason for _, reason in investigation.flagged_positions]
    for c in investigation.proposed_changes:
        texts += [c.current_value, c.suggested_value, c.rationale]
    return texts


def _check(
    investigation: AIInvestigationResult,
    context: str,
    shares: list[float],
    labelled: list[factcheck.Source],
) -> factcheck.FactCheckResult:
    """Fact-check the answer's fields, or its raw text when it was not valid JSON
    (2026-10-02: the model wrote arithmetic inside its JSON, and the answer used
    to go unchecked with a null status)."""
    if investigation.parse_error:
        checked = factcheck.check([investigation.raw_response or ""], context, shares, labelled)
        checked.parsed = False
        return checked
    return factcheck.check(_answer_texts(investigation), context, shares, labelled)


def _factcheck_context(messages: list[dict[str, Any]]) -> str:
    """Everything the model saw: the prompt and every full tool result (the
    stored trace keeps only a truncated summary)."""
    return "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") in ("system", "user", "tool")
    )


def _labelled_sources(
    messages: list[dict[str, Any]], trace: list[ToolCallTrace], shares: list[float]
) -> list[factcheck.Source]:
    """The numbers whose meaning is known: the explain and history in the user
    turn, and each what_if_reprice result (tool messages follow the trace's order)."""
    explain_text = next((str(m["content"]) for m in messages if m.get("role") == "user"), "")
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    what_ifs = []
    for t, m in zip(trace, tool_messages, strict=False):
        if t.tool_name != "what_if_reprice":
            continue
        try:
            payload = json.loads(str(m.get("content", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            what_ifs.append(payload)
    return factcheck.labelled_sources(explain_text, what_ifs, shares)


def _residual_shares(result: PnLExplainResult) -> list[float]:
    """Each position's share of the total residual, in %: a figure the model may
    legitimately compute from its data."""
    total = sum(result.by_position_residual.values())
    if not total:
        return []
    return [100.0 * v / total for v in result.by_position_residual.values()]


def _corrective_retry(
    messages: list[dict[str, Any]],
    raw: str,
    parse_error: str | None,
    checked: factcheck.FactCheckResult,
) -> str | None:
    """One more turn: say what was wrong (invalid JSON, numbers that are not in the
    model's data, numbers called the wrong quantity) and ask for the same JSON
    corrected (planning/ai_factcheck_plan.md, ai_factcheck_labels_plan.md)."""
    unsupported = checked.unsupported
    problems = []
    if parse_error:
        problems.append(
            f"Your answer was not valid JSON ({parse_error}). Write every value as a plain "
            "number or string: no arithmetic, no comments."
        )
    if unsupported:
        problems.append(
            "These numbers in your answer do not appear in the data you were given, or "
            f"carry the wrong unit: {', '.join(unsupported)}. Dollar residuals and "
            "residuals in bp of NAV are different numbers in the data; use each with the "
            "unit it has there, and do not present a what-if scenario as something that "
            "happened."
        )
    if checked.mislabelled:
        problems.append(
            "These numbers are in your data but your answer calls them a different quantity: "
            f"{'; '.join(checked.mislabelled)}. Describe each number as what it is in the data: "
            "a what-if reprice result is a hypothetical, not an actual or explained P&L, and a "
            "residual is neither the actual nor the explained P&L."
        )
    retry_messages = [
        *messages,
        {"role": "assistant", "content": raw},
        {"role": "user", "content": f"{' '.join(problems)} {_FINAL_JSON_INSTRUCTIONS}"},
    ]
    message = ollama_client.chat(retry_messages, tools=None)
    if message is None:
        return None
    return message.get("content") or None


def _persist(investigation: AIInvestigationResult, curated_root: Path) -> None:
    fc = investigation.factcheck
    row = {
        "asof_date": investigation.day1,
        "day0": investigation.day0,
        "model": investigation.model,
        "ai_available": investigation.ai_available,
        "summary": investigation.summary,
        "root_cause_hypothesis": investigation.root_cause_hypothesis,
        "confidence": investigation.confidence,
        "raw_response": investigation.raw_response,
        "prompt": investigation.prompt,
        "factcheck_status": None if fc is None else fc.status,
        "factcheck_n_numbers": None if fc is None else fc.n_numbers,
        "factcheck_unsupported": None if fc is None else json.dumps(fc.unsupported),
        "factcheck_mislabelled": None if fc is None else json.dumps(fc.mislabelled),
        "factcheck_retried": investigation.factcheck_retried,
    }
    table = validate(
        pd.DataFrame([row]), AI_INVESTIGATION_SCHEMA, AI_INVESTIGATION_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, curated_root / "ai_investigations", ["asof_date"])

    # A rerun can legitimately produce no flagged positions, changes or tool calls;
    # clear the day's old rows first so they don't outlive the run that wrote them.
    for child in ("ai_flagged_positions", "ai_proposed_changes", "ai_investigation_trace"):
        store.remove_partition(curated_root / child, {"asof_date": investigation.day1})

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
