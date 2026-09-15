"""Daily AI investigation agent — goes beyond the README's original 16-step plan,
at the user's own request to make the project "independent, self-reliant using
AI." Runs every day (not just on a residual breach), reads that day's REAL
already-computed P&L-explain output plus this project's own documented known
limitations, and asks a small local open-weight model (via `agent/ollama_client.py`
— free, no API key, nothing leaves this machine) to write a plain-English summary,
a root-cause hypothesis, positions to flag for human review, and PROPOSED (never
auto-applied) config/threshold changes.

**Grounding, not hallucination:** the prompt is built only from real numbers this
project already computed (`pricing.pnl_explain`'s output, the persisted historical
residual trend) and a literal excerpt of this project's own documented limitations
(`docs/model_documentation.md` Section 4) — the model is asked to reason from
stated facts, not invent its own. **Every output is labeled as an unverified AI
hypothesis**, in the persisted markdown, the CLI render, and the dashboard —
matching this project's own "measure, don't assume" standard rather than
exempting the one part of the system an LLM produces.

**What this deliberately does not do:** never auto-applies a proposed config
change (always a written suggestion for a human to review), never recalibrates
or touches model parameters, never sends any project data anywhere except
`localhost:11434`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eqdrisk.agent import ollama_client
from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    AI_FLAGGED_POSITION_REQUIRED_NOT_NULL,
    AI_FLAGGED_POSITION_SCHEMA,
    AI_INVESTIGATION_REQUIRED_NOT_NULL,
    AI_INVESTIGATION_SCHEMA,
    AI_PROPOSED_CHANGE_REQUIRED_NOT_NULL,
    AI_PROPOSED_CHANGE_SCHEMA,
    validate,
)
from eqdrisk.pricing.pnl_explain import PnLExplainResult

MODEL_DOC_PATH = "docs/model_documentation.md"
HISTORY_LOOKBACK_DAYS = 5
MAX_LIMITATIONS_CHARS = 2500  # keeps the prompt small enough for a fast local 7B reply


@dataclass
class ProposedChange:
    parameter: str
    current_value: str
    suggested_value: str
    rationale: str


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
    raw_response: str = ""
    parse_error: str | None = None

    def render(self) -> str:
        lines = [
            "AI INVESTIGATION — AI-GENERATED HYPOTHESIS, NOT VERIFIED, FOR HUMAN REVIEW",
            f"  {self.day0} -> {self.day1}  (model: {self.model})",
        ]
        if not self.ai_available:
            lines.append("  AI unavailable today (Ollama not reachable) — no investigation run.")
            return "\n".join(lines)
        if self.parse_error:
            lines.append(f"  NOTE: structured parsing failed ({self.parse_error}); raw text below.")
            lines.append(f"  {self.summary}")
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
        return "\n".join(lines)


def _limitations_excerpt(project_root: Path) -> str:
    """A literal excerpt of this project's own documented limitations (Section 4
    of the model doc) — grounds the model in REAL, stated facts about known gaps
    (e.g. the NVDA autocallable's vega-only Greek set) rather than inventing its
    own. Falls back to a short hardcoded summary if the doc can't be read, rather
    than failing the whole investigation over a missing file."""
    doc_path = project_root / MODEL_DOC_PATH
    fallback = (
        "Known limitation: MC-priced positions (barrier, autocall) have a vega-only "
        "Greek set that has been shown to badly overstate true vol sensitivity near "
        "barriers (see Step 13's incident report)."
    )
    if not doc_path.exists():
        return fallback
    text = doc_path.read_text()
    match = re.search(r"## 4\. Assumptions and limitations(.*?)## 5\. Data", text, re.DOTALL)
    if not match:
        return fallback
    return match.group(1).strip()[:MAX_LIMITATIONS_CHARS]


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


def _build_prompt(result: PnLExplainResult, history: str, limitations: str) -> str:
    return f"""You are a risk-desk assistant reviewing one day's real P&L-explain output for an \
equity derivatives book. Use ONLY the real numbers given below — do not invent figures.

TODAY'S P&L EXPLAIN ({result.day0} -> {result.day1}, NAV={result.nav:,.2f}):
{result.render()}

RECENT HISTORY (last {HISTORY_LOOKBACK_DAYS} real day-pairs):
{history}

THIS PROJECT'S OWN DOCUMENTED KNOWN LIMITATIONS (Section 4 of its model documentation):
{limitations}

Respond with ONLY a JSON object (no markdown fences, no other text) with these exact keys:
{{
  "summary": "one paragraph, plain English, what happened today",
  "root_cause_hypothesis": "your best hypothesis for the largest residual/breach today, \
referencing specific positions/steps from the data above, or 'no unusual residual today' if clean",
  "confidence": "low, medium, or high",
  "flagged_positions": [{{"position_id": "...", "reason": "..."}}],
  "proposed_changes": [{{"parameter": "...", "current_value": "...", "suggested_value": "...", \
"rationale": "..."}}]
}}
If nothing is unusual today, flagged_positions and proposed_changes should be empty lists.
"""


def _parse_response(raw: str) -> tuple[dict, str | None]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return {}, f"expected a JSON object, got {type(parsed).__name__}"
        return parsed, None
    except json.JSONDecodeError as exc:
        return {}, f"JSON parse failed: {exc}"


def run_daily_investigation(
    cfg: BaseConfig,
    result: PnLExplainResult,
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
    limitations = _limitations_excerpt(project_root)
    prompt = _build_prompt(result, history, limitations)
    raw = ollama_client.generate(prompt)

    if raw is None:
        investigation = AIInvestigationResult(
            day0=result.day0, day1=result.day1, model=ollama_client.OLLAMA_MODEL, ai_available=False
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


def _write_markdown(investigation: AIInvestigationResult, project_root: Path) -> None:
    out_dir = project_root / "logs" / "ai_investigations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{investigation.day1.isoformat()}.md").write_text(
        f"# AI-GENERATED HYPOTHESIS — NOT VERIFIED, FOR HUMAN REVIEW\n\n"
        f"Generated by a local open-weight model ({investigation.model}, via Ollama), not a human "
        f"analyst or a frontier AI system. Treat everything below as a hypothesis to check, not a "
        f"conclusion.\n\n{investigation.render()}\n"
    )
