"""Agent benchmark (planning/agent_benchmark_plan.md): the daily investigation
agent run on stored real days by any local model, each answer scored against
what that day's own data says the right answer is.

Why: the fact-check (`agent/factcheck.py`) catches wrong numbers, not wrong
reasoning, and every result so far is about one model. An item is one real day:
its stored data (the P&L explain and five-day history the model saw) under
today's prompt (the system prompt and the closing line,
`investigate.closing_line`), so every model gets the same instructions and data,
and a prompt change is tested as the daily run would send it. Each answer
records the `PROMPT_VERSION` it ran under. The loop, fact-check and corrective
retry are the daily run's own (`investigate.tool_loop`,
`investigate.check_and_retry`).

Scores per answer, all computed from the day's data, none judged:
- grounded: the fact-check passed (after its one corrective retry, as daily);
- status_ok: positions flagged on an alert day, none on a clean day (not scored
  on an inconclusive-only day, where flagging for review is a fair call);
  reported separately for the two, since a model that always flags is right on
  every alert day;
- top_named: the largest-residual position is named, on any day not clean;
- noise_ok: Monte Carlo noise is mentioned on a day whose total residual is
  inside its noise band, or that is inconclusive;
- complete: the total residual and, on a day not clean, the top position's
  residual are stated ($ or bp), so a model cannot pass the fact-check by
  writing no numbers;
- called_tool: on an alert day, what_if_reprice was called on the top position,
  as the prompt tells it to.

Days before `TEST_FROM` are the dev split (prompt work may look at them); from it
on, the test split, only scored. Anything trained later uses dev days only.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eqdrisk.agent import factcheck, ollama_client, tools
from eqdrisk.agent.investigate import (
    PROMPT_VERSION,
    AIInvestigationResult,
    Chat,
    ToolCallTrace,
    ToolExecutor,
    check_and_retry,
    closing_line,
    events_section,
    system_prompt,
    tool_loop,
    upcoming_events,
)
from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    AGENT_BENCH_REQUIRED_NOT_NULL,
    AGENT_BENCH_SCHEMA,
    validate,
)

# 23 dev days (2026-09-01..10-21) and 10 test days when the split was set
# (2026-11-09); every later day is test.
TEST_FROM = dt.date(2026, 10, 22)
# A 14B model on a 16 GB Mac takes minutes per turn with this prompt.
BENCH_TIMEOUT_SECONDS = 600.0
USER_TURN_MARKER = "TODAY'S P&L EXPLAIN"

ALERT = "alert"
INCONCLUSIVE = "inconclusive"
CLEAN = "clean"

_SIGNED = r"[-+]?\d[\d,]*(?:\.\d+)?"
_TOTAL = re.compile(
    rf"TOTAL: actual={_SIGNED}\s+explained={_SIGNED}\s+residual=(?P<r>{_SIGNED}) \+- "
    rf"{_SIGNED}\s+\((?P<bp>{_SIGNED}) \+- (?P<band>{_SIGNED})bp of NAV"
)
_POSITION = re.compile(
    rf"^\s+(?P<pid>P\w+): (?P<r>{_SIGNED}) \+- {_SIGNED}\s+"
    rf"\((?P<bp>{_SIGNED}) \+- (?P<band>{_SIGNED})bp of NAV\)",
    re.MULTILINE,
)
_NOISE_WORDS = re.compile(
    r"\bnoise\b|standard error|monte carlo|\bMC\b|inconclusive|\b2 ?SE\b", re.IGNORECASE
)


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


@dataclass
class AnswerKey:
    """What one day's P&L explain says, read from the text the model was given."""

    status: str  # ALERT, INCONCLUSIVE or CLEAN
    total_residual: float
    total_residual_bp: float
    total_band_bp: float  # 2 SE
    position_residual: dict[str, float] = field(default_factory=dict)
    position_residual_bp: dict[str, float] = field(default_factory=dict)

    @property
    def top_position(self) -> str | None:
        nonzero = {p: r for p, r in self.position_residual.items() if r}
        return max(nonzero, key=lambda p: abs(nonzero[p])) if nonzero else None

    @property
    def noise_relevant(self) -> bool:
        return self.status == INCONCLUSIVE or abs(self.total_residual_bp) <= self.total_band_bp

    def shares_pct(self) -> list[float]:
        """Each position's share of the total residual, in %, as the daily run
        allows the model to compute."""
        total = sum(self.position_residual.values())
        return [100.0 * v / total for v in self.position_residual.values()] if total else []


def answer_key(user_turn: str) -> AnswerKey:
    """Parse the day's explain (the part before the recent history)."""
    explain = user_turn.split("RECENT HISTORY")[0]
    total = _TOTAL.search(explain)
    if total is None:
        raise ValueError("no TOTAL line in the explain")
    if "ALERT:" in explain:
        status = ALERT
    elif "INCONCLUSIVE:" in explain:
        status = INCONCLUSIVE
    else:
        status = CLEAN
    positions = list(_POSITION.finditer(explain))
    return AnswerKey(
        status=status,
        total_residual=_num(total.group("r")),
        total_residual_bp=_num(total.group("bp")),
        total_band_bp=_num(total.group("band")),
        position_residual={m.group("pid"): _num(m.group("r")) for m in positions},
        position_residual_bp={m.group("pid"): _num(m.group("bp")) for m in positions},
    )


def _states(claims: list[factcheck.Claim], dollars: float, bp: float) -> bool:
    return any(
        factcheck.matches(c, [bp] if c.unit == factcheck.BP else [dollars])
        for c in claims
        if c.unit in (factcheck.BP, factcheck.PLAIN)
    )


def score(investigation: AIInvestigationResult, key: AnswerKey) -> dict[str, Any]:
    """The computed scores for one answer; None where a score does not apply to
    the day, or when there was no answer."""
    empty: dict[str, Any] = dict.fromkeys(
        ("grounded", "status_ok", "top_named", "noise_ok", "complete", "called_tool")
    )
    if not investigation.ai_available or investigation.factcheck is None:
        return empty
    flagged = [pid for pid, _ in investigation.flagged_positions if pid]
    texts = [investigation.summary, investigation.root_cause_hypothesis]
    texts += [reason for _, reason in investigation.flagged_positions]
    prose = " ".join(texts)
    claims = [c for t in texts for c in factcheck.extract_claims(t)]
    top = key.top_position
    not_clean = key.status != CLEAN and top is not None

    out = dict(empty)
    out["grounded"] = investigation.factcheck.status == "verified"
    if key.status == ALERT:
        out["status_ok"] = bool(flagged)
    elif key.status == CLEAN:
        out["status_ok"] = not flagged
    if not_clean:
        out["top_named"] = top in flagged or re.search(rf"\b{top}\b", prose) is not None
    if key.noise_relevant:
        out["noise_ok"] = _NOISE_WORDS.search(prose) is not None
    complete = _states(claims, key.total_residual, key.total_residual_bp)
    if not_clean and top is not None:
        complete = complete and _states(
            claims, key.position_residual[top], key.position_residual_bp[top]
        )
    out["complete"] = complete
    if investigation.factcheck.status == "unparsed":
        # Not the answer format the daily run reads: no flags, so a clean day
        # would otherwise be scored right for free. Tool use is still scored.
        out = {k: (v if v is None else False) for k, v in out.items()}
    if key.status == ALERT and top is not None:
        out["called_tool"] = any(
            t.tool_name == "what_if_reprice" and json.loads(t.arguments).get("position_id") == top
            for t in investigation.trace
        )
    return out


@dataclass
class BenchItem:
    day0: dt.date
    day1: dt.date
    data_turn: str  # the explain and history, without the closing line
    # The book's upcoming events on day1, rebuilt from the stored closes.
    events: list[str] = field(default_factory=list)

    @property
    def split(self) -> str:
        return "test" if self.day1 >= TEST_FROM else "dev"

    def user_turn(self) -> str:
        """The user turn the daily run would send today for this day's data."""
        key = answer_key(self.data_turn)
        return (
            self.data_turn
            + events_section(self.events)
            + closing_line(key.status == ALERT, key.status == CLEAN, key.top_position)
        )


def data_turn(stored: str) -> str:
    """A stored user turn without its closing line: everything up to the blank
    line after the recent history."""
    history = stored.find("RECENT HISTORY")
    end = stored.find("\n\n", history)
    return stored if history < 0 or end < 0 else stored[: end + 1]


def load_items(cfg: BaseConfig, portfolio_path: str = "configs/portfolio.yaml") -> list[BenchItem]:
    """One item per day with a stored investigation: the data it was given, and
    the day's upcoming events as the daily run would add them now."""
    root = Path(cfg.paths.curated) / "ai_investigations"
    if not root.exists():
        return []
    df = store.query(
        "SELECT asof_date, day0, prompt FROM t ORDER BY asof_date", views={"t": str(root)}
    ).to_pandas()
    items = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"] or "")
        start = prompt.find(USER_TURN_MARKER)
        if start < 0:
            continue
        day1 = _as_date(r["asof_date"])
        items.append(
            BenchItem(
                _as_date(r["day0"]),
                day1,
                data_turn(prompt[start:]),
                upcoming_events(cfg, portfolio_path, day1),
            )
        )
    return items


def _as_date(value: Any) -> dt.date:
    return value if isinstance(value, dt.date) else pd.Timestamp(value).date()


class ToolCache:
    """what_if_reprice results by (day, arguments), kept across runs: a reprice
    loads the day's market state and runs Monte Carlo, and every model asking
    the same question should get the same answer. Errors are not cached."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def executor(
        self, cfg: BaseConfig, portfolio_path: str, day1: dt.date, project_root: Path
    ) -> ToolExecutor:
        def execute(name: str, arguments: dict[str, Any]) -> Any:
            def run() -> Any:
                return tools.execute_tool_call(
                    name, arguments, cfg, portfolio_path, day1, project_root
                )

            if name != "what_if_reprice":
                return run()
            key = json.dumps([day1.isoformat(), arguments], sort_keys=True)
            if key in self.entries:
                return self.entries[key]
            result = run()
            if isinstance(result, dict) and "error" not in result:
                self.entries[key] = result
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True))
            return result

        return execute


def model_key(model: str) -> str:
    """A model name safe as a partition directory: "qwen2.5:7b" -> "qwen2.5-7b"."""
    return re.sub(r"[^\w.]+", "-", model)


def _chat_for(model: str, seed: int) -> Chat:
    def chat(
        messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        return ollama_client.chat(
            messages,
            tools=tool_specs,
            model=model,
            timeout=BENCH_TIMEOUT_SECONDS,
            options={"seed": seed},
        )

    return chat


def run_item(
    item: BenchItem, model: str, sample: int, execute: ToolExecutor
) -> tuple[AIInvestigationResult, AnswerKey, float]:
    """One answer: the daily run's loop and fact-check, this model, seed `sample`."""
    key = answer_key(item.data_turn)
    chat = _chat_for(model, seed=sample)
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": item.user_turn()},
    ]
    start = time.perf_counter()
    raw, trace, messages = tool_loop(messages, execute, chat)
    investigation = check_and_retry(
        item.day0, item.day1, raw, trace, messages, key.shares_pct(), model=model, chat=chat
    )
    return investigation, key, time.perf_counter() - start


def _row(
    item: BenchItem,
    model: str,
    sample: int,
    investigation: AIInvestigationResult,
    key: AnswerKey,
    seconds: float,
) -> dict[str, Any]:
    fc = investigation.factcheck
    return {
        "model_key": model_key(model),
        "prompt_version": PROMPT_VERSION,
        "asof_date": item.day1,
        "model": model,
        "day0": item.day0,
        "sample": sample,
        "split": item.split,
        "day_status": key.status,
        "ai_available": investigation.ai_available,
        "factcheck_status": None if fc is None else fc.status,
        "n_numbers": None if fc is None else fc.n_numbers,
        "n_unsupported": None if fc is None else len(fc.unsupported),
        "n_mislabelled": None if fc is None else len(fc.mislabelled),
        "unsupported": None if fc is None else json.dumps(fc.unsupported),
        "mislabelled": None if fc is None else json.dumps(fc.mislabelled),
        "retried": investigation.factcheck_retried,
        **score(investigation, key),
        "n_tool_calls": len(investigation.trace),
        "seconds": seconds,
        "summary": investigation.summary,
        "root_cause_hypothesis": investigation.root_cause_hypothesis,
        "flagged_positions": json.dumps(investigation.flagged_positions),
        "trace": json.dumps([_trace_dict(t) for t in investigation.trace]),
        "raw_response": investigation.raw_response,
    }


def _trace_dict(t: ToolCallTrace) -> dict[str, Any]:
    return {
        "round": t.round,
        "tool": t.tool_name,
        "arguments": t.arguments,
        "result": t.result_summary,
    }


def run_benchmark(
    cfg: BaseConfig,
    models: list[str],
    samples: int,
    portfolio_path: str,
    days: list[dt.date] | None = None,
    split: str | None = None,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Every (model, day, sample), persisted one (model, day) at a time, so an
    interrupted run keeps what it finished and a rerun replaces that day."""
    project_root = project_root or Path.cwd()
    curated = Path(cfg.paths.curated)
    items = [
        i
        for i in load_items(cfg, portfolio_path)
        if (days is None or i.day1 in days) and (split is None or i.split == split)
    ]
    cache = ToolCache(curated / "agent_bench" / "tool_cache.json")
    rows = []
    for model in models:
        for item in items:
            execute = cache.executor(cfg, portfolio_path, item.day1, project_root)
            day_rows = []
            for sample in range(samples):
                investigation, key, seconds = run_item(item, model, sample, execute)
                row = _row(item, model, sample, investigation, key, seconds)
                day_rows.append(row)
                log(
                    f"{model} {item.day1} #{sample}: {row['factcheck_status'] or 'no answer'}, "
                    f"{len(investigation.trace)} tool calls, {seconds:.0f}s"
                )
            table = validate(
                pd.DataFrame(day_rows), AGENT_BENCH_SCHEMA, AGENT_BENCH_REQUIRED_NOT_NULL
            )
            store.write_partitioned(table, curated / "agent_bench", PARTITION_COLS)
            rows += day_rows
    return pd.DataFrame(rows)


SCORES = ("grounded", "status_ok", "top_named", "noise_ok", "complete", "called_tool")
PARTITION_COLS = ["model_key", "prompt_version", "asof_date"]


def _rate(scores: pd.Series) -> str:
    applies = scores.dropna()
    return f"{applies.astype(bool).mean():.0%} of {len(applies)}" if len(applies) else "-"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, prompt version and split: answers, how many came back, and each
    score's pass rate over the answers it applies to (with that count). The
    status score is shown as alert days flagged and clean days left clean."""
    out = []
    for (model, version, split), g in df.groupby(["model", "prompt_version", "split"]):
        row: dict[str, Any] = {
            "model": model,
            "prompt": version,
            "split": split,
            "answers": len(g),
            "available": int(g["ai_available"].sum()),
        }
        for s in SCORES:
            if s == "status_ok":
                row["alert_flagged"] = _rate(g.loc[g["day_status"] == ALERT, s])
                row["clean_left_clean"] = _rate(g.loc[g["day_status"] == CLEAN, s])
            else:
                row[s] = _rate(g[s])
        row["median_s"] = round(float(g["seconds"].median()))
        out.append(row)
    return pd.DataFrame(out)


def load_results(cfg: BaseConfig) -> pd.DataFrame:
    root = Path(cfg.paths.curated) / "agent_bench"
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame()
    return store.query("SELECT * FROM t", views={"t": str(root)}).to_pandas()
