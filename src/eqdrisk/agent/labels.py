"""Hand labels for the agent benchmark (planning/agent_benchmark_plan.md):
which answers to label, the numbers in each, saving a label, and what the
labels say about the fact-check.

The labelling page (`app/label_bench.py`) is blind: it shows the day's data,
the tool results and the answer, but not the model or the fact-check verdict,
so a label is the labeller's own reading. Each number is marked right (in the
data and called what it is) or wrong; the reasoning is marked good, weak or
wrong. The fact-check verdicts stored with each answer are compared with these
afterwards: its recall (wrong numbers it caught) had never been measured, only
its precision on the numbers it flagged (planning/ai_factcheck_labels_plan.md).
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eqdrisk.agent import factcheck
from eqdrisk.config import BaseConfig
from eqdrisk.io import store
from eqdrisk.io.schemas import (
    AGENT_BENCH_LABEL_REQUIRED_NOT_NULL,
    AGENT_BENCH_LABEL_SCHEMA,
    validate,
)

LABEL_TARGET = 100
# The labels are of the first full run (2026-11-09); later prompt versions are
# scored by the computed scores only.
LABEL_PROMPT_VERSION = "v1"
REASONING_GRADES = ("good", "weak", "wrong")
ANSWER_KEY_COLS = ["model_key", "asof_date", "sample"]


def answer_texts(row: pd.Series | dict[str, Any]) -> list[str]:
    """The fields the fact-check reads, in the order it reads them."""
    flagged = json.loads(row["flagged_positions"] or "[]")
    return [str(row["summary"] or ""), str(row["root_cause_hypothesis"] or "")] + [
        str(reason) for _, reason in flagged
    ]


@dataclass
class NumberInAnswer:
    text: str
    context: str  # the words around it, for the labelling page
    flagged: bool  # the fact-check called it unsupported or mislabelled


def numbers_in(row: pd.Series | dict[str, Any]) -> list[NumberInAnswer]:
    """Every checked number in the answer, with the fact-check's verdict on it.
    The verdicts are lists of number texts (`mislabelled` entries start with
    the text), each used up once, so a figure written twice is matched once."""
    unsupported = list(json.loads(row["unsupported"] or "[]"))
    mislabelled = [m.split(" (called")[0] for m in json.loads(row["mislabelled"] or "[]")]
    out = []
    for text in answer_texts(row):
        cursor = 0
        for claim in factcheck.extract_claims(text):
            at = text.find(claim.text, cursor)
            cursor = at + len(claim.text) if at >= 0 else cursor
            start = max(at, 0)
            context = text[max(0, start - 60) : start + len(claim.text) + 40]
            flagged = False
            for pool in (unsupported, mislabelled):
                if claim.text in pool:
                    pool.remove(claim.text)
                    flagged = True
                    break
            out.append(NumberInAnswer(claim.text, context, flagged))
    return out


def label_sample(results: pd.DataFrame, n: int = LABEL_TARGET, seed: int = 0) -> pd.DataFrame:
    """`n` answers to label, the same every time: an equal share per model,
    spread over days (shuffled day order, one sample per day before repeats),
    then shuffled together so the page does not show one model's run in a row.
    Each model's order has its own seed, so adding a model later only shortens
    the others' shares: what was already labelled stays in the sample. Only
    answers under `LABEL_PROMPT_VERSION` are labelled."""
    answered = results[results["ai_available"].astype(bool)]
    if "prompt_version" in answered:
        answered = answered[answered["prompt_version"] == LABEL_PROMPT_VERSION]
    if answered.empty:
        return answered
    models = sorted(answered["model"].unique())
    per_model, extra = divmod(n, len(models))
    picked = []
    for i, model in enumerate(models):
        rows = answered[answered["model"] == model].sort_values(["asof_date", "sample"])
        days = sorted(rows["asof_date"].unique())
        random.Random(f"{seed}:{model}").shuffle(days)
        order = []
        for sample in sorted(rows["sample"].unique()):
            for day in days:
                match = rows[(rows["asof_date"] == day) & (rows["sample"] == sample)]
                order += list(match.index)
        picked += order[: per_model + (1 if i < extra else 0)]
    random.Random(seed).shuffle(picked)
    return answered.loc[picked]


def save_label(
    cfg: BaseConfig,
    answer: pd.Series,
    number_right: list[bool],
    reasoning: str,
    note: str = "",
) -> None:
    """One answer's labels; saving it again replaces them."""
    if reasoning not in REASONING_GRADES:
        raise ValueError(f"reasoning must be one of {REASONING_GRADES}")
    numbers = numbers_in(answer)
    if len(number_right) != len(numbers):
        raise ValueError(f"{len(numbers)} numbers in the answer, {len(number_right)} labels")
    row = {
        "model_key": answer["model_key"],
        "asof_date": answer["asof_date"],
        "sample": int(answer["sample"]),
        "model": answer["model"],
        "numbers": json.dumps(
            [
                {"text": num.text, "right": right, "flagged": num.flagged}
                for num, right in zip(numbers, number_right, strict=True)
            ]
        ),
        "reasoning": reasoning,
        "note": note,
        "labelled_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
    }
    table = validate(
        pd.DataFrame([row]), AGENT_BENCH_LABEL_SCHEMA, AGENT_BENCH_LABEL_REQUIRED_NOT_NULL
    )
    store.write_partitioned(table, _labels_root(cfg), ANSWER_KEY_COLS)


def _labels_root(cfg: BaseConfig) -> Path:
    return Path(cfg.paths.curated) / "agent_bench_labels"


def load_labels(cfg: BaseConfig) -> pd.DataFrame:
    root = _labels_root(cfg)
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame(columns=AGENT_BENCH_LABEL_SCHEMA.names)
    df = store.query("SELECT * FROM t", views={"t": str(root)}).to_pandas()
    df["asof_date"] = [_as_date(d) for d in df["asof_date"]]
    df["sample"] = df["sample"].astype(int)
    return df


def _as_date(value: Any) -> dt.date:
    return value if isinstance(value, dt.date) else pd.Timestamp(value).date()


@dataclass
class FactCheckAgreement:
    """The fact-check against the hand labels, number by number."""

    numbers: int
    wrong: int  # labelled wrong
    caught: int  # labelled wrong and flagged by the fact-check
    false_alarms: int  # labelled right but flagged

    @property
    def recall(self) -> float | None:
        return self.caught / self.wrong if self.wrong else None

    @property
    def precision(self) -> float | None:
        flagged = self.caught + self.false_alarms
        return self.caught / flagged if flagged else None


def factcheck_agreement(labels: pd.DataFrame) -> FactCheckAgreement:
    numbers = [n for raw in labels["numbers"] for n in json.loads(raw)]
    return FactCheckAgreement(
        numbers=len(numbers),
        wrong=sum(not n["right"] for n in numbers),
        caught=sum(not n["right"] and n["flagged"] for n in numbers),
        false_alarms=sum(n["right"] and n["flagged"] for n in numbers),
    )
