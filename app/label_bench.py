"""Labelling page for the agent benchmark (planning/agent_benchmark_plan.md).

Shows one benchmark answer at a time with everything its model was given (the
day's P&L explain, the history, each tool call and its result) and asks, for
every number in the answer, whether it is right, then grades the reasoning.
Blind on purpose: the model's name and the fact-check verdict are not shown, so
the labels can be used to measure both.

Run via `eqdrisk labelbench`.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from eqdrisk.agent.benchmark import load_items, load_results
from eqdrisk.agent.labels import (
    REASONING_GRADES,
    label_sample,
    load_labels,
    numbers_in,
    save_label,
)
from eqdrisk.config import BaseConfig

CONFIG_PATH = "configs/base.yaml"

GUIDE = """
**A number is right** when it is in the data on the left and the answer calls it what it is
there. Wrong examples: a what-if reprice result described as something that happened; one
step's residual called the book's residual; a dollar figure given as bp; a number not in the
data at all.

**Reasoning:**
- **good**: the root cause fits the data: it names the position carrying the residual, treats a
  residual inside its noise band as noise, and does not present a what-if as real.
- **weak**: nothing in it is wrong, but it is generic or misses the day's main fact.
- **wrong**: it contradicts the data (wrong position, a breach on a clean day, a what-if
  presented as the day's P&L, noise treated as a model failure).
"""


def _key(row: pd.Series) -> tuple[str, object, int]:
    return (row["model_key"], row["asof_date"], int(row["sample"]))


def main() -> None:
    st.set_page_config(page_title="Agent benchmark labels", layout="wide")
    st.title("Agent benchmark: hand labels")
    cfg = BaseConfig.from_yaml(CONFIG_PATH)

    results = load_results(cfg)
    if results.empty:
        st.info("No benchmark answers yet: run `eqdrisk agentbench` first.")
        return
    results["asof_date"] = pd.to_datetime(results["asof_date"]).dt.date
    sample = label_sample(results).reset_index(drop=True)
    labels = load_labels(cfg)
    done = {(r["model_key"], r["asof_date"], int(r["sample"])) for _, r in labels.iterrows()}
    is_done = [_key(r) in done for _, r in sample.iterrows()]

    st.progress(sum(is_done) / len(sample), text=f"{sum(is_done)} of {len(sample)} labelled")
    with st.expander("How to label", expanded=not done):
        st.markdown(GUIDE)

    first_open = next((i for i, d in enumerate(is_done) if not d), None)
    choices = [f"Answer {i + 1}{' (done)' if d else ''}" for i, d in enumerate(is_done)]
    default = first_open if first_open is not None else 0
    picked = st.selectbox(
        "Answer", range(len(sample)), index=default, format_func=choices.__getitem__
    )
    answer = sample.iloc[picked]

    user_turns = {i.day1: i.user_turn for i in load_items(cfg)}
    left, right = st.columns(2)
    with left:
        st.subheader("What the model was given")
        st.code(user_turns.get(answer["asof_date"], "(day's prompt not found)"), language=None)
        for call in json.loads(answer["trace"] or "[]"):
            with st.expander(f"Tool call, round {call['round']}: {call['tool']}", expanded=True):
                st.write("Arguments:")
                st.code(call["arguments"], language="json")
                st.write("Result:")
                st.code(call.get("result", ""), language="json")
    with right:
        st.subheader("The answer")
        st.markdown(f"**Summary.** {answer['summary']}")
        st.markdown(f"**Root cause.** {answer['root_cause_hypothesis']}")
        flagged = json.loads(answer["flagged_positions"] or "[]")
        if flagged:
            st.markdown("**Flagged for review:**")
            for pid, reason in flagged:
                st.markdown(f"- {pid}: {reason}")

        numbers = numbers_in(answer)
        with st.form(f"label-{picked}"):
            st.markdown(f"**Each number** ({len(numbers)})")
            marks = [
                st.radio(
                    f"`{num.text}` ...{num.context}...",
                    ["right", "wrong"],
                    index=None,
                    horizontal=True,
                    key=f"n-{picked}-{i}",
                )
                for i, num in enumerate(numbers)
            ]
            reasoning = st.radio(
                "**Reasoning**", REASONING_GRADES, index=None, horizontal=True, key=f"r-{picked}"
            )
            note = st.text_input("Note (optional)", key=f"note-{picked}")
            if st.form_submit_button("Save and next"):
                if None in marks or reasoning is None:
                    st.error("Mark every number and the reasoning first.")
                else:
                    save_label(cfg, answer, [m == "right" for m in marks], str(reasoning), note)
                    st.rerun()


main()
