import datetime as dt
import json

import pandas as pd
import pytest

from eqdrisk.agent.labels import (
    factcheck_agreement,
    label_sample,
    load_labels,
    numbers_in,
    save_label,
)
from eqdrisk.config import BaseConfig, Paths, Universe

DAY = dt.date(2026, 11, 9)


def _answer(summary, unsupported=(), mislabelled=(), flagged=(), **extra):
    return pd.Series(
        {
            "model_key": "qwen2.5-7b",
            "asof_date": DAY,
            "sample": 0,
            "model": "qwen2.5:7b",
            "ai_available": True,
            "summary": summary,
            "root_cause_hypothesis": "",
            "flagged_positions": json.dumps(list(flagged)),
            "unsupported": json.dumps(list(unsupported)),
            "mislabelled": json.dumps(list(mislabelled)),
            **extra,
        }
    )


def test_numbers_carry_the_fact_check_verdict_once_each():
    answer = _answer(
        "residual +1362.98 for the book; P008 +6,979.95; again +1362.98",
        mislabelled=["+1362.98 (called a residual; in the data it is ...)"],
        flagged=[["P008", "residual 19.1bp"]],
    )
    numbers = numbers_in(answer)
    assert [n.text for n in numbers] == ["+1362.98", "+6,979.95", "+1362.98", "19.1bp"]
    # Flagged once in the verdict list, so only the first occurrence is flagged.
    assert [n.flagged for n in numbers] == [True, False, False, False]
    assert "for the book" in numbers[0].context


def test_label_sample_is_fixed_and_shares_models_evenly():
    rows = [
        {"model": m, "asof_date": DAY - dt.timedelta(days=d), "sample": s, "ai_available": True}
        for m in ("a", "b", "c")
        for d in range(10)
        for s in range(2)
    ]
    rows.append({"model": "a", "asof_date": DAY, "sample": 9, "ai_available": False})
    results = pd.DataFrame(rows)

    picked = label_sample(results, n=10)
    assert picked.equals(label_sample(results, n=10))
    assert sorted(picked["model"].value_counts().tolist()) == [3, 3, 4]
    assert picked["ai_available"].all()
    # Only the labelled prompt version is sampled.
    later = results.assign(prompt_version="v2")
    assert label_sample(pd.concat([results.assign(prompt_version="v1"), later]), n=10).equals(
        label_sample(results.assign(prompt_version="v1"), n=10)
    )
    # Spread over days: no day repeats within a model before every day is used.
    for _, g in picked.groupby("model"):
        assert g["asof_date"].is_unique


def test_adding_a_model_keeps_what_was_already_in_the_sample():
    rows = [
        {"model": m, "asof_date": DAY - dt.timedelta(days=d), "sample": s, "ai_available": True}
        for m in ("a", "b", "c", "d")
        for d in range(10)
        for s in range(2)
    ]
    results = pd.DataFrame(rows)
    before = label_sample(results[results["model"] != "c"], n=12)
    after = label_sample(results, n=12)

    def keys(df):
        return set(zip(df["model"], df["asof_date"], df["sample"], strict=True))

    assert keys(after[after["model"] != "c"]) <= keys(before)


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=DAY,
        universe=Universe(index=["SPX"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path / "curated")),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def test_a_saved_label_replaces_the_last_one_for_that_answer(tmp_path):
    cfg = _cfg(tmp_path)
    answer = _answer("residual +1362.98, P008 +6,979.95", mislabelled=["+1362.98 (called x)"])
    save_label(cfg, answer, [True, True], "good")
    save_label(cfg, answer, [False, True], "wrong", note="a step's residual, not the book's")

    labels = load_labels(cfg)
    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["asof_date"] == DAY and row["sample"] == 0 and row["reasoning"] == "wrong"
    assert json.loads(row["numbers"])[0] == {"text": "+1362.98", "right": False, "flagged": True}

    with pytest.raises(ValueError, match="2 numbers"):
        save_label(cfg, answer, [True], "good")
    with pytest.raises(ValueError, match="reasoning"):
        save_label(cfg, answer, [True, True], "fine")


def test_factcheck_agreement_counts_recall_and_precision():
    numbers = [
        {"right": False, "flagged": True},  # caught
        {"right": False, "flagged": False},  # missed
        {"right": True, "flagged": True},  # false alarm
        {"right": True, "flagged": False},
    ]
    agreement = factcheck_agreement(pd.DataFrame({"numbers": [json.dumps(numbers)]}))
    assert (agreement.numbers, agreement.wrong, agreement.caught) == (4, 2, 1)
    assert agreement.recall == 0.5 and agreement.precision == 0.5
