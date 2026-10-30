import datetime as dt
from unittest.mock import patch

from eqdrisk.agent.factcheck import check, extract_claims, labelled_sources
from eqdrisk.agent.investigate import run_daily_investigation
from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io import store
from eqdrisk.pricing.pnl_explain import PnLExplainResult, StepResult

# The real explain lines the model was given on 2026-10-19.
CONTEXT_2026_10_19 = "\n".join(
    [
        "P&L explain — 2026-10-16 -> 2026-10-19  (NAV=4,355,914.04)",
        "  time: actual=+15,945.19  explained=+9,241.42  residual=+6,703.77 +- 3,478.95 (2 SE)",
        "  spot: actual=-989.10  explained=+9,715.98  residual=-10,705.08 +- 784.24 (2 SE)",
        "  TOTAL: actual=+18,914.19  explained=+23,911.61  residual=-4,997.42 +- 3,369.79"
        "  (-11.5 +- 7.7bp of NAV, 2 SE)",
        "  residual by position:",
        "    P008: -5,020.37 +- 3,379.65  (-11.5 +- 7.8bp of NAV)",
        "  ALERT: total residual -11.5bp of NAV (+- 7.7bp, 2 SE) exceeds the 5bp threshold",
    ]
)


def test_catches_dollars_labelled_as_bp():
    """The real 2026-10-19 failure: correct dollar figures, wrong unit."""
    answer = (
        "a significant residual of -4,997.42bp of NAV, primarily driven by position P008 "
        "with a residual of -5,020.37bp of NAV"
    )
    result = check([answer], CONTEXT_2026_10_19)
    assert result.status == "unsupported_numbers"
    assert result.unsupported == ["-4,997.42bp", "-5,020.37bp"]


def test_accepts_the_same_figures_with_their_real_units():
    answer = (
        "The total residual was -4,997.42 (-11.5bp of NAV), above the 5bp threshold; P008 "
        "(-5,020.37) is 100% of it, with a $10,705.08 shortfall in the spot step."
    )
    result = check([answer], CONTEXT_2026_10_19, extra_pct=[100.46])
    assert result.status == "verified"
    assert result.n_numbers == 6


def test_sign_and_rounding_are_tolerated_but_invented_numbers_are_not():
    assert check(["a loss of 11.5bp"], CONTEXT_2026_10_19).status == "verified"
    assert check(["a residual of about -11.53bp"], CONTEXT_2026_10_19).status == "verified"
    assert check(["a residual of -4,997"], CONTEXT_2026_10_19).status == "verified"
    invented = check(["a residual of -14.2bp and a $250,000 loss"], CONTEXT_2026_10_19)
    assert invented.unsupported == ["-14.2bp", "$250,000"]


def test_percent_claims_match_tool_fractions():
    context = 'what_if_reprice -> {"spot_shock_pct": -0.1, "vol_shock_pct": 0.25}'
    assert check(["NVDA down 10% and vol up 25%"], context).status == "verified"
    assert check(["NVDA down 15%"], context).status == "unsupported_numbers"


def test_dates_ids_and_small_counts_are_not_checked():
    claims = extract_claims("On 2026-10-19 P008 (model qwen2.5:7b) took 3 rounds")
    assert claims == []


# The what_if_reprice result the model got on 2026-10-26 (P008, spot -10%, vol +25%).
WHAT_IF_2026_10_26 = {
    "position_id": "P008",
    "spot_shock_pct": -0.1,
    "vol_shock_pct": 0.25,
    "base_price": 4748524.610855177,
    "shocked_price": 4200504.573801067,
    "price_delta": -548020.0370541103,
    "shocked_delta": 17862.66234384465,
}


def _labelled_check(answer: str, what_if: dict | None = None, shares: list[float] | None = None):
    context = CONTEXT_2026_10_19 + ("\n" + str(what_if) if what_if else "")
    labels = labelled_sources(CONTEXT_2026_10_19, [what_if] if what_if else [], shares)
    return check([answer], context, shares, labels)


def test_what_if_results_called_actual_or_explained_are_mislabelled():
    """The real 2026-10-26 answer: real numbers, but what-if results written up
    as what happened."""
    answer = (
        "The position's actual price change of -548,020.04 differs from the explained "
        "change of 17,862.66."
    )
    result = _labelled_check(answer, WHAT_IF_2026_10_26)
    assert result.status == "mislabelled_numbers"
    assert not result.unsupported
    assert result.mislabelled == [
        "-548,020.04 (called an actual P&L or value; in the data it is a what-if reprice result)",
        "17,862.66 (called an explained P&L; in the data it is a what-if reprice result)",
    ]
    assert "described as the wrong quantity" in result.render()


def test_the_same_numbers_described_as_what_ifs_are_verified():
    answer = (
        "The what_if_reprice call shows a price change of -548,020.04 under a 10% spot fall, "
        "with a shocked delta of 17,862.66; the position is worth 4,748,524.61 today."
    )
    assert _labelled_check(answer, WHAT_IF_2026_10_26).status == "verified"


def test_a_residual_called_the_actual_pnl_is_mislabelled():
    wrong = _labelled_check("P008 had a large negative actual P&L (-5,020.37).")
    assert wrong.mislabelled == [
        "-5,020.37 (called an actual P&L or value; in the data it is a residual)"
    ]
    right = _labelled_check("The book's actual P&L was +18,914.19, with P008's residual -5,020.37.")
    assert right.status == "verified"


def test_threshold_noise_and_unlabelled_wording_fall_back_to_the_plain_check():
    for answer in [
        "The residual exceeds the 5bp threshold.",
        "The total residual is inconclusive, with Monte Carlo noise wider than the 5bp threshold.",
        "an unexpected -4,997.42 move",  # "unexpected" is not "expected"
        "P008 contributed -5,020.37.",  # no quantity named: existence only
        "a residual of -11.5bp with a noise band of 7.7bp",
    ]:
        assert _labelled_check(answer).status == "verified", answer


def test_without_labels_only_existence_and_unit_are_checked():
    assert check(["an actual P&L of -5,020.37"], CONTEXT_2026_10_19).status == "verified"


def _cfg(tmp_path) -> BaseConfig:
    return BaseConfig(
        run_date=dt.date(2026, 10, 16),
        universe=Universe(index=["TEST"], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path)),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _result() -> PnLExplainResult:
    # Residual 200 on NAV 1,000,000 = 2.0bp.
    return PnLExplainResult(
        day0=dt.date(2026, 10, 16),
        day1=dt.date(2026, 10, 19),
        steps=[StepResult(step="vol", actual_pnl=1000.0, explained_pnl=800.0)],
        by_position_residual={"P1": 200.0},
        nav=1_000_000.0,
    )


def _answer(summary: str) -> dict:
    content = (
        f'{{"summary": "{summary}", "root_cause_hypothesis": "vega on P1", '
        '"confidence": "low", "flagged_positions": [], "proposed_changes": []}'
    )
    return {"role": "assistant", "content": content}


def _run(tmp_path, replies):
    with (
        patch("eqdrisk.agent.ollama_client.is_available", return_value=True),
        patch("eqdrisk.agent.investigate.ollama_client.chat", side_effect=replies) as chat,
    ):
        investigation = run_daily_investigation(
            _cfg(tmp_path), _result(), "configs/portfolio.yaml", project_root=tmp_path
        )
    return investigation, chat


def test_clean_answer_is_verified_without_a_retry(tmp_path):
    investigation, chat = _run(tmp_path, [_answer("residual of 200.00, 2.0bp of NAV")])
    assert chat.call_count == 1
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "verified"
    assert not investigation.factcheck_retried
    assert "1,000,000.00" in investigation.prompt


def test_wrong_unit_triggers_one_retry_and_the_correction_is_kept(tmp_path):
    investigation, chat = _run(
        tmp_path,
        [_answer("residual of 200.00bp of NAV"), _answer("residual of 200.00, 2.0bp of NAV")],
    )
    assert chat.call_count == 2
    retry_prompt = chat.call_args_list[1].args[0][-1]["content"]
    assert "200.00bp" in retry_prompt
    assert investigation.factcheck_retried
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "verified"
    assert investigation.summary == "residual of 200.00, 2.0bp of NAV"


def test_still_wrong_after_the_retry_is_flagged_and_stored(tmp_path):
    investigation, _ = _run(
        tmp_path,
        [_answer("residual of 200.00bp of NAV"), _answer("residual of 350bp of NAV")],
    )
    assert investigation.factcheck is not None
    assert investigation.factcheck.unsupported == ["350bp"]
    assert "FACT-CHECK: 1 of" in investigation.render()

    stored = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigations")}
    ).to_pandas()
    row = stored.iloc[0]
    assert row["factcheck_status"] == "unsupported_numbers"
    assert row["factcheck_unsupported"] == '["350bp"]'
    assert bool(row["factcheck_retried"])
    assert "NAV=1,000,000.00" in row["prompt"]


def _broken(summary: str) -> dict:
    # Arithmetic inside the JSON, as qwen2.5 wrote on 2026-10-02.
    content = (
        f'{{"summary": "{summary}", "root_cause_hypothesis": "delta on P1", '
        '"confidence": "low", "flagged_positions": [], "proposed_changes": '
        '[{"parameter": "delta", "current_value": 23231.16, "suggested_value": 23231.16 - 548.97, '
        '"rationale": "x"}]}'
    )
    return {"role": "assistant", "content": content}


def test_invalid_json_uses_the_retry_and_a_valid_retry_is_checked(tmp_path):
    investigation, chat = _run(
        tmp_path, [_broken("residual of 200.00"), _answer("residual of 200.00, 2.0bp of NAV")]
    )
    assert chat.call_count == 2
    assert "not valid JSON" in chat.call_args_list[1].args[0][-1]["content"]
    assert investigation.parse_error is None
    assert investigation.factcheck_retried
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "verified"


def test_still_invalid_json_after_the_retry_is_checked_on_raw_text_and_flagged(tmp_path):
    investigation, chat = _run(
        tmp_path, [_broken("residual of 200.00"), _broken("residual of 200.00")]
    )
    assert chat.call_count == 2
    assert investigation.parse_error is not None
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "unparsed"
    assert investigation.factcheck.n_numbers > 0
    assert sorted(investigation.factcheck.unsupported) == ["23231.16", "23231.16", "548.97"]
    assert "FACT-CHECK: the answer was not valid JSON" in investigation.render()

    stored = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigations")}
    ).to_pandas()
    assert stored.iloc[0]["factcheck_status"] == "unparsed"
    assert not bool(stored.iloc[0]["factcheck_retried"])


def test_invalid_json_with_supported_numbers_is_still_flagged_unparsed():
    from eqdrisk.agent.factcheck import FactCheckResult

    result = FactCheckResult(n_numbers=2, unsupported=[], parsed=False)
    assert result.status == "unparsed"
    assert result.render().startswith("FACT-CHECK:")


def test_mislabelled_numbers_get_the_retry_and_are_stored_if_they_survive(tmp_path):
    # In _result() 200.00 is the residual; the actual P&L is 1,000.00.
    investigation, chat = _run(
        tmp_path, [_answer("an actual P&L of 200.00"), _answer("an actual P&L of 200.00")]
    )
    assert chat.call_count == 2
    retry_prompt = chat.call_args_list[1].args[0][-1]["content"]
    assert "calls them a different quantity" in retry_prompt
    assert "200.00 (called an actual P&L or value; in the data it is a residual)" in retry_prompt
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "mislabelled_numbers"

    stored = store.query(
        "SELECT * FROM t", views={"t": str(tmp_path / "ai_investigations")}
    ).to_pandas()
    row = stored.iloc[0]
    assert row["factcheck_status"] == "mislabelled_numbers"
    assert row["factcheck_unsupported"] == "[]"
    assert "a residual" in row["factcheck_mislabelled"]


def test_a_corrected_label_after_the_retry_is_verified(tmp_path):
    investigation, _ = _run(
        tmp_path,
        [_answer("an actual P&L of 200.00"), _answer("an actual P&L of 1,000.00, residual 200.00")],
    )
    assert investigation.factcheck_retried
    assert investigation.factcheck is not None
    assert investigation.factcheck.status == "verified"
