"""Fact-check the numbers in an AI investigation against what the model was given.

The local model (`agent/investigate.py`) is told to use only real numbers, but
nothing checked that it did. On 2026-10-19 the explain said "residual -4,997.42 ...
(-11.5 +- 7.7bp of NAV)" and the model wrote "a residual of -4,997.42bp of NAV":
a real dollar figure given the wrong unit, overstating the gap 430x, logged as the
day's finding (planning/ai_factcheck_plan.md).

Every number the model writes, with its unit, must appear in its input (the
prompt and the tool results) carrying a compatible unit:
- a bp claim needs a source tagged bp;
- a % claim needs a % source, or a fraction (tool shocks are fractions: -0.1 is
  10%), or a position's share of the total residual;
- a $ or unitless claim needs any dollar or unitless source.
A claim matches within its own printed precision or 0.5% of the source,
ignoring sign (a "loss of 11.5bp" is the same figure as "-11.5bp"). Dates,
position ids and small integer counts are not checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BP = "bp"
PCT = "pct"
PLAIN = "plain"

RELATIVE_TOLERANCE = 0.005
SMALL_COUNT_LIMIT = 10  # integers below this are counts ("3 rounds"), not data

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# A number not glued to a word (so P008 and model names are skipped), optionally
# signed, $-prefixed, comma-grouped, decimal or in exponent form, then its unit.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-+]?)(?P<dollar>\$?)(?P<num>\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)"
    r"\s*(?P<unit>bp\b|basis points|%|percent)?",
    re.IGNORECASE,
)
# "-11.5 +- 7.7bp": the unit after the band applies to the centre value too.
_BAND = re.compile(
    r"(?P<a>[-+]?\d[\d,]*(?:\.\d+)?)\s*\+-\s*(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>bp|%)",
    re.IGNORECASE,
)


@dataclass
class Claim:
    text: str
    value: float
    unit: str
    decimals: int


@dataclass
class FactCheckResult:
    n_numbers: int = 0
    unsupported: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "unsupported_numbers" if self.unsupported else "verified"

    def render(self) -> str:
        if not self.unsupported:
            return f"fact-check: {self.n_numbers} numbers checked, all found in the data given"
        return (
            f"FACT-CHECK: {len(self.unsupported)} of {self.n_numbers} numbers not in the data "
            f"given (or with the wrong unit): {', '.join(self.unsupported)}"
        )


def _unit(raw_unit: str | None) -> str:
    u = (raw_unit or "").lower()
    if u in ("bp", "basis points"):
        return BP
    if u in ("%", "percent"):
        return PCT
    return PLAIN


def extract_claims(text: str) -> list[Claim]:
    """Every checkable number in `text`, with its unit."""
    text = _DATE.sub(" ", text)
    claims = []
    for m in _NUMBER.finditer(text):
        num = m.group("num").replace(",", "")
        value = float(m.group("sign") + num)
        unit = _unit(m.group("unit"))
        decimals = len(num.split(".")[1]) if "." in num and "e" not in num.lower() else 0
        if unit == PLAIN and decimals == 0 and abs(value) < SMALL_COUNT_LIMIT:
            continue
        claims.append(Claim(m.group(0).strip(), value, unit, decimals))
    return claims


def source_values(context: str, extra_pct: list[float] | None = None) -> dict[str, list[float]]:
    """Numbers in the model's input, grouped by the unit they carry there."""
    values: dict[str, list[float]] = {BP: [], PCT: list(extra_pct or []), PLAIN: []}
    context = _DATE.sub(" ", context)
    for m in _BAND.finditer(context):
        unit = _unit(m.group("unit"))
        values[unit] += [float(m.group("a").replace(",", "")), float(m.group("b").replace(",", ""))]
    for m in _NUMBER.finditer(context):
        value = float(m.group("sign") + m.group("num").replace(",", ""))
        unit = _unit(m.group("unit"))
        values[unit].append(value)
        if unit == PLAIN:
            values[PCT].append(100.0 * value)  # fractions such as a -0.1 shock = 10%
    return values


def _matches(claim: Claim, pool: list[float]) -> bool:
    printed = 0.5 * 10.0**-claim.decimals
    return any(
        abs(abs(claim.value) - abs(v)) <= max(printed, RELATIVE_TOLERANCE * abs(v)) for v in pool
    )


def check(
    answer_texts: list[str], context: str, extra_pct: list[float] | None = None
) -> FactCheckResult:
    """Check every number in `answer_texts` against `context` (the prompt plus
    the tool results the model saw). `extra_pct`: derived percentages the model
    may legitimately compute, such as each position's share of the residual."""
    sources = source_values(context, extra_pct)
    allowed = {BP: sources[BP], PCT: sources[PCT], PLAIN: sources[PLAIN]}
    result = FactCheckResult()
    for text in answer_texts:
        for claim in extract_claims(text):
            result.n_numbers += 1
            if not _matches(claim, allowed[claim.unit]):
                result.unsupported.append(claim.text)
    return result
