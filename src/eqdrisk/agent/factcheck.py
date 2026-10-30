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

A number must also be called what it is (2026-10-30, planning/
ai_factcheck_labels_plan.md). On 2026-10-26 the model wrote "the position's actual
price change of -548,020.04": a real number, but a what-if reprice result, not
something that happened. Each source number carries the label it has in the
model's input (actual, explained, residual, noise, threshold, what-if). When the
words just before a claimed number name one of those quantities, the number must
match a source with that label; otherwise only the check above applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

BP = "bp"
PCT = "pct"
PLAIN = "plain"

ACTUAL = "actual"
EXPLAINED = "explained"
RESIDUAL = "residual"
NOISE = "noise"
THRESHOLD = "threshold"
WHAT_IF = "what_if"

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


# Words that name a quantity, looked for just before a claimed number. Word
# boundaries matter: "unexpected" is not "expected". Bare "shock" is left out on
# purpose: "the spot and vol shocks" usually means the day's real market moves.
_QUANTITY_WORDS = {
    ACTUAL: re.compile(r"\b(?:actual|realized|realised|observed)\b", re.IGNORECASE),
    EXPLAINED: re.compile(r"\b(?:explained|expected|predicted|attributed)\b", re.IGNORECASE),
    RESIDUAL: re.compile(r"\b(?:residual|unexplained|gap)\b", re.IGNORECASE),
    NOISE: re.compile(r"\b(?:noise|standard error|band)\b", re.IGNORECASE),
    WHAT_IF: re.compile(
        r"what[-_ ]?if|\b(?:shocked|scenario|hypothetical|would|under an?)\b|\brepric",
        re.IGNORECASE,
    ),
}
# "exceeds the 5bp threshold": the figure is the threshold, whatever else the
# sentence names, so these fall back to the plain check.
_THRESHOLD_WORDS = re.compile(r"\b(?:threshold|exceed\w*|limit|breach\w*)\b", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(
    r"[.;:!?,]\s|\n|\b(?:compared (?:to|with)|versus|vs\.?|while|whereas|but)\b", re.IGNORECASE
)
LABEL_WINDOW_WORDS = 6
# "noise wider than the 5bp threshold": the threshold word can follow the number.
THRESHOLD_LOOKAHEAD_WORDS = 3

# How a label reads in the corrective retry and the stored list.
LABEL_NAMES = {
    ACTUAL: "an actual P&L or value",
    EXPLAINED: "an explained P&L",
    RESIDUAL: "a residual",
    NOISE: "a noise band",
    THRESHOLD: "the alert threshold",
    WHAT_IF: "a what-if reprice result",
}

# The explain render (pnl_explain.PnLExplainResult.render) and the history lines.
_KEYED = re.compile(r"(?P<key>actual|explained|residual)=(?P<v>[-+]?\d[\d,]*(?:\.\d+)?)")
_KEYED_NOISE = re.compile(r"residual=[-+]?\d[\d,]*(?:\.\d+)?\s*\+-\s*(?P<v>\d[\d,]*(?:\.\d+)?)")
_POSITION_LINE = re.compile(
    r"^\s*P\w+:\s*(?P<r>[-+]?\d[\d,]*(?:\.\d+)?)\s*\+-\s*(?P<se>\d[\d,]*(?:\.\d+)?)", re.MULTILINE
)
_BP_BAND = re.compile(r"(?P<r>[-+]?\d[\d,]*(?:\.\d+)?)\s*\+-\s*(?P<se>\d[\d,]*(?:\.\d+)?)bp of NAV")
_BP_RESIDUAL = re.compile(
    r"residual (?P<r>[-+]?\d[\d,]*(?:\.\d+)?)bp of NAV(?: \(\+- (?P<se>\d[\d,]*(?:\.\d+)?)bp)?"
)
_BP_THRESHOLD = re.compile(r"(?P<v>\d+(?:\.\d+)?)bp threshold|threshold (?P<w>\d+(?:\.\d+)?)bp")


@dataclass
class Source:
    value: float
    unit: str
    label: str


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
    # Numbers that are in the data but described as a different quantity, each
    # with what it really is: "-548,020.04 (called actual; ...what-if...)".
    mislabelled: list[str] = field(default_factory=list)
    # False when the answer was not valid JSON even after the corrective retry:
    # its numbers were checked in the raw text, but its fields could not be read,
    # so it is flagged whatever the numbers say.
    parsed: bool = True

    @property
    def status(self) -> str:
        if not self.parsed:
            return "unparsed"
        if self.unsupported:
            return "unsupported_numbers"
        return "mislabelled_numbers" if self.mislabelled else "verified"

    def render(self) -> str:
        if not self.parsed:
            found = f": {', '.join(self.unsupported)}" if self.unsupported else ""
            return (
                "FACT-CHECK: the answer was not valid JSON, so its fields could not be read; "
                f"in its raw text {len(self.unsupported)} of {self.n_numbers} numbers are not "
                f"in the data given{found}"
            )
        lines = []
        if self.unsupported:
            lines.append(
                f"FACT-CHECK: {len(self.unsupported)} of {self.n_numbers} numbers not in the "
                f"data given (or with the wrong unit): {', '.join(self.unsupported)}"
            )
        if self.mislabelled:
            lines.append(
                f"FACT-CHECK: {len(self.mislabelled)} of {self.n_numbers} numbers described as "
                f"the wrong quantity: {'; '.join(self.mislabelled)}"
            )
        if not lines:
            return (
                f"fact-check: {self.n_numbers} numbers checked, all found in the data given "
                "and described as what they are"
            )
        return "\n  ".join(lines)


def _unit(raw_unit: str | None) -> str:
    u = (raw_unit or "").lower()
    if u in ("bp", "basis points"):
        return BP
    if u in ("%", "percent"):
        return PCT
    return PLAIN


def _claims_with_windows(text: str) -> list[tuple[Claim, str, str]]:
    """Every checkable number in `text`, with its unit, the words just before it
    (back to the previous number or a clause break, at most `LABEL_WINDOW_WORDS`
    words) and the first few words after it (up to a clause break)."""
    text = _DATE.sub(lambda m: " " * len(m.group(0)), text)  # keep offsets
    out = []
    prev_end = 0
    for m in _NUMBER.finditer(text):
        before = text[prev_end : m.start()]
        prev_end = m.end()
        num = m.group("num").replace(",", "")
        value = float(m.group("sign") + num)
        unit = _unit(m.group("unit"))
        decimals = len(num.split(".")[1]) if "." in num and "e" not in num.lower() else 0
        if unit == PLAIN and decimals == 0 and abs(value) < SMALL_COUNT_LIMIT:
            continue
        breaks = list(_CLAUSE_BREAK.finditer(before))
        if breaks:
            before = before[breaks[-1].end() :]
        window = " ".join(before.split()[-LABEL_WINDOW_WORDS:])
        after = text[m.end() :]
        cut = _CLAUSE_BREAK.search(after)
        after = " ".join(after[: cut.start() if cut else None].split()[:THRESHOLD_LOOKAHEAD_WORDS])
        out.append((Claim(m.group(0).strip(), value, unit, decimals), window, after))
    return out


def extract_claims(text: str) -> list[Claim]:
    """Every checkable number in `text`, with its unit."""
    return [claim for claim, _, _ in _claims_with_windows(text)]


def named_quantities(window: str, after: str = "") -> set[str]:
    """The quantities the words before a number name; empty when they name none,
    or when the number is the alert threshold (named before or just after it)."""
    if _THRESHOLD_WORDS.search(window) or _THRESHOLD_WORDS.search(after):
        return set()
    return {label for label, words in _QUANTITY_WORDS.items() if words.search(window)}


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


def labelled_sources(
    explain_text: str,
    what_if_results: list[dict[str, Any]] | None = None,
    residual_shares_pct: list[float] | None = None,
) -> list[Source]:
    """Every number the model was given whose meaning is known, with its label:
    the explain lines and history (`explain_text`, in the render format of
    `pnl_explain.PnLExplainResult.render`), each `what_if_reprice` result, and
    each position's share of the residual."""
    out: list[Source] = []
    for m in _KEYED.finditer(explain_text):
        out.append(Source(_num(m.group("v")), PLAIN, m.group("key")))
    for m in _KEYED_NOISE.finditer(explain_text):
        out.append(Source(_num(m.group("v")), PLAIN, NOISE))
    for m in _POSITION_LINE.finditer(explain_text):
        out += [
            Source(_num(m.group("r")), PLAIN, RESIDUAL),
            Source(_num(m.group("se")), PLAIN, NOISE),
        ]
    for pattern in (_BP_BAND, _BP_RESIDUAL):
        for m in pattern.finditer(explain_text):
            out.append(Source(_num(m.group("r")), BP, RESIDUAL))
            if m.group("se"):
                out.append(Source(_num(m.group("se")), BP, NOISE))
    for m in _BP_THRESHOLD.finditer(explain_text):
        out.append(Source(_num(m.group("v") or m.group("w")), BP, THRESHOLD))
    for result in what_if_results or []:
        for key, value in result.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            # base_price is the position's real value today; everything else
            # (shocked price, price change, shocked Greeks, the shocks) is what-if.
            label = ACTUAL if key == "base_price" else WHAT_IF
            out.append(Source(float(value), PLAIN, label))
            if key.endswith("_pct"):
                out.append(Source(100.0 * float(value), PCT, WHAT_IF))
    out += [Source(v, PCT, RESIDUAL) for v in residual_shares_pct or []]
    return out


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


def _describe(claim: Claim, named: set[str], labels: set[str]) -> str:
    called = " or ".join(sorted(LABEL_NAMES[n] for n in named))
    if labels:
        really = " or ".join(sorted(LABEL_NAMES[lab] for lab in labels))
    else:
        really = "a figure of another kind (such as NAV or the model documentation)"
    return f"{claim.text} (called {called}; in the data it is {really})"


def check(
    answer_texts: list[str],
    context: str,
    extra_pct: list[float] | None = None,
    labelled: list[Source] | None = None,
) -> FactCheckResult:
    """Check every number in `answer_texts` against `context` (the prompt plus
    the tool results the model saw). `extra_pct`: derived percentages the model
    may legitimately compute, such as each position's share of the residual.
    `labelled`: the sources whose meaning is known (`labelled_sources`); a number
    whose preceding words name a quantity must match one carrying that label;
    without it only existence and unit are checked."""
    sources = source_values(context, extra_pct)
    allowed = {BP: sources[BP], PCT: sources[PCT], PLAIN: sources[PLAIN]}
    result = FactCheckResult()
    for text in answer_texts:
        for claim, window, after in _claims_with_windows(text):
            result.n_numbers += 1
            if not _matches(claim, allowed[claim.unit]):
                result.unsupported.append(claim.text)
                continue
            named = named_quantities(window, after)
            if labelled is None or not named:
                continue
            same_unit = [s for s in labelled if s.unit == claim.unit]
            if _matches(claim, [s.value for s in same_unit if s.label in named]):
                continue
            labels = {s.label for s in same_unit if _matches(claim, [s.value])}
            result.mislabelled.append(_describe(claim, named, labels))
    return result
