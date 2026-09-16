"""Real tools the AI investigation agent can call mid-investigation (README's
"beyond the 16 steps" section, take 2 — `planning/ai_agent_tool_use_plan.md`).
Both tools do REAL work — an actual pricing-engine reprice, an actual file
read — not a simulated or narrated answer, and both fail soft (return an
`{"error": ...}` dict or an honest "not found" string) rather than raise, so
one bad tool call from a small local model can't sink the whole investigation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eqdrisk.config import BaseConfig
from eqdrisk.portfolio.mark import load_market_state, mark_with_state
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.stress.hypothetical_grid import GRID_MC_SETTINGS
from eqdrisk.stress.shock import MarketShock

MODEL_DOC_PATH = "docs/model_documentation.md"
MAX_SECTION_CHARS = 3000
SPOT_SHOCK_BOUNDS = (-0.5, 0.5)
VOL_SHOCK_BOUNDS = (-0.5, 1.0)

MODEL_DOC_SECTIONS = {
    1: "Purpose and scope",
    2: "Product coverage",
    3: "Methodology",
    4: "Assumptions and limitations",
    5: "Data",
    6: "Implementation",
    7: "Testing and validation evidence",
    8: "Model risk assessment",
    9: "Ongoing monitoring plan",
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "what_if_reprice",
            "description": (
                "Reprice the book under a hypothetical spot/vol shock and return one "
                "position's price delta and shocked Greeks. Real pricing-engine call — "
                "use it to check whether a position's actual sensitivity matches what "
                "the day's P&L explain residual would imply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position_id": {"type": "string", "description": "e.g. 'P008'"},
                    "spot_shock_pct": {
                        "type": "number",
                        "description": "e.g. -0.1 for a 10% spot decline. Clamped to [-0.5, 0.5].",
                    },
                    "vol_shock_pct": {
                        "type": "number",
                        "description": (
                            "e.g. 0.25 for a 25% relative parallel vol increase. "
                            "Clamped to [-0.5, 1.0]."
                        ),
                    },
                },
                "required": ["position_id", "spot_shock_pct", "vol_shock_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_model_doc_section",
            "description": (
                "Read a section of this project's own model documentation. Sections: "
                + ", ".join(f"{n}={title}" for n, title in MODEL_DOC_SECTIONS.items())
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "integer", "description": "Section number, 1-9."}
                },
                "required": ["section"],
            },
        },
    },
]


def what_if_reprice(
    cfg: BaseConfig,
    portfolio_path: str,
    day1: Any,
    position_id: str,
    spot_shock_pct: float,
    vol_shock_pct: float,
) -> dict[str, Any]:
    spot_shock_pct = max(SPOT_SHOCK_BOUNDS[0], min(SPOT_SHOCK_BOUNDS[1], float(spot_shock_pct)))
    vol_shock_pct = max(VOL_SHOCK_BOUNDS[0], min(VOL_SHOCK_BOUNDS[1], float(vol_shock_pct)))

    try:
        portfolio = Portfolio.from_yaml(portfolio_path)
        state = load_market_state(cfg, day1, portfolio)
        if state is None:
            return {"error": f"no market state available for {day1}"}

        base = mark_with_state(cfg, day1, portfolio, state, mc_settings=GRID_MC_SETTINGS)
        shock = MarketShock(spot_shock_pct=spot_shock_pct, vol_shock_pct=vol_shock_pct)
        shocked = mark_with_state(
            cfg, day1, portfolio, state, shock=shock, mc_settings=GRID_MC_SETTINGS
        )

        base_mark = next((m for m in base.marks if m.position_id == position_id), None)
        shocked_mark = next((m for m in shocked.marks if m.position_id == position_id), None)
        if base_mark is None or shocked_mark is None:
            reason = (
                base.skipped.get(position_id)
                or shocked.skipped.get(position_id)
                or f"position {position_id} not found in portfolio"
            )
            return {"error": reason}

        return {
            "position_id": position_id,
            "spot_shock_pct": spot_shock_pct,
            "vol_shock_pct": vol_shock_pct,
            "base_price": base_mark.price,
            "shocked_price": shocked_mark.price,
            "price_delta": shocked_mark.price - base_mark.price,
            "shocked_delta": shocked_mark.delta,
            "shocked_gamma": shocked_mark.gamma,
            "shocked_vega": shocked_mark.vega,
        }
    except Exception as exc:  # noqa: BLE001 - a bad tool call must not sink the investigation
        return {"error": f"reprice failed: {exc}"}


def read_model_doc_section(project_root: Path, section: int) -> str:
    title = MODEL_DOC_SECTIONS.get(int(section))
    if title is None:
        return f"Section {section} does not exist. Valid sections: 1-9."

    doc_path = project_root / MODEL_DOC_PATH
    if not doc_path.exists():
        return "Model documentation file not found."

    text = doc_path.read_text()
    next_section = int(section) + 1
    if next_section in MODEL_DOC_SECTIONS:
        pattern = rf"## {section}\. {re.escape(title)}(.*?)## {next_section}\."
    else:
        pattern = rf"## {section}\. {re.escape(title)}(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return f"Section {section} ({title}) not found in the document."
    return match.group(1).strip()[:MAX_SECTION_CHARS]


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    cfg: BaseConfig,
    portfolio_path: str,
    day1: Any,
    project_root: Path,
) -> Any:
    """Dispatches a tool call by name. Returns an honest error dict/string for
    an unrecognized tool name rather than raising — the loop must be able to
    feed that back to the model and continue."""
    if name == "what_if_reprice":
        return what_if_reprice(
            cfg,
            portfolio_path,
            day1,
            str(arguments.get("position_id", "")),
            float(arguments.get("spot_shock_pct", 0.0)),
            float(arguments.get("vol_shock_pct", 0.0)),
        )
    if name == "read_model_doc_section":
        return read_model_doc_section(project_root, int(arguments.get("section", 0)))
    return {"error": f"unknown tool '{name}'"}
