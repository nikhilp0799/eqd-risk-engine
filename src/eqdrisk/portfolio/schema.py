"""Pydantic portfolio schema (README Step 7) — a discriminated union of position
types matching `configs/portfolio.yaml` exactly, same "no magic parsing" convention
as `config.py`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class _Lifecycle(BaseModel):
    """Fields shared by every position (planning/position_lifecycle_plan.md).
    `active_from`: first date the position is in the book (a roll's trade date);
    None = in the book from the start of history."""

    active_from: dt.date | None = None


class VanillaPosition(_Lifecycle):
    id: str
    type: Literal["vanilla"] = "vanilla"
    underlying: str
    cp: Literal["C", "P"]
    strike: float
    expiry: dt.date
    qty: float


class VarSwapPosition(_Lifecycle):
    id: str
    type: Literal["varswap"] = "varswap"
    underlying: str
    expiry: dt.date
    vega_notional: float


class BarrierPosition(_Lifecycle):
    id: str
    type: Literal["barrier"] = "barrier"
    underlying: str
    sub: Literal["down_and_in_put"] = "down_and_in_put"
    strike: float
    barrier: float
    expiry: dt.date
    qty: float
    # Knock-in is monitored on daily closes from this date (the trade date when
    # known); None = from the first stored day.
    monitoring_start: dt.date | None = None


class AutocallPosition(_Lifecycle):
    id: str
    type: Literal["autocall"] = "autocall"
    underlying: str
    notional: float
    autocall_barrier: float
    coupon_barrier: float
    put_barrier: float
    coupon: float
    obs: Literal["quarterly"] = "quarterly"
    expiry: dt.date
    # The issue date and initial fixing the barriers are fractions of. None for
    # initial_level = a note issued today (struck at today's spot).
    issue_date: dt.date | None = None
    initial_level: float | None = None


class EquityPosition(_Lifecycle):
    id: str
    type: Literal["equity"] = "equity"
    underlying: str
    qty: float


Position = Annotated[
    VanillaPosition | VarSwapPosition | BarrierPosition | AutocallPosition | EquityPosition,
    Field(discriminator="type"),
]


class Portfolio(BaseModel):
    positions: list[Position]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Portfolio:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
