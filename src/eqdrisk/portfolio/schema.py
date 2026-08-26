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


class VanillaPosition(BaseModel):
    id: str
    type: Literal["vanilla"] = "vanilla"
    underlying: str
    cp: Literal["C", "P"]
    strike: float
    expiry: dt.date
    qty: float


class VarSwapPosition(BaseModel):
    id: str
    type: Literal["varswap"] = "varswap"
    underlying: str
    expiry: dt.date
    vega_notional: float


class BarrierPosition(BaseModel):
    id: str
    type: Literal["barrier"] = "barrier"
    underlying: str
    sub: Literal["down_and_in_put"] = "down_and_in_put"
    strike: float
    barrier: float
    expiry: dt.date
    qty: float


class AutocallPosition(BaseModel):
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


class EquityPosition(BaseModel):
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
