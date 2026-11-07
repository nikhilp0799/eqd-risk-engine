"""Pydantic schema for configs/base.yaml — no magic numbers in code."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import yaml
from pydantic import BaseModel


class Universe(BaseModel):
    index: list[str]
    single_names: list[str]


class Paths(BaseModel):
    raw: str
    curated: str


class ChainCatchUp(BaseModel):
    """When the daily run re-fetches a missing option chain. Measured 2026-11-06
    against the 16:30 snapshot: SPX quotes still passed the filters at 17:00 and
    had no bids by 17:30; AAPL/NVDA quotes stayed unchanged to 21:00
    (planning/ingest_catch_up_retry_plan.md)."""

    index_cutoff: time = time(17, 0, 0)
    single_name_cutoff: time = time(21, 0, 0)
    index_interval_minutes: int = 10
    single_name_interval_minutes: int = 30


class BaseConfig(BaseModel):
    run_date: date
    universe: Universe
    paths: Paths
    calendar: str
    daycount: str
    canonical_snap_time: time = time(16, 0, 0)
    snap_tolerance_minutes: int = 15
    chain_catch_up: ChainCatchUp = ChainCatchUp()

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaseConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
