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


class BaseConfig(BaseModel):
    run_date: date
    universe: Universe
    paths: Paths
    calendar: str
    daycount: str
    canonical_snap_time: time = time(16, 0, 0)
    snap_tolerance_minutes: int = 15

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaseConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
