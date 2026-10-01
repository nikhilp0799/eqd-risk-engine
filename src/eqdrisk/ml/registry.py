"""Saved deep-hedging policies (Phase 7, `planning/deep_hedging_plan.md`).

Phase 6 stored only each trained policy's scores, so a policy could never be
re-evaluated on a new scenario without retraining it. `run_deep_hedge` now saves
every trained network here, keyed the same way as `deep_hedge_results` rows:
(asof_date, instrument, loss_type, seed). About 20KB per network.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import torch

from eqdrisk.ml.hedge_model import HedgeNet

MODELS_TABLE = "deep_hedge_models"


def model_path(
    curated_root: Path, asof: dt.date, instrument: str, loss_type: str, seed: int
) -> Path:
    return (
        curated_root
        / MODELS_TABLE
        / f"asof_date={asof.isoformat()}"
        / f"{instrument}__{loss_type}__seed{seed}.pt"
    )


def save_model(
    net: HedgeNet,
    curated_root: Path,
    asof: dt.date,
    instrument: str,
    loss_type: str,
    seed: int,
) -> Path:
    path = model_path(curated_root, asof, instrument, loss_type, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = net.net[0].out_features
    torch.save({"hidden": hidden, "state_dict": net.state_dict()}, path)
    return path


def load_model(
    curated_root: Path, asof: dt.date, instrument: str, loss_type: str, seed: int
) -> HedgeNet | None:
    """The saved network, or `None` if that combination was never trained and
    saved (e.g. results from before Phase 7)."""
    path = model_path(curated_root, asof, instrument, loss_type, seed)
    if not path.exists():
        return None
    # weights_only: the file holds only tensors and an int, never arbitrary objects.
    blob = torch.load(path, weights_only=True)
    net = HedgeNet(hidden=int(blob["hidden"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net
