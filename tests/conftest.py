"""Top-level fixtures shared across test directories (unit/regression/integration)."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def mocked_sources(monkeypatch):
    """Patches every network-touching function in `io.sources` with synthetic data
    (see `tests/regression/synthetic_fixture.py`) — confirmed, by reading
    `io/snapshot.py` (the only caller), to be exactly these five functions, all
    only ever invoked from `run_snapshot` (the `ingest` stage). Every stage
    downstream of ingest reads only already-curated local data, so patching these
    five is sufficient to run the full pipeline with zero network dependency."""
    from eqdrisk.io import sources
    from synthetic_fixture import (
        build_synthetic_chain,
        build_synthetic_ohlc,
        build_synthetic_rates,
    )

    monkeypatch.setattr(sources, "fetch_option_chain", lambda *a, **k: build_synthetic_chain())
    monkeypatch.setattr(sources, "fetch_dividends", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sources, "fetch_underlying_ohlc", lambda *a, **k: build_synthetic_ohlc())
    monkeypatch.setattr(sources, "fetch_rates", lambda *a, **k: build_synthetic_rates())
    monkeypatch.setattr(sources, "fetch_vol_indices", lambda *a, **k: pd.DataFrame())
