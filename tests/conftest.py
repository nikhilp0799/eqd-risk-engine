"""Top-level fixtures shared across test directories (unit/regression/integration)."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def mocked_sources(monkeypatch):
    """Patches every network-touching call the full pipeline can make with
    synthetic/canned data, so `eqdrisk run` has zero real network dependency
    regardless of the machine it runs on. Two independent things are patched:

    1. `io.sources` — confirmed, by reading `io/snapshot.py` (the only caller),
       to be exactly these five functions, all only ever invoked from
       `run_snapshot` (the `ingest` stage).
    2. `agent.ollama_client` — the `ai_investigate` stage calls a REAL local
       Ollama server if one happens to be reachable (`is_available()` isn't
       mocked away, this makes the test's outcome depend on whether Ollama is
       running on whatever machine runs the suite, which is exactly the kind
       of environment-dependent flakiness `io.sources` mocking above already
       exists to avoid). Patched to a scripted, deterministic final answer —
       exercising the real persistence code path, not skipping it."""
    from eqdrisk.agent import ollama_client
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

    monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(
        ollama_client,
        "chat",
        lambda *a, **k: {
            "role": "assistant",
            "content": '{"summary": "synthetic fixture, no real residual to investigate", '
            '"root_cause_hypothesis": "n/a", "confidence": "high", '
            '"flagged_positions": [], "proposed_changes": []}',
        },
    )
