"""Thin REST client for a locally-running Ollama server — no API key, no external
account, no per-call cost, and no project data ever leaves this machine. Chosen
over a hosted API specifically so the AI investigation agent (`agent/investigate.py`)
costs nothing to run daily and has no external dependency to fail silently against.

Deliberately returns `None` rather than raising when Ollama isn't reachable or a
call fails: the daily pipeline must degrade gracefully (an honest "AI unavailable"
note) if the local model service happens to be down, not crash the whole run over
an optional add-on.
"""

from __future__ import annotations

import requests

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT_SECONDS = 120.0


def is_available(host: str = OLLAMA_HOST) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """One-shot, non-streaming completion. Returns `None` (not an exception) on
    any connection failure, timeout, or non-200 response — callers must treat
    that as "AI unavailable today," not a pipeline failure."""
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        return str(resp.json().get("response", "")) or None
    except requests.exceptions.RequestException:
        return None
