"""Full daily pipeline orchestration (README Step 16 integration/reproducibility bar):
`eqdrisk run --date X` used to be a stub even though every individual stage it needs
(`ingest`/`curves`/`iv`/`calibrate`/`price`/`varswap`/`riskfactors`/`portfolio`/`explainpnl`)
already existed and runs daily via `scripts/daily_ingest.sh`. This module gives that
sequence a real, in-process, testable implementation so `make reproduce DATE=X` and the
README's own "eqdrisk run --date X completes and writes all artifacts" integration-test
bar are both genuinely true, not just documented as intent.

Each stage's failure is caught and reported independently rather than aborting the whole
run — the same resilience principle `stress/replay.py` already uses for its five
independent historical episodes ("one stage failing must not sink the others").
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from eqdrisk.config import BaseConfig
from eqdrisk.marketdata.calendar import last_n_trading_days

_LIBRARIES_IN_MANIFEST = ("numpy", "scipy", "pandas", "numba", "pydantic", "typer")


@dataclass
class StageResult:
    stage: str
    ok: bool
    detail: str = ""


@dataclass
class PipelineResult:
    asof: dt.date
    stages: list[StageResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def render(self) -> str:
        lines = [f"Full daily pipeline — {self.asof}"]
        for s in self.stages:
            status = "OK" if s.ok else "FAILED"
            lines.append(f"  [{status}] {s.stage}" + (f": {s.detail}" if s.detail else ""))
        return "\n".join(lines)


def _git_sha(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    cfg: BaseConfig,
    asof: dt.date,
    config_path: str,
    portfolio_path: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    project_root: Path,
) -> Path:
    """A reproducibility record for one `run_daily_pipeline` invocation (README Step 16):
    git SHA, config/portfolio file hashes, timestamps, and key library versions. Fixed MC
    seeds are not re-recorded here — they're already named constants in
    `portfolio/mark.py` (`MC_SEED`), not a per-run value."""
    manifest = {
        "asof_date": asof.isoformat(),
        "git_sha": _git_sha(project_root),
        "config_path": config_path,
        "config_sha256": _file_sha256(project_root / config_path),
        "portfolio_path": portfolio_path,
        "portfolio_sha256": _file_sha256(project_root / portfolio_path),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "library_versions": {
            lib: importlib.metadata.version(lib) for lib in _LIBRARIES_IN_MANIFEST
        },
    }
    manifest_dir = Path(cfg.paths.curated) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{asof.isoformat()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run_daily_pipeline(
    cfg: BaseConfig,
    asof: dt.date,
    config_path: str,
    portfolio_path: str,
    project_root: Path | None = None,
) -> PipelineResult:
    from eqdrisk.io.snapshot import run_snapshot
    from eqdrisk.marketdata.forward import run_forward_construction
    from eqdrisk.portfolio.mark import mark_portfolio
    from eqdrisk.pricing.engine import run_pricing
    from eqdrisk.pricing.pnl_explain import run_pnl_explain
    from eqdrisk.pricing.varswap_engine import run_varswap
    from eqdrisk.vol.implied import run_iv_extraction
    from eqdrisk.vol.risk_factors import run_risk_factor_grid
    from eqdrisk.vol.surface import run_surface_calibration

    project_root = project_root or Path.cwd()
    started_at = dt.datetime.now(dt.UTC)
    result = PipelineResult(asof=asof)

    def _stage(name: str, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            output = fn()
            detail = output.render() if hasattr(output, "render") else ""
            result.stages.append(
                StageResult(stage=name, ok=True, detail=detail.splitlines()[0] if detail else "")
            )
        except Exception as exc:  # noqa: BLE001 - one stage's failure must not sink the rest
            result.stages.append(StageResult(stage=name, ok=False, detail=str(exc)))

    _stage("ingest", lambda: run_snapshot(cfg, asof))
    _stage("curves", lambda: run_forward_construction(cfg, asof))
    _stage("iv", lambda: run_iv_extraction(cfg, asof))
    _stage("calibrate", lambda: run_surface_calibration(cfg, asof))
    _stage("price", lambda: run_pricing(cfg, asof))
    _stage("varswap", lambda: run_varswap(cfg, asof))
    _stage("riskfactors", lambda: run_risk_factor_grid(cfg, asof))
    _stage("portfolio", lambda: mark_portfolio(cfg, asof, portfolio_path))

    day0 = last_n_trading_days(asof, 2, cfg.calendar)[0]
    _stage("explainpnl", lambda: run_pnl_explain(cfg, day0, asof, portfolio_path))

    finished_at = dt.datetime.now(dt.UTC)
    write_manifest(cfg, asof, config_path, portfolio_path, started_at, finished_at, project_root)
    return result
