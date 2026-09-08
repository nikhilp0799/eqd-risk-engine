"""Integration test (README Step 16's testing pyramid): `eqdrisk run --date X`
must complete end to end and write every stage's curated artifact, with zero
network dependency (see `tests/synthetic_fixture.py`). Before this test existed,
`run_snapshot` (the `ingest` stage) had no test coverage at all beyond its
private helper functions — this is the first test that exercises the actual
network-calling orchestration path, even though the network itself is mocked.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from eqdrisk.cli import app
from synthetic_fixture import ASOF, EXPIRIES, UNDERLYING

runner = CliRunner()


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    curated = tmp_path / "curated"
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        f"run_date: '{ASOF.isoformat()}'\n"
        f"universe: {{index: [{UNDERLYING}], single_names: []}}\n"
        f"paths: {{raw: '{tmp_path / 'raw'}', curated: '{curated}'}}\n"
        "calendar: NYSE\n"
        "daycount: ACT/365F\n"
    )
    portfolio_path = tmp_path / "portfolio.yaml"
    portfolio_path.write_text(
        "positions:\n"
        f"  - {{id: T1, type: vanilla, underlying: {UNDERLYING}, cp: C, strike: 100, "
        f"expiry: {EXPIRIES[0][0].isoformat()}, qty: 10}}\n"
    )
    return config_path, portfolio_path


def test_run_completes_and_writes_every_stages_artifact(tmp_path, mocked_sources):
    config_path, portfolio_path = _write_configs(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--date",
            ASOF.isoformat(),
            "--config",
            str(config_path),
            "--portfolio",
            str(portfolio_path),
        ],
    )

    assert result.exit_code == 0, result.output
    for stage in (
        "ingest",
        "curves",
        "iv",
        "calibrate",
        "price",
        "varswap",
        "riskfactors",
        "portfolio",
        "explainpnl",
    ):
        assert f"[OK] {stage}" in result.output, result.output

    curated = tmp_path / "curated"
    for table in (
        "chains",
        "discount_curves",
        "forwards",
        "implied_vols",
        "vol_surface",
        "greeks",
        "varswap",
        "risk_factors",
        "portfolio_marks",
        "manifests",
    ):
        table_root = curated / table
        assert table_root.exists(), f"expected {table} to be written, found nothing"


def test_run_writes_a_reproducibility_manifest(tmp_path, mocked_sources):
    config_path, portfolio_path = _write_configs(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--date",
            ASOF.isoformat(),
            "--config",
            str(config_path),
            "--portfolio",
            str(portfolio_path),
        ],
    )
    assert result.exit_code == 0, result.output

    manifest_path = tmp_path / "curated" / "manifests" / f"{ASOF.isoformat()}.json"
    assert manifest_path.exists()

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["asof_date"] == ASOF.isoformat()
    assert manifest["config_sha256"] is not None
    assert manifest["portfolio_sha256"] is not None
    assert "numpy" in manifest["library_versions"]
    assert manifest["duration_seconds"] >= 0
