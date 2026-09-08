"""Golden-file regression test (README Step 16's testing pyramid): the full
pipeline, run against a FIXED synthetic scenario (`tests/synthetic_fixture.py`),
must reproduce a checked-in set of key output values exactly. "Fixed date" is
read as "fixed synthetic scenario" — see that module's docstring for why.

If this test fails after a deliberate model/calibration change, regenerate the
golden file by running this module directly
(`PYTHONPATH=tests python tests/regression/test_golden_pipeline.py`) and
reviewing the diff before committing the new golden file — never regenerate
just to make a red test pass without understanding why the numbers moved.
"""

from __future__ import annotations

import json
from pathlib import Path

from eqdrisk.config import BaseConfig, Paths, Universe
from eqdrisk.io.snapshot import run_snapshot
from eqdrisk.marketdata.forward import run_forward_construction
from eqdrisk.portfolio.mark import mark_portfolio
from eqdrisk.vol.implied import run_iv_extraction
from eqdrisk.vol.surface import run_surface_calibration
from synthetic_fixture import ASOF, EXPIRIES, UNDERLYING

GOLDEN_PATH = Path(__file__).parent / "golden_pipeline_output.json"


def _cfg(tmp_path: Path) -> BaseConfig:
    return BaseConfig(
        run_date=ASOF,
        universe=Universe(index=[UNDERLYING], single_names=[]),
        paths=Paths(raw=str(tmp_path / "raw"), curated=str(tmp_path / "curated")),
        calendar="NYSE",
        daycount="ACT/365F",
    )


def _run_pipeline(tmp_path: Path) -> dict:
    cfg = _cfg(tmp_path)
    portfolio_path = tmp_path / "portfolio.yaml"
    portfolio_path.write_text(
        "positions:\n"
        f"  - {{id: T1, type: vanilla, underlying: {UNDERLYING}, cp: C, strike: 100, "
        f"expiry: {EXPIRIES[0][0].isoformat()}, qty: 10}}\n"
    )

    run_snapshot(cfg, ASOF)
    run_forward_construction(cfg, ASOF)
    run_iv_extraction(cfg, ASOF)
    calib = run_surface_calibration(cfg, ASOF)
    mark = mark_portfolio(cfg, ASOF, str(portfolio_path))

    slices = sorted(calib.slices, key=lambda s: s.T)
    return {
        "slices": [
            {
                "expiry": s.expiry.isoformat(),
                "T": s.T,
                "model": s.model,
                "params": {k: round(v, 10) for k, v in s.params.items()},
                "n_points": s.n_points,
                "rmse_vol_points": round(s.rmse_vol_points, 8),
                "butterfly_violations": s.butterfly_violations,
                "calendar_violated": s.calendar_violated,
            }
            for s in slices
        ],
        "portfolio": {
            "total_value": round(mark.total_value(), 6),
            "marks": [
                {
                    "position_id": m.position_id,
                    "price": round(m.price, 6),
                    "delta": round(m.delta, 6),
                    "gamma": round(m.gamma, 6),
                    "vega": round(m.vega, 6),
                }
                for m in mark.marks
            ],
        },
    }


def test_pipeline_reproduces_golden_output(tmp_path, mocked_sources):
    actual = _run_pipeline(tmp_path)
    golden = json.loads(GOLDEN_PATH.read_text())
    assert actual == golden, (
        "Pipeline output drifted from the checked-in golden file. If this is an "
        "intentional model/calibration change, regenerate golden_pipeline_output.json "
        "by running this module directly and review the diff before committing."
    )


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    import pandas as pd

    from eqdrisk.io import sources
    from synthetic_fixture import build_synthetic_chain, build_synthetic_ohlc, build_synthetic_rates

    with tempfile.TemporaryDirectory() as tmp:
        with (
            mock.patch.object(
                sources, "fetch_option_chain", lambda *a, **k: build_synthetic_chain()
            ),
            mock.patch.object(sources, "fetch_dividends", lambda *a, **k: pd.DataFrame()),
            mock.patch.object(
                sources, "fetch_underlying_ohlc", lambda *a, **k: build_synthetic_ohlc()
            ),
            mock.patch.object(sources, "fetch_rates", lambda *a, **k: build_synthetic_rates()),
            mock.patch.object(sources, "fetch_vol_indices", lambda *a, **k: pd.DataFrame()),
        ):
            output = _run_pipeline(Path(tmp))
    GOLDEN_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {GOLDEN_PATH}")
