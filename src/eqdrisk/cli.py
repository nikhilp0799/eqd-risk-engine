"""Typer entrypoint for the eqdrisk pipeline.

Commands are implemented step by step; unimplemented ones raise NotImplementedError.
"""

from __future__ import annotations

import datetime as dt

import typer

from eqdrisk.config import BaseConfig

app = typer.Typer(help="eqd-risk-engine: equity derivatives risk analytics")


@app.command()
def ingest(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    config: str = typer.Option("configs/base.yaml", help="Path to base config"),
) -> None:
    """Pull raw data, validate schema, write curated Parquet."""
    from eqdrisk.io.snapshot import run_snapshot

    cfg = BaseConfig.from_yaml(config)
    asof = dt.date.fromisoformat(date)
    result = run_snapshot(cfg, asof)
    typer.echo(result.qc.render())


@app.command()
def curves(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    config: str = typer.Option("configs/base.yaml", help="Path to base config"),
) -> None:
    """Bootstrap the discount curve and fit implied forwards/dividends (Step 2).

    Not in the README's original CLI table — added because Step 2's outputs
    need to be independently runnable/inspectable, same as `ingest`.
    """
    from eqdrisk.marketdata.forward import run_forward_construction

    cfg = BaseConfig.from_yaml(config)
    asof = dt.date.fromisoformat(date)
    result = run_forward_construction(cfg, asof)
    typer.echo(result.render())


@app.command()
def iv(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    config: str = typer.Option("configs/base.yaml", help="Path to base config"),
) -> None:
    """Extract implied vols with the quality filter chain (Step 3).

    Also not in the README's original CLI table, same reasoning as `curves`.
    """
    from eqdrisk.vol.implied import run_iv_extraction

    cfg = BaseConfig.from_yaml(config)
    asof = dt.date.fromisoformat(date)
    result = run_iv_extraction(cfg, asof)
    typer.echo(result.render())


@app.command()
def calibrate(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    underlying: str = typer.Option(..., help="Underlying ticker"),
    config: str = typer.Option("configs/base.yaml", help="Path to base config"),
) -> None:
    """Calibrate the volatility surface (SVI/SSVI + SABR comparison) for one underlying."""
    from eqdrisk.vol.surface import run_surface_calibration

    cfg = BaseConfig.from_yaml(config)
    asof = dt.date.fromisoformat(date)
    result = run_surface_calibration(cfg, asof, underlyings=[underlying])
    typer.echo(result.render())


@app.command()
def price(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    portfolio: str = typer.Option(..., help="Path to portfolio config"),
) -> None:
    """Mark the portfolio to model."""
    raise NotImplementedError("Steps 5-7 not yet implemented")


@app.command()
def var(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    method: str = typer.Option("both", help="full-reval | taylor | both"),
    confidence: float = typer.Option(0.99),
) -> None:
    """Compute VaR / Expected Shortfall."""
    raise NotImplementedError("Step 9 not yet implemented")


@app.command()
def backtest(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD"),
) -> None:
    """Backtest the VaR model (Kupiec, Christoffersen, traffic light, PLA)."""
    raise NotImplementedError("Step 10 not yet implemented")


@app.command()
def stress(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    scenarios: str = typer.Option(..., help="Path to stress config"),
) -> None:
    """Run historical, hypothetical, and conditional stress tests."""
    raise NotImplementedError("Step 11 not yet implemented")


@app.command()
def reverse(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    loss_target: float = typer.Option(..., help="Target loss, e.g. -5000000"),
) -> None:
    """Solve for the most plausible shock producing a target loss."""
    raise NotImplementedError("Step 11 not yet implemented")


@app.command()
def explain(date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD")) -> None:
    """Daily P&L explain with residual monitoring."""
    raise NotImplementedError("Step 12 not yet implemented")


@app.command()
def run(date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD")) -> None:
    """Run the full daily pipeline end to end."""
    raise NotImplementedError("Full pipeline not yet implemented")


@app.command()
def dashboard() -> None:
    """Launch the Streamlit risk dashboard."""
    raise NotImplementedError("Step 14 not yet implemented")


if __name__ == "__main__":
    app()
