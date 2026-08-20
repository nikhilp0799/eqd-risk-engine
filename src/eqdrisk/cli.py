"""Typer entrypoint for the eqdrisk pipeline.

Each command is a stub for now — Step 0 only requires the CLI to exist
and print help; commands are implemented step by step.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="eqd-risk-engine: equity derivatives risk analytics")


@app.command()
def ingest(date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD")) -> None:
    """Pull raw data, validate schema, write curated Parquet."""
    raise NotImplementedError("Step 1 not yet implemented")


@app.command()
def calibrate(
    date: str = typer.Option(..., help="Snapshot date, YYYY-MM-DD"),
    underlying: str = typer.Option(..., help="Underlying ticker"),
) -> None:
    """Calibrate the volatility surface for one underlying."""
    raise NotImplementedError("Step 4 not yet implemented")


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
