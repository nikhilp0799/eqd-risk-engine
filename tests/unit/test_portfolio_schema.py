from pathlib import Path

from eqdrisk.portfolio.schema import (
    AutocallPosition,
    BarrierPosition,
    EquityPosition,
    Portfolio,
    VanillaPosition,
    VarSwapPosition,
)

REPO_PORTFOLIO = Path(__file__).parents[2] / "configs" / "portfolio.yaml"


def test_readme_portfolio_yaml_parses_into_the_expected_discriminated_types():
    portfolio = Portfolio.from_yaml(REPO_PORTFOLIO)
    assert len(portfolio.positions) == 9

    by_id = {p.id: p for p in portfolio.positions}
    assert isinstance(by_id["P001"], VanillaPosition)
    assert by_id["P001"].cp == "P"
    assert by_id["P001"].qty == -150
    assert isinstance(by_id["P006"], VarSwapPosition)
    assert by_id["P006"].vega_notional == 250000
    assert isinstance(by_id["P007"], BarrierPosition)
    assert by_id["P007"].sub == "down_and_in_put"
    assert isinstance(by_id["P008"], AutocallPosition)
    assert by_id["P008"].autocall_barrier == 1.00
    assert isinstance(by_id["P009"], EquityPosition)
    assert by_id["P009"].qty == -80


def test_position_types_round_trip_through_yaml_shaped_dicts():
    raw = {
        "positions": [
            {
                "id": "X1",
                "type": "vanilla",
                "underlying": "TEST",
                "cp": "C",
                "strike": 100.0,
                "expiry": "2027-01-15",
                "qty": 10,
            },
            {"id": "X2", "type": "equity", "underlying": "TEST", "qty": -5},
        ]
    }
    portfolio = Portfolio.model_validate(raw)
    assert isinstance(portfolio.positions[0], VanillaPosition)
    assert isinstance(portfolio.positions[1], EquityPosition)
