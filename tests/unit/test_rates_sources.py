import datetime as dt
import io

import pandas as pd

from eqdrisk.io import sources

D = dt.date


def _rows(day: str, source: str, **rates: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": D.fromisoformat(day),
            "tenor": list(rates),
            "rate": list(rates.values()),
            "source": source,
        }
    )


def test_treasury_fills_the_days_fred_has_not_posted_and_sofr_is_carried():
    """2026-11-13: FRED's last Treasury yields were 11-10's and its last SOFR
    11-12's; Treasury already had 11-12 and 11-13."""
    fred = pd.concat(
        [
            _rows("2026-11-10", "FRED", SOFR=3.89, **{"2Y": 4.73}),
            _rows("2026-11-12", "FRED", SOFR=3.90),
        ]
    )
    treasury = pd.concat(
        [
            _rows("2026-11-10", "Treasury", **{"2Y": 4.73}),
            _rows("2026-11-12", "Treasury", **{"2Y": 4.86}),
            _rows("2026-11-13", "Treasury", **{"2Y": 4.88}),
        ]
    )
    out = sources.combine_rates(fred, treasury)
    got = {(r.asof_date.isoformat(), r.tenor): (r.rate, r.source) for r in out.itertuples()}
    assert got == {
        ("2026-11-10", "2Y"): (4.73, "Treasury"),
        ("2026-11-10", "SOFR"): (3.89, "FRED"),
        ("2026-11-12", "2Y"): (4.86, "Treasury"),
        ("2026-11-12", "SOFR"): (3.90, "FRED"),
        ("2026-11-13", "2Y"): (4.88, "Treasury"),
        ("2026-11-13", "SOFR"): (3.90, sources.SOFR_CARRIED),
    }


def test_a_treasury_failure_leaves_fred_alone(monkeypatch):
    fred = _rows("2026-11-10", "FRED", SOFR=3.89, **{"2Y": 4.73})
    monkeypatch.setattr(sources, "fetch_fred_rates", lambda *a: fred)

    def down(*a):
        raise ConnectionError("treasury.gov")

    monkeypatch.setattr(sources, "fetch_treasury_rates", down)
    assert sources.fetch_rates(D(2026, 11, 1), D(2026, 11, 13)).equals(fred)


def test_treasury_csv_parsed_to_the_fred_tenors_within_the_window(monkeypatch):
    csv = (
        'Date,"1 Mo","1.5 Month","3 Mo","6 Mo","1 Yr","2 Yr","5 Yr","10 Yr","30 Yr"\n'
        "11/13/2026,4.10,4.18,4.32,4.42,4.65,4.88,4.98,5.15,5.47\n"
        "10/30/2026,4.00,4.05,4.20,4.25,4.35,4.55,4.65,4.85,5.20\n"
    )
    urls = []
    real_read_csv = pd.read_csv

    def read_csv(url):
        urls.append(url)
        return real_read_csv(io.StringIO(csv))

    monkeypatch.setattr(sources.pd, "read_csv", read_csv)
    out = sources.fetch_treasury_rates(D(2026, 11, 1), D(2026, 11, 13))
    assert len(urls) == 1 and "2026" in urls[0]
    assert set(out["asof_date"]) == {D(2026, 11, 13)}
    assert dict(zip(out["tenor"], out["rate"], strict=True)) == {
        "1M": 4.10,
        "3M": 4.32,
        "6M": 4.42,
        "1Y": 4.65,
        "2Y": 4.88,
        "5Y": 4.98,
        "10Y": 5.15,
    }
    assert set(out["source"]) == {"Treasury"}
