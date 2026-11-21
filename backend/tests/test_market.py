from datetime import date

import pytest
from app.models.price import Price
from app.models.stock import Stock
from sqlalchemy.exc import IntegrityError


def test_create_stock_insert_price_and_query(client) -> None:
    ingest_response = client.post(
        "/market/ingest",
        json={
            "symbol": "NABIL",
            "name": "Nabil Bank Limited",
            "sector": "Commercial Banks",
            "date": "2026-06-04",
            "open": "1000.00",
            "high": "1020.00",
            "low": "990.00",
            "close": "1010.00",
            "volume": 10000,
        },
    )

    assert ingest_response.status_code == 201
    assert ingest_response.json()["symbol"] == "NABIL"

    prices_response = client.get("/market/prices?symbol=NABIL")

    assert prices_response.status_code == 200
    assert prices_response.json()[0]["close"] == "1010.0000"


def test_price_unique_constraint_is_enforced(db_session) -> None:
    stock = Stock(symbol="NICA", name="NIC Asia Bank Limited", sector="Commercial Banks")
    db_session.add(stock)
    db_session.commit()
    db_session.refresh(stock)

    db_session.add_all(
        [
            Price(stock_id=stock.id, date=date(2026, 6, 4), close=900, volume=1000),
            Price(stock_id=stock.id, date=date(2026, 6, 4), close=901, volume=1001),
        ]
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_live_market_route_searches_and_sorts_provider_rows(client, monkeypatch) -> None:
    import app.api.routes.market as market_routes

    snapshot = {
        "source": "YONEPSE",
        "source_url": "https://github.com/Shubhamnpk/yonepse",
        "attribution": "Unofficial test source",
        "fetched_at": "2026-11-21T10:00:00+05:45",
        "as_of": "2026-11-21T09:59:00+05:45",
        "age_minutes": 1.0,
        "freshness": "fresh",
        "market_status": {"is_open": True, "last_checked": "2026-11-21T10:00:00"},
        "index": {"name": "NEPSE Index", "value": 2700.0, "percent_change": 0.5},
        "summary": {"stock_count": 2, "total_volume": 3000, "total_turnover": 900000.0},
        "stocks": [
            {"symbol": "NABIL", "name": "Nabil Bank Limited", "turnover": 500000.0},
            {"symbol": "NICA", "name": "NIC Asia Bank Limited", "turnover": 400000.0},
            {"symbol": "AHPC", "name": "Arun Valley Hydropower", "turnover": 900000.0},
        ],
    }
    monkeypatch.setattr(
        market_routes.yonepse_market_data,
        "get_snapshot",
        lambda: snapshot,
    )

    response = client.get("/market/live?search=bank&sort_by=turnover&limit=1")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "YONEPSE"
    assert data["freshness"] == "fresh"
    assert data["total_results"] == 2
    assert len(data["stocks"]) == 1
    assert data["stocks"][0]["symbol"] == "NABIL"
