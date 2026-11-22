def _create_portfolio_with_nabil_position(client) -> int:
    created = client.post(
        "/portfolio/account",
        json={"name": "Market Valuation Test", "initial_capital": 10000.0},
    )
    assert created.status_code == 200
    portfolio_id = created.json()["portfolio_id"]

    trade = client.post(
        f"/portfolio/account/trade?portfolio_id={portfolio_id}",
        json={
            "symbol": "NABIL",
            "action": "BUY",
            "quantity": 10,
            "price": 500.0,
            "fees": 0.0,
        },
    )
    assert trade.status_code == 200
    return portfolio_id


def test_portfolio_uses_latest_session_quote_when_market_is_closed(
    client, monkeypatch
) -> None:
    from app.services.yonepse_market import yonepse_market_data

    monkeypatch.setattr(
        yonepse_market_data,
        "get_snapshot",
        lambda: {
            "source": "YONEPSE",
            "as_of": "2026-11-20T15:16:44+05:45",
            "freshness": "market_closed",
            "freshness_reason": (
                "The exchange is closed; this is the latest available session price."
            ),
            "stocks": [
                {
                    "symbol": "NABIL",
                    "ltp": 550.0,
                    "last_updated_at": "2026-11-20T14:59:36+05:45",
                    "age_minutes": 3071.7,
                    "freshness": "market_closed",
                    "freshness_reason": (
                        "The exchange is closed; this is the latest available session price."
                    ),
                }
            ],
        },
    )
    portfolio_id = _create_portfolio_with_nabil_position(client)

    response = client.get(
        f"/portfolio/account/snapshot?portfolio_id={portfolio_id}"
    )

    assert response.status_code == 200
    snapshot = response.json()
    position = snapshot["positions"]["NABIL"]
    assert snapshot["cash"] == 5000.0
    assert position["current_price"] == 550.0
    assert position["market_value"] == 5500.0
    assert position["price_source"] == "YONEPSE"
    assert position["price_freshness"] == "market_closed"
    assert position["price_as_of"] == "2026-11-20T14:59:36+05:45"
    assert snapshot["equity"] == 10500.0
    assert snapshot["market_data_freshness"] == "market_closed"


def test_portfolio_rejects_stale_symbol_quote_even_when_index_is_fresh(
    client, monkeypatch
) -> None:
    from app.services.yonepse_market import yonepse_market_data

    monkeypatch.setattr(
        yonepse_market_data,
        "get_snapshot",
        lambda: {
            "source": "YONEPSE",
            "as_of": "2026-11-23T11:09:00+05:45",
            "freshness": "fresh",
            "freshness_reason": "The latest index timestamp is current.",
            "stocks": [
                {
                    "symbol": "NABIL",
                    "ltp": 550.0,
                    "last_updated_at": "2026-11-20T14:59:36+05:45",
                    "age_minutes": 4000.0,
                    "freshness": "stale",
                    "freshness_reason": (
                        "This symbol's quote is older than the allowed data-age window."
                    ),
                }
            ],
        },
    )
    portfolio_id = _create_portfolio_with_nabil_position(client)

    response = client.get(
        f"/portfolio/account/snapshot?portfolio_id={portfolio_id}"
    )

    assert response.status_code == 200
    snapshot = response.json()
    position = snapshot["positions"]["NABIL"]
    assert position["current_price"] == 500.0
    assert position["market_value"] == 5000.0
    assert position["price_source"] == "COST_BASIS"
    assert position["price_freshness"] == "stale"
    assert snapshot["equity"] == 10000.0
