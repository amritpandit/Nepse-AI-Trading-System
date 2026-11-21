# backend/tests/test_portfolio_integration.py

from decimal import Decimal


def test_portfolio_account_creation_and_trade(client):
    """Test creating a portfolio account and executing trades."""
    # Create portfolio account
    response = client.post("/portfolio/account", json={"initial_capital": 50000.0})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Portfolio account created successfully"
    assert data["initial_capital"] == 50000.0

    # Get initial snapshot
    response = client.get("/portfolio/account/snapshot")
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["equity"] == 50000.0
    assert snapshot["cash"] == 50000.0
    assert len(snapshot["positions"]) == 0

    # Buy some stocks
    response = client.post(
        "/portfolio/account/trade",
        json={"symbol": "NBL", "action": "BUY", "quantity": 100, "price": 250.0, "fees": 10.0},
    )
    assert response.status_code == 200
    trade_data = response.json()
    assert trade_data["message"] == "BUY order recorded successfully in paper portfolio"
    assert trade_data["symbol"] == "NBL"
    assert trade_data["quantity"] == 100.0
    assert trade_data["price"] == 250.0
    assert trade_data["fees"] == 10.0
    assert trade_data["total_value"] == 25000.0  # 100 * 250
    # Cash should be: 50000 - 25000 - 10 = 24990
    assert trade_data["new_cash_balance"] == 24990.0

    # Get updated snapshot
    response = client.get("/portfolio/account/snapshot")
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["equity"] == 49990.0  # Equity decreased by $10 fees
    assert snapshot["cash"] == 24990.0
    assert len(snapshot["positions"]) == 1
    assert snapshot["positions"]["NBL"]["quantity"] == 100.0
    assert snapshot["positions"]["NBL"]["average_price"] == 250.0

    # Get allocation
    response = client.get("/portfolio/account/allocation")
    assert response.status_code == 200
    allocation = response.json()
    # Cash allocation: 24990/49990 * 100 = 49.998%
    # NBL allocation: 25000/49990 * 100 = 50.002%
    assert abs(allocation["allocation"]["CASH"] - 49.998) < 0.01
    assert abs(allocation["allocation"]["NBL"] - 50.002) < 0.01
    assert abs(allocation["total_equity"] - 49990.0) < 0.01
    assert abs(allocation["cash"] - 24990.0) < 0.01
    assert abs(allocation["positions_value"] - 25000.0) < 0.01

    # Sell some stocks
    response = client.post(
        "/portfolio/account/trade",
        json={"symbol": "NBL", "action": "SELL", "quantity": 50, "price": 260.0, "fees": 5.0},
    )
    assert response.status_code == 200
    trade_data = response.json()
    assert trade_data["message"] == "SELL order recorded successfully in paper portfolio"
    assert trade_data["symbol"] == "NBL"
    assert trade_data["quantity"] == 50.0
    assert trade_data["price"] == 260.0
    assert trade_data["fees"] == 5.0
    assert trade_data["total_value"] == 13000.0  # 50 * 260
    # Cash should be: 24990 + 13000 - 5 = 37985
    assert trade_data["new_cash_balance"] == 37985.0

    # Get final snapshot
    response = client.get("/portfolio/account/snapshot")
    assert response.status_code == 200
    snapshot = response.json()
    # Equity = cash + position value = 37985 + (50 * 250) = 50485
    # But fees were $15 total, so equity = 50000 - 10 - 5 = 49985
    # Actually: equity = cash + (quantity * average_price) = 37985 + 12500 = 50485
    assert snapshot["equity"] == 50485.0
    assert snapshot["cash"] == 37985.0
    assert len(snapshot["positions"]) == 1
    assert snapshot["positions"]["NBL"]["quantity"] == 50.0
    assert snapshot["positions"]["NBL"]["average_price"] == 250.0  # Average price unchanged
    # Unrealized P&L: since current_price defaults to average_price in snapshot, P&L = 0
    assert snapshot["positions"]["NBL"]["unrealized_pnl"] == 0.0


def test_portfolio_optimization_endpoints(client):
    """Test portfolio optimization endpoints."""
    # Test equal weight allocation
    response = client.post(
        "/portfolio/optimize",
        json={
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "expected_returns": [0.15, 0.12, 0.18],
            "covariance_matrix": [[0.04, 0.01, 0.005], [0.01, 0.09, 0.003], [0.005, 0.003, 0.16]],
            "method": "equal_weight",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "equal_weight"
    assert data["success"]
    assert len(data["weights"]) == 3
    # Each weight should be approximately 1/3
    for weight in data["weights"].values():
        assert abs(weight - 1.0 / 3.0) < 0.0001
    # Calculate expected return: (0.15 + 0.12 + 0.18) / 3 = 0.15
    assert abs(data["expected_return"] - 0.15) < 0.0001

    # Test risk parity allocation
    response = client.post(
        "/portfolio/optimize",
        json={
            "symbols": ["STOCK_A", "STOCK_B"],
            "expected_returns": [0.10, 0.15],
            "covariance_matrix": [
                [0.04, 0.01],  # Stock A: 20% volatility
                [0.01, 0.09],  # Stock B: 30% volatility
            ],
            "method": "risk_parity",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "risk_parity"
    assert data["success"]
    assert len(data["weights"]) == 2
    # Risk parity should give higher weight to lower volatility stock (STOCK_A)
    assert data["weights"]["STOCK_A"] > data["weights"]["STOCK_B"]
    # Weights should sum to 1
    assert abs(sum(data["weights"].values()) - 1.0) < 0.0001

    # Test mean-variance optimization
    response = client.post(
        "/portfolio/optimize",
        json={
            "symbols": ["STOCK_X", "STOCK_Y"],
            "expected_returns": [0.20, 0.10],
            "covariance_matrix": [[0.04, 0.005], [0.005, 0.09]],
            "method": "mean_variance",
            "constraints": {"long_only": True, "max_weight": 0.8},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "mean_variance"
    assert data["success"]
    assert len(data["weights"]) == 2
    # Weights should sum to 1
    assert abs(sum(data["weights"].values()) - 1.0) < 0.0001
    # All weights should be between 0 and 0.8 (due to max_weight constraint)
    for weight in data["weights"].values():
        assert 0 <= weight <= 0.8 + 0.0001


def test_portfolio_health_endpoint(client):
    """Test portfolio health endpoint."""
    response = client.get("/portfolio/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "portfolio"
    assert data["version"] == "1.0.0"

def test_portfolios_are_persisted_and_isolated_by_account(client) -> None:
    first = client.post(
        "/portfolio/account",
        json={"name": "Long Term", "initial_capital": 20000.0},
    )
    second = client.post(
        "/portfolio/account",
        json={"name": "Dividend", "initial_capital": 15000.0},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_id = first.json()["portfolio_id"]
    second_id = second.json()["portfolio_id"]
    assert first_id != second_id

    listed = client.get("/portfolio/accounts")
    assert listed.status_code == 200
    assert {item["name"] for item in listed.json()} == {"Long Term", "Dividend"}

    trade = client.post(
        f"/portfolio/account/trade?portfolio_id={first_id}",
        json={
            "symbol": "NABIL",
            "action": "BUY",
            "quantity": 20,
            "price": 300.0,
            "fees": 50.0,
        },
    )
    assert trade.status_code == 200
    assert trade.json()["execution_type"] == "paper_portfolio_record_only"
    assert trade.json()["new_cash_balance"] == 13950.0

    # Separate API requests read the same saved rows, not an in-memory session object.
    first_snapshot = client.get(f"/portfolio/account/snapshot?portfolio_id={first_id}")
    second_snapshot = client.get(f"/portfolio/account/snapshot?portfolio_id={second_id}")
    assert first_snapshot.status_code == 200
    assert first_snapshot.json()["cash"] == 13950.0
    assert first_snapshot.json()["positions"]["NABIL"]["quantity"] == 20.0
    assert second_snapshot.status_code == 200
    assert second_snapshot.json()["cash"] == 15000.0
    assert second_snapshot.json()["positions"] == {}

    history = client.get(f"/portfolio/account/transactions?portfolio_id={first_id}")
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["symbol"] == "NABIL"


def test_portfolio_id_cannot_be_read_by_another_user(client, db_session) -> None:
    from app.models.user import User
    from app.services.persistent_portfolio import persistent_portfolio_service

    other_user = User(
        id="another-user",
        email="another@example.com",
        hashed_password="x",
        role="researcher",
    )
    db_session.add(other_user)
    db_session.commit()
    other_account = persistent_portfolio_service.create_account(
        db_session,
        user_id=other_user.id,
        name="Private Portfolio",
        initial_capital=Decimal("10000.00"),
    )

    response = client.get(f"/portfolio/account/snapshot?portfolio_id={other_account.id}")
    assert response.status_code == 404

def test_portfolio_falls_back_to_cost_basis_when_market_feed_is_stale(
    client, monkeypatch
) -> None:
    from app.services.persistent_portfolio import persistent_portfolio_service

    monkeypatch.setattr(
        persistent_portfolio_service,
        "_live_quotes",
        lambda positions: (
            {},
            {
                "source": "YONEPSE",
                "as_of": "2026-11-18T15:00:00+05:45",
                "freshness": "stale",
            },
        ),
    )
    created = client.post(
        "/portfolio/account",
        json={"name": "Stale Price Test", "initial_capital": 10000.0},
    )
    portfolio_id = created.json()["portfolio_id"]
    bought = client.post(
        f"/portfolio/account/trade?portfolio_id={portfolio_id}",
        json={"symbol": "NABIL", "action": "BUY", "quantity": 10, "price": 100.0},
    )
    assert bought.status_code == 200

    snapshot = client.get(f"/portfolio/account/snapshot?portfolio_id={portfolio_id}")
    assert snapshot.status_code == 200
    position = snapshot.json()["positions"]["NABIL"]
    assert position["current_price"] == 100.0
    assert position["market_value"] == 1000.0
    assert snapshot.json()["market_data_freshness"] == "stale"
