"""Tests for Phase 5: Strategy and Backtesting MVP."""

from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from app.models.benchmark import NEPSEIndex, SectorIndex
from app.models.price import Price
from app.models.stock import Stock
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.services.backtest import BacktestEngine, VectorizedBacktestEngine
from app.services.benchmark import BenchmarkService
from app.services.strategy import StrategyService
from sqlalchemy import select
from sqlalchemy.orm import Session


def _seed_strategy(
    db: Session,
    name: str = "test_strategy",
    version: str = "v1.0.0",
    config: dict | None = None,
) -> Strategy:
    if config is None:
        config = {
            "symbols": ["TEST"],
            "entry_rules": [{"rule": "rsi_oversold", "params": {"threshold": 30}}],
            "exit_rules": [{"rule": "rsi_exit_long", "params": {"threshold": 70}}],
            "risk_rules": [{"rule": "max_position_size", "params": {"max_pct": 0.1}}],
        }
    strategy = Strategy(name=name, version=version, config=config)
    db.add(strategy)
    db.flush()
    db.refresh(strategy)
    return strategy


def _seed_price_data(
    db: Session,
    symbol: str,
    start_date: str,
    count: int = 60,
    base_price: float = 100.0,
) -> int:
    stock = db.scalar(select(Stock).where(Stock.symbol == symbol.upper()))
    if stock is None:
        stock = Stock(symbol=symbol.upper(), is_active=True)
        db.add(stock)
        db.flush()

    start = date.fromisoformat(start_date)
    for i in range(count):
        price_date = start + timedelta(days=i)
        close = base_price + (i - count / 2) * 0.5
        high = close + 2.0
        low = close - 2.0
        db.add(
            Price(
                stock_id=stock.id,
                date=price_date,
                open=Decimal(str(close * 0.99)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=1000 + i * 10,
            )
        )
    db.flush()
    return stock.id


class TestStrategyService:
    def test_create_strategy(self, db_session: Session) -> None:
        service = StrategyService(session=db_session)
        strategy = service.create_strategy(
            name="momentum",
            version="v1.0.0",
            config={"symbols": ["NEPSE"]},
        )

        assert strategy.id is not None
        assert strategy.name == "momentum"
        assert strategy.version == "v1.0.0"

    def test_get_strategy(self, db_session: Session) -> None:
        strategy = _seed_strategy(db_session)

        service = StrategyService(session=db_session)
        retrieved = service.get_strategy(strategy.id)

        assert retrieved is not None
        assert retrieved.id == strategy.id

    def test_list_strategies(self, db_session: Session) -> None:
        _seed_strategy(db_session, "strat1")
        _seed_strategy(db_session, "strat2")

        service = StrategyService(session=db_session)
        strategies = service.list_strategies(limit=10)

        assert len(strategies) >= 2

    def test_evaluate_entry_signal_rsi_oversold(self) -> None:
        service = StrategyService()
        features = {"rsi_14": 25.0, "close": 100.0}
        entry_rules = [{"rule": "rsi_oversold", "params": {"threshold": 30}}]

        result = service.evaluate_entry_signal("TEST", "2024-01-01", features, entry_rules)
        assert result is True

    def test_evaluate_entry_signal_rsi_not_oversold(self) -> None:
        service = StrategyService()
        features = {"rsi_14": 50.0, "close": 100.0}
        entry_rules = [{"rule": "rsi_oversold", "params": {"threshold": 30}}]

        result = service.evaluate_entry_signal("TEST", "2024-01-01", features, entry_rules)
        assert result is False

    def test_evaluate_exit_signal(self) -> None:
        service = StrategyService()
        features = {"rsi_14": 75.0, "close": 110.0}
        exit_rules = [{"rule": "rsi_exit_long", "params": {"threshold": 70}}]

        class MockPosition:
            quantity = 100

        result = service.evaluate_exit_signal(
            "TEST", "2024-01-01", features, exit_rules, MockPosition()
        )
        assert result is True

    def test_check_risk_rules(self) -> None:
        service = StrategyService()
        portfolio = {"cash": 1000000, "price": 100.0}
        risk_rules = [{"rule": "max_position_size", "params": {"max_pct": 0.1}}]

        result = service.check_risk_rules(portfolio, risk_rules, "TEST")
        assert "max_quantity" in result
        assert result["max_quantity"] > 0


class TestBacktestEngine:
    def test_apply_slippage_buy(self) -> None:
        engine = BacktestEngine(slippage_bps=Decimal("5.0"))
        price = Decimal("100.0")

        result = engine._apply_slippage(price, "buy")
        assert result > price
        assert result == Decimal("100.05")

    def test_apply_slippage_sell(self) -> None:
        engine = BacktestEngine(slippage_bps=Decimal("5.0"))
        price = Decimal("100.0")

        result = engine._apply_slippage(price, "sell")
        assert result < price
        assert result == Decimal("99.95")

    def test_calculate_fill_rate_high_volume(self) -> None:
        engine = BacktestEngine(partial_fill_threshold=Decimal("0.3"))
        result = engine._calculate_fill_rate(10000, 100)
        assert result == Decimal("1.0")

    def test_calculate_fill_rate_low_volume(self) -> None:
        engine = BacktestEngine(partial_fill_threshold=Decimal("0.3"))
        result = engine._calculate_fill_rate(100, 500)
        assert result < Decimal("1.0")

    def test_calculate_max_drawdown(self) -> None:
        engine = BacktestEngine()
        equity_curve = [
            {"equity": 100000, "cash": 0, "positions_value": 100000},
            {"equity": 90000, "cash": 0, "positions_value": 90000},
            {"equity": 110000, "cash": 0, "positions_value": 110000},
        ]

        result = engine._calculate_max_drawdown(equity_curve)
        assert result == 0.1

    def test_calculate_sharpe(self) -> None:
        engine = BacktestEngine()
        returns = [0.01, 0.02, -0.005, 0.015, 0.008]

        result = engine._calculate_sharpe(returns)
        assert result is not None
        assert abs(result) > 0

    def test_calculate_sharpe_uses_sample_standard_deviation(self) -> None:
        engine = BacktestEngine()
        returns = [0.01, -0.01, 0.02, 0.0]
        expected = float(np.mean(returns) / np.std(returns, ddof=1) * (252**0.5))
        assert engine._calculate_sharpe(returns) == expected


    def test_sortino_uses_downside_deviation_across_all_days(self) -> None:
        engine = BacktestEngine()
        returns = [0.10, -0.05, 0.03, -0.02]
        downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(returns, 0.0)))))
        expected = float(np.mean(returns) / downside_deviation * np.sqrt(252))
        curve = [
            {"date": "2024-01-01", "equity": Decimal("100"), "cash": Decimal("100"), "positions_value": Decimal("0")},
            {"date": "2024-01-02", "equity": Decimal("110"), "cash": Decimal("110"), "positions_value": Decimal("0")},
            {"date": "2024-01-03", "equity": Decimal("104.5"), "cash": Decimal("104.5"), "positions_value": Decimal("0")},
            {"date": "2024-01-04", "equity": Decimal("107.635"), "cash": Decimal("107.635"), "positions_value": Decimal("0")},
            {"date": "2024-01-05", "equity": Decimal("105.4823"), "cash": Decimal("105.4823"), "positions_value": Decimal("0")},
        ]
        metrics = engine._calculate_metrics(curve, [])
        assert metrics["sortino_ratio"] == round(expected, 4)

    def test_metrics_use_configured_initial_capital_as_baseline(self) -> None:
        engine = BacktestEngine()
        curve = [
            {"date": "2024-01-01", "equity": Decimal("105"), "cash": Decimal("5"), "positions_value": Decimal("100")},
            {"date": "2024-01-02", "equity": Decimal("95"), "cash": Decimal("5"), "positions_value": Decimal("90")},
        ]
        metrics = engine._calculate_metrics(curve, [], initial_capital=Decimal("100"))
        assert metrics["total_return"] == -0.05
        assert metrics["max_drawdown"] == round((105 - 95) / 105, 6)

    def test_annualized_return_is_compounded(self) -> None:
        engine = BacktestEngine()
        equity_curve = [
            {"date": "2024-01-01", "equity": Decimal("100"), "cash": Decimal("100"), "positions_value": Decimal("0")},
            {"date": "2024-01-02", "equity": Decimal("110"), "cash": Decimal("110"), "positions_value": Decimal("0")},
            {"date": "2024-01-03", "equity": Decimal("121"), "cash": Decimal("121"), "positions_value": Decimal("0")},
        ]
        metrics = engine._calculate_metrics(equity_curve, [])
        expected = (1.21 ** (252 / 2)) - 1
        assert metrics["annualized_return"] == round(expected, 6)
        assert metrics["trading_days"] == 3
 
    def test_exit_fills_next_bar_open_and_keeps_unfilled_shares(
        self, db_session: Session
    ) -> None:
        _seed_price_data(db_session, "TEST", "2024-01-01", count=3)
        db_session.commit()

        prices = db_session.scalars(
            select(Price)
            .join(Stock, Price.stock_id == Stock.id)
            .where(Stock.symbol == "TEST")
            .order_by(Price.date)
        ).all()
        prices[0].open = Decimal("100")
        prices[0].close = Decimal("100")
        prices[0].volume = 1000
        prices[1].open = Decimal("105")
        prices[1].close = Decimal("105")
        prices[1].volume = 1000
        prices[2].open = Decimal("90")
        prices[2].close = Decimal("90")
        prices[2].volume = 10
        db_session.commit()

        engine = BacktestEngine(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0"),
            slippage_bps=Decimal("0"),
            data_quality_gate=None,
        )
        engine.gate.assert_safe_for_backtest = lambda symbol, date_str: None
        engine._get_features = lambda symbol, date_str: {}
        engine._should_enter = lambda symbol, date_str, features, rules: date_str == "2024-01-01"
        engine._should_exit = (
            lambda symbol, date_str, features, rules, position, price_row:
            date_str == "2024-01-02"
        )
        strategy = SimpleNamespace(
            id=1,
            name="next-open-exit-test",
            version="1",
            config={"symbols": ["TEST"], "risk_rules": [], "entry_rules": [], "exit_rules": []},
        )

        result = engine.run(strategy, ["TEST"], "2024-01-01", "2024-01-03")
        sells = [trade for trade in result["trades"] if trade["action"] == "SELL"]

        assert len(sells) == 1
        assert sells[0]["timestamp"].startswith("2024-01-03")
        assert sells[0]["price"] == 90.0
        assert sells[0]["quantity"] == 3.0
        assert result["equity_curve"][-1]["positions_value"] == 8730.0


class TestBenchmarkService:
    def test_backtest_refuses_to_run_without_historical_prices(self) -> None:
        engine = BacktestEngine()
        engine._get_trading_dates = lambda start_date, end_date, symbols: []
        strategy = SimpleNamespace(
            id=1,
            name="empty-history-test",
            version="1",
            config={"symbols": ["TEST"]},
        )
        with pytest.raises(ValueError, match="No historical price observations"):
            engine.run(strategy, ["TEST"], "2024-01-01", "2024-12-31")

    def test_get_trading_dates_uses_observed_price_dates(self, db_session: Session) -> None:
        _seed_price_data(db_session, "TEST", "2024-01-01", count=5)
        engine = BacktestEngine(session=db_session)
        dates = engine._get_trading_dates("2024-01-01", "2024-01-05", ["TEST"])
        assert dates == [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ]

    def test_calculate_benchmark_metrics(self) -> None:
        service = BenchmarkService()
        equity_curve = [
            {"equity": 100000, "cash": 0, "positions_value": 100000},
            {"equity": 105000, "cash": 0, "positions_value": 105000},
            {"equity": 110000, "cash": 0, "positions_value": 110000},
        ]

        result = service._calculate_benchmark_metrics(equity_curve)
        assert result["period_return"] == 0.1
        assert "annualized_return" in result

    def test_calculate_profit_factor(self) -> None:
        engine = BacktestEngine()
        trades = [
            {"action": "BUY", "symbol": "A", "quantity": 100, "price": 100.0},
            {"action": "SELL", "symbol": "A", "quantity": 100, "price": 120.0},
            {"action": "BUY", "symbol": "B", "quantity": 100, "price": 100.0},
            {"action": "SELL", "symbol": "B", "quantity": 100, "price": 80.0},
        ]
        result = engine._calculate_profit_factor(trades)
        assert result == 1.0

    def test_calculate_expectancy(self) -> None:
        engine = BacktestEngine()
        trades = [
            {"action": "BUY", "symbol": "A", "quantity": 100, "price": 100.0},
            {"action": "SELL", "symbol": "A", "quantity": 100, "price": 105.0},
        ]
        result = engine._calculate_expectancy(trades)
        assert result == 500.0


class TestVectorizedBacktestEngine:
    """Tests for the vectorized backtest engine (Phase 10)."""

    def test_init(self) -> None:
        """VectorizedBacktestEngine initializes with same params as BacktestEngine."""
        vec_engine = VectorizedBacktestEngine(
            initial_capital=Decimal("500000"),
            commission_rate=Decimal("0.01"),
            slippage_bps=Decimal("10"),
        )

        assert vec_engine.initial_capital == Decimal("500000")
        assert vec_engine.commission_rate == Decimal("0.01")
        assert vec_engine.slippage_bps == Decimal("10")

    def test_import_polars(self) -> None:
        """Polars is available for vectorized operations."""
        assert pl is not None
        # Basic Polars operation test
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert df.shape == (3, 2)


class TestModelsIntegration:
    def test_trade_model_has_transaction_fields(self) -> None:
        trade = Trade()
        assert hasattr(trade, "transaction_cost")
        assert hasattr(trade, "fill_rate")

    def test_benchmark_models_exist(self) -> None:
        assert NEPSEIndex is not None
        assert SectorIndex is not None
        assert hasattr(NEPSEIndex, "close")
        assert hasattr(SectorIndex, "sector")
