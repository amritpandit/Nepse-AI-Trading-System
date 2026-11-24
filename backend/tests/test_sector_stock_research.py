from __future__ import annotations

import json

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, cast
from unittest.mock import patch

import httpx
import numpy as np
import sqlalchemy as sa
from app.models.benchmark import NEPSEIndex
from app.models.price import Price
from app.models.signal import Signal
from app.models.stock import Stock
from app.api.routes import research as research_routes
from app.services.nepal_market_data import (
    INDEX_PATH,
    REFERENCE_PATH,
    NepalMarketDataImporter,
)
from app.services.stock_sector_research import (
    StockSectorResearchService,
    _chart_series,
    _indicator_snapshot,
    _historical_score_validation,
    _rsi,
    _score_signal,
    _smc_snapshot,
    _wilder,
)
from sqlalchemy.orm import Session


def _mock_client() -> httpx.Client:
    reference = (
        "symbol,name,source_category,instrument_type,sector,status\n"
        "NABIL,Nabil Bank Limited,Commercial Bank,ordinary,Commercial Bank,listed\n"
        "NABILP,Nabil Promoter Share,Promoter Share,promoter_share,,listed\n"
        "OLDCO,Old Company,Merged,ordinary,,merged\n"
    )
    nabil = (
        "published_date,open,high,low,close,per_change,traded_quantity,traded_amount,status\n"
        "2024-01-01,500,510,495,505,1,1000,505000,A\n"
        "2024-01-02,505,515,500,512,1.39,1200,614400,A\n"
    )
    index = (
        "published_date,open,high,low,close,per_change,traded_quantity,traded_amount,status\n"
        "2024-01-01,2000,2010,1990,2005,0.25,,0,A\n"
        "2024-01-02,2005,2020,2000,2015,0.50,,0,A\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(REFERENCE_PATH):
            return httpx.Response(200, text=reference)
        if request.url.path.endswith(INDEX_PATH):
            return httpx.Response(200, text=index)
        if request.url.path.endswith("/data/nepse/NABIL.csv"):
            return httpx.Response(200, text=nabil)
        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_catalogue_excludes_nonordinary_and_merged() -> None:
    client = _mock_client()
    importer = NepalMarketDataImporter(client=client)
    try:
        rows = importer.list_tradable_equities()
    finally:
        client.close()
    assert [row["symbol"] for row in rows] == ["NABIL"]


def test_imports_ohlcv_and_metadata_idempotently(db_session: Session) -> None:
    client = _mock_client()
    importer = NepalMarketDataImporter(session=db_session, client=client)
    try:
        first = importer.sync_equities(batch_size=1)
        second = importer.sync_equities(batch_size=1)
    finally:
        client.close()

    assert first["symbols"] == ["NABIL"]
    assert first["rows_inserted"] == 2
    assert second["rows_inserted"] == 0
    assert second["rows_updated"] == 2

    stock = db_session.scalar(sa.select(Stock).where(Stock.symbol == "NABIL"))
    assert stock is not None
    assert stock.instrument_type == "ordinary"
    assert stock.sector == "Commercial Bank"
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Price)) == 2


def test_import_dry_run_writes_nothing(db_session: Session) -> None:
    client = _mock_client()
    importer = NepalMarketDataImporter(session=db_session, client=client)
    try:
        result = importer.sync_equities(batch_size=1, dry_run=True)
    finally:
        client.close()

    assert result["dry_run"] is True
    assert result["rows_inserted"] == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Stock)) == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Price)) == 0


def test_imports_nepse_index_as_benchmark_only(db_session: Session) -> None:
    client = _mock_client()
    importer = NepalMarketDataImporter(session=db_session, client=client)
    try:
        result = importer.sync_nepse_index()
    finally:
        client.close()

    assert result["series"] == "NEPSE_INDEX"
    assert result["role"] == "benchmark_only"
    assert result["inserted"] == 2
    assert db_session.scalar(sa.select(sa.func.count()).select_from(NEPSEIndex)) == 2


def test_stock_analysis_only_emits_stock_trigger_and_sector_context(db_session: Session) -> None:
    stock = Stock(
        symbol="TEST",
        name="Test Commercial Bank",
        sector="Commercial Bank",
        instrument_type="ordinary",
        is_active=True,
    )
    db_session.add(stock)
    db_session.flush()
    start = date(2024, 1, 1)
    index_level = 2000.0
    for i in range(230):
        close = 100.0 + i * 0.2
        day = start + timedelta(days=i)
        db_session.add(
            Price(
                stock_id=stock.id,
                date=day,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000,
            )
        )
        index_level += 1.0
        db_session.add(
            NEPSEIndex(
                date=day,
                open=index_level - 1,
                high=index_level + 1,
                low=index_level - 2,
                close=index_level,
                volume=100000,
            )
        )
    db_session.commit()

    result = StockSectorResearchService(db_session).analyze_stock("TEST")
    assert result["symbol"] == "TEST"
    assert result["market_context"]["signal_role"] == "benchmark_only_no_buy_or_sell_orders"
    assert result["sector_context"]["sector"] == "Commercial Bank"
    assert "breakout event study" in result["historical_event_study"]["method"]
    assert result["data_quality"]["source_prices_corporate_action_adjusted"] is False


def test_promoter_share_is_not_eligible_for_stock_triggers(db_session: Session) -> None:
    db_session.add(
        Stock(symbol="TESTP", instrument_type="promoter_share", is_active=True)
    )
    db_session.commit()
    service = StockSectorResearchService(db_session)
    try:
        service.analyze_stock("TESTP")
    except ValueError as exc:
        assert "ordinary shares" in str(exc)
    else:
        raise AssertionError("promoter share must not receive ordinary-share triggers")



def test_weighted_score_model_has_100_points_and_is_not_a_probability() -> None:
    indicators = {
        "close": 150.0,
        "sma_20": 140.0,
        "sma_50": 130.0,
        "sma_200": 120.0,
        "rsi_14": 62.0,
        "macd": 4.0,
        "macd_signal": 2.0,
        "macd_histogram": 2.0,
        "previous_macd_histogram": 1.5,
        "adx_14": 32.0,
        "stoch_k_14": 65.0,
        "stoch_d_3": 60.0,
        "obv_rising_20d": True,
        "volume_ratio_20d": 2.0,
        "daily_return": 0.04,
        "breakout_20d": True,
        "breakdown_20d": False,
        "return_60d": 0.12,
        "return_120d": 0.24,
        "return_200d": 0.35,
    }
    smc = {"bullish_points": 15, "bearish_points": 0}
    market = {"regime": "bullish"}
    sector = {
        "relative_strength_vs_nepse_20d": 0.04,
        "relative_strength_vs_nepse_60d": 0.07,
        "relative_strength_vs_nepse_120d": 0.11,
    }

    result = _score_signal(indicators, smc, market, sector)

    assert sum(result["weights"].values()) == 100
    assert result["buy_score"] == 100
    assert result["sell_score"] == 0
    assert "not a probability" in result["score_type"]


def test_weighted_score_model_handles_bearish_evidence() -> None:
    indicators = {
        "close": 80.0,
        "sma_20": 90.0,
        "sma_50": 100.0,
        "sma_200": 110.0,
        "rsi_14": 32.0,
        "macd": -4.0,
        "macd_signal": -2.0,
        "macd_histogram": -2.0,
        "previous_macd_histogram": -1.5,
        "adx_14": 32.0,
        "stoch_k_14": 25.0,
        "stoch_d_3": 40.0,
        "obv_rising_20d": False,
        "volume_ratio_20d": 2.0,
        "daily_return": -0.04,
        "breakout_20d": False,
        "breakdown_20d": True,
        "return_60d": -0.10,
        "return_120d": -0.20,
        "return_200d": -0.30,
    }
    result = _score_signal(
        indicators,
        {"bullish_points": 0, "bearish_points": 15},
        {"regime": "bearish"},
        {
            "relative_strength_vs_nepse_20d": -0.05,
            "relative_strength_vs_nepse_60d": -0.08,
            "relative_strength_vs_nepse_120d": -0.12,
        },
    )
    assert result["sell_score"] == 100
    assert result["buy_score"] == 0
    assert result["score_margin"] == -100


def test_wilder_rma_uses_initial_period_mean_and_recursive_smoothing() -> None:
    result = _wilder(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 2.0
    assert abs(result[3] - (8.0 / 3.0)) < 1e-12
    assert abs(result[4] - (31.0 / 9.0)) < 1e-12


def test_wilder_rsi_handles_rising_falling_and_flat_prices() -> None:
    rising = _rsi(np.arange(20.0), 14)
    falling = _rsi(np.arange(20.0, 0.0, -1.0), 14)
    flat = _rsi(np.full(20, 100.0), 14)

    assert rising[14] == 100.0
    assert falling[14] == 0.0
    assert flat[14] == 50.0


def test_smc_rules_do_not_emit_structure_without_minimum_history() -> None:
    result = _smc_snapshot([])
    assert result["available"] is False
    assert result["directional_points"] == 0


def test_chart_series_contains_candles_trend_macd_and_rsi(db_session: Session) -> None:
    stock = Stock(symbol="CHART", instrument_type="ordinary", is_active=True)
    db_session.add(stock)
    db_session.flush()
    for index in range(230):
        close = Decimal(str(100.0 + index * 0.3))
        db_session.add(
            Price(
                stock_id=stock.id,
                date=date(2024, 1, 1) + timedelta(days=index),
                open=close - Decimal("0.1"),
                high=close + Decimal("0.2"),
                low=close - Decimal("0.2"),
                close=close,
                volume=1000 + index * 5,
            )
        )
    db_session.flush()
    rows = db_session.scalars(
        sa.select(Price).where(Price.stock_id == stock.id).order_by(Price.date)
    ).all()
    price_rows = list(rows)
    indicators = _indicator_snapshot(price_rows)
    chart = _chart_series(price_rows, _smc_snapshot(price_rows))

    assert indicators["sma_20"] is not None
    assert indicators["macd_histogram"] is not None
    assert indicators["adx_14"] is not None
    assert indicators["stoch_k_14"] is not None
    assert indicators["stoch_d_3"] is not None
    assert indicators["obv"] is not None
    assert len(chart) == 220
    json.dumps(chart, allow_nan=False)
    assert {
        "open", "high", "low", "close", "volume", "sma_20", "sma_50", "sma_200",
        "macd", "macd_signal", "rsi_14", "adx_14", "stoch_k_14", "stoch_d_3", "obv",
    } <= set(chart[-1])



def test_historical_score_validation_uses_chronological_holdout() -> None:
    stock_rows = []
    sector_rows = []
    benchmark_rows = []
    start = date(2024, 1, 1)
    for index in range(300):
        day = start + timedelta(days=index)
        stock_close = Decimal(str(100.0 + index * 0.2 + index * index * 0.001))
        sector_close = Decimal(str(80.0 + index * 0.25 + index * index * 0.0015))
        index_close = Decimal(str(2000.0 + index * 0.2))
        stock_rows.append(
            Price(
                stock_id=1,
                date=day,
                open=stock_close - Decimal("0.1"),
                high=stock_close + Decimal("0.1"),
                low=stock_close - Decimal("0.3"),
                close=stock_close,
                volume=1000,
            )
        )
        sector_rows.append(
            Price(
                stock_id=2,
                date=day,
                open=sector_close - Decimal("0.1"),
                high=sector_close + Decimal("0.3"),
                low=sector_close - Decimal("0.3"),
                close=sector_close,
                volume=1200,
            )
        )
        benchmark_rows.append(
            NEPSEIndex(
                date=day,
                open=index_close - Decimal("0.1"),
                high=index_close + Decimal("0.2"),
                low=index_close - Decimal("0.2"),
                close=index_close,
                volume=100000,
            )
        )

    result = _historical_score_validation(
        stock_rows,
        benchmark_rows,
        [sector_rows],
    )

    assert result["status"] == "evaluated"
    assert result["validation_method"].startswith("chronological holdout")
    assert result["test_observations"] == 90
    assert result["buy_trigger_count"] > 0
    assert result["results"]["forward_20d"]["buy_sample_count"] > 0
    assert result["assumptions"]["nepse_index_role"].startswith("benchmark only")



def test_saved_score_signal_is_idempotent_and_not_probability(db_session: Session) -> None:
    stock = Stock(
        symbol="LEDGER",
        name="Ledger Test Limited",
        sector="Commercial Bank",
        instrument_type="ordinary",
        is_active=True,
    )
    db_session.add(stock)
    db_session.commit()

    base_item = {
        "symbol": "LEDGER",
        "as_of": "2026-11-23",
        "signal": "BUY_TRIGGER",
        "buy_score": 80,
        "sell_score": 20,
        "score_margin": 60,
        "score_model": {"score_type": "evidence points, not probability"},
        "market_context": {"regime": "bullish"},
        "sector_context": {"sector": "Commercial Bank"},
        "data_freshness": {"status": "current"},
        "data_quality": {"status": "usable"},
        "signal_reason": "Test reason",
        "reference_target_level": 120.0,
        "reference_stop_level": 90.0,
    }
    scan_result = {
        "as_of": "2026-11-23",
        "market_context": {"regime": "bullish"},
        "stocks_analyzed": 1,
        "signals": [base_item],
    }

    with patch.object(
        StockSectorResearchService,
        "scan_stocks",
        return_value=scan_result,
    ):
        first = research_routes.refresh_and_save_stock_signals(
            user=cast(Any, None),
            db=db_session,
            limit=300,
        )

    count_after_first = db_session.scalar(sa.select(sa.func.count()).select_from(Signal))
    assert first["snapshots_saved"] == 1
    assert first["value_semantics"].startswith("signed score margin")
    assert count_after_first == 1

    updated_item = {**base_item, "signal": "SELL_TRIGGER", "buy_score": 15, "sell_score": 85, "score_margin": -70}
    scan_result["signals"] = [updated_item]
    with patch.object(
        StockSectorResearchService,
        "scan_stocks",
        return_value=scan_result,
    ):
        second = research_routes.refresh_and_save_stock_signals(
            user=cast(Any, None),
            db=db_session,
            limit=300,
        )

    rows = db_session.scalars(sa.select(Signal)).all()
    assert second["snapshots_saved"] == 1
    assert len(rows) == 1
    assert rows[0].signal_type == "SELL_TRIGGER"
    assert abs(float(rows[0].value) + 0.7) < 1e-8
    assert rows[0].confidence is None
