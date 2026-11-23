"""Market-context, sector ranking and tradable-stock research routes."""
from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import sqlalchemy as sa
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.signal import Signal
from app.models.stock import Stock
from app.models.user import User
from app.services.stock_sector_research import StockSectorResearchService
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

router = APIRouter(prefix="/research", tags=["research"])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.get("/market-regime")
def market_regime(db: DbSession) -> dict:
    """NEPSE Index is context/benchmark only; it is never a stock signal."""
    service = StockSectorResearchService(db)
    return service.sector_ranking()["market_context"]


@router.get("/sectors")
def sector_ranking(db: DbSession) -> dict:
    """Rank sectors by member-stock median returns, breadth and NEPSE relative strength."""
    return StockSectorResearchService(db).sector_ranking()


@router.post("/signals/refresh")
def refresh_and_save_stock_signals(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=300)] = 300,
) -> dict:
    """Run a current stock scan and idempotently save one research snapshot per stock/session."""
    scan = StockSectorResearchService(db).scan_stocks(limit=limit)
    signal_rows = scan.get("signals", [])
    symbols = [str(item.get("symbol", "")).upper() for item in signal_rows]
    stocks = db.scalars(sa.select(Stock).where(Stock.symbol.in_(symbols))).all() if symbols else []
    stock_by_symbol = {item.symbol.upper(): item for item in stocks}

    saved = 0
    counts: dict[str, int] = {}
    for item in signal_rows:
        stock = stock_by_symbol.get(str(item.get("symbol", "")).upper())
        if stock is None:
            continue
        as_of = date.fromisoformat(str(item["as_of"]))
        signal_type = str(item.get("signal", "HOLD"))
        buy_score = int(item.get("buy_score", 0))
        sell_score = int(item.get("sell_score", 0))
        score_margin = int(item.get("score_margin", 0))
        rule_params = {
            "model_version": "sector-stock-score-v1",
            "buy_score": buy_score,
            "sell_score": sell_score,
            "score_margin": score_margin,
            "score_type": "evidence points, not a probability",
            "score_model": item.get("score_model"),
            "market_context": item.get("market_context"),
            "sector_context": item.get("sector_context"),
            "data_freshness": item.get("data_freshness"),
            "data_quality": item.get("data_quality"),
            "signal_reason": item.get("signal_reason"),
        }
        existing = db.scalar(
            sa.select(Signal).where(
                Signal.stock_id == stock.id,
                Signal.date == as_of,
                Signal.provider == "research_score",
                Signal.model_source == "sector-stock-score-v1",
            )
        )
        values = {
            "signal_type": signal_type,
            "value": float(score_margin) / 100.0,
            "confidence": None,
            "provider": "research_score",
            "model_source": "sector-stock-score-v1",
            "rule_params": json.dumps(rule_params, default=str),
            "target_price": item.get("reference_target_level"),
            "take_profit": item.get("reference_target_level"),
            "stop_loss": item.get("reference_stop_level"),
        }
        if existing is None:
            db.add(Signal(stock_id=stock.id, date=as_of, **values))
        else:
            for key, value in values.items():
                setattr(existing, key, value)
        saved += 1
        counts[signal_type] = counts.get(signal_type, 0) + 1

    db.commit()
    return {
        "as_of": scan.get("as_of"),
        "market_context": scan.get("market_context"),
        "stocks_analyzed": scan.get("stocks_analyzed", 0),
        "snapshots_saved": saved,
        "signal_counts": counts,
        "provider": "research_score",
        "model_version": "sector-stock-score-v1",
        "value_semantics": "signed score margin divided by 100; not a probability",
    }


@router.get("/signals/history")
def get_saved_stock_signal_history(
    db: DbSession,
    symbol: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 300,
) -> dict:
    """Retrieve persisted score-model signal snapshots without running a new scan."""
    query = (
        sa.select(Signal, Stock)
        .join(Stock, Signal.stock_id == Stock.id)
        .where(
            Signal.provider == "research_score",
            Signal.model_source == "sector-stock-score-v1",
        )
        .order_by(Signal.date.desc(), Stock.symbol)
        .limit(limit)
    )
    if symbol:
        query = query.where(Stock.symbol == symbol.strip().upper())
    rows = db.execute(query).all()
    history = []
    for signal, stock in rows:
        try:
            score_details = json.loads(signal.rule_params or "{}")
        except (TypeError, ValueError):
            score_details = {}
        history.append(
            {
                "date": signal.date.isoformat(),
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector,
                "signal": signal.signal_type,
                "score_margin": round(float(signal.value) * 100, 2),
                "buy_score": score_details.get("buy_score"),
                "sell_score": score_details.get("sell_score"),
                "signal_reason": score_details.get("signal_reason"),
                "stop_reference": float(signal.stop_loss) if signal.stop_loss is not None else None,
                "target_reference": float(signal.target_price) if signal.target_price is not None else None,
            }
        )
    return {"count": len(history), "signals": history}


@router.get("/stocks/{symbol}")
def stock_research(symbol: str, db: DbSession) -> dict:
    """Detailed technical trigger, market/sector confirmation and historical event study."""
    try:
        return StockSectorResearchService(db).analyze_stock(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/signals")
def scan_stock_signals(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=300)] = 100,
) -> dict:
    """Scan eligible ordinary shares; only individual stocks can have BUY/SELL triggers."""
    return StockSectorResearchService(db).scan_stocks(limit=limit)
