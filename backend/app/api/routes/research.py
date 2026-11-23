"""Market-context, sector ranking and tradable-stock research routes."""
from __future__ import annotations

from typing import Annotated

from app.db.session import get_db
from app.services.stock_sector_research import StockSectorResearchService
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

router = APIRouter(prefix="/research", tags=["research"])
DbSession = Annotated[Session, Depends(get_db)]


@router.get("/market-regime")
def market_regime(db: DbSession) -> dict:
    """NEPSE Index is context/benchmark only; it is never a stock signal."""
    service = StockSectorResearchService(db)
    return service.sector_ranking()["market_context"]


@router.get("/sectors")
def sector_ranking(db: DbSession) -> dict:
    """Rank sectors by member-stock median returns, breadth and NEPSE relative strength."""
    return StockSectorResearchService(db).sector_ranking()


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
