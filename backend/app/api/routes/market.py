import logging
from typing import Annotated, Literal

import httpx
import sqlalchemy as sa
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.price import Price
from app.models.stock import Stock
from app.models.user import User
from app.schemas.ingestion import IngestionRequest
from app.schemas.market import PriceIngestRequest, PriceResponse
from app.services.csv_ingestion import CsvIngestionService
from app.services.market import ingest_price, list_prices
from app.services.nepal_market_data import NepalMarketDataImporter
from app.services.yonepse_market import (
    MarketDataUnavailable,
    yonepse_market_data,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
PriceLimit = Annotated[int, Query(ge=1, le=1000)]
PriceOffset = Annotated[int, Query(ge=0)]


@router.get("/prices", response_model=list[PriceResponse])
def get_prices(
    db: DbSession,
    symbol: str | None = None,
    limit: PriceLimit = 100,
    offset: PriceOffset = 0,
) -> list[PriceResponse]:
    rows = list_prices(db, symbol=symbol, limit=limit, offset=offset)
    return [
        PriceResponse(
            symbol=stock.symbol,
            date=price.date,
            open=price.open,
            high=price.high,
            low=price.low,
            close=price.close,
            volume=price.volume,
        )
        for price, stock in rows
    ]


@router.post("/ingest", response_model=PriceResponse, status_code=status.HTTP_201_CREATED)
def ingest_market_price(
    payload: PriceIngestRequest, user: CurrentUser, db: DbSession
) -> PriceResponse:
    price = ingest_price(db, payload)
    return PriceResponse(
        symbol=payload.symbol.upper(),
        date=price.date,
        open=price.open,
        high=price.high,
        low=price.low,
        close=price.close,
        volume=price.volume,
    )


@router.post("/ingest/batch", status_code=status.HTTP_202_ACCEPTED)
def ingest_batch(
    payload: IngestionRequest,
    user: CurrentUser,
    db: DbSession,
    symbol: str | None = None,
) -> dict[str, object]:
    service = CsvIngestionService(session=db, source=payload.source)
    result = service.run(
        symbol=symbol,
        start_date=payload.start_date,
        end_date=payload.end_date,
        dry_run=payload.dry_run,
    )
    return {"symbol": symbol.upper() if symbol else "ALL", **result}


@router.post("/research-data/equities/sync")
def sync_research_equities(
    user: CurrentUser,
    db: DbSession,
    offset: Annotated[int, Query(ge=0)] = 0,
    batch_size: Annotated[int, Query(ge=1, le=15)] = 5,
    symbols: list[str] | None = Query(default=None),
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Import a bounded batch of listed ordinary shares from the public history dataset."""
    importer = NepalMarketDataImporter(session=db)
    try:
        return importer.sync_equities(
            offset=offset,
            batch_size=batch_size,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            dry_run=dry_run,
        )
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        importer.close()


@router.post("/research-data/index/sync")
def sync_research_index(
    user: CurrentUser,
    db: DbSession,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Import NEPSE Index history as a benchmark series, never as a tradable stock."""
    importer = NepalMarketDataImporter(session=db)
    try:
        return importer.sync_nepse_index(
            start_date=start_date,
            end_date=end_date,
            dry_run=dry_run,
        )
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        importer.close()


@router.get("/live")
def get_live_market(
    symbol: str | None = None,
    search: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 500,
    sort_by: Literal["turnover", "volume", "percent_change", "market_cap", "ltp"] = "turnover",
    order: Literal["asc", "desc"] = "desc",
) -> dict:
    """Return the latest YONEPSE market snapshot and searchable stock rows."""
    try:
        snapshot = yonepse_market_data.get_snapshot()
    except MarketDataUnavailable as exc:
        logger.warning("YONEPSE market snapshot unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Market data feed is temporarily unavailable",
        ) from exc

    rows = snapshot["stocks"]
    if symbol:
        requested_symbol = symbol.strip().upper()
        rows = [row for row in rows if row["symbol"] == requested_symbol]
    if search:
        query = search.strip().casefold()
        rows = [
            row for row in rows
            if query in row["symbol"].casefold() or query in row["name"].casefold()
        ]

    rows = sorted(
        rows,
        key=lambda row: (
            row.get(sort_by)
            if row.get(sort_by) is not None
            else (float("-inf") if order == "desc" else float("inf"))
        ),
        reverse=order == "desc",
    )
    return {
        **{key: value for key, value in snapshot.items() if key != "stocks"},
        "total_results": len(rows),
        "stocks": rows[:limit],
    }


@router.get("/overview")
def get_market_overview(db: DbSession) -> dict:
    """Get market overview using the public market feed, with DB fallback for volume."""
    total_symbols = db.scalar(sa.select(sa.func.count()).select_from(Stock)) or 0
    latest_date = db.scalar(sa.select(sa.func.max(Price.date)))
    db_volume = 0
    if latest_date is not None:
        db_volume = db.scalar(
            sa.select(sa.func.coalesce(sa.func.sum(Price.volume), 0)).where(Price.date == latest_date)
        ) or 0

    snapshot = None
    try:
        snapshot = yonepse_market_data.get_snapshot()
    except MarketDataUnavailable as exc:
        logger.warning("Live market overview unavailable; using stored figures only: %s", exc)

    index_data = snapshot.get("index") if snapshot else None
    summary = snapshot.get("summary", {}) if snapshot else {}
    index_value = index_data.get("value") if isinstance(index_data, dict) else None
    index_change = index_data.get("percent_change") if isinstance(index_data, dict) else None
    market_data_date = snapshot.get("as_of") if snapshot else None
    market_status = snapshot.get("market_status", {}) if snapshot else {}

    return {
        "total_symbols": summary.get("stock_count", total_symbols),
        "nse_value": index_value,
        "nse_change": index_change,
        "total_volume": summary.get("total_volume", int(db_volume)),
        "volume_change": None,
        "market_data_date": market_data_date or (latest_date.isoformat() if latest_date else None),
        "market_data_freshness": snapshot.get("freshness") if snapshot else "unavailable",
        "market_data_freshness_reason": snapshot.get("freshness_reason") if snapshot else None,
        "market_data_source": snapshot.get("source") if snapshot else None,
        "market_status_freshness": market_status.get("freshness"),
        "market_status_age_minutes": market_status.get("age_minutes"),
        "market_data_source_url": snapshot.get("source_url") if snapshot else None,
        "market_data_attribution": snapshot.get("attribution") if snapshot else None,
        "market_is_open": market_status.get("is_open"),
        "market_status_checked_at": market_status.get("last_checked"),
        "active_signals": 0,
        "new_signals_today": 0,
        "new_symbols_today": 0,
    }
