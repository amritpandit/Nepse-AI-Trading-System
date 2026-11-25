"""Importer for the public, versioned NEPSE daily OHLCV research dataset.

Prices are stored exactly as published (not corporate-action adjusted). The
research endpoints must disclose that limitation and flag suspicious price
discontinuities rather than silently adjusting historical prices.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import sqlalchemy as sa
from app.db.session import session_scope
from app.models.benchmark import NEPSEIndex
from app.models.data_source import IngestionLog
from app.models.price import Price
from app.models.stock import Stock
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DATASET_BASE_URL = "https://raw.githubusercontent.com/binayabaral/nepal-market-data/main"
REFERENCE_PATH = "data/reference/nepse-symbols.csv"
INDEX_PATH = "data/nepse/NEPSE_INDEX.csv"
DATASET_ATTRIBUTION = (
    "nepal-market-data by Binaya Baral; public MIT-licensed dataset; "
    "daily prices are raw and are not adjusted for corporate actions."
)
MISSING = {"", "-", "--", "n/a", "na", "nan", "none"}


def _decimal(value: Any) -> Decimal | None:
    text = str(value or "").strip().replace(",", "")
    if text.lower() in MISSING:
        return None
    try:
        result = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _int_value(value: Any) -> int | None:
    number = _decimal(value)
    if number is None:
        return None
    try:
        return int(number)
    except (ValueError, OverflowError):
        return None


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


class NepalMarketDataImporter:
    """Download the public symbol catalogue and ingest bounded batches of OHLCV."""

    def __init__(
        self,
        session: Session | None = None,
        client: httpx.Client | None = None,
        base_url: str = DATASET_BASE_URL,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._session = session
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Nepse-AI-Research/1.0"},
        )
        self._owns_client = client is None
        self.base_url = base_url.rstrip("/")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get_csv(self, path: str) -> list[dict[str, str]]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self._client.get(url)
        response.raise_for_status()
        return _read_csv(response.text)

    def list_tradable_equities(self) -> list[dict[str, str]]:
        """Return only listed ordinary shares; exclude debentures/promoters/merged."""
        rows = self._get_csv(REFERENCE_PATH)
        eligible = [
            row
            for row in rows
            if row.get("status", "").strip().lower() == "listed"
            and row.get("instrument_type", "").strip().lower() == "ordinary"
            and row.get("symbol", "").strip()
        ]
        return sorted(eligible, key=lambda row: row["symbol"].strip().upper())

    def sync_equities(
        self,
        offset: int = 0,
        batch_size: int = 5,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Import one bounded catalogue batch; repeat with next_offset until done."""
        if offset < 0 or not 1 <= batch_size <= 15:
            raise ValueError("offset must be >= 0 and batch_size must be between 1 and 15")
        if symbols and len([symbol for symbol in symbols if symbol.strip()]) > batch_size:
            raise ValueError("explicit symbols cannot exceed batch_size")
        start = _parse_date(start_date) if start_date else None
        end = _parse_date(end_date) if end_date else None
        if start_date and start is None:
            raise ValueError("start_date must use YYYY-MM-DD")
        if end_date and end is None:
            raise ValueError("end_date must use YYYY-MM-DD")
        if start and end and start > end:
            raise ValueError("start_date must not be after end_date")

        started = datetime.now(UTC)
        catalogue = self.list_tradable_equities()
        catalogue_count = len(catalogue)
        if symbols:
            requested = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
            catalogue = [
                row for row in catalogue
                if row["symbol"].strip().upper() in requested
            ]
            missing = sorted(requested - {row["symbol"].strip().upper() for row in catalogue})
        else:
            missing = []
            catalogue = catalogue[offset : offset + batch_size]

        fetched_rows = inserted = updated = rejected = duplicate_rows = 0
        successful: list[str] = []
        errors: list[dict[str, str]] = []

        with session_scope(self._session) as session:
            for meta in catalogue:
                symbol = meta["symbol"].strip().upper()
                file_symbol = symbol.replace("/", "-")
                try:
                    rows = self._get_csv(f"data/nepse/{file_symbol}.csv")
                except httpx.HTTPError as exc:
                    errors.append({"symbol": symbol, "error": f"download_failed: {exc}"})
                    continue

                stock: Stock | None = None
                if not dry_run:
                    stock = session.scalar(sa.select(Stock).where(Stock.symbol == symbol))
                    if stock is None:
                        stock = Stock(symbol=symbol)
                        session.add(stock)
                        session.flush()
                    stock.name = meta.get("name") or stock.name
                    stock.sector = meta.get("sector") or stock.sector
                    stock.instrument_type = "ordinary"
                    stock.is_active = True

                parsed: dict[date, dict[str, Any]] = {}
                for row in rows:
                    row_date = _parse_date(row.get("published_date"))
                    if row_date is None:
                        rejected += 1
                        continue
                    if (start and row_date < start) or (end and row_date > end):
                        continue
                    fetched_rows += 1
                    o = _decimal(row.get("open"))
                    h = _decimal(row.get("high"))
                    low = _decimal(row.get("low"))
                    close = _decimal(row.get("close"))
                    volume = _int_value(row.get("traded_quantity")) or 0
                    ohlc_values = [value for value in (o, h, low, close) if value is not None]
                    valid = (
                        len(ohlc_values) == 4
                        and all(value > 0 for value in ohlc_values)
                        and h is not None
                        and low is not None
                        and h >= max(ohlc_values)
                        and low <= min(ohlc_values)
                        and volume >= 0
                    )
                    if not valid:
                        rejected += 1
                        continue
                    if row_date in parsed:
                        duplicate_rows += 1
                    parsed[row_date] = {
                        "open": o,
                        "high": h,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }

                if dry_run:
                    successful.append(symbol)
                    continue

                if stock is None:
                    raise RuntimeError(f"Stock metadata was not initialized for {symbol}")
                if parsed:
                    existing_rows = session.scalars(
                        sa.select(Price).where(
                            Price.stock_id == stock.id,
                            Price.date.in_(list(parsed)),
                        )
                    ).all()
                    existing = {item.date: item for item in existing_rows}
                    for row_date, values in parsed.items():
                        current = existing.get(row_date)
                        if current is None:
                            session.add(
                                Price(
                                    stock_id=stock.id,
                                    date=row_date,
                                    **values,
                                )
                            )
                            inserted += 1
                        else:
                            for field, value in values.items():
                                setattr(current, field, value)
                            updated += 1
                successful.append(symbol)

            if not dry_run:
                session.add(
                    IngestionLog(
                        source="nepal-market-data",
                        source_id=None,
                        status="failed" if errors else "success",
                        records_fetched=fetched_rows,
                        records_inserted=inserted,
                        records_rejected=rejected,
                        duration_seconds=(datetime.now(UTC) - started).total_seconds(),
                        errors=str(errors[:20]) if errors else None,
                        started_at=started,
                        completed_at=datetime.now(UTC),
                    )
                )
                session.commit()

        next_offset = None
        if not symbols and offset + len(catalogue) < catalogue_count:
            next_offset = offset + len(catalogue)

        return {
            "source": "nepal-market-data",
            "attribution": DATASET_ATTRIBUTION,
            "corporate_action_adjusted": False,
            "catalogue_tradable_equities": catalogue_count,
            "batch_offset": offset,
            "batch_size": batch_size,
            "symbols_requested": len(catalogue),
            "symbols_succeeded": len(successful),
            "symbols": successful,
            "symbols_missing_from_catalogue": missing,
            "rows_fetched": fetched_rows,
            "rows_inserted": 0 if dry_run else inserted,
            "rows_updated": 0 if dry_run else updated,
            "rows_rejected": rejected,
            "duplicate_dates_in_source": duplicate_rows,
            "errors": errors,
            "dry_run": dry_run,
            "next_offset": next_offset,
        }

    def sync_nepse_index(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Import NEPSE Index history as a benchmark series, never as a tradable stock."""
        start = _parse_date(start_date) if start_date else None
        end = _parse_date(end_date) if end_date else None
        if start_date and start is None:
            raise ValueError("start_date must use YYYY-MM-DD")
        if end_date and end is None:
            raise ValueError("end_date must use YYYY-MM-DD")
        rows = self._get_csv(INDEX_PATH)
        parsed: dict[date, dict[str, Any]] = {}
        rejected = duplicates = 0
        for row in rows:
            row_date = _parse_date(row.get("published_date"))
            close = _decimal(row.get("close"))
            if row_date is None or close is None or close <= 0:
                rejected += 1
                continue
            if (start and row_date < start) or (end and row_date > end):
                continue
            values: dict[str, float | int] = {
                "open": float(_decimal(row.get("open")) or close),
                "high": float(_decimal(row.get("high")) or close),
                "low": float(_decimal(row.get("low")) or close),
                "close": float(close),
                "volume": int(_int_value(row.get("traded_quantity")) or 0),
            }
            if values["high"] < max(values["open"], values["close"]):
                rejected += 1
                continue
            if values["low"] > min(values["open"], values["close"]):
                rejected += 1
                continue
            if row_date in parsed:
                duplicates += 1
            parsed[row_date] = values

        inserted = updated = 0
        if not dry_run:
            with session_scope(self._session) as session:
                existing_rows = session.scalars(
                    sa.select(NEPSEIndex).where(NEPSEIndex.date.in_(list(parsed)))
                ).all() if parsed else []
                existing = {item.date: item for item in existing_rows}
                for row_date, values in parsed.items():
                    current = existing.get(row_date)
                    if current is None:
                        session.add(NEPSEIndex(date=row_date, **values))
                        inserted += 1
                    else:
                        for field, value in values.items():
                            setattr(current, field, value)
                        updated += 1
                session.add(
                    IngestionLog(
                        source="nepal-market-data-index",
                        source_id=None,
                        status="success",
                        records_fetched=len(rows),
                        records_inserted=inserted,
                        records_rejected=rejected,
                        duration_seconds=0.0,
                        errors=None,
                        started_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC),
                    )
                )
                session.commit()

        return {
            "source": "nepal-market-data",
            "series": "NEPSE_INDEX",
            "role": "benchmark_only",
            "attribution": DATASET_ATTRIBUTION,
            "records_fetched": len(rows),
            "valid_sessions": len(parsed),
            "inserted": 0 if dry_run else inserted,
            "updated": 0 if dry_run else updated,
            "rejected": rejected,
            "duplicate_dates_in_source": duplicates,
            "dry_run": dry_run,
        }
