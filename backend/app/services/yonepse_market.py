"""Read-only market snapshot client for the YONEPSE public JSON feed.

YONEPSE is an unofficial, third-party data aggregator. The service keeps the
provider attribution, source timestamps, and freshness assessment with every
snapshot. It does not place orders or write provider data into the database.
"""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings


NPT = ZoneInfo("Asia/Kathmandu")
UTC = timezone.utc

SOURCE_NAME = "YONEPSE"
SOURCE_URL = "https://github.com/Shubhamnpk/yonepse"
ATTRIBUTION = (
    "Market data aggregated by YONEPSE from NEPSE and other credited sources. "
    "Unofficial third-party data; verify important prices independently."
)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # YONEPSE timestamps without offsets are treated as Nepal local time.
        parsed = parsed.replace(tzinfo=NPT)
    return parsed.astimezone(UTC)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


class MarketDataUnavailable(RuntimeError):
    """Raised when the external market feed is disabled or cannot be read."""


class YonepseMarketDataService:
    """Fetch and normalize YONEPSE market JSON with a small process-local cache."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        cache_seconds: int | None = None,
        stale_after_minutes: int | None = None,
        delayed_after_minutes: int = 1440,
        market_status_max_age_minutes: int | None = None,
        market_closed_max_age_hours: int | None = None,
        enabled: bool | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.yonepse_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.yonepse_timeout_seconds
        self.cache_seconds = (
            settings.yonepse_cache_seconds if cache_seconds is None else cache_seconds
        )
        self.stale_after_minutes = (
            settings.yonepse_stale_after_minutes
            if stale_after_minutes is None
            else stale_after_minutes
        )
        self.delayed_after_minutes = delayed_after_minutes
        self.market_status_max_age_minutes = (
            settings.yonepse_market_status_max_age_minutes
            if market_status_max_age_minutes is None
            else market_status_max_age_minutes
        )
        self.market_closed_max_age_hours = (
            settings.yonepse_market_closed_max_age_hours
            if market_closed_max_age_hours is None
            else market_closed_max_age_hours
        )
        self.enabled = settings.yonepse_enabled if enabled is None else enabled
        self._client = client
        self._cache: dict[str, Any] | None = None
        self._cache_until = 0.0
        self._lock = threading.Lock()

    def _get_json(self, client: httpx.Client, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MarketDataUnavailable(f"Could not retrieve market data from {path}") from exc

    def _classify_freshness(
        self,
        timestamp: datetime | None,
        *,
        now: datetime,
        market_is_open: bool | None,
        status_checked_at: datetime | None,
    ) -> tuple[str, float | None, str]:
        """Classify one timestamp without mistaking a normal market closure for a failure."""
        if timestamp is None:
            return "unknown", None, "The provider did not supply a valid price timestamp."

        now_npt = now.astimezone(NPT)
        timestamp_npt = timestamp.astimezone(NPT)
        age_minutes = max(
            0.0,
            (now_npt.astimezone(UTC) - timestamp_npt.astimezone(UTC)).total_seconds()
            / 60,
        )

        status_age_minutes = None
        if status_checked_at is not None:
            status_age_minutes = max(
                0.0,
                (now_npt.astimezone(UTC) - status_checked_at.astimezone(UTC)).total_seconds()
                / 60,
            )

        status_confirms_closed = (
            market_is_open is False
            and status_age_minutes is not None
            and status_age_minutes <= self.market_status_max_age_minutes
        )
        weekend_closure = (
            market_is_open is False
            and now_npt.weekday() in (5, 6)
            and timestamp_npt.weekday() < 5
        )
        closed_market_max_age = self.market_closed_max_age_hours * 60

        if (
            age_minutes <= closed_market_max_age
            and (status_confirms_closed or weekend_closure)
        ):
            return (
                "market_closed",
                round(age_minutes, 1),
                "The exchange is closed; this is the latest available session price.",
            )
        if age_minutes <= self.stale_after_minutes:
            return (
                "fresh",
                round(age_minutes, 1),
                "Price timestamp is within the fresh-data window.",
            )
        if age_minutes <= self.delayed_after_minutes:
            return (
                "delayed",
                round(age_minutes, 1),
                "Price timestamp is older than the fresh-data window.",
            )
        return (
            "stale",
            round(age_minutes, 1),
            "Price timestamp exceeds the permitted data-age window.",
        )

    def get_snapshot(
        self, *, now: datetime | None = None, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Return latest normalized market data and an explicit freshness label."""
        if not self.enabled:
            raise MarketDataUnavailable("YONEPSE market feed is disabled")

        # Use the cache only for normal requests; tests and operators can force a refresh.
        with self._lock:
            if (
                now is None
                and not force_refresh
                and self._cache is not None
                and time.monotonic() < self._cache_until
            ):
                return copy.deepcopy(self._cache)

            if self._client is not None:
                raw = self._fetch_all(self._client)
            else:
                try:
                    with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                        raw = self._fetch_all(client)
                except httpx.HTTPError as exc:
                    raise MarketDataUnavailable("YONEPSE market feed is unavailable") from exc

            snapshot = self._normalize(raw, now=now or datetime.now(NPT))
            self._cache = copy.deepcopy(snapshot)
            self._cache_until = time.monotonic() + max(0, self.cache_seconds)
            return snapshot

    def _fetch_all(self, client: httpx.Client) -> dict[str, Any]:
        return {
            "stocks": self._get_json(client, "data/nepse_data.json"),
            "indices": self._get_json(client, "data/market/indices.json"),
            "status": self._get_json(client, "data/market/status.json"),
        }

    def _normalize(self, raw: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        raw_stocks = raw.get("stocks")
        raw_indices = raw.get("indices")
        raw_status = raw.get("status")

        if not isinstance(raw_stocks, list) or not isinstance(raw_indices, list):
            raise MarketDataUnavailable("YONEPSE returned an unexpected market-data format")

        if now.tzinfo is None:
            now = now.replace(tzinfo=NPT)
        now_npt = now.astimezone(NPT)

        market_status = raw_status if isinstance(raw_status, dict) else {}
        market_is_open = (
            market_status.get("is_open")
            if isinstance(market_status.get("is_open"), bool)
            else None
        )
        status_checked_at = _parse_datetime(market_status.get("last_checked"))
        status_age_minutes = None
        if status_checked_at is not None:
            status_age_minutes = max(
                0.0,
                (now_npt.astimezone(UTC) - status_checked_at).total_seconds() / 60,
            )

        market_status_data = {
            "is_open": market_is_open,
            "last_checked": market_status.get("last_checked"),
            "last_checked_at": (
                status_checked_at.astimezone(NPT).isoformat()
                if status_checked_at is not None
                else None
            ),
            "age_minutes": (
                round(status_age_minutes, 1) if status_age_minutes is not None else None
            ),
            "freshness": (
                "fresh"
                if status_age_minutes is not None
                and status_age_minutes <= self.market_status_max_age_minutes
                else "stale" if status_checked_at is not None else "unknown"
            ),
        }

        stocks: list[dict[str, Any]] = []
        timestamps: list[datetime] = []
        for item in raw_stocks:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol or item.get("asset_type") == "open_ended_mutual_fund":
                continue
            last_updated = item.get("last_updated")
            parsed_time = _parse_datetime(last_updated)
            row = {
                "symbol": symbol,
                "name": str(item.get("name") or symbol),
                "ltp": _number(item.get("ltp")),
                "previous_close": _number(item.get("previous_close")),
                "change": _number(item.get("change")),
                "percent_change": _number(item.get("percent_change")),
                "high": _number(item.get("high")),
                "low": _number(item.get("low")),
                "volume": int(_number(item.get("volume")) or 0),
                "turnover": _number(item.get("turnover")) or 0.0,
                "trades": int(_number(item.get("trades")) or 0),
                "market_cap": _number(item.get("market_cap")),
                "last_updated": last_updated,
                "last_updated_at": (
                    parsed_time.astimezone(NPT).isoformat() if parsed_time is not None else None
                ),
            }
            row_freshness, row_age, row_reason = self._classify_freshness(
                parsed_time,
                now=now_npt,
                market_is_open=market_is_open,
                status_checked_at=status_checked_at,
            )
            row["freshness"] = row_freshness
            row["age_minutes"] = row_age
            row["freshness_reason"] = row_reason
            stocks.append(row)
            if parsed_time is not None:
                timestamps.append(parsed_time)

        nepse_index = next(
            (
                item
                for item in raw_indices
                if isinstance(item, dict)
                and str(item.get("index", "")).strip().lower() == "nepse index"
            ),
            None,
        )
        index_data: dict[str, Any] | None = None
        if nepse_index is not None:
            generated_time = nepse_index.get("generatedTime")
            parsed_index_time = _parse_datetime(generated_time)
            index_data = {
                "name": "NEPSE Index",
                "value": _number(nepse_index.get("currentValue"))
                or _number(nepse_index.get("close")),
                "change": _number(nepse_index.get("change")),
                "percent_change": _number(nepse_index.get("perChange")),
                "high": _number(nepse_index.get("high")),
                "low": _number(nepse_index.get("low")),
                "generated_time": generated_time,
                "generated_time_at": (
                    parsed_index_time.astimezone(NPT).isoformat()
                    if parsed_index_time is not None
                    else None
                ),
            }
            if parsed_index_time is not None:
                timestamps.append(parsed_index_time)

        latest_time = max(timestamps) if timestamps else None
        freshness, age_minutes, freshness_reason = self._classify_freshness(
            latest_time,
            now=now_npt,
            market_is_open=market_is_open,
            status_checked_at=status_checked_at,
        )

        total_volume = sum(row["volume"] for row in stocks)
        total_turnover = sum(row["turnover"] for row in stocks)

        return {
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "attribution": ATTRIBUTION,
            "fetched_at": now_npt.isoformat(),
            "as_of": latest_time.astimezone(NPT).isoformat() if latest_time else None,
            "age_minutes": age_minutes,
            "freshness": freshness,
            "freshness_reason": freshness_reason,
            "market_status": market_status_data,
            "index": index_data,
            "summary": {
                "stock_count": len(stocks),
                "total_volume": total_volume,
                "total_turnover": round(total_turnover, 2),
            },
            "stocks": stocks,
        }


yonepse_market_data = YonepseMarketDataService()
