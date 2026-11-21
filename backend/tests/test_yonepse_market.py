from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.services.yonepse_market import MarketDataUnavailable, YonepseMarketDataService

NPT = ZoneInfo("Asia/Kathmandu")


def _mock_client() -> httpx.Client:
    payloads = {
        "/data/nepse_data.json": [
            {
                "symbol": "NABIL",
                "name": "Nabil Bank Limited",
                "ltp": 500.0,
                "previous_close": 495.0,
                "change": 5.0,
                "percent_change": 1.01,
                "high": 501.0,
                "low": 496.0,
                "volume": 1200,
                "turnover": 600000.0,
                "trades": 45,
                "market_cap": 50000.0,
                "last_updated": "2026-11-21T09:51:00",
            },
            {
                "symbol": "TESTMF",
                "name": "Test Open Ended Fund",
                "asset_type": "open_ended_mutual_fund",
                "ltp": 10.2,
                "volume": 100,
                "last_updated": "2026-11-21T09:50:00",
            },
        ],
        "/data/market/indices.json": [
            {
                "index": "NEPSE Index",
                "currentValue": 2700.5,
                "change": 12.5,
                "perChange": 0.46,
                "high": 2705.0,
                "low": 2680.0,
                "generatedTime": "2026-11-21T09:50:00",
            }
        ],
        "/data/market/status.json": {
            "is_open": True,
            "last_checked": "2026-11-21T09:52:00",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = payloads.get(request.url.path)
        if payload is None:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=payload, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_market_snapshot_normalizes_data_and_attributes_source() -> None:
    service = YonepseMarketDataService(
        base_url="https://feed.test",
        enabled=True,
        cache_seconds=0,
        client=_mock_client(),
    )
    snapshot = service.get_snapshot(
        now=datetime(2026, 11, 21, 10, 0, tzinfo=NPT)
    )

    assert snapshot["source"] == "YONEPSE"
    assert "github.com/Shubhamnpk/yonepse" in snapshot["source_url"]
    assert snapshot["freshness"] == "fresh"
    assert snapshot["index"]["value"] == 2700.5
    assert snapshot["index"]["percent_change"] == 0.46
    assert snapshot["market_status"]["is_open"] is True
    assert snapshot["summary"]["stock_count"] == 1
    assert snapshot["summary"]["total_volume"] == 1200
    assert [row["symbol"] for row in snapshot["stocks"]] == ["NABIL"]


def test_market_snapshot_marks_old_data_as_stale() -> None:
    service = YonepseMarketDataService(
        base_url="https://feed.test",
        enabled=True,
        stale_after_minutes=90,
        cache_seconds=0,
        client=_mock_client(),
    )
    snapshot = service.get_snapshot(
        now=datetime(2026, 11, 23, 10, 0, tzinfo=NPT)
    )

    assert snapshot["freshness"] == "stale"
    assert snapshot["age_minutes"] > 24 * 60


def test_market_feed_rejects_unavailable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    service = YonepseMarketDataService(
        base_url="https://feed.test",
        enabled=True,
        cache_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(MarketDataUnavailable):
        service.get_snapshot()
