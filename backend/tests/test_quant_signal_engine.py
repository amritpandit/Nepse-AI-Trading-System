from __future__ import annotations

import pytest

from app.services.quant_signal_engine import score_ohlcv


def _candles(count: int = 60, trend: float = 1.0) -> list[dict[str, float]]:
    rows = []
    price = 200.0
    for index in range(count):
        previous = price
        price += trend * (0.8 + (index % 3) * 0.1)
        rows.append(
            {
                "open": previous,
                "high": max(previous, price) + 1.0,
                "low": min(previous, price) - 1.0,
                "close": price,
                "volume": 100000.0 + index * 1000.0,
            }
        )
    return rows


def test_quant_score_is_bounded_and_explainable() -> None:
    result = score_ohlcv(_candles())
    assert 0.0 <= result["buy_score"] <= 100.0
    assert 0.0 <= result["sell_score"] <= 100.0
    assert result["buy_score"] + result["sell_score"] == pytest.approx(100.0)
    assert set(result["components"]) == {
        "trend",
        "momentum",
        "smc_structure",
        "volume_confirmation",
        "volatility_risk",
        "candle_price_action",
    }
    assert result["advisory_only"] is True
    assert result["execution_enabled"] is False


def test_low_data_trust_suppresses_directional_signal() -> None:
    result = score_ohlcv(_candles(), data_trust_score=0.42)
    assert result["signal"] == "HOLD"
    assert result["status"] == "DATA_QUALITY_BLOCK"


def test_high_data_trust_allows_scoring() -> None:
    result = score_ohlcv(_candles(), data_trust_score=0.95)
    assert result["status"] == "ADVISORY"
    assert result["data_quality"]["passed"] is True


def test_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="At least 35"):
        score_ohlcv(_candles(20))


def test_rejects_invalid_ohlc() -> None:
    candles = _candles()
    candles[-1]["high"] = 1.0
    with pytest.raises(ValueError, match="invalid high"):
        score_ohlcv(candles)


def test_rejects_invalid_trust_score() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_ohlcv(_candles(), data_trust_score=1.5)
