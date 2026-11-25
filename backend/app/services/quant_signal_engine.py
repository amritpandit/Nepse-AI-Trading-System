"""Explainable NEPSE quantitative signal scoring.

This module is deterministic, dependency-light and advisory-only. SMC concepts are
rule-based approximations over OHLCV candles; they are not claims about institutional
order flow or hidden liquidity.
"""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import mean, pstdev
from typing import Any


def _ema(values: list[float], period: int) -> list[float]:
    alpha = 2.0 / (period + 1.0)
    result: list[float] = []
    current = values[0]
    for value in values:
        current = alpha * value + (1.0 - alpha) * current
        result.append(current)
    return result


def _rsi(values: list[float], period: int = 14) -> float:
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    recent = changes[-period:]
    gains = [max(change, 0.0) for change in recent]
    losses = [max(-change, 0.0) for change in recent]
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _validate_candles(candles: list[dict[str, Any]]) -> list[dict[str, float]]:
    if len(candles) < 35:
        raise ValueError("At least 35 valid OHLCV candles are required.")
    normalized: list[dict[str, float]] = []
    for index, candle in enumerate(candles):
        row = {key: float(candle[key]) for key in ("open", "high", "low", "close", "volume")}
        if not all(isfinite(value) for value in row.values()):
            raise ValueError(f"Candle {index} contains a non-finite value.")
        if min(row["open"], row["high"], row["low"], row["close"]) <= 0:
            raise ValueError(f"Candle {index} contains a non-positive price.")
        if row["volume"] < 0:
            raise ValueError(f"Candle {index} contains negative volume.")
        if row["high"] < max(row["open"], row["close"], row["low"]):
            raise ValueError(f"Candle {index} has an invalid high.")
        if row["low"] > min(row["open"], row["close"], row["high"]):
            raise ValueError(f"Candle {index} has an invalid low.")
        normalized.append(row)
    return normalized


def _swing_points(values: list[float], left: int = 2, right: int = 2) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for index in range(left, len(values) - right):
        window = values[index - left : index + right + 1]
        if values[index] == max(window) and window.count(values[index]) == 1:
            points.append((index, values[index]))
    return points


def _smc_structure(
    highs: list[float], lows: list[float], closes: list[float]
) -> dict[str, Any]:
    swing_highs = _swing_points(highs)
    swing_lows = _swing_points(lows)
    latest_close = closes[-1]
    prior_high = swing_highs[-1][1] if swing_highs else None
    prior_low = swing_lows[-1][1] if swing_lows else None

    bullish_bos = prior_high is not None and latest_close > prior_high
    bearish_bos = prior_low is not None and latest_close < prior_low

    bullish_fvg = len(highs) >= 3 and lows[-1] > highs[-3]
    bearish_fvg = len(lows) >= 3 and highs[-1] < lows[-3]

    # CHoCH is flagged when price closes through the latest opposing confirmed
    # swing after the most recent opposite-side structure. This is a simplified
    # swing-break heuristic, not a full discretionary market-structure engine.
    bullish_choch = bool(
        bullish_bos and swing_lows and swing_highs
        and swing_lows[-1][0] > swing_highs[-1][0]
    )
    bearish_choch = bool(
        bearish_bos and swing_highs and swing_lows
        and swing_highs[-1][0] > swing_lows[-1][0]
    )

    direction = 0.0
    if bullish_bos:
        direction += 0.65
    if bearish_bos:
        direction -= 0.65
    if bullish_choch:
        direction += 0.25
    if bearish_choch:
        direction -= 0.25
    if bullish_fvg:
        direction += 0.10
    if bearish_fvg:
        direction -= 0.10

    return {
        "direction": max(-1.0, min(1.0, direction)),
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "bullish_choch": bullish_choch,
        "bearish_choch": bearish_choch,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "last_swing_high": prior_high,
        "last_swing_low": prior_low,
        "methodology": "confirmed 2-left/2-right swing-break heuristic",
    }


def score_ohlcv(
    candles: list[dict[str, Any]],
    *,
    data_trust_score: float | None = None,
    min_data_trust: float = 0.70,
) -> dict[str, Any]:
    """Return transparent buy/sell scores and ATR-based advisory levels."""
    rows = _validate_candles(candles)
    if data_trust_score is not None and not 0.0 <= data_trust_score <= 1.0:
        raise ValueError("data_trust_score must be between 0 and 1.")

    opens = [row["open"] for row in rows]
    highs = [row["high"] for row in rows]
    lows = [row["low"] for row in rows]
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]

    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    ema26 = _ema(closes, 26)
    ema12 = _ema(closes, 12)
    macd_line = [fast - slow for fast, slow in zip(ema12, ema26)]
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line[-1] - macd_signal[-1]
    rsi = _rsi(closes)
    latest_close = closes[-1]

    typical = closes[-20:]
    bb_mid = mean(typical)
    bb_std = pstdev(typical)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    true_ranges = []
    for index in range(1, len(rows)):
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    atr = mean(true_ranges[-14:])
    atr_pct = atr / latest_close if latest_close else 0.0

    avg_volume = mean(volumes[-20:])
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0.0

    # Each component is normalized to [-1, +1], where positive favours BUY.
    trend = 1.0 if ema9[-1] > ema21[-1] else -1.0
    trend += 0.35 if latest_close > ema21[-1] else -0.35
    trend = max(-1.0, min(1.0, trend))

    momentum = max(-1.0, min(1.0, (50.0 - rsi) / 25.0 * -1.0))
    if macd_hist > 0:
        momentum = min(1.0, momentum + 0.35)
    elif macd_hist < 0:
        momentum = max(-1.0, momentum - 0.35)

    structure = _smc_structure(highs, lows, closes)
    structure_score = structure["direction"]

    volume_direction = 1.0 if closes[-1] > opens[-1] else -1.0 if closes[-1] < opens[-1] else 0.0
    volume_score = volume_direction * max(-1.0, min(1.0, (volume_ratio - 0.8) / 1.2))

    # Penalize extreme volatility; do not force a directional bias from volatility.
    volatility_score = 0.0 if atr_pct <= 0.06 else -min(1.0, (atr_pct - 0.06) / 0.06)

    candle_range = highs[-1] - lows[-1]
    candle_score = (
        (closes[-1] - opens[-1]) / candle_range if candle_range > 0 else 0.0
    )
    candle_score = max(-1.0, min(1.0, candle_score))

    components = {
        "trend": {"weight": 0.20, "score": trend},
        "momentum": {"weight": 0.20, "score": momentum},
        "smc_structure": {"weight": 0.25, "score": structure_score},
        "volume_confirmation": {"weight": 0.15, "score": volume_score},
        "volatility_risk": {"weight": 0.10, "score": volatility_score},
        "candle_price_action": {"weight": 0.10, "score": candle_score},
    }
    composite = sum(item["weight"] * item["score"] for item in components.values())
    buy_score = round(50.0 + 50.0 * composite, 2)
    sell_score = round(50.0 - 50.0 * composite, 2)

    if data_trust_score is not None and data_trust_score < min_data_trust:
        signal = "HOLD"
        status = "DATA_QUALITY_BLOCK"
        reason = "Data trust is below the minimum threshold; directional signals are suppressed."
    elif buy_score >= 65.0:
        signal = "BUY"
        status = "ADVISORY"
        reason = "Weighted bullish evidence exceeds the advisory threshold."
    elif sell_score >= 65.0:
        signal = "SELL"
        status = "ADVISORY"
        reason = "Weighted bearish evidence exceeds the advisory threshold."
    else:
        signal = "HOLD"
        status = "ADVISORY"
        reason = "Evidence is mixed or insufficient for a directional advisory."

    stop_distance = 1.5 * atr
    target_distance = 3.0 * atr
    levels = {
        "atr": round(atr, 4),
        "atr_percent": round(atr_pct * 100.0, 2),
        "reference_close": round(latest_close, 4),
        "buy_stop_loss": round(latest_close - stop_distance, 4),
        "buy_take_profit_2r": round(latest_close + target_distance, 4),
        "sell_stop_loss": round(latest_close + stop_distance, 4),
        "sell_take_profit_2r": round(latest_close - target_distance, 4),
        "risk_reward_ratio": 2.0,
        "note": "Illustrative ATR levels; validate against price limits, liquidity and support/resistance.",
    }

    return {
        "signal": signal,
        "status": status,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "confidence_alignment": round(abs(buy_score - sell_score), 2),
        "reason": reason,
        "components": components,
        "indicators": {
            "rsi_14": round(rsi, 2),
            "ema_9": round(ema9[-1], 4),
            "ema_21": round(ema21[-1], 4),
            "macd": round(macd_line[-1], 6),
            "macd_signal": round(macd_signal[-1], 6),
            "macd_histogram": round(macd_hist, 6),
            "bollinger_middle": round(bb_mid, 4),
            "bollinger_upper": round(bb_upper, 4),
            "bollinger_lower": round(bb_lower, 4),
            "volume_ratio_20": round(volume_ratio, 3),
        },
        "market_structure": structure,
        "risk_levels": levels,
        "data_quality": {
            "trust_score": data_trust_score,
            "minimum_trust_score": min_data_trust,
            "passed": data_trust_score is None or data_trust_score >= min_data_trust,
            "bars_used": len(rows),
        },
        "methodology_version": "nepse-quant-score-v1",
        "advisory_only": True,
        "execution_enabled": False,
    }
