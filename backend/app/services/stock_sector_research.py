"""Sector-relative and stock-level technical research (advisory only)."""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any

import numpy as np
import sqlalchemy as sa
from app.models.benchmark import NEPSEIndex
from app.models.price import Price
from app.models.stock import Stock
from sqlalchemy.orm import Session

# Price gaps beyond the normal daily band are treated as possible corporate
# actions or source errors. Raw source prices are not silently adjusted.
DISCONTINUITY_THRESHOLD = 0.15
RESEARCH_SOURCE_WARNING = (
    "Historical prices are raw and not corporate-action adjusted. "
    "Price discontinuities above 15% are flagged; flagged histories are "
    "not used for BUY triggers until sufficient post-event observations exist."
)
TRADING_COST_RATE = 0.005  # illustrative one-way transaction fee assumption
SLIPPAGE_BPS = 5.0


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size == 0:
        return result
    alpha = 2.0 / (span + 1.0)
    result[0] = values[0]
    for index in range(1, values.size):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _sma(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size < window:
        return result
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size <= period:
        return result
    changes = np.diff(values)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    for index in range(period, values.size):
        gain = avg_gain[index - 1]
        loss = avg_loss[index - 1]
        if loss == 0:
            result[index] = 100.0 if gain > 0 else 50.0
        else:
            result[index] = 100.0 - (100.0 / (1.0 + gain / loss))
    return result


def _valid_history(rows: list[Price]) -> list[Price]:
    return [
        row
        for row in rows
        if row.close is not None
        and float(row.close) > 0
        and row.open is not None
        and row.high is not None
        and row.low is not None
    ]


def _clean_after_latest_discontinuity(rows: list[Price]) -> tuple[list[Price], dict[str, Any]]:
    if len(rows) < 2:
        return rows, {"flagged": False, "event_date": None}
    event_index: int | None = None
    for index in range(1, len(rows)):
        previous = float(rows[index - 1].close)
        current = float(rows[index].close)
        if previous > 0 and abs(current / previous - 1.0) > DISCONTINUITY_THRESHOLD:
            event_index = index
    if event_index is None:
        return rows, {"flagged": False, "event_date": None}
    return rows[event_index:], {
        "flagged": True,
        "event_date": rows[event_index].date.isoformat(),
        "reason": "Possible corporate action or abnormal source price jump",
    }


def _indicator_snapshot(rows: list[Price]) -> dict[str, Any]:
    close = np.asarray([float(row.close) for row in rows], dtype=float)
    high = np.asarray([float(row.high) for row in rows], dtype=float)
    low = np.asarray([float(row.low) for row in rows], dtype=float)
    volume = np.asarray([float(row.volume or 0) for row in rows], dtype=float)

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    rsi = _rsi(close, 14)

    true_range = np.full(close.shape, np.nan, dtype=float)
    if close.size:
        true_range[0] = high[0] - low[0]
    if close.size > 1:
        true_range[1:] = np.maximum.reduce(
            [high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])]
        )
    atr14 = _sma(true_range, 14)

    avg_volume20 = _sma(volume, 20)
    volume_ratio = (
        float(volume[-1] / avg_volume20[-1])
        if avg_volume20.size and avg_volume20[-1] > 0
        else None
    )
    previous_high20 = float(np.max(high[-21:-1])) if close.size >= 21 else None
    previous_low20 = float(np.min(low[-21:-1])) if close.size >= 21 else None
    current_hist = float(macd[-1] - macd_signal[-1]) if np.isfinite(macd[-1]) and np.isfinite(macd_signal[-1]) else None
    previous_hist = float(macd[-2] - macd_signal[-2]) if close.size >= 2 and np.isfinite(macd[-2]) and np.isfinite(macd_signal[-2]) else None
    average_turnover20 = _sma(close * volume, 20)
    daily_return = float(close[-1] / close[-2] - 1.0) if close.size >= 2 and close[-2] > 0 else None
    bollinger_mid = float(sma20[-1]) if np.isfinite(sma20[-1]) else None
    bollinger_std = float(np.std(close[-20:], ddof=0)) if close.size >= 20 else None
    previous_open = float(rows[-2].open) if len(rows) >= 2 else None
    previous_close = float(close[-2]) if close.size >= 2 else None
    current_open = float(rows[-1].open)
    current_high = float(high[-1])
    current_low = float(low[-1])
    current_close = float(close[-1])
    previous_bearish = previous_open is not None and previous_close is not None and previous_close < previous_open
    previous_bullish = previous_open is not None and previous_close is not None and previous_close > previous_open
    current_bullish = current_close > current_open
    current_bearish = current_close < current_open
    bullish_engulfing = (
        previous_bearish and current_bullish
        and current_open <= float(previous_close)
        and current_close >= float(previous_open)
    )
    bearish_engulfing = (
        previous_bullish and current_bearish
        and current_open >= float(previous_close)
        and current_close <= float(previous_open)
    )
    candle_body = abs(current_close - current_open)
    lower_wick = min(current_open, current_close) - current_low
    upper_wick = current_high - max(current_open, current_close)
    hammer = candle_body > 0 and lower_wick >= 2 * candle_body and upper_wick <= candle_body
    shooting_star = candle_body > 0 and upper_wick >= 2 * candle_body and lower_wick <= candle_body
    return {
        "close": float(close[-1]),
        "sma_20": float(sma20[-1]) if np.isfinite(sma20[-1]) else None,
        "previous_sma_20": float(sma20[-2]) if sma20.size >= 2 and np.isfinite(sma20[-2]) else None,
        "previous_close": float(close[-2]) if close.size >= 2 else None,
        "sma_50": float(sma50[-1]) if np.isfinite(sma50[-1]) else None,
        "sma_200": float(sma200[-1]) if np.isfinite(sma200[-1]) else None,
        "rsi_14": float(rsi[-1]) if np.isfinite(rsi[-1]) else None,
        "macd": float(macd[-1]) if np.isfinite(macd[-1]) else None,
        "macd_signal": float(macd_signal[-1]) if np.isfinite(macd_signal[-1]) else None,
        "macd_histogram": current_hist,
        "previous_macd_histogram": previous_hist,
        "atr_14": float(atr14[-1]) if np.isfinite(atr14[-1]) else None,
        "bollinger_middle": bollinger_mid,
        "bollinger_upper": bollinger_mid + 2 * bollinger_std if bollinger_mid is not None and bollinger_std is not None else None,
        "bollinger_lower": bollinger_mid - 2 * bollinger_std if bollinger_mid is not None and bollinger_std is not None else None,
        "average_traded_value_20d": float(average_turnover20[-1]) if average_turnover20.size and np.isfinite(average_turnover20[-1]) else None,
        "daily_return": daily_return,
        "breakout_20d": previous_high20 is not None and close[-1] > previous_high20,
        "breakdown_20d": previous_low20 is not None and close[-1] < previous_low20,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "volume_ratio_20d": volume_ratio,
        "previous_high_20d": previous_high20,
        "previous_low_20d": previous_low20,
        "return_20d": float(close[-1] / close[-21] - 1.0) if close.size >= 21 else None,
        "return_60d": float(close[-1] / close[-61] - 1.0) if close.size >= 61 else None,
        "bars": int(close.size),
    }



def _smc_snapshot(rows: list[Price]) -> dict[str, Any]:
    """Deterministic daily-bar approximations of common SMC concepts.

    These are explicitly rule-defined proxies, not a claim to identify actual
    institutional orders. Swing points are confirmed only after two later bars.
    """
    if len(rows) < 25:
        return {
            "available": False,
            "reason": "At least 25 completed sessions required for SMC structure",
            "directional_points": 0,
        }

    opens = np.asarray([float(row.open) for row in rows], dtype=float)
    highs = np.asarray([float(row.high) for row in rows], dtype=float)
    lows = np.asarray([float(row.low) for row in rows], dtype=float)
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    latest = len(rows) - 1

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for index in range(2, len(rows) - 2):
        if highs[index] >= max(highs[index - 2 : index]) and highs[index] > max(
            highs[index + 1 : index + 3]
        ):
            swing_highs.append((index, float(highs[index])))
        if lows[index] <= min(lows[index - 2 : index]) and lows[index] < min(
            lows[index + 1 : index + 3]
        ):
            swing_lows.append((index, float(lows[index])))

    last_swing_high = swing_highs[-1][1] if swing_highs else None
    last_swing_low = swing_lows[-1][1] if swing_lows else None
    previous_close = float(closes[-2])
    current_close = float(closes[-1])
    bullish_bos = (
        last_swing_high is not None
        and previous_close <= last_swing_high
        and current_close > last_swing_high
    )
    bearish_bos = (
        last_swing_low is not None
        and previous_close >= last_swing_low
        and current_close < last_swing_low
    )

    prior_low = float(np.min(lows[-21:-1]))
    prior_high = float(np.max(highs[-21:-1]))
    candle_mid = (float(highs[-1]) + float(lows[-1])) / 2.0
    bullish_sweep = (
        float(lows[-1]) < prior_low
        and current_close > prior_low
        and current_close >= candle_mid
    )
    bearish_sweep = (
        float(highs[-1]) > prior_high
        and current_close < prior_high
        and current_close <= candle_mid
    )
    bullish_fvg = float(lows[-1]) > float(highs[-3])
    bearish_fvg = float(highs[-1]) < float(lows[-3])

    # A simple demand/supply block proxy: the last opposite-colour candle
    # preceding a structure break. It is contextual, and not scored again
    # unless price retests and rejects that zone.
    bullish_order_block: dict[str, float] | None = None
    bearish_order_block: dict[str, float] | None = None
    if bullish_bos:
        for index in range(latest - 1, max(-1, latest - 12), -1):
            if closes[index] < opens[index]:
                bullish_order_block = {
                    "low": float(lows[index]),
                    "high": float(opens[index]),
                }
                break
    if bearish_bos:
        for index in range(latest - 1, max(-1, latest - 12), -1):
            if closes[index] > opens[index]:
                bearish_order_block = {
                    "low": float(opens[index]),
                    "high": float(highs[index]),
                }
                break

    bullish_ob_retest = (
        bullish_order_block is not None
        and float(lows[-1]) <= bullish_order_block["high"]
        and current_close >= bullish_order_block["low"]
        and current_close >= opens[-1]
    )
    bearish_ob_retest = (
        bearish_order_block is not None
        and float(highs[-1]) >= bearish_order_block["low"]
        and current_close <= bearish_order_block["high"]
        and current_close <= opens[-1]
    )

    bullish_points = (
        (4 if bullish_sweep else 0)
        + (5 if bullish_bos else 0)
        + (3 if bullish_fvg else 0)
        + (3 if bullish_ob_retest else 0)
    )
    bearish_points = (
        (4 if bearish_sweep else 0)
        + (5 if bearish_bos else 0)
        + (3 if bearish_fvg else 0)
        + (3 if bearish_ob_retest else 0)
    )
    return {
        "available": True,
        "confirmed_swing_high": last_swing_high,
        "confirmed_swing_low": last_swing_low,
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "bullish_liquidity_sweep": bullish_sweep,
        "bearish_liquidity_sweep": bearish_sweep,
        "bullish_fair_value_gap_proxy": bullish_fvg,
        "bearish_fair_value_gap_proxy": bearish_fvg,
        "bullish_order_block_proxy": bullish_order_block,
        "bearish_order_block_proxy": bearish_order_block,
        "bullish_order_block_retest": bullish_ob_retest,
        "bearish_order_block_retest": bearish_ob_retest,
        "bullish_points": bullish_points,
        "bearish_points": bearish_points,
        "directional_points": bullish_points - bearish_points,
        "methodology": (
            "Daily OHLC rule-based approximation: two-bar confirmed pivots, "
            "20-session liquidity sweeps, close-through structure breaks, "
            "three-candle gaps and opposite-candle block retests. Not actual order-flow data."
        ),
    }


def _score_signal(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    sector: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create separate 0-100 BUY and SELL evidence scores; not probabilities."""
    buy = 0
    sell = 0
    components: dict[str, dict[str, Any]] = {}

    regime = market.get("regime")
    if regime == "bullish":
        buy += 10
        components["market_regime"] = {"weight": 10, "side": "buy", "reason": "NEPSE trend bullish"}
    elif regime == "bearish":
        sell += 10
        components["market_regime"] = {"weight": 10, "side": "sell", "reason": "NEPSE trend bearish"}
    else:
        components["market_regime"] = {"weight": 10, "side": "neutral", "reason": "No confirmed market trend"}

    relative_strength = sector.get("relative_strength_vs_nepse_20d") if sector else None
    if relative_strength is not None and relative_strength > 0:
        buy += 10
        components["sector_strength"] = {"weight": 10, "side": "buy", "value": relative_strength}
    elif relative_strength is not None and relative_strength < 0:
        sell += 10
        components["sector_strength"] = {"weight": 10, "side": "sell", "value": relative_strength}
    else:
        components["sector_strength"] = {
            "weight": 10,
            "side": "neutral",
            "reason": "Sector relative strength unavailable or flat",
        }

    close = indicators["close"]
    sma20, sma50, sma200 = (
        indicators.get("sma_20"),
        indicators.get("sma_50"),
        indicators.get("sma_200"),
    )
    if sma20 is not None and sma50 is not None and close > sma20 > sma50:
        buy += 20 if sma200 is None or sma50 > sma200 else 15
        components["trend_structure"] = {"weight": 20, "side": "buy", "reason": "Close above SMA20 and SMA20 above SMA50"}
    elif sma20 is not None and sma50 is not None and close < sma20 < sma50:
        sell += 20 if sma200 is None or sma50 < sma200 else 15
        components["trend_structure"] = {"weight": 20, "side": "sell", "reason": "Close below SMA20 and SMA20 below SMA50"}
    elif sma20 is not None and close > sma20:
        buy += 8
        components["trend_structure"] = {"weight": 20, "side": "buy", "reason": "Close above SMA20 only"}
    elif sma20 is not None and close < sma20:
        sell += 8
        components["trend_structure"] = {"weight": 20, "side": "sell", "reason": "Close below SMA20 only"}
    else:
        components["trend_structure"] = {"weight": 20, "side": "neutral"}

    rsi = indicators.get("rsi_14")
    if rsi is not None and 55 <= rsi <= 70:
        buy += 10
        components["rsi_momentum"] = {"weight": 10, "side": "buy", "value": rsi}
    elif rsi is not None and rsi < 40:
        sell += 10
        components["rsi_momentum"] = {"weight": 10, "side": "sell", "value": rsi}
    elif rsi is not None and rsi > 75:
        sell += 5
        components["rsi_momentum"] = {"weight": 10, "side": "sell", "value": rsi, "reason": "Overbought risk"}
    else:
        components["rsi_momentum"] = {"weight": 10, "side": "neutral", "value": rsi}

    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    previous_hist = indicators.get("previous_macd_histogram")
    current_hist = indicators.get("macd_histogram")
    if macd is not None and macd_signal is not None and macd > macd_signal and (
        current_hist is None or previous_hist is None or current_hist >= previous_hist
    ):
        buy += 10
        components["macd"] = {"weight": 10, "side": "buy", "histogram": current_hist}
    elif macd is not None and macd_signal is not None and macd < macd_signal and (
        current_hist is None or previous_hist is None or current_hist <= previous_hist
    ):
        sell += 10
        components["macd"] = {"weight": 10, "side": "sell", "histogram": current_hist}
    else:
        components["macd"] = {"weight": 10, "side": "neutral", "histogram": current_hist}

    volume_ratio = indicators.get("volume_ratio_20d")
    daily_change = indicators.get("daily_return")
    if volume_ratio is not None and volume_ratio >= 1.5 and daily_change is not None and daily_change > 0:
        buy += 10
        components["volume_confirmation"] = {"weight": 10, "side": "buy", "ratio": volume_ratio}
    elif volume_ratio is not None and volume_ratio >= 1.2 and daily_change is not None and daily_change < 0:
        sell += 10
        components["volume_confirmation"] = {"weight": 10, "side": "sell", "ratio": volume_ratio}
    else:
        components["volume_confirmation"] = {"weight": 10, "side": "neutral", "ratio": volume_ratio}

    if indicators.get("breakout_20d"):
        buy += 15
        components["price_action"] = {"weight": 15, "side": "buy", "reason": "Close above previous 20-session high"}
    elif indicators.get("breakdown_20d"):
        sell += 15
        components["price_action"] = {"weight": 15, "side": "sell", "reason": "Close below previous 20-session low"}
    elif indicators.get("bullish_engulfing") or indicators.get("hammer"):
        buy += 5
        components["price_action"] = {"weight": 15, "side": "buy", "reason": "Bullish reversal candle pattern"}
    elif indicators.get("bearish_engulfing") or indicators.get("shooting_star"):
        sell += 5
        components["price_action"] = {"weight": 15, "side": "sell", "reason": "Bearish reversal candle pattern"}
    else:
        components["price_action"] = {"weight": 15, "side": "neutral"}

    smc_buy = int(smc.get("bullish_points", 0))
    smc_sell = int(smc.get("bearish_points", 0))
    buy += smc_buy
    sell += smc_sell
    components["smc_price_action"] = {
        "weight": 15,
        "buy_points": smc_buy,
        "sell_points": smc_sell,
        "methodology": "Rule-based OHLC proxies; not observed institutional orders",
    }

    return {
        "buy_score": min(100, buy),
        "sell_score": min(100, sell),
        "score_margin": buy - sell,
        "score_type": "evidence points out of 100; not a probability of profit",
        "components": components,
        "weights": {
            "market_regime": 10,
            "sector_relative_strength": 10,
            "trend_structure": 20,
            "rsi_momentum": 10,
            "macd": 10,
            "volume_confirmation": 10,
            "price_action_breakout_breakdown": 15,
            "smc_rule_based_proxies": 15,
        },
    }


def _chart_series(rows: list[Price], smc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return JSON-safe daily chart data with indicator overlays."""
    sample = rows[-220:]
    close = np.asarray([float(row.close) for row in sample], dtype=float)
    highs = np.asarray([float(row.high) for row in sample], dtype=float)
    lows = np.asarray([float(row.low) for row in sample], dtype=float)
    volume = np.asarray([float(row.volume or 0) for row in sample], dtype=float)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd, 9)
    rsi = _rsi(close, 14)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(sample):
        result.append(
            {
                "date": row.date.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(volume[index]),
                "sma_20": float(sma20[index]) if np.isfinite(sma20[index]) else None,
                "sma_50": float(sma50[index]) if np.isfinite(sma50[index]) else None,
                "sma_200": float(sma200[index]) if np.isfinite(sma200[index]) else None,
                "macd": float(macd[index]) if np.isfinite(macd[index]) else None,
                "macd_signal": float(macd_signal[index]) if np.isfinite(macd_signal[index]) else None,
                "macd_histogram": (
                    float(macd[index] - macd_signal[index])
                    if np.isfinite(macd[index]) and np.isfinite(macd_signal[index])
                    else None
                ),
                "rsi_14": float(rsi[index]) if np.isfinite(rsi[index]) else None,
            }
        )
    return result


def _market_regime(session: Session) -> dict[str, Any]:
    rows = session.scalars(
        sa.select(NEPSEIndex).order_by(NEPSEIndex.date.desc()).limit(220)
    ).all()
    rows = list(reversed(rows))
    if len(rows) < 20:
        return {
            "regime": "unavailable",
            "reason": "NEPSE benchmark history is not yet imported or has fewer than 20 sessions",
            "signal_role": "market_context_only",
        }
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    sma50 = float(np.mean(closes[-50:])) if closes.size >= 50 else None
    sma200 = float(np.mean(closes[-200:])) if closes.size >= 200 else None
    return20 = float(closes[-1] / closes[-21] - 1.0) if closes.size >= 21 else None
    regime = "sideways"
    if sma50 is not None and return20 is not None:
        if closes[-1] > sma50 and return20 > 0 and (sma200 is None or sma50 > sma200):
            regime = "bullish"
        elif closes[-1] < sma50 and return20 < 0 and (sma200 is None or sma50 < sma200):
            regime = "bearish"
    return {
        "regime": regime,
        "as_of": rows[-1].date.isoformat(),
        "index_close": float(closes[-1]),
        "return_20d": return20,
        "sma_50": sma50,
        "sma_200": sma200,
        "observations": int(closes.size),
        "signal_role": "benchmark_only_no_buy_or_sell_orders",
    }


def _buy_trigger_at(index: int, closes: np.ndarray, highs: np.ndarray, volumes: np.ndarray) -> bool:
    if index < 50:
        return False
    prior_high = float(np.max(highs[index - 20 : index]))
    avg_volume = float(np.mean(volumes[index - 20 : index]))
    sma50 = float(np.mean(closes[index - 50 : index]))
    return (
        closes[index] > prior_high
        and avg_volume > 0
        and volumes[index] >= 1.5 * avg_volume
        and closes[index] > sma50
    )


def _sell_trigger_at(index: int, closes: np.ndarray, volumes: np.ndarray) -> bool:
    if index < 21:
        return False
    current_sma20 = float(np.mean(closes[index - 19 : index + 1]))
    prior_sma20 = float(np.mean(closes[index - 20 : index]))
    avg_volume = float(np.mean(volumes[index - 20 : index]))
    return (
        closes[index] < current_sma20
        and closes[index - 1] >= prior_sma20
        and avg_volume > 0
        and volumes[index] >= 1.2 * avg_volume
    )


def _historical_event_study(
    rows: list[Price], benchmark_rows: list[NEPSEIndex] | None = None
) -> dict[str, Any]:
    """Evaluate stock BUY/SELL events at next open and compare BUYs with NEPSE."""
    if len(rows) < 75:
        return {"sample_count": 0, "reason": "At least 75 clean observations required"}
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    highs = np.asarray([float(row.high) for row in rows], dtype=float)
    volumes = np.asarray([float(row.volume or 0) for row in rows], dtype=float)
    horizons = (5, 10, 20)
    outcomes: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
    benchmark_outcomes: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
    excess_outcomes: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
    sell_following: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
    benchmark_by_date = {row.date: row for row in benchmark_rows or []}
    trigger_dates: list[str] = []
    sell_trigger_dates: list[str] = []
    for index in range(50, len(rows) - max(horizons) - 1):
        if not _buy_trigger_at(index, closes, highs, volumes):
            continue
        entry_open = rows[index + 1].open
        if entry_open is None or float(entry_open) <= 0:
            continue
        trigger_dates.append(rows[index].date.isoformat())
        entry_fill = float(entry_open) * (1.0 + SLIPPAGE_BPS / 10_000.0)
        for horizon in horizons:
            exit_close = float(rows[index + horizon].close)
            exit_fill = exit_close * (1.0 - SLIPPAGE_BPS / 10_000.0)
            gross_return = exit_fill / entry_fill - 1.0
            net_return = (1.0 + gross_return) * ((1.0 - TRADING_COST_RATE) ** 2) - 1.0
            outcomes[str(horizon)].append(net_return)
            benchmark_entry = benchmark_by_date.get(rows[index + 1].date)
            benchmark_exit = benchmark_by_date.get(rows[index + horizon].date)
            if (
                benchmark_entry is not None
                and benchmark_entry.open is not None
                and benchmark_entry.open > 0
                and benchmark_exit is not None
            ):
                benchmark_return = float(benchmark_exit.close) / float(benchmark_entry.open) - 1.0
                benchmark_outcomes[str(horizon)].append(benchmark_return)
                excess_outcomes[str(horizon)].append(net_return - benchmark_return)

    for index in range(21, len(rows) - max(horizons) - 1):
        if not _sell_trigger_at(index, closes, volumes):
            continue
        sell_trigger_dates.append(rows[index].date.isoformat())
        for horizon in horizons:
            entry_open = rows[index + 1].open
            exit_close = float(rows[index + horizon].close)
            if entry_open is not None and float(entry_open) > 0:
                sell_following[str(horizon)].append(exit_close / float(entry_open) - 1.0)

    summary: dict[str, Any] = {}
    for horizon, values in outcomes.items():
        benchmark_values = benchmark_outcomes[horizon]
        excess_values = excess_outcomes[horizon]
        sell_values = sell_following[horizon]
        summary[f"forward_{horizon}d"] = {
            "sample_count": len(values),
            "win_rate": (
                sum(1 for value in values if value > 0) / len(values) if values else None
            ),
            "mean_net_return": float(statistics.mean(values)) if values else None,
            "median_net_return": float(statistics.median(values)) if values else None,
            "best_net_return": max(values) if values else None,
            "worst_net_return": min(values) if values else None,
            "nepse_benchmark_sample_count": len(benchmark_values),
            "mean_nepse_index_return": (
                float(statistics.mean(benchmark_values)) if benchmark_values else None
            ),
            "mean_excess_vs_nepse": (
                float(statistics.mean(excess_values)) if excess_values else None
            ),
            "sell_trigger_following_return_sample_count": len(sell_values),
            "mean_return_after_sell_trigger": (
                float(statistics.mean(sell_values)) if sell_values else None
            ),
            "negative_return_share_after_sell_trigger": (
                sum(1 for value in sell_values if value < 0) / len(sell_values)
                if sell_values else None
            ),
        }
    return {
        "method": "breakout event study; entry at next session open; fixed-horizon close exits",
        "trigger_count": len(trigger_dates),
        "first_trigger_date": trigger_dates[0] if trigger_dates else None,
        "last_trigger_date": trigger_dates[-1] if trigger_dates else None,
        "sell_trigger_count": len(sell_trigger_dates),
        "first_sell_trigger_date": sell_trigger_dates[0] if sell_trigger_dates else None,
        "last_sell_trigger_date": sell_trigger_dates[-1] if sell_trigger_dates else None,
        "assumptions": {
            "one_way_commission_rate": TRADING_COST_RATE,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "corporate_action_adjusted": False,
            "benchmark": "NEPSE Index; context only, not a tradable security",
        },
        **summary,
    }


class StockSectorResearchService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _load_histories(
        self, stocks: list[Stock], limit: int
    ) -> dict[int, list[Price]]:
        stock_ids = [stock.id for stock in stocks if stock.id is not None]
        if not stock_ids:
            return {}
        ranked = (
            sa.select(
                Price.id.label("price_id"),
                sa.func.row_number()
                .over(
                    partition_by=Price.stock_id,
                    order_by=(Price.date.desc(), Price.id.desc()),
                )
                .label("row_number"),
            )
            .where(Price.stock_id.in_(stock_ids))
            .subquery()
        )
        rows = self.session.scalars(
            sa.select(Price)
            .join(ranked, Price.id == ranked.c.price_id)
            .where(ranked.c.row_number <= limit)
            .order_by(Price.stock_id, Price.date)
        ).all()
        grouped: dict[int, list[Price]] = {}
        for row in rows:
            grouped.setdefault(row.stock_id, []).append(row)
        return {stock_id: _valid_history(history) for stock_id, history in grouped.items()}

    def _load_stock_history(self, stock: Stock, limit: int = 800) -> list[Price]:
        return self._load_histories([stock], limit).get(stock.id, [])

    def _sector_summary(
        self,
        eligible_stocks: list[Stock] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        stocks = eligible_stocks or self.session.scalars(
            sa.select(Stock).where(
                Stock.instrument_type == "ordinary",
                Stock.is_active.is_(True),
                Stock.sector.is_not(None),
            )
        ).all()
        histories = self._load_histories(stocks, 70)
        grouped: dict[str, list[float]] = {}
        grouped60: dict[str, list[float]] = {}
        for stock in stocks:
            history = histories.get(stock.id, [])
            clean, event = _clean_after_latest_discontinuity(history)
            if event["flagged"]:
                continue
            metrics = _indicator_snapshot(clean) if len(clean) >= 61 else {}
            ret20 = metrics.get("return_20d")
            ret60 = metrics.get("return_60d")
            if ret20 is not None:
                grouped.setdefault(str(stock.sector), []).append(float(ret20))
            if ret60 is not None:
                grouped60.setdefault(str(stock.sector), []).append(float(ret60))

        market = market_context if market_context is not None else _market_regime(self.session)
        market_return = market.get("return_20d")
        result: dict[str, dict[str, Any]] = {}
        for sector, values in grouped.items():
            med = float(statistics.median(values))
            result[sector] = {
                "sector": sector,
                "construction": "median member-stock raw price return (not an official sector index)",
                "return_20d": med,
                "return_60d": (
                    float(statistics.median(grouped60[sector])) if grouped60.get(sector) else None
                ),
                "relative_strength_vs_nepse_20d": (
                    med - float(market_return) if market_return is not None else None
                ),
                "positive_stock_breadth": sum(value > 0 for value in values) / len(values),
                "stocks_covered": len(values),
                "corporate_action_flagged_stocks_excluded": True,
            }
        return result

    def analyze_stock(
        self,
        symbol: str,
        sector_summaries: dict[str, dict[str, Any]] | None = None,
        history: list[Price] | None = None,
        stock_obj: Stock | None = None,
        market_context: dict[str, Any] | None = None,
        benchmark_rows: list[NEPSEIndex] | None = None,
    ) -> dict[str, Any]:
        stock = stock_obj or self.session.scalar(
            sa.select(Stock).where(Stock.symbol == symbol.strip().upper())
        )
        if stock is None:
            raise ValueError(f"Stock {symbol} not found")
        if stock.instrument_type != "ordinary" or not stock.is_active:
            raise ValueError("Only active listed ordinary shares are eligible for stock triggers")

        stock_history = history if history is not None else self._load_stock_history(stock)
        clean, discontinuity = _clean_after_latest_discontinuity(stock_history)
        if not clean:
            raise ValueError("No valid OHLCV history is available for this stock")

        indicators = _indicator_snapshot(clean)
        market = market_context if market_context is not None else _market_regime(self.session)
        sectors = (
            sector_summaries
            if sector_summaries is not None
            else self._sector_summary(market_context=market)
        )
        sector = sectors.get(str(stock.sector)) if stock.sector else None
        sector_rs = sector.get("relative_strength_vs_nepse_20d") if sector else None

        warnings: list[str] = []
        if discontinuity["flagged"]:
            warnings.append(
                "A price discontinuity was detected; indicators use only observations after "
                f"{discontinuity['event_date']}."
            )
        if len(clean) < 200:
            warnings.append("Fewer than 200 clean sessions: the 200-day trend filter is unavailable.")
        if indicators["bars"] < 50:
            warnings.append("Insufficient clean history for a confirmed 50-day trend trigger.")
        if market["regime"] == "unavailable":
            warnings.append("NEPSE benchmark history is unavailable; market confirmation is disabled.")
        if not sector:
            warnings.append("Sector classification or clean sector history is unavailable.")

        smc = _smc_snapshot(clean)
        score_model = _score_signal(indicators, smc, market, sector)
        buy_score = int(score_model["buy_score"])
        sell_score = int(score_model["sell_score"])
        score_margin = int(score_model["score_margin"])

        sma20 = indicators.get("sma_20")
        volume_ratio = indicators.get("volume_ratio_20d")
        sell_price_confirmation = bool(
            indicators.get("breakdown_20d")
            or (
                sma20 is not None
                and indicators["close"] < sma20
                and volume_ratio is not None
                and volume_ratio >= 1.2
            )
        )
        latest_age_days = max((date.today() - clean[-1].date).days, 0)
        data_freshness = "current" if latest_age_days <= 7 else "stale"

        trigger = "HOLD"
        trigger_reason = "Neither directional score reached the trigger threshold."
        buy_gates = {
            "market_history_available": market.get("regime") != "unavailable",
            "market_not_bearish": market.get("regime") in ("bullish", "sideways"),
            "sector_history_available": sector_rs is not None,
            "sector_not_underperforming": sector_rs is not None and sector_rs >= 0,
            "at_least_50_clean_sessions": len(clean) >= 50,
            "no_recent_price_discontinuity": not discontinuity["flagged"],
            "latest_data_within_7_calendar_days": data_freshness == "current",
        }
        if data_freshness == "stale":
            trigger = "STALE_DATA"
            trigger_reason = (
                f"Latest usable completed session is {latest_age_days} calendar days old; "
                "current stock triggers are suppressed."
            )
            warnings.append("Price history is more than seven calendar days old.")
        elif (
            buy_score >= 65
            and score_margin >= 25
            and all(buy_gates.values())
        ):
            trigger = "BUY_TRIGGER"
            trigger_reason = (
                "Composite BUY score reached 65/100 with a 25-point margin and all "
                "market, sector, history and data-quality gates passed."
            )
        elif (
            sell_score >= 65
            and -score_margin >= 25
            and sell_price_confirmation
            and data_freshness == "current"
        ):
            trigger = "SELL_TRIGGER"
            trigger_reason = (
                "Composite SELL score reached 65/100 with bearish price/volume confirmation. "
                "This is a review/exit alert for an existing holding, not a short-sale instruction."
            )
        elif buy_score >= 45 or sell_score >= 45:
            trigger = "WATCH"
            trigger_reason = (
                "Directional evidence is building, but the score threshold or one or more "
                "confirmation gates have not been satisfied."
            )

        confirmations = {
            "price_above_sma20": sma20 is not None and indicators["close"] > sma20,
            "sma20_above_sma50": sma20 is not None and indicators["sma_50"] is not None and sma20 > indicators["sma_50"],
            "rsi_positive_momentum": indicators.get("rsi_14") is not None and 50 <= indicators["rsi_14"] <= 70,
            "macd_above_signal": indicators.get("macd") is not None and indicators.get("macd_signal") is not None and indicators["macd"] > indicators["macd_signal"],
            "sector_outperforming": sector_rs is not None and sector_rs > 0,
            "market_not_bearish": market.get("regime") in ("bullish", "sideways"),
        }
        score_strength = max(buy_score, sell_score) / 100.0
        if benchmark_rows is None:
            benchmark_rows = self.session.scalars(
                sa.select(NEPSEIndex)
                .where(
                    NEPSEIndex.date >= clean[0].date,
                    NEPSEIndex.date <= clean[-1].date,
                )
                .order_by(NEPSEIndex.date)
            ).all()
        atr = indicators["atr_14"]
        suggested_stop = (
            round(indicators["close"] - 2.0 * atr, 2) if atr is not None else None
        )
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "instrument_type": stock.instrument_type,
            "as_of": clean[-1].date.isoformat(),
            "signal": trigger,
            "signal_reason": trigger_reason,
            "confidence_score": round(score_strength, 4),
            "buy_score": buy_score,
            "sell_score": sell_score,
            "score_margin": score_margin,
            "score_model": score_model,
            "confirmation_gates": buy_gates,
            "confirmation_checks": confirmations,
            "smart_money_concepts": smc,
            "market_context": market,
            "sector_context": sector,
            "indicators": indicators,
            "price_action": {
                "bullish_engulfing": indicators.get("bullish_engulfing"),
                "bearish_engulfing": indicators.get("bearish_engulfing"),
                "hammer": indicators.get("hammer"),
                "shooting_star": indicators.get("shooting_star"),
                "prior_20_session_high": indicators.get("previous_high_20d"),
                "prior_20_session_low": indicators.get("previous_low_20d"),
            },
            "reference_stop_level": suggested_stop,
            "reference_target_level": (
                round(indicators["close"] + 4.0 * atr, 2) if atr is not None else None
            ),
            "price_history_for_chart": _chart_series(clean, smc),
            "data_freshness": {
                "status": data_freshness,
                "latest_session": clean[-1].date.isoformat(),
                "calendar_days_old": latest_age_days,
                "max_age_calendar_days_for_new_trigger": 7,
            },
            "historical_event_study": _historical_event_study(clean, list(benchmark_rows)),
            "data_quality": {
                "status": "warning" if warnings or discontinuity["flagged"] else "usable",
                "warnings": warnings,
                "price_discontinuity": discontinuity,
                "source_prices_corporate_action_adjusted": False,
                "source_disclaimer": RESEARCH_SOURCE_WARNING,
            },
            "execution_note": (
                "Signals are calculated after a completed daily close. Any paper entry or exit "
                "must use a subsequent session's opening price, subject to liquidity and fees."
            ),
        }

    def scan_stocks(self, limit: int = 100) -> dict[str, Any]:
        stocks = self.session.scalars(
            sa.select(Stock)
            .where(
                Stock.instrument_type == "ordinary",
                Stock.is_active.is_(True),
            )
            .order_by(Stock.symbol)
        ).all()
        sectors = self._sector_summary(stocks)
        histories = self._load_histories(stocks, 800)
        results: list[dict[str, Any]] = []
        for stock in stocks:
            try:
                analysis = self.analyze_stock(
                    stock.symbol,
                    sector_summaries=sectors,
                    history=histories.get(stock.id, []),
                )
                analysis["sector_context"] = sectors.get(str(stock.sector)) if stock.sector else None
                results.append(analysis)
            except ValueError:
                continue
        priority = {"BUY_TRIGGER": 0, "SELL_TRIGGER": 1, "SETUP": 2, "WATCH": 3, "HOLD": 4}
        results.sort(key=lambda item: (priority.get(item["signal"], 9), -item["confidence_score"]))
        return {
            "as_of": max((item["as_of"] for item in results), default=None),
            "market_context": _market_regime(self.session),
            "stocks_analyzed": len(results),
            "signals": results[:limit],
            "data_disclaimer": RESEARCH_SOURCE_WARNING,
        }

    def sector_ranking(self) -> dict[str, Any]:
        market = _market_regime(self.session)
        sectors = list(self._sector_summary().values())
        sectors.sort(
            key=lambda item: (
                item["relative_strength_vs_nepse_20d"] is not None,
                item["relative_strength_vs_nepse_20d"] or -999,
            ),
            reverse=True,
        )
        return {
            "as_of": market.get("as_of"),
            "market_context": market,
            "sector_count": len(sectors),
            "sectors": sectors,
            "methodology_warning": (
                "Sector returns are medians of member-stock raw price returns, not official "
                "capitalization-weighted sector indices. Stocks with recent >15% price jumps "
                "are excluded pending corporate-action/source review."
            ),
        }
