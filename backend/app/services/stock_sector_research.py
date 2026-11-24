"""Sector-relative and stock-level technical research (advisory only)."""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
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


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA, seeded by the first complete period of finite values."""
    result = np.full(values.shape, np.nan, dtype=float)
    if period <= 0 or values.size < period:
        return result
    seed_index: int | None = None
    for start in range(values.size - period + 1):
        window = values[start : start + period]
        if np.isfinite(window).all():
            seed_index = start + period - 1
            result[seed_index] = float(np.mean(window))
            break
    if seed_index is None:
        return result
    for index in range(seed_index + 1, values.size):
        if np.isfinite(values[index]) and np.isfinite(result[index - 1]):
            result[index] = (
                result[index - 1] * (period - 1) + values[index]
            ) / period
    return result


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI using the standard initial mean and recursive smoothing."""
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size <= period:
        return result
    changes = np.diff(values)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    avg_gain = _wilder(gains, period)
    avg_loss = _wilder(losses, period)
    for index in range(period, values.size):
        gain = avg_gain[index - 1]
        loss = avg_loss[index - 1]
        if not np.isfinite(gain) or not np.isfinite(loss):
            continue
        if loss == 0:
            result[index] = 100.0 if gain > 0 else 50.0
        elif gain == 0:
            result[index] = 0.0
        else:
            result[index] = 100.0 - (100.0 / (1.0 + gain / loss))
    return result


def _adx_series(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Wilder-smoothed ATR, +DI, -DI and ADX for completed OHLC bars."""
    size = close.size
    true_range = np.full(size, np.nan, dtype=float)
    plus_dm = np.zeros(size, dtype=float)
    minus_dm = np.zeros(size, dtype=float)
    if size:
        true_range[0] = high[0] - low[0]
    if size > 1:
        true_range[1:] = np.maximum.reduce(
            [
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ]
        )
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm[1:] = np.where(
            (up_move > down_move) & (up_move > 0), up_move, 0.0
        )
        minus_dm[1:] = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )

    atr = _wilder(true_range, period)
    smoothed_plus_dm = _wilder(plus_dm, period)
    smoothed_minus_dm = _wilder(minus_dm, period)
    plus_di = np.full(size, np.nan, dtype=float)
    minus_di = np.full(size, np.nan, dtype=float)
    valid_atr = np.isfinite(atr) & (atr > 0)
    plus_di[valid_atr] = 100.0 * smoothed_plus_dm[valid_atr] / atr[valid_atr]
    minus_di[valid_atr] = 100.0 * smoothed_minus_dm[valid_atr] / atr[valid_atr]

    di_sum = plus_di + minus_di
    dx = np.full(size, np.nan, dtype=float)
    valid_di = np.isfinite(di_sum) & (di_sum > 0)
    dx[valid_di] = (
        100.0
        * np.abs(plus_di[valid_di] - minus_di[valid_di])
        / di_sum[valid_di]
    )
    adx = _wilder(dx, period)
    return atr, plus_di, minus_di, adx


def _valid_history(rows: list[Price]) -> list[Price]:
    """Keep rows with complete, positive closing prices and valid OHLC fields."""
    return [
        row
        for row in rows
        if row.close is not None
        and float(row.close) > 0
        and row.open is not None
        and row.high is not None
        and row.low is not None
    ]


def _clean_after_latest_discontinuity(
    rows: list[Price],
) -> tuple[list[Price], dict[str, Any]]:
    """Return observations after the latest material raw-price discontinuity."""
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
    """Calculate the latest stock indicators from completed, valid OHLCV bars."""
    if not rows:
        raise ValueError("At least one valid price row is required")

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
    atr14, plus_di, minus_di, adx14 = _adx_series(close, high, low, 14)

    stoch_k = np.full(close.shape, np.nan, dtype=float)
    stoch_d = np.full(close.shape, np.nan, dtype=float)
    for index in range(13, close.size):
        highest = float(np.max(high[index - 13 : index + 1]))
        lowest = float(np.min(low[index - 13 : index + 1]))
        stoch_k[index] = (
            50.0
            if highest == lowest
            else 100.0 * (close[index] - lowest) / (highest - lowest)
        )
        if index >= 15:
            stoch_d[index] = float(np.mean(stoch_k[index - 2 : index + 1]))

    obv = np.zeros(close.shape, dtype=float)
    for index in range(1, close.size):
        if close[index] > close[index - 1]:
            obv[index] = obv[index - 1] + volume[index]
        elif close[index] < close[index - 1]:
            obv[index] = obv[index - 1] - volume[index]
        else:
            obv[index] = obv[index - 1]
    obv_sma20 = _sma(obv, 20)
    obv_rising = bool(close.size >= 20 and obv[-1] > obv[-20])

    prior_avg_volume20 = (
        float(np.mean(volume[-21:-1])) if volume.size >= 21 else None
    )
    volume_ratio = (
        float(volume[-1] / prior_avg_volume20)
        if prior_avg_volume20 is not None and prior_avg_volume20 > 0
        else None
    )
    previous_high20 = float(np.max(high[-21:-1])) if close.size >= 21 else None
    previous_low20 = float(np.min(low[-21:-1])) if close.size >= 21 else None
    current_hist = (
        float(macd[-1] - macd_signal[-1])
        if np.isfinite(macd[-1]) and np.isfinite(macd_signal[-1])
        else None
    )
    previous_hist = (
        float(macd[-2] - macd_signal[-2])
        if close.size >= 2
        and np.isfinite(macd[-2])
        and np.isfinite(macd_signal[-2])
        else None
    )
    average_turnover20 = _sma(close * volume, 20)
    daily_return = (
        float(close[-1] / close[-2] - 1.0)
        if close.size >= 2 and close[-2] > 0
        else None
    )
    bollinger_mid = float(sma20[-1]) if np.isfinite(sma20[-1]) else None
    bollinger_std = (
        float(np.std(close[-20:], ddof=0)) if close.size >= 20 else None
    )
    previous_open = float(rows[-2].open) if len(rows) >= 2 else None
    previous_close = float(close[-2]) if close.size >= 2 else None
    current_open = float(rows[-1].open)
    current_high = float(high[-1])
    current_low = float(low[-1])
    current_close = float(close[-1])
    previous_bearish = (
        previous_open is not None
        and previous_close is not None
        and previous_close < previous_open
    )
    previous_bullish = (
        previous_open is not None
        and previous_close is not None
        and previous_close > previous_open
    )
    current_bullish = current_close > current_open
    current_bearish = current_close < current_open
    bullish_engulfing = (
        previous_bearish
        and current_bullish
        and previous_close is not None
        and previous_open is not None
        and current_open <= float(previous_close)
        and current_close >= float(previous_open)
    )
    bearish_engulfing = (
        previous_bullish
        and current_bearish
        and previous_close is not None
        and previous_open is not None
        and current_open >= float(previous_close)
        and current_close <= float(previous_open)
    )
    candle_body = abs(current_close - current_open)
    lower_wick = min(current_open, current_close) - current_low
    upper_wick = current_high - max(current_open, current_close)
    hammer = (
        candle_body > 0
        and lower_wick >= 2 * candle_body
        and upper_wick <= candle_body
    )
    shooting_star = (
        candle_body > 0
        and upper_wick >= 2 * candle_body
        and lower_wick <= candle_body
    )
    return {
        "close": float(close[-1]),
        "sma_20": float(sma20[-1]) if np.isfinite(sma20[-1]) else None,
        "previous_sma_20": (
            float(sma20[-2])
            if sma20.size >= 2 and np.isfinite(sma20[-2])
            else None
        ),
        "previous_close": previous_close,
        "sma_50": float(sma50[-1]) if np.isfinite(sma50[-1]) else None,
        "sma_200": float(sma200[-1]) if np.isfinite(sma200[-1]) else None,
        "rsi_14": float(rsi[-1]) if np.isfinite(rsi[-1]) else None,
        "stoch_k_14": float(stoch_k[-1]) if np.isfinite(stoch_k[-1]) else None,
        "stoch_d_3": float(stoch_d[-1]) if np.isfinite(stoch_d[-1]) else None,
        "adx_14": float(adx14[-1]) if np.isfinite(adx14[-1]) else None,
        "plus_di_14": float(plus_di[-1]) if np.isfinite(plus_di[-1]) else None,
        "minus_di_14": float(minus_di[-1]) if np.isfinite(minus_di[-1]) else None,
        "obv": float(obv[-1]),
        "obv_sma_20": float(obv_sma20[-1]) if np.isfinite(obv_sma20[-1]) else None,
        "obv_rising_20d": obv_rising,
        "macd": float(macd[-1]) if np.isfinite(macd[-1]) else None,
        "macd_signal": (
            float(macd_signal[-1]) if np.isfinite(macd_signal[-1]) else None
        ),
        "macd_histogram": current_hist,
        "previous_macd_histogram": previous_hist,
        "atr_14": float(atr14[-1]) if np.isfinite(atr14[-1]) else None,
        "bollinger_middle": bollinger_mid,
        "bollinger_upper": (
            bollinger_mid + 2 * bollinger_std
            if bollinger_mid is not None and bollinger_std is not None
            else None
        ),
        "bollinger_lower": (
            bollinger_mid - 2 * bollinger_std
            if bollinger_mid is not None and bollinger_std is not None
            else None
        ),
        "average_traded_value_20d": (
            float(average_turnover20[-1])
            if average_turnover20.size and np.isfinite(average_turnover20[-1])
            else None
        ),
        "daily_return": daily_return,
        "breakout_20d": (
            previous_high20 is not None and close[-1] > previous_high20
        ),
        "breakdown_20d": (
            previous_low20 is not None and close[-1] < previous_low20
        ),
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "volume_ratio_20d": volume_ratio,
        "previous_high_20d": previous_high20,
        "previous_low_20d": previous_low20,
        "high_60d": float(np.max(high[-60:])) if close.size >= 60 else None,
        "low_60d": float(np.min(low[-60:])) if close.size >= 60 else None,
        "return_20d": float(close[-1] / close[-21] - 1.0) if close.size >= 21 else None,
        "return_60d": float(close[-1] / close[-61] - 1.0) if close.size >= 61 else None,
        "return_120d": float(close[-1] / close[-121] - 1.0) if close.size >= 121 else None,
        "return_200d": float(close[-1] / close[-201] - 1.0) if close.size >= 201 else None,
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
        "bullish_fvg_zone": (
            {"low": float(highs[-3]), "high": float(lows[-1])} if bullish_fvg else None
        ),
        "bearish_fvg_zone": (
            {"low": float(highs[-1]), "high": float(lows[-3])} if bearish_fvg else None
        ),
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

    def record(
        name: str,
        weight: int,
        buy_points: int = 0,
        sell_points: int = 0,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        nonlocal buy, sell
        buy += buy_points
        sell += sell_points
        components[name] = {
            "weight": weight,
            "buy_points": buy_points,
            "sell_points": sell_points,
            **(evidence or {}),
        }

    regime = market.get("regime")
    if regime == "bullish":
        record("market_regime", 10, buy_points=10, evidence={"reason": "NEPSE trend bullish"})
    elif regime == "bearish":
        record("market_regime", 10, sell_points=10, evidence={"reason": "NEPSE trend bearish"})
    else:
        record("market_regime", 10, evidence={"reason": "No confirmed market trend"})

    sector_relative_strengths: list[float] = []
    if sector is not None:
        for key in (
            "relative_strength_vs_nepse_20d",
            "relative_strength_vs_nepse_60d",
            "relative_strength_vs_nepse_120d",
        ):
            value = sector.get(key)
            if value is not None:
                sector_relative_strengths.append(float(value))
    sector_positive_count = sum(value > 0 for value in sector_relative_strengths)
    sector_negative_count = sum(value < 0 for value in sector_relative_strengths)
    if sector_positive_count >= 2:
        sector_points, sector_side = 10, "BUY"
    elif sector_positive_count == 1:
        sector_points, sector_side = 5, "BUY"
    elif sector_negative_count >= 2:
        sector_points, sector_side = 10, "SELL"
    elif sector_negative_count == 1:
        sector_points, sector_side = 5, "SELL"
    else:
        sector_points, sector_side = 0, "NEUTRAL"
    record(
        "sector_strength",
        10,
        buy_points=sector_points if sector_side == "BUY" else 0,
        sell_points=sector_points if sector_side == "SELL" else 0,
        evidence={
            "relative_strength_vs_nepse": sector_relative_strengths,
            "positive_horizons": sector_positive_count,
            "negative_horizons": sector_negative_count,
            "method": "20/60/120-session median constituent returns versus NEPSE",
        },
    )

    close = indicators["close"]
    sma20 = indicators.get("sma_20")
    sma50 = indicators.get("sma_50")
    sma200 = indicators.get("sma_200")
    adx = indicators.get("adx_14")
    trend_strength_confirmed = adx is not None and adx >= 20.0
    if sma20 is not None and sma50 is not None and close > sma20 > sma50:
        points = 15 if sma200 is not None and sma50 > sma200 and trend_strength_confirmed else 8
        record(
            "trend_structure",
            15,
            buy_points=points,
            evidence={
                "reason": "Close above SMA20 and SMA20 above SMA50",
                "adx_14": adx,
                "adx_trend_strength_confirmed": trend_strength_confirmed,
            },
        )
    elif sma20 is not None and sma50 is not None and close < sma20 < sma50:
        points = 15 if sma200 is not None and sma50 < sma200 and trend_strength_confirmed else 8
        record(
            "trend_structure",
            15,
            sell_points=points,
            evidence={
                "reason": "Close below SMA20 and SMA20 below SMA50",
                "adx_14": adx,
                "adx_trend_strength_confirmed": trend_strength_confirmed,
            },
        )
    elif sma20 is not None and close > sma20:
        record("trend_structure", 15, buy_points=6, evidence={"reason": "Close above SMA20 only"})
    elif sma20 is not None and close < sma20:
        record("trend_structure", 15, sell_points=6, evidence={"reason": "Close below SMA20 only"})
    else:
        record("trend_structure", 15, evidence={"reason": "No usable moving-average trend"})

    medium_term_returns = [
        indicators.get("return_60d"),
        indicators.get("return_120d"),
        indicators.get("return_200d"),
    ]
    valid_medium_term_returns = [
        float(value) for value in medium_term_returns if value is not None
    ]
    positive_momentum_count = sum(value > 0 for value in valid_medium_term_returns)
    negative_momentum_count = sum(value < 0 for value in valid_medium_term_returns)
    if positive_momentum_count >= 2:
        medium_points, medium_side = 10, "BUY"
    elif positive_momentum_count == 1:
        medium_points, medium_side = 5, "BUY"
    elif negative_momentum_count >= 2:
        medium_points, medium_side = 10, "SELL"
    elif negative_momentum_count == 1:
        medium_points, medium_side = 5, "SELL"
    else:
        medium_points, medium_side = 0, "NEUTRAL"
    record(
        "medium_term_momentum",
        10,
        buy_points=medium_points if medium_side == "BUY" else 0,
        sell_points=medium_points if medium_side == "SELL" else 0,
        evidence={
            "return_60d": indicators.get("return_60d"),
            "return_120d": indicators.get("return_120d"),
            "return_200d": indicators.get("return_200d"),
            "positive_horizons": positive_momentum_count,
            "negative_horizons": negative_momentum_count,
            "method": "multi-horizon time-series momentum",
        },
    )

    rsi = indicators.get("rsi_14")
    stoch_k = indicators.get("stoch_k_14")
    stoch_d = indicators.get("stoch_d_3")
    stoch_bullish = stoch_k is not None and stoch_d is not None and stoch_k >= stoch_d
    stoch_bearish = stoch_k is not None and stoch_d is not None and stoch_k <= stoch_d
    if rsi is not None and 55 <= rsi <= 70:
        points = 10 if stoch_bullish else 5
        record(
            "rsi_momentum", 10, buy_points=points,
            evidence={"rsi_14": rsi, "stoch_k_14": stoch_k, "stoch_d_3": stoch_d},
        )
    elif rsi is not None and rsi < 40:
        points = 10 if stoch_bearish or stoch_k is None else 5
        record(
            "rsi_momentum", 10, sell_points=points,
            evidence={"rsi_14": rsi, "stoch_k_14": stoch_k, "stoch_d_3": stoch_d},
        )
    elif rsi is not None and rsi > 75:
        record(
            "rsi_momentum",
            10,
            sell_points=5,
            evidence={"rsi_14": rsi, "reason": "Overbought risk"},
        )
    else:
        record("rsi_momentum", 10, evidence={"rsi_14": rsi})

    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    previous_hist = indicators.get("previous_macd_histogram")
    current_hist = indicators.get("macd_histogram")
    if macd is not None and macd_signal is not None and current_hist is not None and previous_hist is not None and macd > macd_signal:
        points = 10 if current_hist >= previous_hist else 5
        record("macd", 10, buy_points=points, evidence={"macd_histogram": current_hist})
    elif macd is not None and macd_signal is not None and current_hist is not None and previous_hist is not None and macd < macd_signal:
        points = 10 if current_hist <= previous_hist else 5
        record("macd", 10, sell_points=points, evidence={"macd_histogram": current_hist})
    else:
        record("macd", 10, evidence={"macd_histogram": current_hist})

    volume_ratio = indicators.get("volume_ratio_20d")
    daily_change = indicators.get("daily_return")
    obv_rising = indicators.get("obv_rising_20d")
    if volume_ratio is not None and volume_ratio >= 1.5 and daily_change is not None and daily_change > 0:
        points = 10 if obv_rising is True else 5
        record(
            "volume_confirmation",
            10,
            buy_points=points,
            evidence={"volume_ratio_20d": volume_ratio, "obv_rising_20d": obv_rising},
        )
    elif volume_ratio is not None and volume_ratio >= 1.2 and daily_change is not None and daily_change < 0:
        points = 10 if obv_rising is False else 5
        record(
            "volume_confirmation",
            10,
            sell_points=points,
            evidence={"volume_ratio_20d": volume_ratio, "obv_rising_20d": obv_rising},
        )
    else:
        record(
            "volume_confirmation",
            10,
            evidence={"volume_ratio_20d": volume_ratio},
        )

    if indicators.get("breakout_20d"):
        record(
            "price_action",
            15,
            buy_points=15,
            evidence={"reason": "Close above previous 20-session high"},
        )
    elif indicators.get("breakdown_20d"):
        record(
            "price_action",
            15,
            sell_points=15,
            evidence={"reason": "Close below previous 20-session low"},
        )
    elif indicators.get("bullish_engulfing") or indicators.get("hammer"):
        record(
            "price_action",
            15,
            buy_points=5,
            evidence={"reason": "Bullish reversal candle pattern"},
        )
    elif indicators.get("bearish_engulfing") or indicators.get("shooting_star"):
        record(
            "price_action",
            15,
            sell_points=5,
            evidence={"reason": "Bearish reversal candle pattern"},
        )
    else:
        record("price_action", 15, evidence={"reason": "No defined breakout/breakdown candle"})

    # SMC is assigned a deliberately smaller weight because these are
    # discretionary concepts approximated from daily OHLC, not order-flow data.
    smc_buy = min(10, round(int(smc.get("bullish_points", 0)) * 10 / 15))
    smc_sell = min(10, round(int(smc.get("bearish_points", 0)) * 10 / 15))
    record(
        "smc_price_action",
        10,
        buy_points=smc_buy,
        sell_points=smc_sell,
        evidence={"methodology": "Rule-based OHLC proxies; not observed institutional orders"},
    )

    return {
        "buy_score": min(100, buy),
        "sell_score": min(100, sell),
        "score_margin": buy - sell,
        "score_type": "evidence points out of 100; not a probability of profit",
        "components": components,
        "weights": {
            "market_regime": 10,
            "sector_relative_strength": 10,
            "trend_structure": 15,
            "medium_term_momentum": 10,
            "rsi_momentum": 10,
            "macd": 10,
            "volume_confirmation": 10,
            "price_action_breakout_breakdown": 15,
            "smc_rule_based_proxies": 10,
        },
    }


def _chart_series(rows: list[Price], smc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return JSON-safe daily chart data with indicator overlays."""
    sample = rows[-220:]
    close = np.asarray([float(row.close) for row in sample], dtype=float)
    volume = np.asarray([float(row.volume or 0) for row in sample], dtype=float)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd, 9)
    rsi = _rsi(close, 14)
    # Keep chart formulas consistent with the stock-detail scoring snapshot.
    high = np.asarray([float(row.high) for row in sample], dtype=float)
    low = np.asarray([float(row.low) for row in sample], dtype=float)
    # Keep chart overlays identical to scoring: Wilder ATR and ADX.
    atr14, plus_di, minus_di, adx14 = _adx_series(close, high, low, 14)
    stoch_k = np.full(close.shape, np.nan, dtype=float)
    stoch_d = np.full(close.shape, np.nan, dtype=float)
    for index in range(13, close.size):
        highest = float(np.max(high[index - 13 : index + 1]))
        lowest = float(np.min(low[index - 13 : index + 1]))
        stoch_k[index] = (
            50.0 if highest == lowest
            else 100.0 * (close[index] - lowest) / (highest - lowest)
        )
        if index >= 15:
            stoch_d[index] = float(np.mean(stoch_k[index - 2 : index + 1]))
    obv = np.zeros(close.shape, dtype=float)
    for index in range(1, close.size):
        if close[index] > close[index - 1]:
            obv[index] = obv[index - 1] + volume[index]
        elif close[index] < close[index - 1]:
            obv[index] = obv[index - 1] - volume[index]
        else:
            obv[index] = obv[index - 1]
    rolling_std20 = np.full(close.shape, np.nan, dtype=float)
    for index in range(19, close.size):
        rolling_std20[index] = float(np.std(close[index - 19 : index + 1], ddof=0))
    result: list[dict[str, Any]] = []
    for index, row in enumerate(sample):
        previous_open = float(sample[index - 1].open) if index >= 1 else None
        previous_close = float(sample[index - 1].close) if index >= 1 else None
        current_open = float(row.open)
        current_close = float(row.close)
        current_high = float(row.high)
        current_low = float(row.low)
        bullish_engulfing = (
            index >= 1
            and previous_close is not None
            and previous_open is not None
            and previous_close < previous_open
            and current_close > current_open
            and current_open <= previous_close
            and current_close >= previous_open
        )
        bearish_engulfing = (
            index >= 1
            and previous_close is not None
            and previous_open is not None
            and previous_close > previous_open
            and current_close < current_open
            and current_open >= previous_close
            and current_close <= previous_open
        )
        body = abs(current_close - current_open)
        lower_wick = min(current_open, current_close) - current_low
        upper_wick = current_high - max(current_open, current_close)
        hammer = body > 0 and lower_wick >= 2 * body and upper_wick <= body
        shooting_star = body > 0 and upper_wick >= 2 * body and lower_wick <= body
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
                "ema_9": float(ema9[index]) if np.isfinite(ema9[index]) else None,
                "ema_21": float(ema21[index]) if np.isfinite(ema21[index]) else None,
                "bollinger_upper": float(sma20[index] + 2 * rolling_std20[index]) if np.isfinite(sma20[index]) and np.isfinite(rolling_std20[index]) else None,
                "bollinger_lower": float(sma20[index] - 2 * rolling_std20[index]) if np.isfinite(sma20[index]) and np.isfinite(rolling_std20[index]) else None,
                "macd": float(macd[index]) if np.isfinite(macd[index]) else None,
                "macd_signal": float(macd_signal[index]) if np.isfinite(macd_signal[index]) else None,
                "macd_histogram": (
                    float(macd[index] - macd_signal[index])
                    if np.isfinite(macd[index]) and np.isfinite(macd_signal[index])
                    else None
                ),
                "rsi_14": float(rsi[index]) if np.isfinite(rsi[index]) else None,
                "atr_14": float(atr14[index]) if np.isfinite(atr14[index]) else None,
                "adx_14": float(adx14[index]) if np.isfinite(adx14[index]) else None,
                "plus_di_14": float(plus_di[index]) if np.isfinite(plus_di[index]) else None,
                "minus_di_14": float(minus_di[index]) if np.isfinite(minus_di[index]) else None,
                "stoch_k_14": float(stoch_k[index]) if np.isfinite(stoch_k[index]) else None,
                "stoch_d_3": float(stoch_d[index]) if np.isfinite(stoch_d[index]) else None,
                "obv": float(obv[index]),
                "bullish_engulfing": bullish_engulfing,
                "bearish_engulfing": bearish_engulfing,
                "hammer": hammer,
                "shooting_star": shooting_star,
            }
        )
    return result


def _market_regime(session: Session) -> dict[str, Any]:
    rows = session.scalars(
        sa.select(NEPSEIndex).order_by(NEPSEIndex.date.desc()).limit(260)
    ).all()
    rows = list(reversed(rows))
    if len(rows) < 50:
        return {
            "regime": "unavailable",
            "reason": "At least 50 completed NEPSE Index sessions are required for a market-regime filter",
            "signal_role": "market_context_only",
        }
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    sma50 = float(np.mean(closes[-50:])) if closes.size >= 50 else None
    sma200 = float(np.mean(closes[-200:])) if closes.size >= 200 else None
    return20 = float(closes[-1] / closes[-21] - 1.0) if closes.size >= 21 else None
    return60 = float(closes[-1] / closes[-61] - 1.0) if closes.size >= 61 else None
    return120 = float(closes[-1] / closes[-121] - 1.0) if closes.size >= 121 else None
    return200 = float(closes[-1] / closes[-201] - 1.0) if closes.size >= 201 else None
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
        "return_60d": return60,
        "return_120d": return120,
        "return_200d": return200,
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
    last_buy_event = -max(horizons)
    last_sell_event = -max(horizons)
    for index in range(50, len(rows) - max(horizons) - 1):
        if index - last_buy_event < max(horizons):
            continue
        if not _buy_trigger_at(index, closes, highs, volumes):
            continue
        last_buy_event = index
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
        if index - last_sell_event < max(horizons):
            continue
        if not _sell_trigger_at(index, closes, volumes):
            continue
        last_sell_event = index
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
        "method": "20-session breakout event study; entry at next session open; fixed-horizon close exits",
        "scope_warning": (
            "This historical event study evaluates a separate breakout rule, not the full "
            "current multi-factor score. Full score-model walk-forward validation must be run "
            "after sector history and corporate-action adjustments are complete."
        ),
        "overlap_control": f"At least {max(horizons)} sessions between same-direction events",
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


def _market_context_at_date(
    benchmark_rows: list[NEPSEIndex], as_of: date
) -> dict[str, Any]:
    history = [row for row in benchmark_rows if row.date <= as_of]
    if len(history) < 50:
        return {"regime": "unavailable", "return_20d": None, "as_of": as_of.isoformat()}
    closes = np.asarray([float(row.close) for row in history[-260:]], dtype=float)
    sma50 = float(np.mean(closes[-50:]))
    sma200 = float(np.mean(closes[-200:])) if closes.size >= 200 else None
    return20 = float(closes[-1] / closes[-21] - 1.0)
    return60 = float(closes[-1] / closes[-61] - 1.0) if closes.size >= 61 else None
    return120 = float(closes[-1] / closes[-121] - 1.0) if closes.size >= 121 else None
    return200 = float(closes[-1] / closes[-201] - 1.0) if closes.size >= 201 else None
    regime = "sideways"
    if closes[-1] > sma50 and return20 > 0 and (sma200 is None or sma50 > sma200):
        regime = "bullish"
    elif closes[-1] < sma50 and return20 < 0 and (sma200 is None or sma50 < sma200):
        regime = "bearish"
    return {
        "regime": regime,
        "return_20d": return20,
        "return_60d": return60,
        "return_120d": return120,
        "return_200d": return200,
        "index_close": float(closes[-1]),
        "as_of": history[-1].date.isoformat(),
    }


def _historical_score_validation(
    rows: list[Price],
    benchmark_rows: list[NEPSEIndex],
    sector_histories: list[list[Price]],
) -> dict[str, Any]:
    """Evaluate fixed score weights on a chronological 30% holdout.

    Every historical score uses only bars available through that signal date.
    Entries occur at the next recorded session's open. The NEPSE Index is only
    a benchmark; sector strength is a same-date cross-sectional median.
    """
    if len(rows) < 250:
        return {
            "status": "insufficient_history",
            "reason": "At least 250 clean stock sessions are required for a 70/30 holdout",
            "scoring_model_version": "sector-stock-score-v1",
        }
    if not benchmark_rows:
        return {
            "status": "benchmark_missing",
            "reason": "NEPSE Index history is required for market-regime and benchmark comparisons",
            "scoring_model_version": "sector-stock-score-v1",
        }

    split_index = max(200, int(len(rows) * 0.70))
    horizons = (5, 10, 20)
    buy_results: dict[str, list[float]] = {str(h): [] for h in horizons}
    benchmark_results: dict[str, list[float]] = {str(h): [] for h in horizons}
    excess_results: dict[str, list[float]] = {str(h): [] for h in horizons}
    sell_following: dict[str, list[float]] = {str(h): [] for h in horizons}
    buy_dates: list[str] = []
    sell_dates: list[str] = []

    # Match the live scoring model's 20/60/120-session sector-relative
    # strength inputs. A single 20-session value is not enough to represent
    # the three-horizon sector component used by _score_signal.
    sector_returns: dict[int, dict[date, list[float]]] = {
        sector_horizon: {} for sector_horizon in (20, 60, 120)
    }
    for member_history in sector_histories:
        member_rows = _valid_history(member_history)
        for sector_horizon in (20, 60, 120):
            for index in range(sector_horizon, len(member_rows)):
                earlier = float(member_rows[index - sector_horizon].close)
                current = float(member_rows[index].close)
                if earlier > 0:
                    sector_returns[sector_horizon].setdefault(
                        member_rows[index].date, []
                    ).append(current / earlier - 1.0)
    sector_medians_by_horizon = {
        sector_horizon: {
            day: float(statistics.median(values))
            for day, values in dated_values.items()
            if values
        }
        for sector_horizon, dated_values in sector_returns.items()
    }
    benchmark_by_date = {row.date: row for row in benchmark_rows}

    last_buy_index = split_index - max(horizons)
    last_sell_index = split_index - max(horizons)
    for index in range(split_index, len(rows) - max(horizons) - 1):
        if index - last_buy_index >= max(horizons):
            prefix = rows[: index + 1]
            indicators = _indicator_snapshot(prefix)
            smc = _smc_snapshot(prefix)
            market = _market_context_at_date(benchmark_rows, rows[index].date)
            sector: dict[str, Any] | None = None
            sector_relative_strength: dict[str, float] = {}
            for horizon in (20, 60, 120):
                sector_return = sector_medians_by_horizon[horizon].get(
                    rows[index].date
                )
                market_return = market.get(f"return_{horizon}d")
                if sector_return is not None and market_return is not None:
                    sector_relative_strength[f"relative_strength_vs_nepse_{horizon}d"] = (
                        sector_return - float(market_return)
                    )
            if sector_relative_strength:
                sector = sector_relative_strength
            score = _score_signal(indicators, smc, market, sector)
            buy_score = int(score["buy_score"])
            sell_score = int(score["sell_score"])
            margin = int(score["score_margin"])
            buy_gates = (
                market.get("regime") in ("bullish", "sideways")
                and sector is not None
                and float(sector["relative_strength_vs_nepse_20d"]) >= 0
                and indicators.get("sma_50") is not None
                and indicators["close"] > indicators["sma_50"]
            )
            if buy_score >= 65 and margin >= 25 and buy_gates:
                entry_open = rows[index + 1].open
                if entry_open is not None and float(entry_open) > 0:
                    last_buy_index = index
                    buy_dates.append(rows[index].date.isoformat())
                    entry_fill = float(entry_open) * (1.0 + SLIPPAGE_BPS / 10_000.0)
                    benchmark_entry = benchmark_by_date.get(rows[index + 1].date)
                    for horizon in horizons:
                        exit_close = float(rows[index + horizon].close)
                        exit_fill = exit_close * (1.0 - SLIPPAGE_BPS / 10_000.0)
                        gross = exit_fill / entry_fill - 1.0
                        net = (1.0 + gross) * ((1.0 - TRADING_COST_RATE) ** 2) - 1.0
                        buy_results[str(horizon)].append(net)
                        benchmark_exit = benchmark_by_date.get(rows[index + horizon].date)
                        if (
                            benchmark_entry is not None
                            and benchmark_entry.open is not None
                            and benchmark_entry.open > 0
                            and benchmark_exit is not None
                        ):
                            benchmark_return = (
                                float(benchmark_exit.close) / float(benchmark_entry.open) - 1.0
                            )
                            benchmark_results[str(horizon)].append(benchmark_return)
                            excess_results[str(horizon)].append(net - benchmark_return)

            sell_price_confirmation = bool(
                indicators.get("breakdown_20d")
                or (
                    indicators.get("sma_20") is not None
                    and indicators["close"] < indicators["sma_20"]
                    and indicators.get("volume_ratio_20d") is not None
                    and indicators["volume_ratio_20d"] >= 1.2
                )
            )
            if (
                index - last_sell_index >= max(horizons)
                and sell_score >= 65
                and -margin >= 25
                and sell_price_confirmation
            ):
                entry_open = rows[index + 1].open
                if entry_open is not None and float(entry_open) > 0:
                    last_sell_index = index
                    sell_dates.append(rows[index].date.isoformat())
                    for horizon in horizons:
                        exit_close = float(rows[index + horizon].close)
                        sell_following[str(horizon)].append(
                            exit_close / float(entry_open) - 1.0
                        )

    results: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        buy_values = buy_results[key]
        benchmark_values = benchmark_results[key]
        excess_values = excess_results[key]
        sell_values = sell_following[key]
        results[f"forward_{horizon}d"] = {
            "buy_sample_count": len(buy_values),
            "buy_win_rate": (
                sum(value > 0 for value in buy_values) / len(buy_values)
                if buy_values else None
            ),
            "buy_mean_net_return": (
                float(statistics.mean(buy_values)) if buy_values else None
            ),
            "buy_median_net_return": (
                float(statistics.median(buy_values)) if buy_values else None
            ),
            "nepse_benchmark_sample_count": len(benchmark_values),
            "nepse_mean_return": (
                float(statistics.mean(benchmark_values)) if benchmark_values else None
            ),
            "mean_excess_vs_nepse": (
                float(statistics.mean(excess_values)) if excess_values else None
            ),
            "sell_sample_count": len(sell_values),
            "negative_return_share_after_sell": (
                sum(value < 0 for value in sell_values) / len(sell_values)
                if sell_values else None
            ),
            "mean_forward_return_after_sell": (
                float(statistics.mean(sell_values)) if sell_values else None
            ),
        }

    return {
        "status": "evaluated",
        "scoring_model_version": "sector-stock-score-v1",
        "validation_method": "chronological holdout; first 70% warm-up/development, final 30% evaluated",
        "training_observations_for_warmup": split_index,
        "test_observations": len(rows) - split_index,
        "test_start_date": rows[split_index].date.isoformat(),
        "test_end_date": rows[-1].date.isoformat(),
        "buy_trigger_count": len(buy_dates),
        "sell_trigger_count": len(sell_dates),
        "buy_first_trigger_date": buy_dates[0] if buy_dates else None,
        "buy_last_trigger_date": buy_dates[-1] if buy_dates else None,
        "sell_first_trigger_date": sell_dates[0] if sell_dates else None,
        "sell_last_trigger_date": sell_dates[-1] if sell_dates else None,
        "assumptions": {
            "initial_signal": "completed daily close",
            "entry": "next recorded session open",
            "buy_exit": "fixed horizon close (5, 10 or 20 sessions from signal date)",
            "one_way_commission_rate": TRADING_COST_RATE,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "short_selling": False,
            "corporate_action_adjusted": False,
            "sector_return": "same-date median member-stock 20-session raw price return",
            "nepse_index_role": "benchmark only; no index BUY/SELL orders",
        },
        "score_thresholds": {"minimum_buy_score": 65, "minimum_sell_score": 65, "minimum_margin": 25},
        "results": results,
        "warning": (
            "Research holdout only, not live profitability proof. It assumes the source's raw "
            "prices; corporate-action adjustment and survivorship-bias controls remain required."
        ),
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
        history_map: dict[int, list[Price]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        stocks = eligible_stocks or self.session.scalars(
            sa.select(Stock).where(
                Stock.instrument_type == "ordinary",
                Stock.is_active.is_(True),
                Stock.sector.is_not(None),
            )
        ).all()
        # Keep enough observations for 3- to 10-month relative-strength features.
        histories = history_map if history_map is not None else self._load_histories(list(stocks), 260)
        grouped: dict[str, list[float]] = {}
        grouped60: dict[str, list[float]] = {}
        grouped120: dict[str, list[float]] = {}
        grouped200: dict[str, list[float]] = {}
        for stock in stocks:
            history = histories.get(stock.id, [])
            clean, event = _clean_after_latest_discontinuity(history)
            if event["flagged"]:
                continue
            metrics = _indicator_snapshot(clean) if len(clean) >= 21 else {}
            ret20 = metrics.get("return_20d")
            ret60 = metrics.get("return_60d")
            ret120 = metrics.get("return_120d")
            ret200 = metrics.get("return_200d")
            if ret20 is not None:
                grouped.setdefault(str(stock.sector), []).append(float(ret20))
            if ret60 is not None:
                grouped60.setdefault(str(stock.sector), []).append(float(ret60))
            if ret120 is not None:
                grouped120.setdefault(str(stock.sector), []).append(float(ret120))
            if ret200 is not None:
                grouped200.setdefault(str(stock.sector), []).append(float(ret200))

        market = market_context if market_context is not None else _market_regime(self.session)
        market_returns = {
            "20d": market.get("return_20d"),
            "60d": market.get("return_60d"),
            "120d": market.get("return_120d"),
            "200d": market.get("return_200d"),
        }
        result: dict[str, dict[str, Any]] = {}
        for sector, values in grouped.items():
            med = float(statistics.median(values))
            med60 = float(statistics.median(grouped60[sector])) if grouped60.get(sector) else None
            med120 = float(statistics.median(grouped120[sector])) if grouped120.get(sector) else None
            med200 = float(statistics.median(grouped200[sector])) if grouped200.get(sector) else None
            result[sector] = {
                "sector": sector,
                "construction": "median member-stock raw price returns (not an official sector index)",
                "return_20d": med,
                "return_60d": med60,
                "return_120d": med120,
                "return_200d": med200,
                "relative_strength_vs_nepse_20d": (
                    med - float(market_returns["20d"])
                    if market_returns["20d"] is not None else None
                ),
                "relative_strength_vs_nepse_60d": (
                    med60 - float(market_returns["60d"])
                    if med60 is not None and market_returns["60d"] is not None else None
                ),
                "relative_strength_vs_nepse_120d": (
                    med120 - float(market_returns["120d"])
                    if med120 is not None and market_returns["120d"] is not None else None
                ),
                "relative_strength_vs_nepse_200d": (
                    med200 - float(market_returns["200d"])
                    if med200 is not None and market_returns["200d"] is not None else None
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
        sector_price_histories: list[list[Price]] | None = None,
        include_historical_validation: bool = True,
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
        nepal_today = datetime.now(ZoneInfo("Asia/Kathmandu")).date()
        latest_age_days = max((nepal_today - clean[-1].date).days, 0)
        data_freshness = "current" if latest_age_days <= 7 else "stale"

        trigger = "HOLD"
        trigger_reason = "Neither directional score reached the trigger threshold."
        buy_gates = {
            "market_history_available": market.get("regime") != "unavailable",
            "market_not_bearish": market.get("regime") in ("bullish", "sideways"),
            "sector_history_available": sector_rs is not None,
            "sector_not_underperforming": sector_rs is not None and sector_rs >= 0,
            "at_least_50_clean_sessions": len(clean) >= 50,
            "at_least_200_clean_sessions_for_medium_term_momentum": len(clean) >= 200,
            "no_recent_price_discontinuity": (
                not discontinuity["flagged"] or len(clean) >= 200
            ),
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
            and len(clean) >= 200
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
            benchmark_rows = list(self.session.scalars(
                sa.select(NEPSEIndex)
                .where(
                    NEPSEIndex.date >= clean[0].date,
                    NEPSEIndex.date <= clean[-1].date,
                )
                .order_by(NEPSEIndex.date)
            ).all())
        if include_historical_validation:
            if sector_price_histories is None and stock.sector:
                sector_stocks = self.session.scalars(
                    sa.select(Stock).where(
                        Stock.instrument_type == "ordinary",
                        Stock.is_active.is_(True),
                        Stock.sector == stock.sector,
                    )
                ).all()
                sector_history_map = self._load_histories(list(sector_stocks), 800)
                sector_price_histories = [
                    sector_history_map.get(member.id, [])
                    for member in sector_stocks
                    if member.id != stock.id
                ]
            score_validation = _historical_score_validation(
                clean,
                list(benchmark_rows or []),
                sector_price_histories or [],
            )
        else:
            score_validation = {
                "status": "not_run_for_bulk_scan",
                "reason": "Run stock detail to calculate per-stock chronological holdout validation",
            }

        atr = indicators["atr_14"]
        last_close = float(indicators["close"])
        if atr is None or atr <= 0:
            suggested_stop = None
            suggested_target = None
        elif trigger == "SELL_TRIGGER":
            # SELL is a reduce/exit review for an existing long; never a short-sale order.
            suggested_stop = round(last_close + 2.0 * atr, 2)
            suggested_target = round(last_close - 4.0 * atr, 2)
        elif trigger == "BUY_TRIGGER":
            suggested_stop = round(last_close - 2.0 * atr, 2)
            suggested_target = round(last_close + 4.0 * atr, 2)
        else:
            suggested_stop = None
            suggested_target = None
        high60 = indicators.get("high_60d")
        low60 = indicators.get("low_60d")
        high60_value = float(high60) if isinstance(high60, (int, float)) else None
        low60_value = float(low60) if isinstance(low60, (int, float)) else None
        fib_range = (
            high60_value - low60_value
            if high60_value is not None and low60_value is not None
            else None
        )
        fibonacci_levels = (
            {
                "retracement_38_2": round(high60_value - 0.382 * fib_range, 2),
                "retracement_50_0": round(high60_value - 0.5 * fib_range, 2),
                "retracement_61_8": round(high60_value - 0.618 * fib_range, 2),
            }
            if fib_range is not None and fib_range > 0
            else {}
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
                "high_60d": high60,
                "low_60d": low60,
                "fibonacci_retracement_levels": fibonacci_levels,
            },
            "reference_stop_level": suggested_stop,
            "reference_target_level": suggested_target,
            "reference_risk_reward_ratio": 2.0 if suggested_stop is not None else None,
            "price_history_for_chart": _chart_series(clean, smc),
            "data_freshness": {
                "status": data_freshness,
                "latest_session": clean[-1].date.isoformat(),
                "calendar_days_old": latest_age_days,
                "max_age_calendar_days_for_new_trigger": 7,
            },
            "historical_event_study": _historical_event_study(clean, list(benchmark_rows or [])),
            "historical_score_validation": score_validation,
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
        market = _market_regime(self.session)
        # A bulk scan needs 200-session trend/momentum plus a 220-bar chart;
        # reuse the same bounded batch for sector and individual-stock calculations.
        histories = self._load_histories(list(stocks), 260)
        sectors = self._sector_summary(
            list(stocks),
            market_context=market,
            history_map=histories,
        )
        benchmark_rows = self.session.scalars(
            sa.select(NEPSEIndex).order_by(NEPSEIndex.date.desc()).limit(800)
        ).all()
        benchmark_rows = list(reversed(benchmark_rows))
        results: list[dict[str, Any]] = []
        for stock in stocks:
            try:
                analysis = self.analyze_stock(
                    stock.symbol,
                    sector_summaries=sectors,
                    history=histories.get(stock.id, []),
                    stock_obj=stock,
                    market_context=market,
                    benchmark_rows=benchmark_rows,
                    include_historical_validation=False,
                )
                analysis["sector_context"] = sectors.get(str(stock.sector)) if stock.sector else None
                results.append(analysis)
            except ValueError:
                continue
        priority = {"BUY_TRIGGER": 0, "SELL_TRIGGER": 1, "SETUP": 2, "WATCH": 3, "HOLD": 4}
        results.sort(key=lambda item: (priority.get(item["signal"], 9), -item["confidence_score"]))
        return {
            "as_of": max((item["as_of"] for item in results), default=None),
            "market_context": market,
            "stocks_analyzed": len(results),
            "signals": results[:limit],
            "data_disclaimer": RESEARCH_SOURCE_WARNING,
        }

    def sector_ranking(self) -> dict[str, Any]:
        market = _market_regime(self.session)
        sectors = list(self._sector_summary(market_context=market).values())
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
