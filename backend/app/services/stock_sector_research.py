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
    return {
        "close": float(close[-1]),
        "sma_20": float(sma20[-1]) if np.isfinite(sma20[-1]) else None,
        "sma_50": float(sma50[-1]) if np.isfinite(sma50[-1]) else None,
        "sma_200": float(sma200[-1]) if np.isfinite(sma200[-1]) else None,
        "rsi_14": float(rsi[-1]) if np.isfinite(rsi[-1]) else None,
        "macd": float(macd[-1]) if np.isfinite(macd[-1]) else None,
        "macd_signal": float(macd_signal[-1]) if np.isfinite(macd_signal[-1]) else None,
        "atr_14": float(atr14[-1]) if np.isfinite(atr14[-1]) else None,
        "volume_ratio_20d": volume_ratio,
        "previous_high_20d": previous_high20,
        "previous_low_20d": previous_low20,
        "return_20d": float(close[-1] / close[-21] - 1.0) if close.size >= 21 else None,
        "return_60d": float(close[-1] / close[-61] - 1.0) if close.size >= 61 else None,
        "bars": int(close.size),
    }


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


def _historical_event_study(rows: list[Price]) -> dict[str, Any]:
    """Evaluate breakout entries at next open with 5/10/20-session outcomes."""
    if len(rows) < 75:
        return {"sample_count": 0, "reason": "At least 75 clean observations required"}
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    highs = np.asarray([float(row.high) for row in rows], dtype=float)
    volumes = np.asarray([float(row.volume or 0) for row in rows], dtype=float)
    horizons = (5, 10, 20)
    outcomes: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
    trigger_dates: list[str] = []
    for index in range(50, len(rows) - max(horizons) - 1):
        if not _buy_trigger_at(index, closes, highs, volumes):
            continue
        entry_open = rows[index + 1].open
        if entry_open is None or float(entry_open) <= 0:
            continue
        trigger_dates.append(rows[index].date.isoformat())
        entry_price = float(entry_open) * (1.0 + SLIPPAGE_BPS / 10_000.0)
        for horizon in horizons:
            exit_close = float(rows[index + horizon].close)
            gross_return = exit_close / entry_price - 1.0
            net_return = gross_return - (2.0 * TRADING_COST_RATE) - (2.0 * SLIPPAGE_BPS / 10_000.0)
            outcomes[str(horizon)].append(net_return)

    summary: dict[str, Any] = {}
    for horizon, values in outcomes.items():
        summary[f"forward_{horizon}d"] = {
            "sample_count": len(values),
            "win_rate": (
                sum(1 for value in values if value > 0) / len(values) if values else None
            ),
            "mean_net_return": float(statistics.mean(values)) if values else None,
            "median_net_return": float(statistics.median(values)) if values else None,
            "best_net_return": max(values) if values else None,
            "worst_net_return": min(values) if values else None,
        }
    return {
        "method": "breakout event study; entry at next session open; fixed-horizon close exits",
        "trigger_count": len(trigger_dates),
        "first_trigger_date": trigger_dates[0] if trigger_dates else None,
        "last_trigger_date": trigger_dates[-1] if trigger_dates else None,
        "assumptions": {
            "one_way_commission_rate": TRADING_COST_RATE,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "corporate_action_adjusted": False,
        },
        **summary,
    }


class StockSectorResearchService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _load_stock_history(self, stock: Stock, limit: int = 800) -> list[Price]:
        rows = self.session.scalars(
            sa.select(Price)
            .where(Price.stock_id == stock.id)
            .order_by(Price.date.desc())
            .limit(limit)
        ).all()
        return _valid_history(list(reversed(rows)))

    def _sector_summary(self) -> dict[str, dict[str, Any]]:
        stocks = self.session.scalars(
            sa.select(Stock).where(
                Stock.instrument_type == "ordinary",
                Stock.is_active.is_(True),
                Stock.sector.is_not(None),
            )
        ).all()
        grouped: dict[str, list[float]] = {}
        grouped60: dict[str, list[float]] = {}
        for stock in stocks:
            history = self._load_stock_history(stock, limit=70)
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

        market = _market_regime(self.session)
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

    def analyze_stock(self, symbol: str) -> dict[str, Any]:
        stock = self.session.scalar(
            sa.select(Stock).where(Stock.symbol == symbol.strip().upper())
        )
        if stock is None:
            raise ValueError(f"Stock {symbol} not found")
        if stock.instrument_type != "ordinary" or not stock.is_active:
            raise ValueError("Only active listed ordinary shares are eligible for stock triggers")

        history = self._load_stock_history(stock)
        clean, discontinuity = _clean_after_latest_discontinuity(history)
        if not clean:
            raise ValueError("No valid OHLCV history is available for this stock")

        indicators = _indicator_snapshot(clean)
        market = _market_regime(self.session)
        sectors = self._sector_summary()
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

        breakout = False
        if indicators["previous_high_20d"] is not None and indicators["sma_50"] is not None:
            breakout = (
                indicators["close"] > indicators["previous_high_20d"]
                and (indicators["volume_ratio_20d"] or 0) >= 1.5
                and indicators["close"] > indicators["sma_50"]
            )
        sma20 = indicators["sma_20"]
        rsi = indicators["rsi_14"]
        macd = indicators["macd"]
        macd_signal = indicators["macd_signal"]
        sell_trigger = (
            sma20 is not None
            and indicators["close"] < sma20
            and indicators["volume_ratio_20d"] is not None
            and indicators["volume_ratio_20d"] >= 1.2
        )

        trigger = "HOLD"
        trigger_reason = "No confirmed breakout or breakdown trigger on the latest completed session."
        if sell_trigger:
            trigger = "SELL_TRIGGER"
            trigger_reason = "Close below SMA20 with elevated trading volume; review an existing holding."
        elif breakout:
            if market["regime"] == "bearish":
                trigger = "WATCH"
                trigger_reason = "Stock breakout is present, but the NEPSE market regime is bearish."
            elif sector_rs is not None and sector_rs < 0:
                trigger = "WATCH"
                trigger_reason = "Stock breakout is present, but its sector is underperforming NEPSE."
            elif market["regime"] == "unavailable" or sector_rs is None:
                trigger = "SETUP"
                trigger_reason = "Stock breakout is present; market or sector confirmation is missing."
            elif len(clean) < 50:
                trigger = "SETUP"
                trigger_reason = "Stock breakout is present, but clean history is insufficient."
            else:
                trigger = "BUY_TRIGGER"
                trigger_reason = "20-session breakout with volume expansion and market/sector confirmation."

        confirmations = {
            "price_above_sma20": sma20 is not None and indicators["close"] > sma20,
            "sma20_above_sma50": sma20 is not None and indicators["sma_50"] is not None and sma20 > indicators["sma_50"],
            "rsi_positive_momentum": rsi is not None and 50 <= rsi <= 70,
            "macd_above_signal": macd is not None and macd_signal is not None and macd > macd_signal,
            "sector_outperforming": sector_rs is not None and sector_rs > 0,
            "market_not_bearish": market["regime"] in ("bullish", "sideways"),
        }
        score = sum(1 for value in confirmations.values() if value)
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
            "confidence_score": round(score / len(confirmations), 4),
            "confirmation_checks": confirmations,
            "market_context": market,
            "sector_context": sector,
            "indicators": indicators,
            "reference_stop_level": suggested_stop,
            "historical_event_study": _historical_event_study(clean),
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
        sectors = self._sector_summary()
        results: list[dict[str, Any]] = []
        for stock in stocks:
            try:
                analysis = self.analyze_stock(stock.symbol)
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
