# NEPSE market, sector and stock-research model

## Purpose and boundaries

The system has three distinct research layers:

1. **NEPSE Index:** market-regime filter and benchmark only. The index is not a tradable stock in this application and never receives BUY or SELL orders.
2. **Sector:** relative strength and breadth calculated from listed ordinary-share constituents available in the dataset. This is a research basket, not an official NEPSE sector index.
3. **Stock:** individual ordinary-share trigger, score, chart, risk references and historical evidence.

The system is research-only. A BUY trigger is a watchlist/research signal; a SELL trigger is a review or exit alert for an existing long holding. It does not short-sell and it does not route broker orders.

## Directional score model (100 evidence points)

Every stock receives two separate scores: BUY evidence and SELL evidence. These are **weighted rule points, not probabilities, expected returns, or win probabilities**. Missing information receives no points; it is not silently imputed as bullish or bearish.

| Component | Weight | BUY-side evidence | SELL-side evidence |
|---|---:|---|---|
| NEPSE regime | 10 | Index close above its 50-session mean, positive 20-session return, and supportive trend | Corresponding bearish regime |
| Sector relative strength | 10 | Sector median 20/60/120-session returns outperform NEPSE over at least two available horizons | Sector median underperforms NEPSE over at least two available horizons |
| Trend structure | 15 | Close > SMA20 > SMA50; SMA200 and ADX strengthen confirmation | Close < SMA20 < SMA50; SMA200 and ADX strengthen confirmation |
| Medium-term momentum | 10 | Positive stock returns over at least two of 60/120/200 sessions | Negative stock returns over at least two of 60/120/200 sessions |
| RSI / Stochastic momentum | 10 | RSI 55–70 with %K not below %D | RSI below 40 with %K below %D; overbought conditions can add limited risk points |
| MACD | 10 | MACD above signal with non-deteriorating histogram | MACD below signal with non-improving histogram |
| Volume / OBV | 10 | Positive session with volume expansion; rising OBV is additional confirmation | Negative session with volume expansion; falling OBV is additional confirmation |
| Price action | 15 | 20-session breakout, bullish engulfing or hammer | 20-session breakdown, bearish engulfing or shooting star |
| SMC rule proxies | 10 | Bullish structure break, liquidity sweep, fair-value-gap or order-block retest proxy | Bearish equivalents |

The code keeps the total weights at 100. The SMC weighting is deliberately smaller than the more observable trend, multi-horizon momentum, market, sector, volume and price-action evidence. Sector and index relative strength use 20-, 60- and 120-session horizons; stock medium-term momentum uses 60-, 120- and 200-session returns.

### Trigger rules

- **BUY_TRIGGER:** BUY score at least 65/100, BUY minus SELL margin at least 25 points, and all confirmation gates pass: NEPSE regime available and not bearish, sector relative strength available and non-negative, at least 50 clean stock sessions, acceptable discontinuity history, and latest price data within seven calendar days.
- **SELL_TRIGGER:** SELL score at least 65/100, SELL advantage at least 25 points, bearish price/volume confirmation, and current data. This is an exit/review alert for an existing holding, not a short-sale instruction.
- **WATCH:** directional evidence is developing but the threshold or a required gate is not met.
- **HOLD:** neither side has sufficient evidence.
- **STALE_DATA:** current triggers are suppressed when the latest clean stock price observation is more than seven calendar days old.

A score of 80 does not mean an 80% chance of profit. Signal strength is only the sum of the defined evidence points. The model version is sector-stock-score-v1.

## Indicators and chart

The stock-detail chart contains daily candlesticks, volume, SMA20/50/200, EMA9/21, Bollinger Bands, MACD(12,26,9) and signal line with histogram, RSI(14), Stochastic %K(14)/%D(3), ADX(14) with +DI/-DI, OBV, and candle-pattern markers. RSI and the ATR/ADX directional-movement smoothers use Wilder's recursive moving average; ATR(14) is used for illustrative stop/target distances. Bollinger Bands use a 20-session simple moving average and two population standard deviations. Price references include recent support/resistance and Fibonacci retracement levels.

The displayed price/volume and indicator values are derived from completed daily bars. Signals are evaluated after the close; any simulated or paper fill must use a later recorded session. The dashboard does not imply an intraday entry at a price that was not available at the time.

## Smart Money Concepts (SMC): explicit proxy definitions

SMC terms are not directly observable from daily OHLCV alone. In this system they are deterministic, reproducible proxies—not evidence of actual institutional orders, hidden liquidity, or bank activity.

- **Swing high/low:** a local pivot that is only considered confirmed after two later bars.
- **Break of structure (BOS):** a completed close crosses the most recent confirmed swing level.
- **Liquidity sweep proxy:** the day trades beyond a prior 20-session extreme and then closes back through that level, with close location in the relevant half of the candle.
- **Fair-value-gap proxy:** a three-candle non-overlap, using the current low versus the high two bars earlier (bullish) or current high versus the low two bars earlier (bearish).
- **Order-block proxy:** the last opposite-colour candle in the recent lookback before a defined structure break; a retest/rejection is required before it contributes score points.

These definitions are intentionally transparent and testable. They are not a claim that a chart can identify institutional orders. SMC points are capped at 10 out of 100 and must be independently validated against the simpler baseline.

## Historical validation and benchmarking

The per-stock historical score validation uses a chronological 70/30 split: earlier observations are used for indicator warm-up and model development context; the final 30% is the holdout. At each historical signal date, features use only that date and earlier bars. Simulated entries occur at the next recorded session's open, with explicit illustrative commission and slippage. BUY outcomes are compared with the NEPSE Index over 5-, 10- and 20-session horizons. The index is used only as a benchmark.

This is a first-stage holdout, not a final research-grade proof of profitability. A production-grade validation programme should add rolling/anchored walk-forward folds, purged/embargoed sampling for overlapping outcomes, parameter stability, liquidity/impact constraints, universe point-in-time membership, corporate-action adjustment, survivorship-bias controls, and multiple-testing corrections.

The score thresholds and weights are fixed rules, not trained parameters. They must not be optimised repeatedly on the final holdout. Performance reports should disclose sample sizes, missing benchmark matches, gross and net returns, win rates, median outcomes, drawdowns, transaction assumptions and the evaluation window.

## Evidence base and limitations

The core trend/momentum premise has stronger published evidence than any one chart pattern:

- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers*, **The Journal of Finance**, 48(1), 65–91. The paper documents stock momentum over 3–12 month formation/holding periods. DOI: https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
- Moskowitz, Ooi and Pedersen (2012), *Time Series Momentum*, **Journal of Financial Economics**, 104(2), 228–250. The study documents 1–12 month time-series momentum across 58 liquid futures/forward markets. DOI: https://doi.org/10.1016/j.jfineco.2011.11.003

These studies motivate trend and momentum as research features; they do **not** validate these exact thresholds or guarantee that the same returns exist in NEPSE, particularly after Nepal-specific liquidity, circuit-breaker, price-limit, settlement, fee and impact constraints.

Technical rules require stringent validation. Sullivan, Timmermann and White (1999), *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, **The Journal of Finance**, 54(5), 1647–1691, explicitly addresses data-snooping when testing large universes of technical rules. DOI: https://doi.org/10.1111/0022-1082.00163. A 2023 study of developed and emerging equity markets reports that technical-rule predictability diminished out of sample and was sensitive to transaction costs: https://doi.org/10.1007/s11408-023-00433-2.

RSI, MACD, moving averages, Bollinger Bands, ADX, Stochastic and candle patterns are complementary measurements—not independent proofs. They are correlated transformations of price/volume history. The score therefore must be validated as a combined model and by component ablation, not by adding up indicator labels and assuming each is a separate edge.

No generally accepted academic evidence establishes SMC concepts as a universally profitable algorithm. The SMC rules here are exploratory OHLC-based proxies and receive a deliberately limited score weight. Their incremental contribution must be tested against a baseline with SMC disabled.

## Data constraints

The public historical dataset describes its prices as **raw and not adjusted for bonus shares, rights issues or stock splits**. Large discontinuities are flagged and the clean-indicator window is restarted after the most recent flagged jump; this is a protective heuristic, not a substitute for authoritative corporate-action adjustment. The data is independent third-party material, not an official NEPSE feed. Source attribution must be retained.

Sources and dataset limitations are displayed in the dashboard/API. No trigger should be treated as current if the feed is stale, a symbol is not an active ordinary share, or the price history fails quality checks.
