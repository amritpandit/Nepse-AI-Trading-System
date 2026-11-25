# NEPSE Market, Sector and Stock Research Model

## Purpose and scope

The NEPSE Index is a **market-regime benchmark only**. It is not a tradable stock and never receives BUY/SELL instructions. Sector analysis is a cross-sectional ranking of eligible listed ordinary shares. Individual stock triggers are advisory research outputs and do not route broker orders.

Only active listed **ordinary shares** are eligible for stock BUY/SELL triggers. Debentures, promoter shares, merged/inactive listings and unknown instrument types are excluded.

## Current trigger score

Each side receives a separate evidence score from 0 to 100. These are **points, not probabilities, target returns or calibrated confidence levels**.

| Evidence group | Maximum points | Rule family |
|---|---:|---|
| NEPSE market regime | 10 | 50/200-session trend and 20-session direction |
| Sector relative strength | 10 | Sector median 20-session return less NEPSE 20-session return |
| Stock trend structure | 20 | Close vs SMA20, SMA20 vs SMA50, SMA50 vs SMA200 |
| RSI momentum | 10 | RSI(14) momentum band and overbought/weakness penalty |
| MACD | 10 | MACD(12,26,9) line/signal and histogram direction |
| Volume confirmation | 10 | Current volume vs 20-session average and daily direction |
| Price action | 15 | 20-session breakout/breakdown; engulfing/hammer/shooting-star proxies |
| SMC-style structure | 15 | Confirmed swing BOS, liquidity sweep, 3-candle FVG and order-block retest proxies |
| **Total** | **100** | |

### Trigger rules

- **BUY_TRIGGER:** BUY score at least 65, BUY-minus-SELL margin at least 25, with market non-bearish, sector not underperforming, at least 50 clean observations, recent data and acceptable corporate-action history.
- **SELL_TRIGGER:** SELL score at least 65, SELL-minus-BUY margin at least 25, plus bearish price/volume confirmation. It means review/exit an existing holding; it is not a short-sale signal.
- **WATCH:** meaningful directional evidence but a score threshold or confirmation gate is missing.
- **HOLD:** neither directional score is strong enough.
- **STALE_DATA:** the last usable completed daily bar is more than seven calendar days old; new BUY/SELL triggers are suppressed.

A score can be high while a trigger remains WATCH because a required market, sector, history or data-quality gate failed. Missing information is not treated as positive evidence.

## Indicators and chart overlays

The stock detail chart includes daily OHLC candlesticks, SMA20/50/200, EMA9/21, Bollinger Bands (20, 2 standard deviations), volume, MACD(12,26,9), and RSI(14). The API also returns ATR(14), 20-session relative volume, average traded value, and 20/60-session price returns. A reference stop level is based on two ATRs below the latest close; a reference target is four ATRs above it. These are reference levels, not guaranteed fills.

## SMC-style rules

SMC is not a single standardized exchange data field. This implementation translates commonly used chart concepts into transparent daily-OHLC rules:
- **Confirmed swing:** a local high/low with two subsequent bars on both sides; the two later bars are needed to avoid using an unconfirmed pivot.
- **Break of structure (BOS):** a completed close crosses the most recently confirmed swing high or low.
- **Liquidity sweep:** price trades beyond the preceding 20-session high/low and closes back inside the range in the corresponding candle half.
- **Fair value gap proxy:** a three-candle high/low gap.
- **Order-block proxy:** the last opposite-colour candle before a structure break; a retest/rejection is required for points.

These OHLC proxies do **not** observe actual order-book depth, institutional orders, broker flow, or true liquidity pools. SMC contributes at most 15 points and should be evaluated separately in the NEPSE holdout before it is relied upon.

## Historical validation

The per-stock historical score validation:
1. Uses the first 70% of clean observations as warm-up/development history.
2. Evaluates the final 30% chronologically, with no shuffling.
3. Calculates each signal using only price, benchmark and sector observations available up to that signal date.
4. Requires the same score thresholds and market/sector confirmation used by the current signal.
5. Assumes stock entry at the next recorded session's open; BUY outcome is measured at 5, 10 and 20-session close horizons.
6. Applies an illustrative 0.5% one-way commission and 5 basis points slippage on each stock side.
7. Compares matched NEPSE Index returns as a gross benchmark; the index is not assumed tradable.
8. Separates SELL-trigger forward returns as a risk-warning study; it does not simulate short sales.

The separate 20-session breakout event study is a simpler baseline. It is **not** evidence that the full weighted score is profitable. Both are historical research, not a promise of future returns.

## Data quality controls and known limitations

The selected public historical dataset states that its stock prices are raw and are **not adjusted** for bonus shares, rights issues or splits. A large discontinuity can therefore be a corporate action rather than genuine loss/gain. The system flags price jumps above 15% and only scores the subsequent clean history; a BUY requires 200 clean observations after a detected discontinuity. Corporate-action events should be sourced and adjusted before relying on long-period compounded returns.

Sector returns are the median constituent-stock returns, not official NEPSE capitalization-weighted sector indices. Only ordinary listed shares are included. Delisted/merged survivorship bias, symbol changes, thin trading, stale quotes, missing sessions and data-source revisions can still affect results.

Costs in the research engine are configurable assumptions, not a claim that every NEPSE investor pays the same charge. A production-quality portfolio backtest should separately parameterize brokerage commission by turnover band, SEBON/DP charges, capital-gains tax, market impact, price limits, circuit halts and partial fills using verified current rules.

## Research references

- Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following Investing*, Journal of Portfolio Management. Long historical evidence across asset classes supports studying trend-following, but it is not a direct validation of this long-only NEPSE stock score: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026
- Psaradellis, Laws, Pantelous & Sermpinis (2023), *Technical Analysis, Spread Trading, and Data Snooping Control*, International Journal of Forecasting. Uses a large rule universe and false-discovery controls: https://doi.org/10.1016/j.ijforecast.2021.10.002
- Rink (2023), *The Predictive Ability of Technical Trading Rules: An Empirical Analysis of Developed and Emerging Equity Markets*. Reports sensitivity to transaction costs and weak persistence of selected rules out of sample: https://doi.org/10.1007/s11408-023-00433-2

The implementation therefore favors transparent scoring, costs, chronological holdouts and disclosed uncertainty over claims that an indicator bundle is universally proven.
