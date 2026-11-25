from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import requests  # type: ignore[import-untyped]
import streamlit as st
from plotly.subplots import make_subplots
from common import API_BASE, auth_headers, render_hero, render_table


def _get_json(path: str, params: dict | None = None) -> dict | None:
    try:
        response = requests.get(
            f"{API_BASE}{path}", params=params, headers=auth_headers(), timeout=45
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except requests.RequestException as exc:
        st.error(f"Research API request failed: {exc}")
        return None


def page_research() -> None:
    """Market regime, sector relative strength, stock triggers and history import."""
    render_hero(
        "Sector & Stock Research",
        "NEPSE is the benchmark. BUY/SELL triggers are produced only for eligible listed ordinary shares.",
    )
    st.info(
        "Research-only output. No NEPSE Index BUY/SELL orders are generated and no broker orders are placed."
    )

    tab_market, tab_sectors, tab_stocks, tab_detail, tab_history, tab_data = st.tabs(
        ["Market Regime", "Sector Ranking", "Stock Signals", "Stock Detail & History", "Saved Signal Ledger", "Historical Data"]
    )

    with tab_market:
        if st.button("Refresh Market Regime", key="research_market_refresh"):
            data = _get_json("/research/market-regime")
            if data:
                c1, c2, c3 = st.columns(3)
                c1.metric("Market Regime", str(data.get("regime", "unavailable")).upper())
                c2.metric("NEPSE Index", data.get("index_close") or "—")
                return20 = data.get("return_20d")
                c3.metric("20-session return", f"{return20 * 100:.2f}%" if return20 is not None else "—")
                st.caption(
                    f"Benchmark date: {data.get('as_of') or 'not imported'} · "
                    "NEPSE is context only, not a tradable stock."
                )
                st.json(data)

    with tab_sectors:
        if st.button("Rank Sectors", key="research_sector_refresh"):
            data = _get_json("/research/sectors")
            if data:
                st.caption(f"Benchmark regime: {data.get('market_context', {}).get('regime', 'unavailable')}")
                sectors = data.get("sectors", [])
                if sectors:
                    frame = pd.DataFrame(sectors)
                    render_table(
                        frame,
                        key="research_sectors",
                        name="sector_strength",
                        height=420,
                        empty_msg="No sector history available yet.",
                    )
                else:
                    st.warning("No sector results. Import historical ordinary-share prices first.")
                st.caption(data.get("methodology_warning", ""))

    with tab_stocks:
        scan_limit = st.slider("Number of stocks to return", 20, 300, 100, 20, key="research_scan_limit")
        if st.button("Scan Stock BUY / SELL Triggers", key="research_scan_refresh"):
            data = _get_json("/research/signals", params={"limit": scan_limit})
            if data:
                st.write(
                    f"Stocks analyzed: {data.get('stocks_analyzed', 0)} · "
                    f"Data date: {data.get('as_of') or '—'} · "
                    f"Market regime: {data.get('market_context', {}).get('regime', 'unavailable')}"
                )
                signals = data.get("signals", [])
                if signals:
                    rows = [
                        {
                            "Symbol": item.get("symbol"),
                            "Company": item.get("name"),
                            "Sector": item.get("sector"),
                            "Signal": item.get("signal"),
                            "As of": item.get("as_of"),
                            "Score": item.get("confidence_score"),
                            "Reason": item.get("signal_reason"),
                            "Stop reference": item.get("reference_stop_level"),
                        }
                        for item in signals
                    ]
                    render_table(
                        pd.DataFrame(rows),
                        key="research_stock_signals",
                        name="stock_signals",
                        height=520,
                        empty_msg="No stock signals.",
                    )
                else:
                    st.warning("No eligible ordinary shares with usable history were found.")
                st.caption(data.get("data_disclaimer", ""))
        if st.button("Scan and Save Daily Signals", key="research_save_signals"):
            try:
                response = requests.post(
                    f"{API_BASE}/research/signals/refresh",
                    params={"limit": scan_limit},
                    headers=auth_headers(),
                    timeout=120,
                )
                response.raise_for_status()
                st.success("Daily research signal snapshots saved.")
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Could not save signal snapshots: {exc}")

    with tab_detail:
        symbol = st.text_input("Listed stock symbol", value="NABIL", key="research_detail_symbol").strip().upper()
        if st.button("Analyze Stock & Historical Triggers", key="research_detail_refresh"):
            data = _get_json(f"/research/stocks/{symbol}")
            if data:
                signal = str(data.get("signal", "HOLD"))
                if signal == "BUY_TRIGGER":
                    st.success(f"{symbol}: BUY TRIGGER")
                elif signal == "SELL_TRIGGER":
                    st.error(f"{symbol}: SELL TRIGGER")
                elif signal == "WATCH":
                    st.warning(f"{symbol}: WATCH")
                else:
                    st.info(f"{symbol}: {signal}")
                st.write(data.get("signal_reason", ""))
                score_buy, score_sell, score_margin = st.columns(3)
                score_buy.metric("BUY score / 100", data.get("buy_score", 0))
                score_sell.metric("SELL score / 100", data.get("sell_score", 0))
                score_margin.metric("BUY − SELL margin", data.get("score_margin", 0))
                st.caption("Scores are weighted evidence points out of 100, not probability of profit.")
                indicators_snapshot = data.get("indicators", {})
                risk_cols = st.columns(4)
                risk_cols[0].metric(
                    "Last completed close",
                    f"NPR {float(indicators_snapshot.get('close')):,.2f}"
                    if indicators_snapshot.get("close") is not None else "—",
                )
                risk_cols[1].metric(
                    "Reference stop",
                    f"NPR {float(data.get('reference_stop_level')):,.2f}"
                    if data.get("reference_stop_level") is not None else "—",
                )
                risk_cols[2].metric(
                    "Reference target",
                    f"NPR {float(data.get('reference_target_level')):,.2f}"
                    if data.get("reference_target_level") is not None else "—",
                )
                risk_cols[3].metric(
                    "Risk / reward",
                    f"1 : {float(data.get('reference_risk_reward_ratio')):.1f}"
                    if data.get("reference_risk_reward_ratio") is not None else "—",
                )
                st.caption(
                    "Stop/target are ATR-based research references for a confirmed trigger only. "
                    "They are not guaranteed fills or personalized investment advice."
                )
                score_model = data.get("score_model", {})
                components = score_model.get("components", {})
                if components:
                    st.subheader("Scoring evidence")
                    component_rows = []
                    for component, item in components.items():
                        component_rows.append(
                            {
                                "Factor": component.replace("_", " ").title(),
                                "BUY points": item.get("buy_points", item.get("weight") if item.get("side") == "buy" else 0),
                                "SELL points": item.get("sell_points", item.get("weight") if item.get("side") == "sell" else 0),
                                "Evidence": item.get("reason") or item.get("methodology") or item.get("relative_strength_20d") or item.get("rsi_14") or item.get("macd_histogram") or item.get("volume_ratio_20d") or "",
                            }
                        )
                    score_frame = pd.DataFrame(component_rows)
                    render_table(
                        score_frame,
                        key="research_score_components",
                        name=f"{symbol}_score_components",
                        height=300,
                    )
                    score_figure = go.Figure()
                    score_figure.add_trace(
                        go.Bar(
                            y=score_frame["Factor"],
                            x=score_frame["BUY points"],
                            orientation="h",
                            name="BUY evidence",
                        )
                    )
                    score_figure.add_trace(
                        go.Bar(
                            y=score_frame["Factor"],
                            x=-score_frame["SELL points"],
                            orientation="h",
                            name="SELL evidence (negative axis)",
                        )
                    )
                    score_figure.update_layout(
                        barmode="relative",
                        height=360,
                        title="Directional evidence by model component",
                        xaxis_title="SELL points ← 0 → BUY points",
                        yaxis_title="",
                        legend=dict(orientation="h", y=1.02, x=0),
                    )
                    st.plotly_chart(score_figure, use_container_width=True)
                    st.caption(
                        "Scores are weighted evidence points, not a calibrated probability or expected return."
                    )
                chart_rows = data.get("price_history_for_chart", [])
                if chart_rows:
                    st.subheader("Price action, indicators and SMC levels")
                    frame = pd.DataFrame(chart_rows)
                    fig = make_subplots(
                        rows=5,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.025,
                        row_heights=[0.50, 0.17, 0.11, 0.11, 0.11],
                        specs=[
                            [{"secondary_y": True}],
                            [{}],
                            [{}],
                            [{}],
                            [{"secondary_y": True}],
                        ],
                    )
                    fig.add_trace(
                        go.Candlestick(
                            x=frame["date"],
                            open=frame["open"],
                            high=frame["high"],
                            low=frame["low"],
                            close=frame["close"],
                            name="OHLC",
                        ),
                        row=1,
                        col=1,
                        secondary_y=False,
                    )
                    for column, label in (
                        ("sma_20", "SMA 20"),
                        ("sma_50", "SMA 50"),
                        ("sma_200", "SMA 200"),
                        ("ema_9", "EMA 9"),
                        ("ema_21", "EMA 21"),
                        ("bollinger_upper", "Bollinger upper"),
                        ("bollinger_lower", "Bollinger lower"),
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=frame["date"],
                                y=frame[column],
                                mode="lines",
                                name=label,
                            ),
                            row=1,
                            col=1,
                            secondary_y=False,
                        )
                    fig.add_trace(
                        go.Bar(
                            x=frame["date"],
                            y=frame["volume"],
                            name="Volume",
                            opacity=0.22,
                        ),
                        row=1,
                        col=1,
                        secondary_y=True,
                    )
                    fig.add_trace(
                        go.Bar(
                            x=frame["date"],
                            y=frame["macd_histogram"],
                            name="MACD histogram",
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["macd"], mode="lines", name="MACD"),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=frame["date"],
                            y=frame["macd_signal"],
                            mode="lines",
                            name="MACD signal",
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["rsi_14"], mode="lines", name="RSI 14"),
                        row=3,
                        col=1,
                    )
                    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["stoch_k_14"], mode="lines", name="Stochastic %K"),
                        row=4,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["stoch_d_3"], mode="lines", name="Stochastic %D"),
                        row=4,
                        col=1,
                    )
                    fig.add_hline(y=80, line_dash="dot", row=4, col=1)
                    fig.add_hline(y=20, line_dash="dot", row=4, col=1)
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["adx_14"], mode="lines", name="ADX 14"),
                        row=5,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["plus_di_14"], mode="lines", name="+DI"),
                        row=5,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["minus_di_14"], mode="lines", name="-DI"),
                        row=5,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=frame["date"], y=frame["obv"], mode="lines", name="OBV"),
                        row=5,
                        col=1,
                        secondary_y=True,
                    )
                    for flag, label, marker, column in (
                        ("bullish_engulfing", "Bullish engulfing", "triangle-up", "low"),
                        ("hammer", "Hammer", "triangle-up", "low"),
                        ("bearish_engulfing", "Bearish engulfing", "triangle-down", "high"),
                        ("shooting_star", "Shooting star", "triangle-down", "high"),
                    ):
                        marked = frame[frame[flag].fillna(False)]
                        if not marked.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=marked["date"],
                                    y=marked[column],
                                    mode="markers",
                                    marker={"symbol": marker, "size": 10},
                                    name=label,
                                ),
                                row=1,
                                col=1,
                                secondary_y=False,
                            )
                    smc = data.get("smart_money_concepts", {})
                    for key, label in (
                        ("confirmed_swing_high", "Confirmed swing high"),
                        ("confirmed_swing_low", "Confirmed swing low"),
                    ):
                        level = smc.get(key)
                        if level is not None:
                            fig.add_hline(
                                y=float(level),
                                line_dash="dash",
                                annotation_text=label,
                                row=1,
                                col=1,
                            )
                    for key, label in (
                        ("bullish_fvg_zone", "Bullish fair-value-gap proxy"),
                        ("bearish_fvg_zone", "Bearish fair-value-gap proxy"),
                    ):
                        zone = smc.get(key)
                        if isinstance(zone, dict):
                            fig.add_hrect(
                                y0=float(zone["low"]),
                                y1=float(zone["high"]),
                                opacity=0.10,
                                line_width=0,
                                annotation_text=label,
                                row=1,
                                col=1,
                            )
                    for key, label in (
                        ("bullish_order_block_proxy", "Bullish order-block proxy"),
                        ("bearish_order_block_proxy", "Bearish order-block proxy"),
                    ):
                        zone = smc.get(key)
                        if isinstance(zone, dict):
                            fig.add_hrect(
                                y0=float(zone["low"]),
                                y1=float(zone["high"]),
                                opacity=0.12,
                                line_width=0,
                                annotation_text=label,
                                row=1,
                                col=1,
                            )
                    price_action = data.get("price_action", {})
                    for key, label in (
                        ("retracement_38_2", "Fib 38.2%"),
                        ("retracement_50_0", "Fib 50%"),
                        ("retracement_61_8", "Fib 61.8%"),
                    ):
                        level = price_action.get("fibonacci_retracement_levels", {}).get(key)
                        if level is not None:
                            fig.add_hline(
                                y=float(level),
                                line_dash="dot",
                                annotation_text=label,
                                row=1,
                                col=1,
                            )
                    for key, label in (
                        ("high_60d", "60-session resistance"),
                        ("low_60d", "60-session support"),
                    ):
                        level = price_action.get(key)
                        if level is not None:
                            fig.add_hline(
                                y=float(level),
                                line_dash="dash",
                                annotation_text=label,
                                row=1,
                                col=1,
                            )
                    if smc.get("bullish_liquidity_sweep") or smc.get("bearish_liquidity_sweep"):
                        latest = frame.iloc[-1]
                        fig.add_trace(
                            go.Scatter(
                                x=[latest["date"]],
                                y=[latest["low"] if smc.get("bullish_liquidity_sweep") else latest["high"]],
                                mode="markers+text",
                                text=["Bullish liquidity sweep" if smc.get("bullish_liquidity_sweep") else "Bearish liquidity sweep"],
                                textposition="top center",
                                marker={"size": 12},
                                name="SMC liquidity sweep",
                            ),
                            row=1,
                            col=1,
                        )
                    fig.update_layout(
                        height=1100,
                        xaxis_rangeslider_visible=False,
                        title=f"{symbol} — price action, indicators and SMC proxies",
                        legend=dict(orientation="h", y=1.02, x=0),
                    )
                    fig.update_yaxes(title_text="Price (NPR)", row=1, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
                    fig.update_yaxes(title_text="MACD", row=2, col=1)
                    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
                    fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=4, col=1)
                    fig.update_yaxes(title_text="ADX / Directional Index", row=5, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="OBV", row=5, col=1, secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Candlesticks with SMA/EMA, Bollinger Bands, volume, MACD, RSI, Stochastic, ADX, "
                        "OBV and candle-pattern markers. SMC overlays are deterministic OHLC proxies, "
                        "not verified institutional order-flow data."
                    )
                st.json(
                    {
                        "market_context": data.get("market_context"),
                        "sector_context": data.get("sector_context"),
                        "indicators": data.get("indicators"),
                        "data_quality": data.get("data_quality"),
                        "execution_note": data.get("execution_note"),
                    }
                )
                st.subheader("Historical score-model validation")
                validation = data.get("historical_score_validation", {})
                st.write(
                    f"Validation status: {validation.get('status', 'unavailable')} · "
                    f"Out-of-sample period: {validation.get('test_start_date') or '—'} "
                    f"to {validation.get('test_end_date') or '—'}"
                )
                if validation.get("status") == "evaluated":
                    st.caption(
                        f"Score model: {validation.get('scoring_model_version')} · "
                        f"Warm-up/development observations: {validation.get('training_observations_for_warmup')} · "
                        f"Test observations: {validation.get('test_observations')} · "
                        f"BUY triggers: {validation.get('buy_trigger_count')} · "
                        f"SELL triggers: {validation.get('sell_trigger_count')}"
                    )
                    validation_rows = []
                    validation_results = validation.get("results", {})
                    for horizon in (5, 10, 20):
                        item = validation_results.get(f"forward_{horizon}d", {})
                        validation_rows.append(
                            {
                                "Horizon": f"{horizon} sessions",
                                "BUY samples": item.get("buy_sample_count"),
                                "BUY win rate": item.get("buy_win_rate"),
                                "BUY mean net return": item.get("buy_mean_net_return"),
                                "NEPSE mean return": item.get("nepse_mean_return"),
                                "Excess vs NEPSE": item.get("mean_excess_vs_nepse"),
                                "SELL samples": item.get("sell_sample_count"),
                                "Negative return share after SELL": item.get("negative_return_share_after_sell"),
                            }
                        )
                    render_table(
                        pd.DataFrame(validation_rows),
                        key="research_score_validation",
                        name=f"{symbol}_score_validation",
                        height=230,
                    )
                    st.caption(validation.get("warning", ""))
                else:
                    st.info(validation.get("reason", "Historical score validation is not available."))

                st.subheader("Historical breakout event study (separate baseline)")
                history = data.get("historical_event_study", {})
                summary_rows = []
                for horizon in (5, 10, 20):
                    item = history.get(f"forward_{horizon}d", {})
                    summary_rows.append(
                        {
                            "Horizon": f"{horizon} sessions",
                            "BUY samples": item.get("sample_count"),
                            "BUY win rate": item.get("win_rate"),
                            "Mean net return": item.get("mean_net_return"),
                            "NEPSE mean return": item.get("mean_nepse_index_return"),
                            "Excess vs NEPSE": item.get("mean_excess_vs_nepse"),
                            "SELL sample count": item.get("sell_trigger_following_return_sample_count"),
                            "Mean return after SELL": item.get("mean_return_after_sell_trigger"),
                        }
                    )
                render_table(
                    pd.DataFrame(summary_rows),
                    key="research_event_study",
                    name=f"{symbol}_historical_events",
                    height=220,
                )
                st.caption(
                    "Event study: breakout signal at close, entry at the next session open, "
                    "fixed-horizon close exits; assumes the displayed commission and slippage. "
                    "It is not a full portfolio backtest and does not guarantee future results."
                )

    with tab_history:
        st.subheader("Saved daily stock signals")
        history_symbol = st.text_input(
            "Filter by stock symbol (optional)", value="", key="research_history_symbol"
        ).strip().upper()
        history_params = {"symbol": history_symbol} if history_symbol else None
        if st.button("Load Saved Signal History", key="research_history_refresh"):
            data = _get_json("/research/signals/history", params=history_params)
            if data:
                history_rows = data.get("signals", [])
                render_table(
                    pd.DataFrame(history_rows),
                    key="research_saved_signals",
                    name="saved_stock_signal_history",
                    height=520,
                    empty_msg="No saved research signals yet. Run and save a stock scan first.",
                )

    with tab_data:
        st.subheader("Historical data import")
        st.caption(
            "The public dataset includes daily OHLCV for listed ordinary shares and a separate "
            "NEPSE Index benchmark file. Prices are raw, not corporate-action adjusted."
        )
        col1, col2 = st.columns(2)
        with col1:
            dry_run = st.checkbox("Dry-run only (no database changes)", value=True, key="research_import_dry_run")
        with col2:
            batch_size = st.selectbox("Symbols per batch", [5, 10, 15], index=0, key="research_import_batch")
        if st.button("Preview First Batch", key="research_import_preview"):
            try:
                response = requests.post(
                    f"{API_BASE}/market/research-data/equities/sync",
                    params={"offset": 0, "batch_size": int(batch_size), "dry_run": True},
                    headers=auth_headers(),
                    timeout=60,
                )
                response.raise_for_status()
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Preview failed: {exc}")

        if st.button("Import NEPSE Index Benchmark", key="research_import_index"):
            try:
                response = requests.post(
                    f"{API_BASE}/market/research-data/index/sync",
                    headers=auth_headers(),
                    timeout=60,
                )
                response.raise_for_status()
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Index import failed: {exc}")

        confirm_all = st.checkbox(
            "I understand this imports raw historical prices for ordinary shares only.",
            value=False,
            key="research_import_confirm",
        )
        if st.button(
            "Import All Listed Ordinary Shares",
            disabled=not confirm_all or dry_run,
            key="research_import_all",
        ):
            progress = st.progress(0.0)
            status = st.empty()
            offset = 0
            total_inserted = total_updated = total_rejected = 0
            failed_symbols: list[str] = []
            import_errors: list[dict] = []
            batch_number = 0
            try:
                while True:
                    response = requests.post(
                        f"{API_BASE}/market/research-data/equities/sync",
                        params={"offset": offset, "batch_size": int(batch_size)},
                        headers=auth_headers(),
                        timeout=90,
                    )
                    response.raise_for_status()
                    result = response.json()
                    batch_number += 1
                    total_inserted += int(result.get("rows_inserted", 0))
                    total_updated += int(result.get("rows_updated", 0))
                    total_rejected += int(result.get("rows_rejected", 0))
                    batch_errors = result.get("errors", [])
                    import_errors.extend(
                        item for item in batch_errors if isinstance(item, dict)
                    )
                    failed_symbols.extend(
                        str(item.get("symbol"))
                        for item in batch_errors
                        if isinstance(item, dict) and item.get("symbol")
                    )
                    count = max(int(result.get("catalogue_tradable_equities", 1)), 1)
                    next_offset = result.get("next_offset")
                    progress.progress(min((offset + int(result.get("symbols_requested", 0))) / count, 1.0))
                    status.write(
                        f"Batch {batch_number}: {result.get('symbols_succeeded', 0)} symbols; "
                        f"{total_inserted:,} inserted, {total_updated:,} updated."
                    )
                    if next_offset is None:
                        break
                    offset = int(next_offset)
                progress.progress(1.0)
                if failed_symbols:
                    st.warning(
                        f"Import finished with {len(set(failed_symbols))} symbol download failures. "
                        "The successful rows were saved; retry the listed symbols before trusting "
                        "a complete-universe scan."
                    )
                    st.write("Symbols requiring retry:", sorted(set(failed_symbols)))
                    st.json(import_errors[:20])
                else:
                    st.success(
                        f"Import completed. Inserted {total_inserted:,} rows, "
                        f"updated {total_updated:,} rows, rejected {total_rejected:,} rows."
                    )
            except requests.RequestException as exc:
                st.error(
                    f"Import stopped at offset {offset}. {exc} "
                    "You can resume by calling the sync endpoint with this offset."
                )
