from __future__ import annotations

import pandas as pd
import requests  # type: ignore[import-untyped]
import streamlit as st
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

    tab_market, tab_sectors, tab_stocks, tab_detail, tab_data = st.tabs(
        ["Market Regime", "Sector Ranking", "Stock Signals", "Stock Detail & History", "Historical Data"]
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
                st.json(
                    {
                        "market_context": data.get("market_context"),
                        "sector_context": data.get("sector_context"),
                        "indicators": data.get("indicators"),
                        "data_quality": data.get("data_quality"),
                        "execution_note": data.get("execution_note"),
                    }
                )
                st.subheader("Historical BUY/SELL event outcomes")
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
            batch_size = st.selectbox("Symbols per batch", [5, 10, 15], index=2, key="research_import_batch")
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
                st.success(
                    f"Import completed. Inserted {total_inserted:,} rows, "
                    f"updated {total_updated:,} rows, rejected {total_rejected:,} rows."
                )
            except requests.RequestException as exc:
                st.error(
                    f"Import stopped at offset {offset}. {exc} "
                    "You can resume by calling the sync endpoint with this offset."
                )
