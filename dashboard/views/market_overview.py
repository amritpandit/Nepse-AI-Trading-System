from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from common import (
    _safe_dict,
    _safe_float,
    get_daily_quality_report,
    get_trust_score,
    load_live_market,
    load_market_overview,
)


def page_market_overview():
    """Market overview with data quality metrics."""  # type: ignore[annotation-unchecked]

    st.header("📊 Market Overview")

    # Load market data
    overview = load_market_overview()
    quality_report = get_daily_quality_report()
    overview_data = _safe_dict(overview)
    quality_data = _safe_dict(quality_report)

    # Key metrics row
    if overview_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            nse_value = overview_data.get("nse_value")
            nse_change = overview_data.get("nse_change")
            st.metric(
                "NEPSE Index",
                f"{nse_value:,.2f}" if isinstance(nse_value, int | float) else "N/A",
                delta=f"{nse_change:+.2f}%" if isinstance(nse_change, int | float) else None,
                delta_color="normal",
                help="Index snapshot from the attributed YONEPSE third-party feed; cross-check important figures with NEPSE.",
            )

        with col2:
            st.metric(
                "Tracked Symbols",
                overview_data.get("total_symbols", 0),
                delta=overview_data.get("new_symbols_today", 0),
            )

        with col3:
            vol_change = overview_data.get("volume_change")
            st.metric(
                "Total Volume",
                f"{_safe_float(overview_data.get('total_volume', 0)):,.0f}",
                delta=f"{vol_change:+.1f}%" if isinstance(vol_change, int | float) else None,
                help=f"Aggregated from the market feed for {overview_data.get('market_data_date') or 'the latest available trading date'}.",
            )

        with col4:
            st.metric(
                "Active Signals",
                overview_data.get("active_signals", 0),
                delta=overview_data.get("new_signals_today", 0),
                delta_color="off",
            )
    else:
        st.info("Market overview unavailable. Ensure backend is running.")

    source = overview_data.get("market_data_source")
    freshness = overview_data.get("market_data_freshness", "unavailable")
    as_of = overview_data.get("market_data_date") or "timestamp unavailable"
    if source:
        st.caption(
            f"Market source: {source} · Data as of: {as_of} · Feed status: {str(freshness).upper()}. "
            "Unofficial third-party feed; verify important prices independently."
        )
        market_open = overview_data.get("market_is_open")
        market_state = (
            "OPEN" if market_open is True else "CLOSED" if market_open is False else "UNKNOWN"
        )
        status_freshness = overview_data.get("market_status_freshness", "unknown")
        status_age = overview_data.get("market_status_age_minutes")
        status_age_label = (
            f"{status_age:.0f} minutes old"
            if isinstance(status_age, int | float)
            else "age unknown"
        )
        st.caption(
            f"Provider market status: {market_state} · "
            f"Last checked: {overview_data.get('market_status_checked_at') or 'not reported'} · "
            f"status feed: {str(status_freshness).upper()} ({status_age_label})"
        )
        st.markdown("[YONEPSE source repository](https://github.com/Shubhamnpk/yonepse)")
        if freshness == "stale":
            st.warning("Market data is stale. Do not use these prices as current quotes.")
        elif freshness == "delayed":
            st.info(
                "The latest available market snapshot is delayed; "
                "verify prices before relying on them."
            )
        elif freshness == "market_closed":
            st.info(
                "The exchange is closed. This is the latest available session snapshot, "
                "not a live market quote."
            )
    else:
        st.warning("Live market feed is unavailable. The index is shown as N/A rather than estimated.")

    st.subheader("🔎 Market Screener")
    screener_col1, screener_col2 = st.columns([2, 1])
    with screener_col1:
        market_search = st.text_input(
            "Search by company or symbol",
            placeholder="e.g. NABIL, Hydro, Bank",
            key="market_screener_search",
        )
    with screener_col2:
        sort_labels = {
            "Turnover": "turnover",
            "Trading Volume": "volume",
            "Daily Change (%)": "percent_change",
            "Market Capitalization": "market_cap",
            "Last Traded Price": "ltp",
        }
        selected_sort = st.selectbox(
            "Rank stocks by", list(sort_labels), key="market_screener_sort"
        )

    live_market = load_live_market(
        search=market_search,
        sort_by=sort_labels[selected_sort],
        order="desc",
        limit=500,
    )
    live_rows = live_market.get("stocks", []) if isinstance(live_market, dict) else []
    if live_rows:
        live_df = pd.DataFrame(live_rows)
        display_columns = [
            column
            for column in [
                "symbol",
                "name",
                "ltp",
                "change",
                "percent_change",
                "volume",
                "turnover",
                "trades",
                "last_updated",
                "freshness",
                "age_minutes",
            ]
            if column in live_df.columns
        ]
        st.dataframe(
            live_df[display_columns],
            use_container_width=True,
            hide_index=True,
            height=420,
        )
        st.caption(
            f"Showing {len(live_rows)} of {live_market.get('total_results', len(live_rows))} matching "
            "listed stocks. Sort values are based on the latest available provider snapshot."
        )
    elif isinstance(live_market, dict):
        st.info("No stocks matched the search.")
    else:
        st.warning("Stock screener data is unavailable. Check the backend market feed.")

    st.divider()

    # Data quality report
    st.subheader("🔍 Data Quality Report")

    if quality_data:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Data Completeness",
                f"{quality_data.get('completeness_score', 0):.1%}",
                help="Percentage of expected data points received",
            )

        with col2:
            st.metric(
                "Validation Pass Rate",
                f"{quality_data.get('validation_pass_rate', 0):.1%}",
                help="Percentage of data passing validation checks",
            )

        with col3:
            st.metric(
                "Avg Trust Score",
                f"{quality_data.get('avg_trust_score', 0):.2f}/1.0",
                help="Average data trustworthiness across all symbols",
            )

        # Detailed quality breakdown
        if "quality_by_symbol" in quality_data:
            df_quality = pd.DataFrame(quality_data["quality_by_symbol"])
            if not df_quality.empty:
                st.subheader("Quality by Symbol")
                # Render score columns as inline progress bars (native mini-chart, §1.1).
                score_cols = [
                    c for c in df_quality.columns if "score" in c.lower() or "rate" in c.lower()
                ]
                col_config = {
                    c: st.column_config.ProgressColumn(
                        c, min_value=0.0, max_value=1.0, format="%.2f"
                    )
                    for c in score_cols
                    if pd.api.types.is_numeric_dtype(df_quality[c])
                    and df_quality[c].dropna().between(0, 1).all()
                }
                st.dataframe(
                    df_quality, use_container_width=True, height=300, column_config=col_config
                )
    else:
        st.warning("Quality report unavailable.")

    st.divider()

    st.session_state.setdefault("trust_symbol", "NABIL")
    st.session_state.setdefault("trust_recent", [])
    recent = st.session_state["trust_recent"]
    if recent:
        picked = st.selectbox("Recent lookups", ["—", *recent], key="trust_recent_pick")
        if picked != "—":
            st.session_state["trust_symbol"] = picked

    col1, col2 = st.columns(2)
    with col1:
        lookup_symbol = st.text_input("Symbol", key="trust_symbol")
    with col2:
        _lookup_date = st.date_input("Date", value=dt.date.today(), key="trust_date")
        lookup_date = _lookup_date.isoformat() if isinstance(_lookup_date, dt.date) else ""

    if st.button("Fetch Trust Score", key="fetch_trust"):
        sym = lookup_symbol.strip().upper()
        if not sym:
            st.warning("Enter a symbol before fetching a trust score.")
            return
        if sym in recent:
            recent.remove(sym)
        recent.insert(0, sym)
        del recent[8:]  # keep the 8 most recent
        trust = get_trust_score(sym, lookup_date)
        trust_data = _safe_dict(trust)

        if trust_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trust Score", f"{trust_data.get('trust_score', 0):.2f}/1.0")
            with col2:
                status = str(trust_data.get("status", "unknown")).upper()
                color = "🟢" if status == "SAFE" else "🟡" if status == "CAUTION" else "🔴"
                st.metric("Status", f"{color} {status}")
            with col3:
                st.metric("Safe?", "✅ Yes" if trust_data.get("safe") else "❌ No")

            if "components" in trust_data:
                st.caption("**Quality Components:**")
                comp_df = pd.DataFrame([trust_data["components"]])
                st.dataframe(comp_df, use_container_width=True)
        else:
            st.error(
                "Failed to fetch trust score. Check symbol/date and ensure backend is running."
            )
