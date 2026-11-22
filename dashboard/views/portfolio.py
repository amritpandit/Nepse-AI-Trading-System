"""Persistent personal paper portfolios backed by the API database."""
from __future__ import annotations

import pandas as pd
import requests  # type: ignore[import-untyped]
import streamlit as st
from common import API_BASE, _format_currency, auth_headers, render_hero


def _api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
        if detail:
            return str(detail)
    except (ValueError, AttributeError):
        pass
    return f"Request failed (HTTP {response.status_code})."


def page_portfolio() -> None:
    """Create portfolios, review holdings, and record paper trades."""
    render_hero(
        "My Portfolio",
        "Track your NEPSE holdings, cash balance, allocation and paper-trade history.",
    )
    st.info(
        "Paper portfolio only: entries record simulated transactions in this system. "
        "They do not place orders with a broker or execute trades on NEPSE."
    )

    headers = auth_headers()
    if not headers:
        st.warning("Please log in from the sidebar to view and manage your saved portfolios.")
        return

    try:
        response = requests.get(f"{API_BASE}/portfolio/accounts", headers=headers, timeout=15)
        response.raise_for_status()
        accounts = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not load your portfolios: {exc}")
        return
    except ValueError:
        st.error("The backend returned an unreadable portfolio response.")
        return

    with st.expander("＋ Create a portfolio", expanded=not bool(accounts)):
        with st.form("create_portfolio_form"):
            name = st.text_input(
                "Portfolio name",
                value="My NEPSE Portfolio",
                max_chars=100,
                help="Choose a distinct name, for example Long-term or Banking Stocks.",
            )
            capital = st.number_input(
                "Starting cash (NPR)",
                min_value=1.0,
                value=100000.0,
                step=10000.0,
                format="%.2f",
            )
            submitted = st.form_submit_button("Create portfolio", type="primary")
        if submitted:
            if not name.strip():
                st.warning("Enter a portfolio name.")
            else:
                try:
                    created = requests.post(
                        f"{API_BASE}/portfolio/account",
                        json={"name": name.strip(), "initial_capital": capital},
                        headers=headers,
                        timeout=15,
                    )
                    if created.status_code == 200:
                        st.success(f"Created {name.strip()} successfully.")
                        st.rerun()
                    else:
                        st.error(_api_error(created))
                except requests.RequestException as exc:
                    st.error(f"Could not create portfolio: {exc}")

    if not accounts:
        st.info("Create your first portfolio above to start tracking holdings.")
        return

    account_labels = {
        f"{item.get('name', 'Portfolio')} · #{item.get('id')}": item.get("id")
        for item in accounts
        if item.get("id") is not None
    }
    if not account_labels:
        st.warning("No valid portfolio accounts were returned.")
        return

    selected_label = st.selectbox("Select portfolio", list(account_labels), key="portfolio_select")
    portfolio_id = int(account_labels[selected_label])
    params = {"portfolio_id": portfolio_id}

    try:
        snapshot_response = requests.get(
            f"{API_BASE}/portfolio/account/snapshot",
            params=params,
            headers=headers,
            timeout=20,
        )
        snapshot_response.raise_for_status()
        snapshot = snapshot_response.json()

        allocation_response = requests.get(
            f"{API_BASE}/portfolio/account/allocation",
            params=params,
            headers=headers,
            timeout=20,
        )
        allocation_response.raise_for_status()
        allocation = allocation_response.json()

        transactions_response = requests.get(
            f"{API_BASE}/portfolio/account/transactions",
            params={**params, "limit": 100},
            headers=headers,
            timeout=20,
        )
        transactions_response.raise_for_status()
        transactions = transactions_response.json()
    except requests.RequestException as exc:
        st.error(f"Could not load portfolio details: {exc}")
        return
    except ValueError:
        st.error("The backend returned an unreadable portfolio response.")
        return

    positions = snapshot.get("positions", {})
    cash = float(snapshot.get("cash", 0) or 0)
    equity = float(snapshot.get("equity", 0) or 0)
    pnl = sum(float(item.get("unrealized_pnl", 0) or 0) for item in positions.values())
    initial_capital = next(
        (
            float(item.get("initial_capital", 0) or 0)
            for item in accounts
            if item.get("id") == portfolio_id
        ),
        0.0,
    )

    st.subheader(snapshot.get("portfolio_name") or selected_label)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Portfolio equity", _format_currency(equity))
    metric_cols[1].metric("Available cash", _format_currency(cash))
    metric_cols[2].metric("Invested value", _format_currency(equity - cash))
    metric_cols[3].metric(
        "Unrealized P&L",
        _format_currency(pnl),
        delta=f"{pnl / initial_capital * 100:+.2f}% of starting cash"
        if initial_capital > 0
        else None,
    )

    freshness = snapshot.get("market_data_freshness", "unavailable")
    as_of = snapshot.get("market_data_as_of") or "timestamp unavailable"
    source = snapshot.get("market_data_source")
    if source:
        st.caption(
            f"Valuation source: {source} · Prices as of: {as_of} · Feed status: {str(freshness).upper()}. "
            "Market prices are indicative and may be delayed."
        )
        if source == "YONEPSE":
            st.markdown("[YONEPSE source repository](https://github.com/Shubhamnpk/yonepse)")
        stale_symbols = [
            symbol
            for symbol, item in positions.items()
            if item.get("price_freshness") in {"stale", "delayed"}
            or item.get("price_source") != "YONEPSE"
        ]
        if stale_symbols:
            st.warning(
                "These holdings use a stored price or cost basis because an eligible "
                "symbol-level quote was unavailable: " + ", ".join(stale_symbols)
            )
        elif freshness == "market_closed":
            st.info(
                "The exchange is closed. Holdings use the latest available session prices; "
                "these are closing/reference prices, not live quotes."
            )
        elif freshness in {"stale", "delayed"}:
            st.warning(
                "The overall market snapshot is delayed or stale. "
                "Check the per-holding price status below before relying on valuations."
            )
        freshness_reason = snapshot.get("market_data_freshness_reason")
        if freshness_reason:
            st.caption(f"Market-data note: {freshness_reason}")
    elif positions:
        st.warning(
            "Live valuation is unavailable. Holdings are valued using their recorded price or cost basis."
        )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Holdings")
        if positions:
            holdings = pd.DataFrame(list(positions.values()))
            columns = [
                col
                for col in [
                    "symbol",
                    "quantity",
                    "average_price",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "price_source",
                    "price_as_of",
                    "price_freshness",
                ]
                if col in holdings.columns
            ]
            st.dataframe(holdings[columns], use_container_width=True, hide_index=True)
        else:
            st.info("No holdings yet. Record a paper BUY transaction below.")
    with right:
        st.subheader("Allocation")
        allocation_rows = allocation.get("allocation", {})
        if allocation_rows:
            allocation_df = pd.DataFrame(
                [{"Asset": symbol, "Weight (%)": value} for symbol, value in allocation_rows.items()]
            ).set_index("Asset")
            st.bar_chart(allocation_df)
        else:
            st.info("Allocation will appear after the portfolio is funded.")

    st.divider()
    st.subheader("Record a paper trade")
    with st.form("portfolio_trade_form"):
        trade_cols = st.columns(4)
        with trade_cols[0]:
            symbol = st.text_input("NEPSE symbol", value="NABIL").strip().upper()
        with trade_cols[1]:
            action = st.selectbox("Action", ["BUY", "SELL"])
        with trade_cols[2]:
            quantity = st.number_input("Quantity (kitta)", min_value=1.0, value=10.0, step=1.0)
        with trade_cols[3]:
            price = st.number_input(
                "Trade price (NPR/share)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
            )
        fees = st.number_input(
            "Fees / charges (NPR)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
        )
        record_trade = st.form_submit_button("Record paper trade", type="primary")
    if record_trade:
        payload = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "fees": fees,
        }
        try:
            trade_response = requests.post(
                f"{API_BASE}/portfolio/account/trade",
                params=params,
                json=payload,
                headers=headers,
                timeout=15,
            )
            if trade_response.status_code == 200:
                st.success(trade_response.json().get("message", "Paper trade recorded."))
                st.rerun()
            else:
                st.error(_api_error(trade_response))
        except requests.RequestException as exc:
            st.error(f"Could not record paper trade: {exc}")

    st.divider()
    st.subheader("Transaction history")
    if transactions:
        history = pd.DataFrame(transactions)
        st.dataframe(history, use_container_width=True, hide_index=True, height=300)
        st.download_button(
            "Download transactions (CSV)",
            history.to_csv(index=False).encode("utf-8"),
            file_name=f"portfolio_{portfolio_id}_transactions.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions recorded for this portfolio yet.")
