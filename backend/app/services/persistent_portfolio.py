"""Database-backed, user-isolated paper portfolio service."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.portfolio import PortfolioAccount, PortfolioPosition, PortfolioTransaction
from app.services.yonepse_market import MarketDataUnavailable, yonepse_market_data

logger = logging.getLogger(__name__)
ZERO = Decimal("0.0000")
DEFAULT_INITIAL_CAPITAL = Decimal("100000.0000")
VALUATION_FRESHNESS = {"fresh", "delayed", "market_closed"}


class PortfolioAccountConflict(ValueError):
    """Raised when a user already has a portfolio with the requested name."""


class PortfolioNotFound(LookupError):
    """Raised when the requested portfolio does not belong to the current user."""


class PersistentPortfolioService:
    """Owns all database reads/writes for persistent paper portfolio accounts."""

    def create_account(
        self,
        db: Session,
        *,
        user_id: str,
        name: str,
        initial_capital: Decimal,
    ) -> PortfolioAccount:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Portfolio name is required")
        if len(clean_name) > 100:
            raise ValueError("Portfolio name must be 100 characters or fewer")
        if initial_capital <= ZERO:
            raise ValueError("Initial capital must be greater than zero")

        account = PortfolioAccount(
            user_id=user_id,
            name=clean_name,
            initial_capital=initial_capital,
            cash=initial_capital,
        )
        db.add(account)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise PortfolioAccountConflict(
                "A portfolio with this name already exists in your account"
            ) from exc
        db.refresh(account)
        return account

    def list_accounts(self, db: Session, *, user_id: str) -> list[PortfolioAccount]:
        return list(
            db.scalars(
                sa.select(PortfolioAccount)
                .where(PortfolioAccount.user_id == user_id)
                .order_by(PortfolioAccount.id)
            ).all()
        )

    def get_account(
        self,
        db: Session,
        *,
        user_id: str,
        portfolio_id: int | None = None,
        create_default: bool = True,
    ) -> PortfolioAccount:
        query = sa.select(PortfolioAccount).where(PortfolioAccount.user_id == user_id)
        if portfolio_id is not None:
            query = query.where(PortfolioAccount.id == portfolio_id)
        else:
            query = query.order_by(PortfolioAccount.id).limit(1)

        account = db.scalar(query)
        if account is not None:
            return account
        if portfolio_id is not None or not create_default:
            raise PortfolioNotFound("Portfolio account was not found")
        return self.create_account(
            db,
            user_id=user_id,
            name="My Portfolio",
            initial_capital=DEFAULT_INITIAL_CAPITAL,
        )

    def _positions(self, db: Session, account_id: int) -> list[PortfolioPosition]:
        return list(
            db.scalars(
                sa.select(PortfolioPosition)
                .where(PortfolioPosition.portfolio_id == account_id)
                .order_by(PortfolioPosition.symbol)
            ).all()
        )

    def _live_quotes(
        self, positions: list[PortfolioPosition]
    ) -> tuple[dict[str, Decimal], dict[str, Any]]:
        if not positions:
            return {}, {
                "source": None,
                "as_of": None,
                "freshness": "not_applicable",
                "freshness_reason": None,
                "quotes": {},
            }
        try:
            snapshot = yonepse_market_data.get_snapshot()
        except MarketDataUnavailable as exc:
            logger.info("Portfolio valuation falling back to stored prices: %s", exc)
            return {}, {
                "source": None,
                "as_of": None,
                "freshness": "unavailable",
                "freshness_reason": str(exc),
                "quotes": {},
            }

        market_meta: dict[str, Any] = {
            "source": snapshot.get("source"),
            "as_of": snapshot.get("as_of"),
            "freshness": str(snapshot.get("freshness", "unknown")),
            "freshness_reason": snapshot.get("freshness_reason"),
            "quotes": {},
        }
        quotes: dict[str, Decimal] = {}
        position_symbols = {position.symbol.upper() for position in positions}

        # Validate each symbol's own timestamp. A newer index or another stock's
        # timestamp must not make an older individual quote appear current.
        for row in snapshot.get("stocks", []):
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol not in position_symbols:
                continue

            quote_freshness = str(row.get("freshness", "unknown"))
            quote_as_of = row.get("last_updated_at") or row.get("last_updated")
            ltp = row.get("ltp")
            quote_detail: dict[str, Any] = {
                "as_of": quote_as_of,
                "freshness": quote_freshness,
                "age_minutes": row.get("age_minutes"),
                "freshness_reason": row.get("freshness_reason"),
                "used_for_valuation": False,
            }
            market_meta["quotes"][symbol] = quote_detail

            if quote_freshness not in VALUATION_FRESHNESS or quote_as_of is None or ltp is None:
                continue
            quote = Decimal(str(ltp))
            if quote <= ZERO:
                continue

            quotes[symbol] = quote
            quote_detail["used_for_valuation"] = True

        return quotes, market_meta

    def snapshot(self, db: Session, account: PortfolioAccount) -> dict[str, Any]:
        positions = self._positions(db, account.id)
        quotes, market_meta = self._live_quotes(positions)
        quote_details = market_meta.get("quotes", {})
        position_data: dict[str, dict[str, float | str | None]] = {}
        positions_value = ZERO

        for position in positions:
            quote_detail = quote_details.get(position.symbol, {})
            if position.symbol in quotes:
                current_price = quotes[position.symbol]
                price_source = "YONEPSE"
            elif position.current_price is not None:
                current_price = position.current_price
                price_source = "STORED"
            else:
                current_price = position.average_price
                price_source = "COST_BASIS"

            market_value = position.quantity * current_price
            unrealized_pnl = position.quantity * (current_price - position.average_price)
            positions_value += market_value
            fallback_freshness = (
                market_meta["freshness"]
                if market_meta["freshness"] == "unavailable"
                else "unknown"
            )
            position_data[position.symbol] = {
                "symbol": position.symbol,
                "quantity": float(position.quantity),
                "average_price": float(position.average_price),
                "current_price": float(current_price),
                "market_value": float(market_value),
                "unrealized_pnl": float(unrealized_pnl),
                "price_source": price_source,
                "price_as_of": quote_detail.get("as_of"),
                "price_freshness": quote_detail.get("freshness", fallback_freshness),
            }

        equity = account.cash + positions_value
        return {
            "id": account.id,
            "portfolio_id": account.id,
            "portfolio_name": account.name,
            "date": datetime.now(timezone.utc).isoformat(),
            "equity": float(equity),
            "cash": float(account.cash),
            "positions": position_data,
            "market_data_source": market_meta["source"],
            "market_data_as_of": market_meta["as_of"],
            "market_data_freshness": market_meta["freshness"],
            "market_data_freshness_reason": market_meta.get("freshness_reason"),
        }

    def allocation(self, db: Session, account: PortfolioAccount) -> dict[str, Any]:
        snapshot = self.snapshot(db, account)
        total_equity = Decimal(str(snapshot["equity"]))
        cash = Decimal(str(snapshot["cash"]))
        allocation: dict[str, float] = {}
        if total_equity > ZERO:
            allocation["CASH"] = float(cash / total_equity * Decimal("100"))
            for symbol, position in snapshot["positions"].items():
                allocation[symbol] = float(
                    Decimal(str(position["market_value"])) / total_equity * Decimal("100")
                )
        return {
            "portfolio_id": account.id,
            "portfolio_name": account.name,
            "allocation": allocation,
            "total_equity": float(total_equity),
            "cash": float(cash),
            "positions_value": float(total_equity - cash),
            "market_data_source": snapshot["market_data_source"],
            "market_data_as_of": snapshot["market_data_as_of"],
            "market_data_freshness": snapshot["market_data_freshness"],
            "market_data_freshness_reason": snapshot["market_data_freshness_reason"],
        }

    def transactions(
        self, db: Session, account: PortfolioAccount, *, limit: int
    ) -> list[PortfolioTransaction]:
        return list(
            db.scalars(
                sa.select(PortfolioTransaction)
                .where(PortfolioTransaction.portfolio_id == account.id)
                .order_by(PortfolioTransaction.timestamp.desc(), PortfolioTransaction.id.desc())
                .limit(limit)
            ).all()
        )

    def execute_trade(
        self,
        db: Session,
        account: PortfolioAccount,
        *,
        symbol: str,
        action: str,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal,
    ) -> dict[str, Any]:
        normalized_symbol = symbol.strip().upper()
        normalized_action = action.strip().upper()
        if not normalized_symbol:
            raise ValueError("Stock symbol is required")
        if normalized_action not in {"BUY", "SELL"}:
            raise ValueError("Trade action must be BUY or SELL")
        if quantity <= ZERO or price <= ZERO or fees < ZERO:
            raise ValueError("Quantity and price must be positive; fees cannot be negative")

        position = db.scalar(
            sa.select(PortfolioPosition).where(
                PortfolioPosition.portfolio_id == account.id,
                PortfolioPosition.symbol == normalized_symbol,
            )
        )

        if normalized_action == "BUY":
            total_cost = quantity * price + fees
            if total_cost > account.cash:
                raise ValueError("Insufficient cash for this purchase")
            if position is None:
                position = PortfolioPosition(
                    portfolio_id=account.id,
                    symbol=normalized_symbol,
                    quantity=quantity,
                    average_price=price,
                )
                db.add(position)
            else:
                total_quantity = position.quantity + quantity
                total_basis = position.quantity * position.average_price + quantity * price
                position.average_price = total_basis / total_quantity
                position.quantity = total_quantity
            account.cash -= total_cost
        else:
            if position is None or position.quantity < quantity:
                raise ValueError("Insufficient shares available for this sale")
            position.quantity -= quantity
            account.cash += quantity * price - fees
            if position.quantity == ZERO:
                db.delete(position)

        transaction = PortfolioTransaction(
            portfolio_id=account.id,
            symbol=normalized_symbol,
            action=normalized_action,
            quantity=quantity,
            price=price,
            fees=fees,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(transaction)
        db.add(account)
        db.commit()
        db.refresh(account)

        return {
            "message": f"{normalized_action} order recorded successfully in paper portfolio",
            "portfolio_id": account.id,
            "portfolio_name": account.name,
            "symbol": normalized_symbol,
            "quantity": float(quantity),
            "price": float(price),
            "fees": float(fees),
            "total_value": float(quantity * price),
            "new_cash_balance": float(account.cash),
            "execution_type": "paper_portfolio_record_only",
        }


persistent_portfolio_service = PersistentPortfolioService()
