"""Portfolio optimization and persistent paper account API routes."""

import logging
from decimal import Decimal

import numpy as np
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.portfolio import (
    AllocationRequest,
    OptimizationResponse,
    PortfolioAccountRequest,
    PortfolioAllocationResponse,
    PortfolioSnapshotResponse,
    TradeRequest,
    TransactionResponse,
)
from app.services.persistent_portfolio import (
    PortfolioAccountConflict,
    PortfolioNotFound,
    persistent_portfolio_service,
)
from app.services.portfolio_optimization import (
    OptimizationConstraints,
    portfolio_optimization_service,
)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _account_or_404(db: Session, user_id: str, portfolio_id: int | None):
    try:
        return persistent_portfolio_service.get_account(
            db, user_id=user_id, portfolio_id=portfolio_id
        )
    except PortfolioNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/accounts", response_model=list[dict])
def list_portfolio_accounts(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List all saved paper portfolios owned by the current user."""
    return [
        {
            "id": account.id,
            "name": account.name,
            "initial_capital": float(account.initial_capital),
            "cash": float(account.cash),
            "created_at": account.created_at.isoformat() if account.created_at else None,
        }
        for account in persistent_portfolio_service.list_accounts(db, user_id=user.id)
    ]


@router.post("/account", response_model=dict)
def create_portfolio_account(
    request: PortfolioAccountRequest | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Create a new named paper portfolio for the current user."""
    initial_capital = (
        Decimal(str(request.initial_capital)) if request else Decimal("100000.00")
    )
    name = request.name if request else "My Portfolio"
    try:
        account = persistent_portfolio_service.create_account(
            db,
            user_id=user.id,
            name=name,
            initial_capital=initial_capital,
        )
    except PortfolioAccountConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Created portfolio account %s for user %s", account.id, user.id)
    return {
        "message": "Portfolio account created successfully",
        "portfolio_id": account.id,
        "name": account.name,
        "initial_capital": float(account.initial_capital),
        "cash": float(account.cash),
        "user_id": user.id,
    }


@router.get("/account/snapshot", response_model=PortfolioSnapshotResponse)
def get_portfolio_snapshot(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    portfolio_id: int | None = Query(default=None, ge=1),
) -> PortfolioSnapshotResponse:
    """Get a saved portfolio snapshot, valued using the latest available market prices."""
    account = _account_or_404(db, user.id, portfolio_id)
    return PortfolioSnapshotResponse(**persistent_portfolio_service.snapshot(db, account))


@router.get("/account/allocation", response_model=PortfolioAllocationResponse)
def get_portfolio_allocation(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    portfolio_id: int | None = Query(default=None, ge=1),
) -> PortfolioAllocationResponse:
    """Get allocation percentages for a saved portfolio."""
    account = _account_or_404(db, user.id, portfolio_id)
    return PortfolioAllocationResponse(
        **persistent_portfolio_service.allocation(db, account)
    )


@router.get("/account/transactions", response_model=list[TransactionResponse])
def get_transaction_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=1000),
    portfolio_id: int | None = Query(default=None, ge=1),
) -> list[TransactionResponse]:
    """Get persisted transaction history for a saved portfolio."""
    account = _account_or_404(db, user.id, portfolio_id)
    rows = persistent_portfolio_service.transactions(db, account, limit=limit)
    return [
        TransactionResponse(
            symbol=row.symbol,
            action=row.action,
            quantity=float(row.quantity),
            price=float(row.price),
            timestamp=row.timestamp.isoformat(),
            fees=float(row.fees),
        )
        for row in rows
    ]


@router.post("/account/trade", response_model=dict)
def execute_trade(
    request: TradeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    portfolio_id: int | None = Query(default=None, ge=1),
) -> dict:
    """Record a manual paper buy/sell trade in the user's saved portfolio."""
    account = _account_or_404(db, user.id, portfolio_id)
    try:
        return persistent_portfolio_service.execute_trade(
            db,
            account,
            symbol=request.symbol,
            action=request.action,
            quantity=Decimal(str(request.quantity)),
            price=Decimal(str(request.price)),
            fees=Decimal(str(request.fees)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/optimize", response_model=OptimizationResponse)
def optimize_portfolio(
    request: AllocationRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Optimize portfolio allocation using various methods."""
    try:
        # Convert inputs to numpy arrays
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)

        # Validate dimensions
        n_symbols = len(request.symbols)
        if len(expected_returns) != n_symbols:
            raise ValueError("Length of expected_returns must match number of symbols")
        if covariance_matrix.shape != (n_symbols, n_symbols):
            raise ValueError("Covariance matrix must be square and match number of symbols")

        # Execute optimization based on method
        if request.method == "equal_weight":
            weights = portfolio_optimization_service.equal_weight(request.symbols)
            # Calculate portfolio metrics
            portfolio_return = np.dot(list(weights.values()), expected_returns)
            portfolio_variance = np.dot(
                list(weights.values()), np.dot(covariance_matrix, list(weights.values()))
            )
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0.0

            return OptimizationResponse(
                weights=weights,
                expected_return=float(portfolio_return),
                expected_risk=float(portfolio_std),
                sharpe_ratio=float(sharpe_ratio),
                method="equal_weight",
                success=True,
                message="Equal weight allocation calculated",
            )

        elif request.method == "risk_parity":
            weights = portfolio_optimization_service.risk_parity(request.symbols, covariance_matrix)
            # Calculate portfolio metrics
            portfolio_return = np.dot(list(weights.values()), expected_returns)
            portfolio_variance = np.dot(
                list(weights.values()), np.dot(covariance_matrix, list(weights.values()))
            )
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0.0

            return OptimizationResponse(
                weights=weights,
                expected_return=float(portfolio_return),
                expected_risk=float(portfolio_std),
                sharpe_ratio=float(sharpe_ratio),
                method="risk_parity",
                success=True,
                message="Risk parity allocation calculated",
            )

        elif request.method == "mean_variance":
            # Parse constraints if provided
            constraints = OptimizationConstraints()
            if request.constraints:
                if "long_only" in request.constraints:
                    constraints.long_only = request.constraints["long_only"]
                if "max_weight" in request.constraints:
                    constraints.max_weight = request.constraints["max_weight"]
                if "min_weight" in request.constraints:
                    constraints.min_weight = request.constraints["min_weight"]
                if "target_return" in request.constraints:
                    constraints.target_return = request.constraints["target_return"]
                if "risk_aversion" in request.constraints:
                    constraints.risk_aversion = request.constraints["risk_aversion"]

            result = portfolio_optimization_service.mean_variance(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                symbols=request.symbols,
                constraints=constraints,
            )

            return OptimizationResponse(
                weights=result.weights,
                expected_return=result.expected_return,
                expected_risk=result.expected_risk,
                sharpe_ratio=result.sharpe_ratio,
                method=result.method,
                success=result.success,
                message=result.message,
            )

        else:
            raise ValueError(f"Unknown optimization method: {request.method}")

    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise HTTPException(
            status_code=400, detail=f"Portfolio optimization failed: {str(e)}"
        ) from e


# Health check endpoint
@router.get("/health")
def portfolio_health_check():
    """Health check for portfolio service."""
    return {"status": "healthy", "service": "portfolio", "version": "1.0.0"}
