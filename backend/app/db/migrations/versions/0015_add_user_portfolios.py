"""Add persistent user-owned portfolio accounts and trade ledgers.

Revision ID: 0015_add_user_portfolios
Revises: 0014_add_external_market_tables
Create Date: 2026-11-21
"""

import sqlalchemy as sa
from alembic import op

revision = "0015_add_user_portfolios"
down_revision = "0014_add_external_market_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portfolio_accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("initial_capital", sa.Numeric(18, 4), nullable=False),
        sa.Column("cash", sa.Numeric(18, 4), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "name", name="uq_portfolio_accounts_user_name"),
    )
    op.create_index("ix_portfolio_accounts_user_id", "portfolio_accounts", ["user_id"])

    op.create_table(
        "portfolio_positions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 4), nullable=False),
        sa.Column("average_price", sa.Numeric(18, 4), nullable=False),
        sa.Column("current_price", sa.Numeric(18, 4), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolio_accounts.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "portfolio_id", "symbol", name="uq_portfolio_positions_portfolio_symbol"
        ),
    )
    op.create_index("ix_portfolio_positions_portfolio_id", "portfolio_positions", ["portfolio_id"])

    op.create_table(
        "portfolio_transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 4), nullable=False),
        sa.Column("price", sa.Numeric(18, 4), nullable=False),
        sa.Column("fees", sa.Numeric(18, 4), server_default="0", nullable=False),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolio_accounts.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_portfolio_transactions_portfolio_id",
        "portfolio_transactions",
        ["portfolio_id"],
    )
    op.create_index(
        "ix_portfolio_transactions_timestamp",
        "portfolio_transactions",
        ["timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_portfolio_transactions_timestamp", table_name="portfolio_transactions")
    op.drop_index("ix_portfolio_transactions_portfolio_id", table_name="portfolio_transactions")
    op.drop_table("portfolio_transactions")
    op.drop_index("ix_portfolio_positions_portfolio_id", table_name="portfolio_positions")
    op.drop_table("portfolio_positions")
    op.drop_index("ix_portfolio_accounts_user_id", table_name="portfolio_accounts")
    op.drop_table("portfolio_accounts")
