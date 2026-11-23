"""Add stock instrument type for tradable-equity screening.

Revision ID: 0016_stock_instrument_type
Revises: 0015_add_user_portfolios
Create Date: 2026-11-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0016_stock_instrument_type"
down_revision = "0015_add_user_portfolios"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stocks",
        sa.Column(
            "instrument_type",
            sa.String(length=30),
            nullable=False,
            server_default="unknown",
        ),
    )


def downgrade() -> None:
    op.drop_column("stocks", "instrument_type")
