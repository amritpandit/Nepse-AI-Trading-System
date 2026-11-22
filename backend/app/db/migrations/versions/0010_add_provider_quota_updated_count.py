"""add missing provider quota updated_count column

Revision ID: 0010_provider_quota_count
Revises: 0009_add_provider_quota_tracking
Create Date: 2026-06-13

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector


revision = "0010_provider_quota_count"
down_revision = "0009_add_provider_quota_tracking"
branch_labels = None
depends_on = None


def _get_inspector() -> Inspector:
    # Inspect on Alembic's active connection to include DDL from earlier
    # revisions in the same PostgreSQL transaction.
    return sa.inspect(op.get_bind())


def _table_exists(table_name: str) -> bool:
    return _get_inspector().has_table(table_name)


def _column_exists(table_name: str, column_name: str) -> bool:
    inspector = _get_inspector()
    return inspector.has_table(table_name) and column_name in {
        column["name"] for column in inspector.get_columns(table_name)
    }


def upgrade() -> None:
    if _table_exists("provider_quota_usage") and not _column_exists(
        "provider_quota_usage", "updated_count"
    ):
        op.add_column(
            "provider_quota_usage",
            sa.Column("updated_count", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    if _table_exists("provider_quota_usage") and _column_exists(
        "provider_quota_usage", "updated_count"
    ):
        op.drop_column("provider_quota_usage", "updated_count")
