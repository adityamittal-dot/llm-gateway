"""Initial schema: tenants, keys, budgets, routing, prices, and the monthly-partitioned usage ledger.

Revision ID: 0001
Revises:
Create Date: 2026-10-18
"""

from alembic import op

from llm_gateway.db import api_keys, budgets, metadata, model_prices, orgs, routing_rules, teams

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

LEDGER_DDL = """
CREATE TABLE usage_ledger (
    id                BIGINT GENERATED ALWAYS AS IDENTITY,
    ts                TIMESTAMPTZ   NOT NULL,
    request_id        VARCHAR(64)   NOT NULL,
    tenant            VARCHAR(200)  NOT NULL,
    session_id        VARCHAR(200),
    provider          VARCHAR(200)  NOT NULL,
    model             VARCHAR(300)  NOT NULL,
    target_model      VARCHAR(300),
    status            INTEGER       NOT NULL,
    stream            BOOLEAN       NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cached_tokens     INTEGER,
    cost_usd          NUMERIC(14,8) NOT NULL,
    latency_ms        INTEGER       NOT NULL,
    ttft_ms           INTEGER,
    cache_status      VARCHAR(16),
    failovers         INTEGER       NOT NULL,
    finish_reason     VARCHAR(32),
    quality           JSONB,
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts)
"""


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[orgs, teams, api_keys, budgets, routing_rules, model_prices])
    op.execute(LEDGER_DDL)
    op.execute("CREATE TABLE usage_ledger_default PARTITION OF usage_ledger DEFAULT")
    op.execute("CREATE INDEX usage_ledger_tenant_ts ON usage_ledger (tenant, ts)")
    op.execute("CREATE INDEX usage_ledger_provider_ts ON usage_ledger (provider, ts)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS usage_ledger CASCADE")
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[model_prices, routing_rules, budgets, api_keys, teams, orgs])
