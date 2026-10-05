"""Postgres: the system of record (README "Data model").

Tables: orgs, teams, api_keys, budgets, routing_rules, model_prices, usage_ledger. The ledger is
append-only and partitioned by month (`usage_ledger_YYYY_MM`), so old months can be detached and
archived cheaply; a default partition catches rows if a month's partition is missing.

Ledger writes never sit on the request path: rows go into a bounded in-memory queue and a
background task inserts them in batches. If the database is slow or down, the queue fills and
new rows are dropped with a warning; completions keep flowing.
"""

import asyncio
import datetime as dt
import logging

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from llm_gateway.pricing import Price

log = logging.getLogger(__name__)
metadata = MetaData()

orgs = Table(
    "orgs", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)  # fmt: skip
teams = Table(
    "teams", metadata,
    Column("id", Integer, primary_key=True),
    Column("org_id", Integer, ForeignKey("orgs.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    UniqueConstraint("org_id", "name"),
)  # fmt: skip
api_keys = Table(
    "api_keys", metadata,
    Column("id", Integer, primary_key=True),
    Column("key_hash", String(64), nullable=False, unique=True),  # sha256 hex; plaintext is never stored
    Column("tenant", String(200), nullable=False),
    Column("team_id", Integer, ForeignKey("teams.id", ondelete="SET NULL")),
    Column("allowed_models", ARRAY(Text)),  # NULL = all models
    Column("rpm", Integer),
    Column("tpm", Integer),
    Column("active", Boolean, nullable=False, server_default=text("true")),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)  # fmt: skip
budgets = Table(
    "budgets", metadata,
    Column("id", Integer, primary_key=True),
    Column("scope_type", String(16), nullable=False),  # org | team | tenant | session
    Column("scope_id", String(200), nullable=False),
    Column("period", String(16), nullable=False, server_default="month"),
    Column("hard_usd", Numeric(14, 6)),
    Column("soft_usd", Numeric(14, 6)),
    UniqueConstraint("scope_type", "scope_id", "period"),
)  # fmt: skip
routing_rules = Table(
    "routing_rules", metadata,
    Column("id", Integer, primary_key=True),
    Column("alias", String(200), nullable=False),
    Column("position", Integer, nullable=False),
    Column("provider", String(200), nullable=False),
    Column("model", String(300), nullable=False),
    UniqueConstraint("alias", "position"),
)  # fmt: skip
model_prices = Table(
    "model_prices", metadata,
    Column("id", Integer, primary_key=True),
    Column("model", String(300), nullable=False),
    Column("region", String(64)),
    Column("input_per_mtok", Numeric(12, 6), nullable=False),
    Column("output_per_mtok", Numeric(12, 6), nullable=False),
    Column("cached_input_per_mtok", Numeric(12, 6)),
    Column("effective_from", DateTime(timezone=True), nullable=False),
    UniqueConstraint("model", "region", "effective_from"),
)  # fmt: skip
# Partitioned by month; created by the migration with raw DDL (PARTITION BY RANGE (ts)).
usage_ledger = Table(
    "usage_ledger", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("request_id", String(64), nullable=False),
    Column("tenant", String(200), nullable=False),
    Column("session_id", String(200)),
    Column("provider", String(200), nullable=False),
    Column("model", String(300), nullable=False),
    Column("target_model", String(300)),
    Column("status", Integer, nullable=False),
    Column("stream", Boolean, nullable=False),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("cached_tokens", Integer),
    Column("cost_usd", Numeric(14, 8), nullable=False),
    Column("latency_ms", Integer, nullable=False),
    Column("ttft_ms", Integer),
    Column("cache_status", String(16)),
    Column("failovers", Integer, nullable=False),
    Column("finish_reason", String(32)),
    Column("quality", JSONB),
)  # fmt: skip


def partition_name(month: dt.date) -> str:
    return f"usage_ledger_{month:%Y_%m}"


def month_start(d: dt.date) -> dt.date:
    return d.replace(day=1)


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    return dt.date(d.year + y, m + 1, 1)


async def ensure_partitions(
    engine: AsyncEngine, months_ahead: int = 2, today: dt.date | None = None
) -> list[str]:
    """Create monthly ledger partitions from this month through `months_ahead` (idempotent)."""
    start = month_start(today or dt.datetime.now(dt.UTC).date())
    created = []
    async with engine.begin() as conn:
        for i in range(months_ahead + 1):
            lo, hi = add_months(start, i), add_months(start, i + 1)
            name = partition_name(lo)
            await conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF usage_ledger "
                f"FOR VALUES FROM ('{lo.isoformat()}') TO ('{hi.isoformat()}')"
            ))  # fmt: skip
            created.append(name)
    return created


async def load_prices(engine: AsyncEngine, now: dt.datetime | None = None) -> dict[str, Price]:
    """Latest price per model whose `effective_from` is not in the future."""
    now = now or dt.datetime.now(dt.UTC)
    query = (
        select(model_prices)
        .where(model_prices.c.effective_from <= now)
        .order_by(model_prices.c.model, model_prices.c.effective_from.desc())
    )
    out: dict[str, Price] = {}
    async with engine.connect() as conn:
        for row in (await conn.execute(query)).mappings():
            if row["model"] in out:
                continue
            cached = row["cached_input_per_mtok"]
            out[row["model"]] = Price(float(row["input_per_mtok"]), float(row["output_per_mtok"]),
                                      float(cached) if cached is not None else None)  # fmt: skip
    return out


class LedgerWriter:
    """Batches ledger rows off the request path."""

    def __init__(
        self, engine: AsyncEngine, max_queue: int = 10_000, batch: int = 200, interval_s: float = 1.0
    ):
        self.engine = engine
        self.queue: asyncio.Queue[dict] = asyncio.Queue(max_queue)
        self.batch, self.interval_s = batch, interval_s
        self.dropped = 0
        self.written = 0
        self._task: asyncio.Task | None = None

    def append(self, row: dict) -> None:
        try:
            self.queue.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 1000 == 1:
                log.warning("ledger queue full; dropping rows", extra={"dropped": self.dropped})

    async def flush(self) -> int:
        rows = []
        while not self.queue.empty() and len(rows) < self.batch * 10:
            rows.append(self.queue.get_nowait())
        if not rows:
            return 0
        # executemany takes its column list from the first row, so give every row every column.
        columns = [c.name for c in usage_ledger.columns if c.name != "id"]
        batch = [{c: row.get(c) for c in columns} for row in rows]
        try:
            async with self.engine.begin() as conn:
                await conn.execute(usage_ledger.insert(), batch)
        except Exception:
            for row in rows:  # put them back for the next attempt (dropping only if the queue is full)
                self.append(row)
            raise
        self.written += len(rows)
        return len(rows)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            try:
                await self.flush()
            except Exception:  # the database being down must not take the gateway down
                log.exception("ledger flush failed")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        try:
            while await self.flush():
                pass
        except Exception:
            log.exception("final ledger flush failed")


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)
