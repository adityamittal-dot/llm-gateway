"""Alembic environment (async). Run: DATABASE_URL=postgresql+asyncpg://... uv run alembic upgrade head"""

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from llm_gateway.db import metadata

target_metadata = metadata


def url() -> str:
    value = context.config.attributes.get("url") or os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("set DATABASE_URL (postgresql+asyncpg://...)")
    return value


def do_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async() -> None:
    engine = create_async_engine(url())
    async with engine.connect() as conn:
        await conn.run_sync(do_migrations)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_async())
