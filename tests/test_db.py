import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from test_providers import FakeProvider

from llm_gateway import db
from llm_gateway.app import create_app
from llm_gateway.config import Settings


async def test_partitions_route_rows_by_month(postgres_url):
    engine = db.make_engine(postgres_url)
    created = await db.ensure_partitions(engine, months_ahead=1, today=dt.date(2026, 12, 15))
    assert created == ["usage_ledger_2026_12", "usage_ledger_2027_01"]
    await db.ensure_partitions(engine, months_ahead=1, today=dt.date(2026, 12, 15))  # idempotent
    writer = db.LedgerWriter(engine)
    for day in (dt.datetime(2026, 12, 31, tzinfo=dt.UTC), dt.datetime(2027, 1, 1, tzinfo=dt.UTC)):
        writer.append({"ts": day, "request_id": "r", "tenant": "t", "provider": "p", "model": "m", "status": 200,
                       "stream": False, "cost_usd": 0.001, "latency_ms": 10, "failovers": 0})  # fmt: skip
    assert await writer.flush() == 2
    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT tableoid::regclass::text AS part, count(*) FROM usage_ledger "
            "WHERE request_id = 'r' GROUP BY 1 ORDER BY 1"))).all()  # fmt: skip
    assert rows == [("usage_ledger_2026_12", 1), ("usage_ledger_2027_01", 1)]
    await engine.dispose()


async def test_latest_effective_price_wins(postgres_url):
    engine = db.make_engine(postgres_url)
    now = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)
    async with engine.begin() as conn:
        await conn.execute(db.model_prices.insert(), [
            {"model": "nova-micro", "input_per_mtok": 0.035, "output_per_mtok": 0.14, "cached_input_per_mtok": None,
             "effective_from": now - dt.timedelta(days=100)},
            {"model": "nova-micro", "input_per_mtok": 0.03, "output_per_mtok": 0.12, "cached_input_per_mtok": 0.0075,
             "effective_from": now - dt.timedelta(days=1)},
            {"model": "nova-micro", "input_per_mtok": 9, "output_per_mtok": 9, "cached_input_per_mtok": None,
             "effective_from": now + dt.timedelta(days=5)},
        ])  # fmt: skip
    prices = await db.load_prices(engine, now)
    assert (
        prices["nova-micro"].input_per_mtok == 0.03 and prices["nova-micro"].cached_input_per_mtok == 0.0075
    )
    await engine.dispose()


async def test_ledger_writer_requeues_on_failure(postgres_url):
    engine = db.make_engine(postgres_url.replace("gateway_test", "does_not_exist"))
    writer = db.LedgerWriter(engine)
    writer.append({"ts": dt.datetime.now(dt.UTC)})
    with pytest.raises(Exception):  # noqa: B017 - any connection error will do
        await writer.flush()
    assert writer.queue.qsize() == 1
    await engine.dispose()


def test_gateway_writes_one_ledger_row_per_request(postgres_url):
    settings = Settings(database_url=postgres_url, keys={"sk-a": "acme"},
                        prices={"m": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})  # fmt: skip
    with TestClient(create_app(settings, providers=[FakeProvider(models=["m"])])) as gw:
        for stream in (False, True):
            gw.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": stream},
                    headers={"Authorization": "Bearer sk-a", "X-Session-Id": "run-42"})  # fmt: skip
    import asyncio

    async def rows():
        engine = db.make_engine(postgres_url)
        async with engine.connect() as conn:
            out = (await conn.execute(text(
                "SELECT tenant, session_id, stream, status, cost_usd FROM usage_ledger "
                "WHERE tenant = 'acme' ORDER BY ts"))).all()  # fmt: skip
        await engine.dispose()
        return out

    result = asyncio.run(rows())
    assert [(r.tenant, r.session_id, r.stream, r.status) for r in result] == [
        ("acme", "run-42", False, 200), ("acme", "run-42", True, 200)]  # fmt: skip
    assert float(result[0].cost_usd) == 2.0 / 1e6  # one completion token at $2 per million


async def test_writer_keeps_creating_partitions_while_running(postgres_url):
    import asyncio

    engine = db.make_engine(postgres_url)
    writer = db.LedgerWriter(engine, interval_s=0.01)
    writer.partition_check_s = 0.0
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"DROP TABLE IF EXISTS {db.partition_name(db.add_months(dt.datetime.now(dt.UTC).date().replace(day=1), 2))}"
            )
        )
    writer.start()
    await asyncio.sleep(0.2)
    await writer.stop()
    async with engine.connect() as conn:
        names = (
            (
                await conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'usage_ledger_2%'")
                )
            )
            .scalars()
            .all()
        )
    assert db.partition_name(db.add_months(dt.datetime.now(dt.UTC).date().replace(day=1), 2)) in names
    await engine.dispose()
