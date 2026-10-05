import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from test_providers import FakeProvider

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.pricing import Price, Pricing
from llm_gateway.ratelimit import Limiter, MemoryLimitStore, RedisLimitStore, estimate_tokens

TENANTS = {"acme": {"tpm": 1000, "rpm": 100, "team": "platform", "org": "acme-inc"}}
TEAMS = {"platform": {"tpm": 1500}}
ORGS = {"acme-inc": {"budget_usd": 1.0, "soft_budget_usd": 0.5}}


class Clock:
    def __init__(self):
        self.t = 1_760_000_000.0

    def __call__(self):
        return self.t


async def stores(valkey_url):
    r = aioredis.from_url(valkey_url)
    await r.flushdb()
    return [MemoryLimitStore(), RedisLimitStore(r)]


async def test_token_bucket_reserves_settles_and_refills(valkey_url):
    for store in await stores(valkey_url):
        clock = Clock()
        lim = Limiter(store, TENANTS, TEAMS, ORGS, clock)
        res, _ = await lim.reserve("acme", 600)
        assert res is not None
        none, wait_ms = await lim.reserve("acme", 600)  # 400 left in the tenant bucket
        assert none is None and 0 < wait_ms <= 60_000 * 200 / 1000 + 1000
        await lim.settle(res, 100)  # used 100 of 600 -> 500 refunded
        res2, _ = await lim.reserve("acme", 600)
        assert res2 is not None
        clock.t += 60  # a full minute refills the bucket
        assert (await lim.reserve("acme", 900))[0] is not None


async def test_team_limit_applies_across_its_tenants_and_all_or_nothing(valkey_url):
    tenants = {"a": {"tpm": 1000, "team": "t"}, "b": {"tpm": 1000, "team": "t"}}
    for store in await stores(valkey_url):
        lim = Limiter(store, tenants, {"t": {"tpm": 1200}}, {}, Clock())
        assert (await lim.reserve("a", 800))[0] is not None
        assert (await lim.reserve("b", 800))[0] is None  # team bucket has 400 left
        assert (await lim.reserve("b", 400))[
            0
        ] is not None  # tenant b's bucket was not charged by the failure


async def test_request_larger_than_capacity_never_fits(valkey_url):
    for store in await stores(valkey_url):
        lim = Limiter(store, TENANTS, TEAMS, ORGS, Clock())
        assert await lim.reserve("acme", 5000) == (None, -1)


async def test_budgets_block_after_hard_limit_and_warn_at_soft(valkey_url):
    for store in await stores(valkey_url):
        lim = Limiter(store, TENANTS, TEAMS, ORGS, Clock())
        assert await lim.over_budget("acme") is None
        warnings = await lim.add_spend("acme", 0.6)
        assert [w[0] for w in warnings] == ["org:acme-inc"]
        await lim.add_spend("acme", 0.5)
        assert await lim.over_budget("acme") == "org:acme-inc"


def test_estimate_and_cost():
    body = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100}
    assert 190 < estimate_tokens(body) < 220
    pricing = Pricing({"m": Price(input_per_mtok=1.0, output_per_mtok=4.0, cached_input_per_mtok=0.1)})
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 400}}
    assert abs(pricing.cost("m", usage) - (600 * 1 + 400 * 0.1 + 500 * 4) / 1e6) < 1e-12
    assert pricing.cost("unknown", usage) == 0


def test_gateway_enforces_limits_and_budgets():
    provider = FakeProvider(models=["m"])
    settings = Settings(
        keys={"sk-a": "acme"},
        tenants={"acme": {"rpm": 2}},
        prices={"m": {"input_per_mtok": 1_000_000, "output_per_mtok": 0}},  # $1 per prompt token
        orgs={},
    )
    with TestClient(create_app(settings, providers=[provider])) as gw:
        auth = {"Authorization": "Bearer sk-a"}
        body = {"model": "m", "messages": []}
        assert gw.post("/v1/chat/completions", json=body, headers=auth).status_code == 200
        assert gw.post("/v1/chat/completions", json=body, headers=auth).status_code == 200
        limited = gw.post("/v1/chat/completions", json=body, headers=auth)
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
        assert gw.post("/v1/chat/completions", json=body).status_code == 200  # other tenant unaffected

    budget = Settings(keys={"sk-a": "acme"}, tenants={"acme": {"budget_usd": 0.5}},
                      prices={"m": {"input_per_mtok": 0, "output_per_mtok": 1_000_000}})  # fmt: skip
    with TestClient(create_app(budget, providers=[FakeProvider(models=["m"])])) as gw:
        auth = {"Authorization": "Bearer sk-a"}
        assert (
            gw.post("/v1/chat/completions", json={"model": "m", "messages": []}, headers=auth).status_code
            == 200
        )
        blocked = gw.post("/v1/chat/completions", json={"model": "m", "messages": []}, headers=auth)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "budget_exceeded"
