import fakeredis
from fastapi.testclient import TestClient
from test_providers import FakeProvider

from llm_gateway.app import create_app
from llm_gateway.breaker import BreakerConfig, CircuitBreaker, MemoryStore, RedisStore
from llm_gateway.config import Settings
from llm_gateway.providers import ErrorKind, ProviderError


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def breakers():
    """The same breaker logic over both stores."""
    for store in (MemoryStore(), RedisStore(fakeredis.FakeAsyncRedis())):
        clock = Clock()
        yield (
            CircuitBreaker(store, BreakerConfig(min_requests=4, failure_ratio=0.5, cooldown_s=30), clock),
            clock,
        )


async def test_breaker_opens_on_failure_ratio_and_recovers_through_a_probe():
    for cb, clock in breakers():
        for _ in range(2):
            await cb.record("p", None)
        await cb.record("p", ErrorKind.OVERLOADED)
        assert await cb.state("p") == "closed"  # 1/3 failed, below min_requests anyway
        await cb.record("p", ErrorKind.TIMEOUT)
        assert await cb.state("p") == "open"  # 2/4 failed
        assert not await cb.allow("p")

        clock.t += 31
        assert await cb.state("p") == "half_open"
        assert await cb.allow("p")  # this caller holds the probe
        assert not await cb.allow("p")  # everyone else waits
        await cb.record("p", ErrorKind.OVERLOADED)  # probe failed -> open again
        assert await cb.state("p") == "open"

        clock.t += 31
        assert await cb.allow("p")
        await cb.record("p", None)  # probe succeeded -> closed, window cleared
        assert await cb.state("p") == "closed"
        assert (await cb.snapshot(["p"]))["p"]["window_failures"] == 0


async def test_bad_request_probe_neither_closes_nor_reopens():
    for cb, clock in breakers():
        for _ in range(4):
            await cb.record("p", ErrorKind.TIMEOUT)
        clock.t += 31
        assert await cb.allow("p")
        await cb.record("p", ErrorKind.BAD_REQUEST)
        assert await cb.state("p") == "half_open"
        assert await cb.allow("p")  # the probe lock was released for the next request


async def test_bad_requests_do_not_trip_the_breaker():
    for cb, _ in breakers():
        for _ in range(10):
            await cb.record("p", ErrorKind.BAD_REQUEST)
        assert await cb.state("p") == "closed"


async def test_failures_age_out_of_the_window():
    for cb, clock in breakers():
        for _ in range(3):
            await cb.record("p", ErrorKind.OVERLOADED)
        clock.t += 120  # beyond the 60 s window
        await cb.record("p", ErrorKind.OVERLOADED)
        assert await cb.state("p") == "closed"


def gateway(*providers, **settings):
    settings.setdefault("retry", {"base_delay_s": 0.0, "max_attempts": 1})
    settings.setdefault("breaker", {"min_requests": 3, "failure_ratio": 0.5, "cooldown_s": 60})
    return TestClient(create_app(Settings(**settings), providers=list(providers)))


ROUTES = {"chat": [{"provider": "primary", "model": "big"}, {"provider": "backup", "model": "small"}]}


def test_failover_to_the_next_provider_and_breaker_opens():
    primary = FakeProvider("primary", models=["big"], error=ProviderError(ErrorKind.OVERLOADED, "down"))
    backup = FakeProvider("backup", models=["small"], reply="from backup")
    with gateway(primary, backup, routes=ROUTES, admin_key="adm") as gw:
        for stream in (False, True, False, False, True):
            resp = gw.post("/v1/chat/completions", json={"model": "chat", "messages": [], "stream": stream})
            assert resp.status_code == 200
            if stream:
                assert '"content": "f"' in resp.text and '"model": "chat"' in resp.text
            else:
                assert resp.json()["choices"][0]["message"]["content"] == "from backup"
                assert resp.json()["model"] == "chat"
        states = gw.get("/admin/breakers", headers={"X-Admin-Key": "adm"}).json()
    assert states["primary"]["state"] == "open"
    assert len(primary.bodies) == 3  # after the breaker opened, the primary was skipped
    assert backup.bodies[0]["model"] == "small"  # the backup gets its own model id


def test_bad_request_does_not_fail_over():
    primary = FakeProvider("primary", models=["big"], error=ProviderError(ErrorKind.BAD_REQUEST, "bad", 422))
    backup = FakeProvider("backup", models=["small"])
    with gateway(primary, backup, routes=ROUTES) as gw:
        resp = gw.post("/v1/chat/completions", json={"model": "chat", "messages": []})
    assert resp.status_code == 422 and backup.bodies == []


def test_all_providers_down_returns_the_last_error_then_503_when_breakers_are_open():
    down = ProviderError(ErrorKind.TIMEOUT, "slow")
    a = FakeProvider("primary", models=["big"], error=down)
    b = FakeProvider("backup", models=["small"], error=down)
    with gateway(a, b, routes=ROUTES) as gw:
        codes = [gw.post("/v1/chat/completions", json={"model": "chat", "messages": []}).status_code
                 for _ in range(5)]  # fmt: skip
    assert codes[:3] == [504, 504, 504]
    assert codes[-1] == 503  # both breakers open: fail fast without calling anyone


def test_routes_are_listed_as_models():
    with gateway(FakeProvider("primary", models=["big"]), FakeProvider("backup", models=["small"]),
                 routes=ROUTES) as gw:  # fmt: skip
        assert [m["id"] for m in gw.get("/v1/models").json()["data"]][:1] == ["chat"]
