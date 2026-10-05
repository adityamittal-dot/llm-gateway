"""Circuit breaker per provider, with state shared across gateway processes through Redis.

States:
  closed     requests flow; outcomes are counted in 10-second buckets over a sliding window
  open       the failure ratio over the window reached `failure_ratio` (with at least
             `min_requests`); requests skip this provider until `cooldown_s` has passed
  half_open  after the cooldown one probe request at a time is let through (a short Redis lock
             makes it one probe across all gateway tasks); success closes the breaker, failure
             re-opens it

Only errors that say something about the provider count as failures: rate limits, overload,
timeouts, connection failures and rejected credentials. A bad request is the client's fault.

Two stores implement the same interface: Redis (shared by all Fargate tasks) and in-process
memory (single process, tests, local runs without Redis).
"""

import time
from dataclasses import dataclass

from llm_gateway.providers.base import ErrorKind

FAILURE_KINDS = {
    ErrorKind.RATE_LIMITED,
    ErrorKind.OVERLOADED,
    ErrorKind.TIMEOUT,
    ErrorKind.UNAVAILABLE,
    ErrorKind.AUTH,
}
BUCKET_S = 10


@dataclass(frozen=True)
class BreakerConfig:
    window_s: int = 60
    min_requests: int = 5
    failure_ratio: float = 0.5
    cooldown_s: float = 30.0
    probe_lock_s: float = 10.0

    @classmethod
    def from_config(cls, cfg: dict | None) -> "BreakerConfig":
        cfg = cfg or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


class MemoryStore:
    def __init__(self):
        self.buckets: dict[tuple[str, int], list[int]] = {}
        self.state: dict[str, dict] = {}
        self.locks: dict[str, float] = {}

    async def add(self, provider: str, bucket: int, failed: bool, ttl: int) -> None:
        counts = self.buckets.setdefault((provider, bucket), [0, 0])
        counts[0] += 1
        counts[1] += int(failed)

    async def window(self, provider: str, buckets: list[int]) -> tuple[int, int]:
        total = failed = 0
        for b in buckets:
            t, f = self.buckets.get((provider, b), (0, 0))
            total, failed = total + t, failed + f
        return total, failed

    async def get_state(self, provider: str) -> dict:
        return dict(self.state.get(provider, {}))

    async def set_state(self, provider: str, state: str, opened_at: float) -> None:
        self.state[provider] = {"state": state, "opened_at": opened_at}

    async def clear_window(self, provider: str) -> None:
        for key in [k for k in self.buckets if k[0] == provider]:
            del self.buckets[key]

    async def try_lock(self, provider: str, ttl: float, now: float) -> bool:
        if self.locks.get(provider, 0) > now:
            return False
        self.locks[provider] = now + ttl
        return True

    async def unlock(self, provider: str) -> None:
        self.locks.pop(provider, None)


class RedisStore:
    """Breaker state in Redis: `cb:{p}:b:{bucket}` hashes (total/failed), `cb:{p}:state`, `cb:{p}:probe`."""

    def __init__(self, redis):
        self.r = redis

    async def add(self, provider: str, bucket: int, failed: bool, ttl: int) -> None:
        key = f"cb:{provider}:b:{bucket}"
        pipe = self.r.pipeline()
        pipe.hincrby(key, "total", 1)
        if failed:
            pipe.hincrby(key, "failed", 1)
        pipe.expire(key, ttl)
        await pipe.execute()

    async def window(self, provider: str, buckets: list[int]) -> tuple[int, int]:
        pipe = self.r.pipeline()
        for b in buckets:
            pipe.hmget(f"cb:{provider}:b:{b}", "total", "failed")
        total = failed = 0
        for t, f in await pipe.execute():
            total += int(t or 0)
            failed += int(f or 0)
        return total, failed

    async def get_state(self, provider: str) -> dict:
        raw = await self.r.hgetall(f"cb:{provider}:state")
        return {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()}  # fmt: skip

    async def set_state(self, provider: str, state: str, opened_at: float) -> None:
        await self.r.hset(f"cb:{provider}:state", mapping={"state": state, "opened_at": str(opened_at)})

    async def clear_window(self, provider: str) -> None:
        keys = [k async for k in self.r.scan_iter(match=f"cb:{provider}:b:*")]
        if keys:
            await self.r.delete(*keys)

    async def try_lock(self, provider: str, ttl: float, now: float) -> bool:
        return bool(await self.r.set(f"cb:{provider}:probe", "1", nx=True, px=int(ttl * 1000)))

    async def unlock(self, provider: str) -> None:
        await self.r.delete(f"cb:{provider}:probe")


class CircuitBreaker:
    def __init__(self, store, config: BreakerConfig | None = None, clock=time.time):
        self.store = store
        self.config = config or BreakerConfig()
        self.clock = clock

    def _buckets(self, now: float) -> list[int]:
        current = int(now // BUCKET_S)
        return list(range(current - self.config.window_s // BUCKET_S + 1, current + 1))

    async def state(self, provider: str) -> str:
        st = await self.store.get_state(provider)
        if st.get("state") == "open" and self.clock() - float(st["opened_at"]) >= self.config.cooldown_s:
            return "half_open"
        return st.get("state", "closed")

    async def allow(self, provider: str) -> bool:
        """May a request go to this provider now? In half-open, only the holder of the probe lock may."""
        state = await self.state(provider)
        if state == "closed":
            return True
        if state == "half_open":
            return await self.store.try_lock(provider, self.config.probe_lock_s, self.clock())
        return False

    async def record(self, provider: str, error: ErrorKind | None) -> None:
        now = self.clock()
        failed = error in FAILURE_KINDS
        state = await self.state(provider)
        if state == "half_open":  # this was the probe
            await self.store.unlock(provider)
            if failed:
                await self.store.set_state(provider, "open", now)
            elif error is None:  # only a real success proves the provider is back
                await self.store.clear_window(provider)
                await self.store.set_state(provider, "closed", 0)
            # A bad request says nothing about provider health: stay half-open for the next probe.
            return
        await self.store.add(provider, int(now // BUCKET_S), failed, self.config.window_s + BUCKET_S)
        if failed and state == "closed":
            total, n_failed = await self.store.window(provider, self._buckets(now))
            if total >= self.config.min_requests and n_failed / total >= self.config.failure_ratio:
                await self.store.set_state(provider, "open", now)

    async def snapshot(self, providers: list[str]) -> dict[str, dict]:
        now = self.clock()
        out = {}
        for p in providers:
            total, failed = await self.store.window(p, self._buckets(now))
            out[p] = {"state": await self.state(p), "window_requests": total, "window_failures": failed}
        return out
