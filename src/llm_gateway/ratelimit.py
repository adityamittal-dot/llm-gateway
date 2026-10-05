"""Token-aware, hierarchical rate limits and spend budgets.

Scopes form a hierarchy: key (tenant) -> team -> org. Each scope may set
  rpm         requests per minute
  tpm         tokens per minute (prompt + completion)
  budget_usd  hard monthly spend limit (requests are rejected once it is reached)
  soft_budget_usd  monthly spend at which a warning is logged

Rate limits are token buckets (capacity = per-minute limit, refilled continuously). A request
reserves its *estimated* tokens up front (prompt estimate + max_tokens) from every bucket of every
scope in one atomic step, all or nothing; when the response finishes, the unused part of the
estimate is refunded. In Redis the reserve is a Lua script, so concurrent gateway tasks cannot
overspend a bucket.
"""

import calendar
import json
import time
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 1024  # reservation when a request sets no max_tokens
CHARS_PER_TOKEN = 4

RESERVE_LUA = """
-- KEYS: bucket keys. ARGV: now, then capacity, rate_per_s, cost for each key.
local now = tonumber(ARGV[1])
local state = {}
local wait = 0
for i, key in ipairs(KEYS) do
  local cap = tonumber(ARGV[2 + (i - 1) * 3])
  local rate = tonumber(ARGV[3 + (i - 1) * 3])
  local cost = tonumber(ARGV[4 + (i - 1) * 3])
  local b = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(b[1]) or cap
  local ts = tonumber(b[2]) or now
  tokens = math.min(cap, tokens + math.max(0, now - ts) * rate)
  if cost > cap then
    return {0, -1}
  end
  if tokens < cost then
    wait = math.max(wait, (cost - tokens) / rate)
  end
  state[i] = tokens
end
if wait > 0 then
  return {0, math.ceil(wait * 1000)}
end
for i, key in ipairs(KEYS) do
  local cap = tonumber(ARGV[2 + (i - 1) * 3])
  local rate = tonumber(ARGV[3 + (i - 1) * 3])
  local cost = tonumber(ARGV[4 + (i - 1) * 3])
  redis.call('HSET', key, 'tokens', state[i] - cost, 'ts', now)
  redis.call('EXPIRE', key, math.ceil(cap / rate) * 2 + 60)
end
return {1, 0}
"""

REFUND_LUA = """
-- KEYS: bucket keys. ARGV: capacity, amount for each key.
for i, key in ipairs(KEYS) do
  local cap = tonumber(ARGV[1 + (i - 1) * 2])
  local amount = tonumber(ARGV[2 + (i - 1) * 2])
  local tokens = tonumber(redis.call('HGET', key, 'tokens'))
  if tokens then
    redis.call('HSET', key, 'tokens', math.min(cap, tokens + amount))
  end
end
return 1
"""


@dataclass(frozen=True)
class Bucket:
    key: str
    capacity: float
    rate_per_s: float


@dataclass
class Reservation:
    buckets: list[Bucket]
    tokens: int  # estimated tokens reserved from every tpm bucket


def estimate_tokens(body: dict) -> int:
    chars = len(json.dumps(body.get("messages") or [])) + len(json.dumps(body.get("tools") or []))
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
    return chars // CHARS_PER_TOKEN + int(max_tokens)


def month_key(now: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(now))


def seconds_to_month_end(now: float) -> int:
    t = time.gmtime(now)
    days = calendar.monthrange(t.tm_year, t.tm_mon)[1]
    end = calendar.timegm((t.tm_year, t.tm_mon, days, 23, 59, 59))
    return max(1, end - int(now))


class MemoryLimitStore:
    """Single-process store with the same semantics as the Redis scripts."""

    def __init__(self):
        self.buckets: dict[str, tuple[float, float]] = {}
        self.spend: dict[str, float] = {}

    async def reserve(self, items: list[tuple[Bucket, int]], now: float) -> tuple[bool, int]:
        state, wait = [], 0.0
        for bucket, cost in items:
            if cost > bucket.capacity:
                return False, -1
            tokens, ts = self.buckets.get(bucket.key, (bucket.capacity, now))
            tokens = min(bucket.capacity, tokens + max(0.0, now - ts) * bucket.rate_per_s)
            if tokens < cost:
                wait = max(wait, (cost - tokens) / bucket.rate_per_s)
            state.append(tokens)
        if wait > 0:
            return False, int(wait * 1000) + 1
        for (bucket, cost), tokens in zip(items, state, strict=True):
            self.buckets[bucket.key] = (tokens - cost, now)
        return True, 0

    async def refund(self, items: list[tuple[Bucket, int]]) -> None:
        for bucket, amount in items:
            if bucket.key in self.buckets:
                tokens, ts = self.buckets[bucket.key]
                self.buckets[bucket.key] = (min(bucket.capacity, tokens + amount), ts)

    async def add_spend(self, key: str, usd: float, ttl: int) -> float:
        self.spend[key] = self.spend.get(key, 0.0) + usd
        return self.spend[key]

    async def get_spend(self, keys: list[str]) -> list[float]:
        return [self.spend.get(k, 0.0) for k in keys]


class RedisLimitStore:
    def __init__(self, redis):
        self.r = redis
        self._reserve = redis.register_script(RESERVE_LUA)
        self._refund = redis.register_script(REFUND_LUA)

    async def reserve(self, items: list[tuple[Bucket, int]], now: float) -> tuple[bool, int]:
        args: list = [now]
        for bucket, cost in items:
            args += [bucket.capacity, bucket.rate_per_s, cost]
        ok, wait_ms = await self._reserve(keys=[b.key for b, _ in items], args=args)
        return bool(ok), int(wait_ms)

    async def refund(self, items: list[tuple[Bucket, int]]) -> None:
        args: list = []
        for bucket, amount in items:
            args += [bucket.capacity, amount]
        await self._refund(keys=[b.key for b, _ in items], args=args)

    async def add_spend(self, key: str, usd: float, ttl: int) -> float:
        pipe = self.r.pipeline()
        pipe.incrbyfloat(key, usd)
        pipe.expire(key, ttl)
        value, _ = await pipe.execute()
        return float(value)

    async def get_spend(self, keys: list[str]) -> list[float]:
        return [float(v or 0) for v in await self.r.mget(keys)]


class Limiter:
    def __init__(self, store, tenants: dict | None = None, teams: dict | None = None, orgs: dict | None = None,
                 clock=time.time):  # fmt: skip
        self.store = store
        self.tenants, self.teams, self.orgs = tenants or {}, teams or {}, orgs or {}
        self.clock = clock

    def scopes(self, tenant: str) -> list[tuple[str, dict]]:
        """(scope id, limits) from the key's tenant up to its org."""
        cfg = self.tenants.get(tenant, {})
        out = [(f"tenant:{tenant}", cfg)]
        if cfg.get("team"):
            out.append((f"team:{cfg['team']}", self.teams.get(cfg["team"], {})))
        if cfg.get("org"):
            out.append((f"org:{cfg['org']}", self.orgs.get(cfg["org"], {})))
        return out

    async def reserve(self, tenant: str, est_tokens: int) -> tuple[Reservation | None, int]:
        """Reserve 1 request and `est_tokens` tokens in every limited scope.
        Returns (reservation, 0) or (None, retry_after_ms); retry_after_ms = -1 means never fits."""
        items: list[tuple[Bucket, int]] = []
        tpm_buckets = []
        for scope, limits in self.scopes(tenant):
            if limits.get("rpm"):
                rpm = float(limits["rpm"])
                items.append((Bucket(f"rl:{scope}:rpm", rpm, rpm / 60), 1))
            if limits.get("tpm"):
                tpm = float(limits["tpm"])
                bucket = Bucket(f"rl:{scope}:tpm", tpm, tpm / 60)
                items.append((bucket, est_tokens))
                tpm_buckets.append(bucket)
        if not items:
            return Reservation([], 0), 0
        ok, wait_ms = await self.store.reserve(items, self.clock())
        return (Reservation(tpm_buckets, est_tokens), 0) if ok else (None, wait_ms)

    async def settle(self, reservation: Reservation, actual_tokens: int | None) -> None:
        """Refund the unused part of the estimate (never charge more than reserved after the fact)."""
        if not reservation.buckets or actual_tokens is None:
            return
        unused = reservation.tokens - actual_tokens
        if unused > 0:
            await self.store.refund([(b, unused) for b in reservation.buckets])

    async def over_budget(self, tenant: str) -> str | None:
        """The first scope whose monthly hard budget is exhausted, if any."""
        month = month_key(self.clock())
        limited = [(scope, float(limits["budget_usd"])) for scope, limits in self.scopes(tenant)
                   if limits.get("budget_usd") is not None]  # fmt: skip
        if not limited:
            return None
        spent = await self.store.get_spend([f"spend:{scope}:{month}" for scope, _ in limited])
        for (scope, budget), value in zip(limited, spent, strict=True):
            if value >= budget:
                return scope
        return None

    async def add_spend(self, tenant: str, usd: float) -> list[tuple[str, float, float]]:
        """Charge every scope; returns (scope, spent, soft budget) for scopes past their soft budget."""
        if usd <= 0:
            return []
        now = self.clock()
        month, ttl = month_key(now), seconds_to_month_end(now) + 86_400
        warnings = []
        for scope, limits in self.scopes(tenant):
            spent = await self.store.add_spend(f"spend:{scope}:{month}", usd, ttl)
            soft = limits.get("soft_budget_usd")
            if soft is not None and spent >= float(soft):
                warnings.append((scope, spent, float(soft)))
        return warnings
