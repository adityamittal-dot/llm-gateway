"""Agent session controls, keyed by the `X-Session-Id` header.

- Session budget: a hard USD cap per session (`session_budget_usd`, per tenant).
- Iteration cap: at most `session_max_requests` model calls per session.
- Loop detection: if the model returns the *same tool call* (name + arguments)
  `loop_repeats` times in a row within a session, the agent is stuck; further calls in that session
  are rejected (`action: block`) or only logged (`action: warn`).

State lives in the limiter's store (Redis when configured) with a 24-hour TTL.
"""

import hashlib
import json
from dataclasses import dataclass

SESSION_TTL_S = 86_400


@dataclass(frozen=True)
class SessionConfig:
    session_budget_usd: float | None = None
    session_max_requests: int | None = None
    loop_repeats: int = 3
    loop_action: str = "block"  # block | warn

    @classmethod
    def for_tenant(cls, tenant_cfg: dict) -> "SessionConfig":
        keys = cls.__dataclass_fields__
        return cls(**{k: tenant_cfg[k] for k in keys if k in tenant_cfg})


def tool_signature(tool_calls: list[dict]) -> str | None:
    if not tool_calls:
        return None
    calls = []
    for c in tool_calls:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            pass
        calls.append([fn.get("name"), args])
    return hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()[:16]


class MemorySessionStore:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.spend: dict[str, float] = {}
        self.calls: dict[str, list[str]] = {}
        self.flags: set[str] = set()

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def get_count(self, key: str) -> int:
        return self.counts.get(key, 0)

    async def add_spend(self, key: str, usd: float) -> float:
        self.spend[key] = self.spend.get(key, 0.0) + usd
        return self.spend[key]

    async def get_spend(self, key: str) -> float:
        return self.spend.get(key, 0.0)

    async def push_call(self, key: str, sig: str, keep: int) -> list[str]:
        items = [sig] + self.calls.get(key, [])
        self.calls[key] = items[:keep]
        return self.calls[key]

    async def flag(self, key: str) -> None:
        self.flags.add(key)

    async def flagged(self, key: str) -> bool:
        return key in self.flags


class RedisSessionStore:
    def __init__(self, redis):
        self.r = redis

    async def incr(self, key: str) -> int:
        pipe = self.r.pipeline()
        pipe.incr(key)
        pipe.expire(key, SESSION_TTL_S)
        n, _ = await pipe.execute()
        return int(n)

    async def get_count(self, key: str) -> int:
        return int(await self.r.get(key) or 0)

    async def add_spend(self, key: str, usd: float) -> float:
        pipe = self.r.pipeline()
        pipe.incrbyfloat(key, usd)
        pipe.expire(key, SESSION_TTL_S)
        value, _ = await pipe.execute()
        return float(value)

    async def get_spend(self, key: str) -> float:
        return float(await self.r.get(key) or 0)

    async def push_call(self, key: str, sig: str, keep: int) -> list[str]:
        pipe = self.r.pipeline()
        pipe.lpush(key, sig)
        pipe.ltrim(key, 0, keep - 1)
        pipe.expire(key, SESSION_TTL_S)
        pipe.lrange(key, 0, keep - 1)
        *_, items = await pipe.execute()
        return [i.decode() if isinstance(i, bytes) else i for i in items]

    async def flag(self, key: str) -> None:
        await self.r.set(key, "1", ex=SESSION_TTL_S)

    async def flagged(self, key: str) -> bool:
        return bool(await self.r.exists(key))


class Sessions:
    def __init__(self, store, tenants: dict | None = None):
        self.store = store
        self.tenants = tenants or {}

    def config(self, tenant: str) -> SessionConfig:
        return SessionConfig.for_tenant(self.tenants.get(tenant, {}))

    @staticmethod
    def _k(kind: str, tenant: str, session: str) -> str:
        return f"sess:{kind}:{tenant}:{session}"

    async def admit(self, tenant: str, session: str | None) -> tuple[str, str] | None:
        """(code, message) if this session must not make another call, else None. Counts the call."""
        if not session:
            return None
        cfg = self.config(tenant)
        if await self.store.flagged(self._k("loop", tenant, session)) and cfg.loop_action == "block":
            return (
                "agent_loop_detected",
                f"Session {session!r} repeated the same tool call {cfg.loop_repeats} times.",
            )
        if cfg.session_budget_usd is not None and (
            await self.store.get_spend(self._k("spend", tenant, session)) >= cfg.session_budget_usd
        ):
            return (
                "session_budget_exceeded",
                f"Session {session!r} reached its ${cfg.session_budget_usd} budget.",
            )
        count = await self.store.get_count(self._k("n", tenant, session))
        if cfg.session_max_requests is not None and count >= cfg.session_max_requests:
            return (
                "session_iterations_exceeded",
                f"Session {session!r} reached {cfg.session_max_requests} calls.",
            )
        await self.store.incr(self._k("n", tenant, session))
        return None

    async def after_response(
        self, tenant: str, session: str | None, cost_usd: float, tool_calls: list[dict]
    ) -> bool:
        """Charge the session and check for a tool-call loop; returns True when a loop was just detected."""
        if not session:
            return False
        cfg = self.config(tenant)
        if cost_usd:
            await self.store.add_spend(self._k("spend", tenant, session), cost_usd)
        sig = tool_signature(tool_calls)
        if sig is None:
            return False
        recent = await self.store.push_call(self._k("calls", tenant, session), sig, cfg.loop_repeats)
        if len(recent) >= cfg.loop_repeats and len(set(recent)) == 1:
            await self.store.flag(self._k("loop", tenant, session))
            return True
        return False
