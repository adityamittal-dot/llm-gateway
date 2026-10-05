"""Retries with exponential backoff, full jitter and a retry budget.

Only transient provider errors are retried on the same provider: rate limits, overload,
timeouts and connection failures. Bad requests fail everywhere, and an auth failure will not
fix itself, so neither is retried (both can still fail over to another provider later).

The retry budget caps retries at a fraction of recent traffic per provider (like gRPC/Finagle
retry budgets): every request deposits `ratio` tokens, every retry withdraws one. When a
provider is down, retries stop after the budget is spent instead of multiplying the load.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from llm_gateway.providers.base import ErrorKind, ProviderError

T = TypeVar("T")
RETRY_SAME_PROVIDER = {ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, ErrorKind.TIMEOUT, ErrorKind.UNAVAILABLE}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    # Never start a retry later than this after the first attempt began.
    max_elapsed_s: float = 30.0

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Full jitter: uniform in [0, min(max, base * 2^attempt)]."""
        return rng.uniform(0, min(self.max_delay_s, self.base_delay_s * 2**attempt))

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RetryPolicy":
        cfg = cfg or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


class RetryBudget:
    def __init__(self, ratio: float = 0.2, min_tokens: float = 3.0, max_tokens: float = 20.0):
        self.ratio, self.max_tokens = ratio, max_tokens
        self.tokens = min_tokens

    def on_request(self) -> None:
        self.tokens = min(self.max_tokens, self.tokens + self.ratio)

    def try_spend(self) -> bool:
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class Retrier:
    def __init__(self, policy: RetryPolicy | None = None, seed: int | None = None, sleep=asyncio.sleep):
        self.policy = policy or RetryPolicy()
        self.budgets: dict[str, RetryBudget] = {}
        self._rng = random.Random(seed)
        self._sleep = sleep

    def budget(self, provider: str) -> RetryBudget:
        return self.budgets.setdefault(provider, RetryBudget())

    async def call(self, provider: str, fn: Callable[[], Awaitable[T]], on_retry=None) -> T:
        """Run `fn`, retrying transient ProviderErrors within the policy and the provider's budget."""
        budget = self.budget(provider)
        budget.on_request()
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                return await fn()
            except ProviderError as err:
                attempt += 1
                if (
                    err.kind not in RETRY_SAME_PROVIDER
                    or attempt >= self.policy.max_attempts
                    or time.monotonic() - started > self.policy.max_elapsed_s
                    or not budget.try_spend()
                ):
                    raise
                delay = self.policy.delay(attempt, self._rng)
                if on_retry:
                    on_retry(attempt, err, delay)
                await self._sleep(delay)
