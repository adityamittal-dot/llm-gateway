import random

import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.providers import ErrorKind, ProviderError
from llm_gateway.retries import Retrier, RetryBudget, RetryPolicy


class Flaky:
    """Fails with `kind` for the first `failures` calls, then succeeds."""

    def __init__(self, kind, failures):
        self.kind, self.failures, self.calls = kind, failures, 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(self.kind, "flaky")
        return "ok"


async def no_sleep(_):
    pass


async def test_transient_errors_are_retried_until_success():
    fn = Flaky(ErrorKind.OVERLOADED, failures=2)
    assert await Retrier(RetryPolicy(max_attempts=3), sleep=no_sleep).call("p", fn) == "ok"
    assert fn.calls == 3


async def test_non_transient_errors_are_not_retried():
    for kind in (ErrorKind.BAD_REQUEST, ErrorKind.AUTH):
        fn = Flaky(kind, failures=1)
        with pytest.raises(ProviderError):
            await Retrier(sleep=no_sleep).call("p", fn)
        assert fn.calls == 1


async def test_attempts_are_capped():
    fn = Flaky(ErrorKind.TIMEOUT, failures=10)
    with pytest.raises(ProviderError):
        await Retrier(RetryPolicy(max_attempts=3), sleep=no_sleep).call("p", fn)
    assert fn.calls == 3


async def test_budget_stops_retry_storms_during_an_outage():
    retrier = Retrier(RetryPolicy(max_attempts=3), sleep=no_sleep)
    calls = 0
    for _ in range(100):
        fn = Flaky(ErrorKind.OVERLOADED, failures=99)
        with pytest.raises(ProviderError):
            await retrier.call("down", fn)
        calls += fn.calls
    # Without a budget: 300 calls. With it: 100 first attempts + ~3 starting tokens + 0.2 per request.
    assert calls < 100 + 3 + 0.2 * 100 + 2
    assert retrier.budget("healthy").tokens == 3  # budgets are per provider


def test_full_jitter_delays_stay_within_bounds():
    policy, rng = RetryPolicy(base_delay_s=0.2, max_delay_s=1.0), random.Random(0)
    for attempt in range(1, 8):
        for _ in range(50):
            assert 0 <= policy.delay(attempt, rng) <= min(1.0, 0.2 * 2**attempt)


def test_budget_refills_with_traffic():
    budget = RetryBudget(ratio=0.5, min_tokens=0, max_tokens=2)
    assert not budget.try_spend()
    budget.on_request(), budget.on_request()
    assert budget.try_spend() and not budget.try_spend()


class FlakyStreamProvider:
    name, models = "flaky", ["*"]

    def __init__(self):
        self.opens = 0

    def serves(self, model):
        return True

    async def stream(self, body):
        self.opens += 1
        if self.opens == 1:
            raise ProviderError(ErrorKind.RATE_LIMITED, "slow down")
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    async def list_models(self):
        return []

    async def aclose(self):
        pass


def test_streams_are_retried_before_the_first_chunk():
    provider = FlakyStreamProvider()
    app = create_app(Settings(retry={"base_delay_s": 0}), providers=[provider])
    with TestClient(app) as gw:
        resp = gw.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 200 and '"content": "ok"' in resp.text and provider.opens == 2


def test_retry_policy_from_config_casts_types():
    assert RetryPolicy.from_config({"max_attempts": "5", "base_delay_s": "0.5", "junk": 1}) == RetryPolicy(
        max_attempts=5, base_delay_s=0.5
    )
