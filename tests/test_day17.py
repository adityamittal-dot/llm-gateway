import random

from fastapi.testclient import TestClient
from test_providers import FakeProvider

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.promptcache import CacheBreakDetector
from llm_gateway.quality import QualityConfig, QualityMonitor
from llm_gateway.sessions import MemorySessionStore, Sessions
from llm_gateway.signals import ResponseSignals

rng = random.Random(0)


def response(faulty: bool = False) -> ResponseSignals:
    tokens = rng.randint(5, 12) if faulty else int(rng.gauss(150, 25))
    return ResponseSignals(ts=0, request_id="r", tenant="t", provider="p", model="m", stream=False, status_code=200,
                           latency_s=1.0, completion_tokens=max(tokens, 1),
                           finish_reason="length" if faulty else "stop")  # fmt: skip


def monitor(**cfg) -> QualityMonitor:
    return QualityMonitor(QualityConfig(warmup=100, **cfg), seed=1)


def test_quality_monitor_stays_healthy_on_healthy_traffic():
    m = monitor()
    for _ in range(3000):
        m.observe("a", "t", response())
    assert m.providers["a"].level == 0


def test_quality_monitor_escalates_on_a_silent_fault_then_recovers():
    m = monitor(recovery_window=30)
    for _ in range(200):
        m.observe("a", "t", response())
    levels = []
    for _ in range(40):
        m.observe("a", "t", response(faulty=True))
        levels.append(m.providers["a"].level)
    assert levels[-1] == 3 and levels.index(2) < 15  # shift within a handful of bad responses
    for _ in range(60):
        m.observe("a", "t", response())
    assert m.providers["a"].level == 0
    assert [e["to"] for e in m.events][-1] == 0


def test_admission_by_level():
    m = monitor(shift_fraction=1.0, canary_fraction=0.0)
    m.providers["a"].level = 2
    assert not m.admit("a", has_fallback=True)
    assert m.admit("a", has_fallback=False)  # never strand a request without a fallback
    m.providers["a"].level = 3
    assert not m.admit("a", has_fallback=True)


def test_tenant_shift_seen_on_two_providers_does_not_blame_the_provider():
    m = monitor()
    for provider in ("a", "b"):
        for _ in range(200):
            m.observe(provider, "chat", response())
    for _ in range(100):
        for provider in ("a", "b"):
            m.observe(provider, "chat", response(faulty=True))
    assert m.shifted_tenants() == {"chat"}
    assert m.providers["a"].level <= 2  # the pooled statistic stops counting the shifted tenant


def test_cache_break_diagnosis_hints():
    d = CacheBreakDetector()
    tools = [{"type": "function", "function": {"name": n}} for n in ("a", "b")]
    base = {
        "tools": tools,
        "messages": [{"role": "system", "content": "You are X."}, {"role": "user", "content": "q"}],
    }
    usage = lambda cached: {"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": cached}}
    assert d.observe("t", "m", base, usage(3500)) is None
    changed = {
        **base,
        "messages": [{"role": "system", "content": "You are X. Time: 12:01"}, base["messages"][1]],
    }
    event = d.observe("t", "m", changed, usage(0))
    assert event["changed_block"] == "messages[0] (system)" and "dynamic values" in event["hint"]
    d.observe("t", "m", base, usage(3500))
    reordered = {**base, "tools": list(reversed(tools))}
    assert "different order" in d.observe("t", "m", reordered, usage(0))["hint"]
    d.observe("t", "m", base, usage(3500))
    assert "expired" in d.observe("t", "m", base, usage(0))["hint"]


async def test_sessions_budget_iterations_and_loop_detection():
    s = Sessions(
        MemorySessionStore(), {"t": {"session_budget_usd": 1.0, "session_max_requests": 5, "loop_repeats": 3}}
    )
    assert await s.admit("t", "s1") is None
    await s.after_response("t", "s1", 1.2, [])
    assert (await s.admit("t", "s1"))[0] == "session_budget_exceeded"
    for _ in range(5):
        assert await s.admit("t", "s2") is None
    assert (await s.admit("t", "s2"))[0] == "session_iterations_exceeded"
    call = [{"function": {"name": "search", "arguments": '{"q": "same"}'}}]
    assert not await s.after_response("t", "s3", 0, call)
    assert not await s.after_response("t", "s3", 0, call)
    assert await s.after_response("t", "s3", 0, call)
    assert (await s.admit("t", "s3"))[0] == "agent_loop_detected"
    assert await s.admit("t", None) is None  # no session header, no session controls


def test_gateway_quality_breaker_moves_traffic_off_a_silently_degraded_provider():
    primary, backup = (
        FakeProvider("primary", models=["m"], reply="x" * 50),
        FakeProvider("backup", models=["m2"]),
    )
    routes = {"chat": [{"provider": "primary", "model": "m"}, {"provider": "backup", "model": "m2"}]}
    settings = Settings(routes=routes, admin_key="adm", quality={"warmup": 20, "canary_fraction": 0.0},
                        retry={"base_delay_s": 0})  # fmt: skip
    with TestClient(create_app(settings, providers=[primary, backup])) as gw:

        def send(n):
            served = []
            for _ in range(n):
                served.append(gw.post("/v1/chat/completions", json={"model": "chat", "messages": []})
                              .headers["x-gateway-provider"])  # fmt: skip
            return served

        assert set(send(60)) == {"primary"}
        primary.reply = "x"  # silent degradation: still 200 OK, but answers collapse to one token...
        primary.finish = "length"
        served = send(80)
        state = gw.get("/admin/quality", headers={"X-Admin-Key": "adm"}).json()
    assert state["providers"]["primary"]["level"] == 3
    assert served[-20:] == ["backup"] * 20
