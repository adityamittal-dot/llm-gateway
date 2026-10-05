import json
import zlib

import fakeredis
import numpy as np
from fastapi.testclient import TestClient
from test_providers import FakeProvider

from llm_gateway.app import create_app
from llm_gateway.cache import CacheConfig, MemoryCacheStore, RedisCacheStore, ResponseCache, replay_as_chunks
from llm_gateway.config import Settings


class BagOfWords:
    """Deterministic fake embedder: hashed bag of words (paraphrases with shared words score high)."""

    async def embed(self, texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("?", "").split():
                out[i, zlib.crc32(w.encode()) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    async def aclose(self):
        pass


def body(question: str, system: str = "Be brief.", **extra) -> dict:
    return {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "temperature": 0} | extra  # fmt: skip


async def test_semantic_hits_only_within_the_same_context():
    for store in (MemoryCacheStore(), RedisCacheStore(fakeredis.FakeAsyncRedis())):
        cache = ResponseCache(store, CacheConfig(threshold=0.8), BagOfWords())
        answer = {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}]}
        await cache.store_response("acme", body("what is the capital of france"), "semantic", answer)
        hit = await cache.lookup("acme", body("what is the capital city of france"), "semantic")
        assert hit is not None and hit.kind == "semantic" and hit.similarity > 0.8
        assert await cache.lookup("other-tenant", body("what is the capital of france"), "semantic") is None
        assert (
            await cache.lookup("acme", body("what is the capital of france", "Answer in French."), "semantic")
            is None
        )
        assert await cache.lookup("acme", body("how tall is the eiffel tower"), "semantic") is None
        exact = await cache.lookup("acme", body("what is the capital of france"), "exact")
        assert exact is not None and exact.kind == "exact"


def test_mode_rules():
    cache = ResponseCache(MemoryCacheStore(), CacheConfig(), None)
    assert cache.mode(None, body("q")) == "off"
    assert cache.mode("semantic", body("q", tools=[{"type": "function"}])) == "exact"
    assert cache.mode("semantic", body("q", temperature=1.2)) == "exact"
    assert cache.mode("bogus", body("q")) == "off"


def test_replayed_chunks_rebuild_the_answer():
    resp = {"id": "x", "model": "m", "choices": [{"message": {"content": "hello big world"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3}}  # fmt: skip
    chunks = replay_as_chunks(resp)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "hello big world" and chunks[-1]["usage"] == {"completion_tokens": 3}


def test_gateway_serves_hits_without_calling_the_provider_or_charging_limits():
    provider = FakeProvider(models=["m"], reply="Paris")
    settings = Settings(keys={"sk-a": "acme"}, tenants={"acme": {"rpm": 1}}, cache={"threshold": 0.8})
    with TestClient(create_app(settings, providers=[provider], embedder=BagOfWords())) as gw:
        auth = {"Authorization": "Bearer sk-a", "x-cache": "semantic"}
        first = gw.post("/v1/chat/completions", json=body("what is the capital of france"), headers=auth)
        assert first.headers["x-cache"] == "miss"
        again = gw.post("/v1/chat/completions", json=body("what is the capital city of france"), headers=auth)
        assert (
            again.headers["x-cache"] == "hit-semantic"
            and again.json()["choices"][0]["message"]["content"] == "Paris"
        )
        streamed = gw.post(
            "/v1/chat/completions", json=body("what is the capital of france", stream=True), headers=auth
        )
        assert streamed.headers["x-cache"] == "hit-exact" and streamed.text.endswith("data: [DONE]\n\n")
        events = [json.loads(e[6:]) for e in streamed.text.split("\n\n") if e.startswith("data: {")]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in events if c["choices"]) == "Paris"
        # rpm=1 was used by the single miss; the hits were not rate limited.
        assert len(provider.bodies) == 1
        miss = gw.post("/v1/chat/completions", json=body("how tall is the eiffel tower"), headers=auth)
        assert miss.status_code == 429


def test_replay_keeps_tool_calls():
    from llm_gateway.signals import StreamAccumulator

    call = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
    resp = {
        "model": "m",
        "choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}],
    }
    acc = StreamAccumulator()
    for chunk in replay_as_chunks(resp):
        acc.add(chunk)
    assert (
        acc.message()["tool_calls"][0]["function"] == call["function"] and acc.finish_reason == "tool_calls"
    )
