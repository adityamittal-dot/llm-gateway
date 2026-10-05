"""Response caching: exact match first, then semantic similarity. Tenant-scoped and opt-in.

Correctness rules (README "Semantic cache correctness"):
- Every key includes the tenant: one customer's answer is never served to another.
- The exact key covers everything that changes the answer: model, all messages, sampling params,
  tools, response format.
- The semantic cache only compares requests that share the same *context* (tenant, model,
  params, system prompt and earlier turns); within that bucket it compares embeddings of the
  last user message. A paraphrased question in a different conversation never matches.
- Opt-in per request with `x-cache: exact` or `x-cache: semantic` (or a per-gateway default).
  Requests with tools, or with temperature above `max_temperature`, are not cached
  semantically: a tool call or a deliberately random answer is not a reusable answer.

Storage: Redis when configured (shared by all gateway tasks), else an in-process LRU. Semantic
candidates are compared by brute-force cosine similarity in the gateway, capped per bucket, which
works on plain Redis/Valkey/ElastiCache with no vector module.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

import httpx
import numpy as np

log = logging.getLogger(__name__)
PARAMS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
          "response_format", "seed", "presence_penalty", "frequency_penalty", "n")  # fmt: skip


@dataclass(frozen=True)
class CacheConfig:
    default_mode: str = "off"  # off | exact | semantic
    ttl_s: int = 3600
    threshold: float = 0.92
    max_temperature: float = 0.7
    max_per_bucket: int = 500
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"

    @classmethod
    def from_config(cls, cfg: dict | None) -> "CacheConfig":
        cfg = cfg or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def exact_key(tenant: str, body: dict) -> str:
    params = {k: body.get(k) for k in PARAMS if k in body}
    return "ce:" + _digest([tenant, body.get("model"), body.get("messages"), params])


def context_key(tenant: str, body: dict) -> tuple[str, str] | None:
    """(bucket key, last user text), or None when the request has no trailing user message."""
    messages = body.get("messages") or []
    if not messages or messages[-1].get("role") != "user":
        return None
    params = {k: body.get(k) for k in PARAMS if k in body}
    bucket = "cs:" + _digest([tenant, body.get("model"), messages[:-1], params])
    return bucket, _text(messages[-1].get("content"))


class MemoryCacheStore:
    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self.values: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self.buckets: dict[str, list[tuple[str, np.ndarray]]] = {}

    async def get(self, key: str) -> bytes | None:
        item = self.values.get(key)
        if item is None or item[0] < time.time():
            return None
        self.values.move_to_end(key)
        return item[1]

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        self.values[key] = (time.time() + ttl, value)
        self.values.move_to_end(key)
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)

    async def candidates(self, bucket: str) -> list[tuple[str, np.ndarray]]:
        return list(self.buckets.get(bucket, []))

    async def add_candidate(
        self, bucket: str, entry_key: str, vector: np.ndarray, cap: int, ttl: int
    ) -> None:
        items = self.buckets.setdefault(bucket, [])
        items.insert(0, (entry_key, vector))
        del items[cap:]


class RedisCacheStore:
    def __init__(self, redis):
        self.r = redis

    async def get(self, key: str) -> bytes | None:
        return await self.r.get(key)

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        await self.r.set(key, value, ex=ttl)

    async def candidates(self, bucket: str) -> list[tuple[str, np.ndarray]]:
        raw = await self.r.lrange(bucket, 0, -1)
        out = []
        for item in raw:
            entry_key, vec = item.split(b"|", 1)
            out.append((entry_key.decode(), np.frombuffer(vec, dtype=np.float32)))
        return out

    async def add_candidate(
        self, bucket: str, entry_key: str, vector: np.ndarray, cap: int, ttl: int
    ) -> None:
        pipe = self.r.pipeline()
        pipe.lpush(bucket, entry_key.encode() + b"|" + vector.astype(np.float32).tobytes())
        pipe.ltrim(bucket, 0, cap - 1)
        pipe.expire(bucket, ttl)
        await pipe.execute()


class Embedder:
    """OpenAI-compatible /embeddings client (Ollama with nomic-embed-text by default)."""

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.model = model
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=10)

    async def embed(self, texts: list[str]) -> np.ndarray:
        resp = await self.client.post("/embeddings", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        vecs = np.array([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    async def aclose(self) -> None:
        await self.client.aclose()


@dataclass
class Hit:
    kind: str  # exact | semantic
    response: dict
    similarity: float = 1.0


class ResponseCache:
    def __init__(self, store, config: CacheConfig, embedder: Embedder | None = None):
        self.store, self.config, self.embedder = store, config, embedder

    def mode(self, header: str | None, body: dict) -> str:
        mode = (header or self.config.default_mode).lower()
        if mode not in ("exact", "semantic"):
            return "off"
        if mode == "semantic" and (
            body.get("tools") or float(body.get("temperature") or 0) > self.config.max_temperature
        ):
            return "exact"
        return mode

    async def lookup(self, tenant: str, body: dict, mode: str) -> Hit | None:
        if mode == "off":
            return None
        raw = await self.store.get(exact_key(tenant, body))
        if raw is not None:
            return Hit("exact", json.loads(raw))
        if mode != "semantic" or self.embedder is None:
            return None
        ctx = context_key(tenant, body)
        if ctx is None:
            return None
        bucket, text = ctx
        candidates = await self.store.candidates(bucket)
        if not candidates:
            return None
        try:
            query = (await self.embedder.embed([text]))[0]
        except (httpx.HTTPError, KeyError, ValueError):
            log.warning("embedding failed; treating as cache miss")
            return None
        keys, vectors = zip(*candidates, strict=True)
        sims = np.stack(vectors) @ query
        best = int(np.argmax(sims))
        if sims[best] < self.config.threshold:
            return None
        raw = await self.store.get(keys[best])
        return Hit("semantic", json.loads(raw), float(sims[best])) if raw is not None else None

    async def store_response(self, tenant: str, body: dict, mode: str, response: dict) -> None:
        if mode == "off":
            return
        key = exact_key(tenant, body)
        await self.store.set(key, json.dumps(response).encode(), self.config.ttl_s)
        if mode != "semantic" or self.embedder is None or (ctx := context_key(tenant, body)) is None:
            return
        bucket, text = ctx
        try:
            vector = (await self.embedder.embed([text]))[0]
        except (httpx.HTTPError, KeyError, ValueError):
            return
        await self.store.add_candidate(bucket, key, vector, self.config.max_per_bucket, self.config.ttl_s)


def completion_from_stream(model: str, message: dict, finish_reason: str | None, usage: dict | None) -> dict:
    msg = {"role": "assistant", "content": message.get("content") or None}
    if message.get("tool_calls"):
        msg["tool_calls"] = message["tool_calls"]
    return {"id": f"chatcmpl-cache-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason or "stop"}],
            "usage": usage or {}}  # fmt: skip


def replay_as_chunks(response: dict) -> list[dict]:
    """A cached completion as streaming chunks (word-sized deltas), ending with usage."""
    choice = response["choices"][0]
    base = {"id": response.get("id", "chatcmpl-cache"), "object": "chat.completion.chunk",
            "created": response.get("created", int(time.time())), "model": response.get("model")}  # fmt: skip
    chunks = [base | {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
    text = choice["message"].get("content") or ""
    words = text.split(" ")
    for i, word in enumerate(words):
        piece = word + (" " if i < len(words) - 1 else "")
        if piece:
            chunks.append(base | {"choices": [{"index": 0, "delta": {"content": piece}}]})
    tool_calls = choice["message"].get("tool_calls") or []
    if tool_calls:  # one delta carrying every call, in the streaming shape (with an index each)
        deltas = [
            {"index": i, **{k: v for k, v in c.items() if k != "index"}} for i, c in enumerate(tool_calls)
        ]
        chunks.append(base | {"choices": [{"index": 0, "delta": {"tool_calls": deltas}}]})
    chunks.append(
        base | {"choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
    )
    chunks.append(base | {"choices": [], "usage": response.get("usage") or {}})
    return chunks
