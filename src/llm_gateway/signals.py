"""Passive per-response quality signals (RESEARCH.md §6).

Signals are computed from the request and response in memory; only the resulting
numbers are stored, never prompt or response text.
"""

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

REFUSAL_RE = re.compile(
    r"^\s*(i'?m sorry|i am sorry|i can(?:'|no)t (?:help|assist|provide|do)|i cannot (?:help|assist|provide|do)"
    r"|i'?m (?:not able|unable) to|as an ai\b)",
    re.IGNORECASE,
)


@dataclass
class ResponseSignals:
    ts: float
    request_id: str
    tenant: str
    provider: str
    model: str
    stream: bool
    status_code: int
    latency_s: float
    ttft_s: float | None = None
    tokens_per_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    output_chars: int = 0
    empty: bool = False
    refusal: bool = False
    repetition: float = 0.0
    tools_offered: bool = False
    tool_calls: int = 0
    tool_call_valid: bool | None = None
    prompt_hash: str = ""
    regenerate: bool = False
    # Ground truth for evaluation only (set by the fault injector); detectors must not read it.
    truth_fault: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        row = asdict(self)
        row["extra"] = json.dumps(row["extra"]) if row["extra"] else None
        return row


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that are repeats; 0 for normal text, near 1 for degenerate loops."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.match(text))


def prompt_hash(body: dict) -> str:
    """Hash of model + messages, used to spot a client re-sending the same prompt (a regenerate)."""
    payload = json.dumps([body.get("model"), body.get("messages")], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_tool_calls(tool_calls: list[dict], tools: list[dict]) -> bool:
    """True if every call names a declared tool and its arguments are a JSON object with the required keys."""
    schemas = {
        t["function"]["name"]: t["function"].get("parameters") or {}
        for t in tools
        if t.get("type", "function") == "function" and "function" in t
    }
    for call in tool_calls:
        fn = call.get("function") or {}
        schema = schemas.get(fn.get("name"))
        if schema is None:
            return False
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return False
        if not isinstance(args, dict):
            return False
        if any(key not in args for key in schema.get("required", [])):
            return False
        props = schema.get("properties")
        if props is not None and any(key not in props for key in args):
            return False
    return True


class RegenerateTracker:
    """Remembers recent (tenant, prompt hash) pairs; a repeat within the window counts as a regenerate."""

    def __init__(self, window_s: float = 600.0, max_entries: int = 50_000):
        self.window_s = window_s
        self.max_entries = max_entries
        self._seen: OrderedDict[tuple[str, str], float] = OrderedDict()

    def seen_recently(self, tenant: str, phash: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        key = (tenant, phash)
        last = self._seen.pop(key, None)
        self._seen[key] = now
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return last is not None and now - last <= self.window_s


class StreamAccumulator:
    """Rebuilds the final message from OpenAI-style SSE chunks while they are relayed."""

    def __init__(self) -> None:
        self._buffer = b""
        self.text_parts: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.chunks = 0

    def feed(self, data: bytes) -> None:
        self._buffer += data
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in event.splitlines():
                if line.startswith(b"data:"):
                    self._handle(line[5:].strip())

    def _handle(self, payload: bytes) -> None:
        if not payload or payload == b"[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            return
        self.chunks += 1
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = self.tool_calls.setdefault(
                    tc.get("index", 0), {"function": {"name": "", "arguments": ""}}
                )
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def message(self) -> dict:
        return {
            "content": "".join(self.text_parts),
            "tool_calls": [self.tool_calls[i] for i in sorted(self.tool_calls)],
        }


def extract(
    *,
    request_id: str,
    tenant: str,
    provider: str,
    body: dict,
    status_code: int,
    started: float,
    finished: float,
    first_token_at: float | None,
    message: dict | None,
    finish_reason: str | None,
    usage: dict | None,
    regenerate: bool,
    truth_fault: str | None = None,
) -> ResponseSignals:
    text = (message or {}).get("content") or ""
    tool_calls = (message or {}).get("tool_calls") or []
    tools = body.get("tools") or []
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    latency = finished - started
    ttft = (first_token_at - started) if first_token_at else None
    decode_time = latency - (ttft or 0.0)
    tps = completion_tokens / decode_time if completion_tokens and decode_time > 0 else None
    return ResponseSignals(
        ts=finished,
        request_id=request_id,
        tenant=tenant,
        provider=provider,
        model=str(body.get("model", "")),
        stream=bool(body.get("stream")),
        status_code=status_code,
        latency_s=latency,
        ttft_s=ttft,
        tokens_per_s=tps,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        finish_reason=finish_reason,
        output_chars=len(text),
        empty=not text.strip() and not tool_calls,
        refusal=is_refusal(text),
        repetition=repetition_score(text),
        tools_offered=bool(tools),
        tool_calls=len(tool_calls),
        tool_call_valid=validate_tool_calls(tool_calls, tools) if tools and tool_calls else None,
        prompt_hash=prompt_hash(body),
        regenerate=regenerate,
        truth_fault=truth_fault,
    )
