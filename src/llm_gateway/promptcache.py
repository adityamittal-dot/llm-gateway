"""Cache-break diagnosis: say *why* a tenant's provider prompt-cache hit ratio collapsed.

Providers cache the processed prefix of a prompt (Anthropic/Bedrock `cache_control`, OpenAI's
automatic caching) and report `cached_tokens`. One innocent change early in the prompt (a
timestamp in the system prompt, tools rendered in a different order, a new tool) invalidates
everything after it, and the only symptom is a bigger bill.

For every (tenant, model) the gateway remembers the block hashes of the previous long prompt
(tools first, then each message). When the cached share of the prompt drops sharply, it reports
the first block that differs from the previous request, with a hint about the likely cause.
Only hashes are kept, never prompt text.
"""

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheBreakConfig:
    min_prompt_tokens: int = 1024  # providers do not cache shorter prompts
    was_cached_ratio: float = 0.5
    now_cached_ratio: float = 0.1


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def blocks(body: dict) -> list[tuple[str, str]]:
    """(label, hash) per prompt block in provider order: tool definitions, then messages."""
    out = []
    tools = body.get("tools") or []
    if tools:
        out.append(("tools", _h(tools)))
    for i, m in enumerate(body.get("messages") or []):
        out.append((f"messages[{i}] ({m.get('role')})", _h(m)))
    return out


def _tool_names(body: dict) -> list[str]:
    return [t.get("function", {}).get("name", "") for t in body.get("tools") or []]


class CacheBreakDetector:
    def __init__(self, config: CacheBreakConfig | None = None, clock=time.time):
        self.config = config or CacheBreakConfig()
        self.last: dict[tuple[str, str], dict] = {}
        self.events: deque = deque(maxlen=200)
        self.clock = clock

    def observe(self, tenant: str, model: str, body: dict, usage: dict | None) -> dict | None:
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        if prompt < self.config.min_prompt_tokens or details.get("cached_tokens") is None:
            return None
        ratio = int(details["cached_tokens"]) / prompt
        key = (tenant, model)
        current = {"blocks": blocks(body), "ratio": ratio, "tools": sorted(_tool_names(body)),
                   "tool_order": _tool_names(body)}  # fmt: skip
        previous = self.last.get(key)
        self.last[key] = current
        if (
            previous is None
            or previous["ratio"] < self.config.was_cached_ratio
            or ratio > self.config.now_cached_ratio
        ):
            return None
        event = {"ts": self.clock(), "tenant": tenant, "model": model, "cached_ratio_before": round(previous["ratio"], 3),
                 "cached_ratio_now": round(ratio, 3), **self._diagnose(previous, current)}  # fmt: skip
        self.events.append(event)
        return event

    @staticmethod
    def _diagnose(prev: dict, cur: dict) -> dict:
        # Only the shared prefix matters: blocks appended after the previous prompt (a conversation
        # that simply grew) cannot have invalidated the cached prefix.
        for i, (label, digest) in enumerate(cur["blocks"][: len(prev["blocks"])]):
            if prev["blocks"][i] != (label, digest):
                if label == "tools":
                    if prev["tools"] == cur["tools"]:
                        hint = "tools are the same but in a different order: render them deterministically"
                    elif set(cur["tools"]) - set(prev["tools"]):
                        hint = f"new tool(s) {sorted(set(cur['tools']) - set(prev['tools']))}: put dynamic tools last"
                    else:
                        hint = "tool definitions changed"
                elif "(system)" in label or "(developer)" in label:
                    hint = (
                        "system prompt changed: move dynamic values (timestamps, ids, user data) to the end"
                    )
                else:
                    hint = "an earlier message was edited (history rewritten or compacted)"
                return {"changed_block": label, "block_index": i, "hint": hint}
        return {"changed_block": None, "block_index": None,
                "hint": "prompt prefix unchanged: the provider's cache likely expired (TTL) or was evicted"}  # fmt: skip
