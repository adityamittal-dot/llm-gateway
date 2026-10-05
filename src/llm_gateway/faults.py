"""Fault injector for the research harness (RESEARCH.md §7).

Faults mimic silent provider-side bugs: they change the request on its way to the provider
(or slow the response down) while the client still gets a normal 200. Each applied fault is
recorded as ground truth in the signal row's `truth_fault`; detectors never read that column.

Faults (name: param):
  drop_system        -           remove system/developer messages
  truncate_context   tokens=64   keep roughly the last N tokens of the conversation
  sampling           temperature=1.8   override temperature, force top_p=1
  output_cap         max_tokens=32     lower max_tokens
  model_substitution model=<name>      send to another model; responses keep the requested name
  quant_swap         model=<name>      same mechanics as model_substitution, labelled separately
  throttle           delay_s=0.05      sleep per streamed chunk (or per 10 tokens when not streaming)
"""

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

FAULTS = (
    "drop_system",
    "truncate_context",
    "sampling",
    "output_cap",
    "model_substitution",
    "quant_swap",
    "throttle",
)
CHARS_PER_TOKEN = 4


@dataclass
class Fault:
    name: str
    p: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in FAULTS:
            raise ValueError(f"unknown fault {self.name!r}; expected one of {FAULTS}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if self.name in ("model_substitution", "quant_swap") and not self.params.get("model"):
            raise ValueError(f"{self.name} needs params.model")


@dataclass
class Applied:
    """What the injector did to one request."""

    body: dict
    fault: str | None = None
    original_model: str | None = None
    substitute_model: str | None = None
    chunk_delay_s: float = 0.0


class FaultInjector:
    def __init__(self, faults: list[Fault] | None = None, seed: int | None = None):
        self.faults = list(faults or [])
        self._rng = random.Random(seed)

    def set(self, faults: list[Fault]) -> None:
        self.faults = list(faults)

    def apply(self, body: dict) -> Applied:
        """Apply at most one fault: the first configured fault whose coin flip lands."""
        for fault in self.faults:
            if self._rng.random() < fault.p:
                return _apply_one(fault, copy.deepcopy(body))
        return Applied(body=body)


def _apply_one(fault: Fault, body: dict) -> Applied:
    applied = Applied(body=body, fault=fault.name)
    messages = body.get("messages") or []
    match fault.name:
        case "drop_system":
            body["messages"] = [m for m in messages if m.get("role") not in ("system", "developer")]
        case "truncate_context":
            body["messages"] = truncate_messages(messages, int(fault.params.get("tokens", 64)))
        case "sampling":
            body["temperature"] = float(fault.params.get("temperature", 1.8))
            body["top_p"] = 1.0
        case "output_cap":
            cap = int(fault.params.get("max_tokens", 32))
            for key in ("max_tokens", "max_completion_tokens"):
                if key in body:
                    body[key] = min(int(body[key]), cap)
            if "max_tokens" not in body and "max_completion_tokens" not in body:
                body["max_tokens"] = cap
        case "model_substitution" | "quant_swap":
            applied.original_model = body.get("model")
            applied.substitute_model = body["model"] = fault.params["model"]
        case "throttle":
            applied.chunk_delay_s = float(fault.params.get("delay_s", 0.05))
    return applied


def truncate_messages(messages: list[dict], max_tokens: int) -> list[dict]:
    """Keep the newest messages within ~max_tokens, cutting the oldest kept message from the front."""
    budget = max_tokens * CHARS_PER_TOKEN
    kept: list[dict] = []
    for message in reversed(messages):
        content = message.get("content")
        # Non-string content (multimodal parts, tool calls) is costed by its JSON size, never free.
        size = len(content) if isinstance(content, str) else len(json.dumps(content or ""))
        if size <= budget:
            kept.append(message)
            budget -= size
            continue
        if budget > 0 and isinstance(content, str):
            kept.append({**message, "content": content[-budget:]})
        break
    return list(reversed(kept)) or messages[-1:]


def parse_faults(raw: list[dict] | None) -> list[Fault]:
    return [
        Fault(name=f["name"], p=float(f.get("p", 1.0)), params=dict(f.get("params") or {})) for f in raw or []
    ]
