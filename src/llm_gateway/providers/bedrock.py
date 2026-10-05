"""Amazon Bedrock adapter using the Converse / ConverseStream API (one format for all Bedrock models).

Translates OpenAI chat requests (messages, tools, sampling params) to Converse and back,
including streamed tool calls. `models` maps the public model name to a Bedrock model ID or
inference profile, e.g. {"nova-micro": "us.amazon.nova-micro-v1:0"}.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.providers.base import ErrorKind, ProviderError

STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}
ERROR_KINDS = {
    "ThrottlingException": ErrorKind.RATE_LIMITED,
    "ServiceQuotaExceededException": ErrorKind.RATE_LIMITED,
    "ServiceUnavailableException": ErrorKind.OVERLOADED,
    "ModelNotReadyException": ErrorKind.OVERLOADED,
    "InternalServerException": ErrorKind.OVERLOADED,
    "ModelStreamErrorException": ErrorKind.OVERLOADED,
    "ModelTimeoutException": ErrorKind.TIMEOUT,
    "AccessDeniedException": ErrorKind.AUTH,
    "UnrecognizedClientException": ErrorKind.AUTH,
    "ValidationException": ErrorKind.BAD_REQUEST,
    "ResourceNotFoundException": ErrorKind.BAD_REQUEST,
    "ModelErrorException": ErrorKind.BAD_REQUEST,
}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def to_converse(body: dict, model_id: str) -> dict:
    """OpenAI chat request -> Converse request kwargs."""
    system, messages = [], []

    def push(role: str, blocks: list[dict]) -> None:
        if messages and messages[-1]["role"] == role:  # Converse requires alternating roles
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            system.append({"text": _text(m.get("content"))})
        elif role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": m.get("tool_call_id", ""),
                            "content": [{"text": _text(m.get("content"))}],
                        }
                    }
                ],
            )
        elif role == "assistant":
            blocks = [{"text": t} for t in [_text(m.get("content"))] if t]
            for call in m.get("tool_calls") or []:
                args = call["function"].get("arguments") or "{}"
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": call.get("id") or uuid.uuid4().hex,
                            "name": call["function"]["name"],
                            "input": json.loads(args) if isinstance(args, str) else args,
                        }
                    }
                )
            push("assistant", blocks or [{"text": ""}])
        else:
            push("user", [{"text": _text(m.get("content"))}])

    inference: dict[str, Any] = {}
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens is not None:
        inference["maxTokens"] = int(max_tokens)
    if body.get("temperature") is not None:
        inference["temperature"] = float(body["temperature"])
    if body.get("top_p") is not None:
        inference["topP"] = float(body["top_p"])
    if body.get("stop"):
        inference["stopSequences"] = [body["stop"]] if isinstance(body["stop"], str) else list(body["stop"])

    kwargs: dict[str, Any] = {"modelId": model_id, "messages": messages}
    if system:
        kwargs["system"] = system
    if inference:
        kwargs["inferenceConfig"] = inference
    tools = [t["function"] for t in body.get("tools") or [] if t.get("type", "function") == "function"]
    if tools and body.get("tool_choice") != "none":
        kwargs["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": f["name"],
                        "description": f.get("description") or f["name"],
                        "inputSchema": {"json": f.get("parameters") or {"type": "object"}},
                    }
                }
                for f in tools
            ]
        }
        choice = body.get("tool_choice")
        if choice == "required":
            kwargs["toolConfig"]["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict) and choice.get("function"):
            kwargs["toolConfig"]["toolChoice"] = {"tool": {"name": choice["function"]["name"]}}
    return kwargs


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt = usage.get("inputTokens", 0)
    completion = usage.get("outputTokens", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": usage.get("cacheReadInputTokens", 0)},
    }


def from_converse(resp: dict, model: str) -> dict:
    """Converse response -> OpenAI chat.completion."""
    text, calls = [], []
    for block in resp.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            text.append(block["text"])
        elif "toolUse" in block:
            use = block["toolUse"]
            calls.append(
                {
                    "id": use["toolUseId"],
                    "type": "function",
                    "function": {"name": use["name"], "arguments": json.dumps(use.get("input", {}))},
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": STOP_REASONS.get(resp.get("stopReason"), "stop"),
            }
        ],
        "usage": _usage(resp.get("usage")),
    }


class BedrockProvider:
    def __init__(self, name: str, models: dict[str, str], region: str | None = None, client_factory=None):
        self.name = name
        self.models = dict(models)
        self.region = region
        self._client_factory = client_factory or self._default_client
        self._session = None

    def _default_client(self):
        import aioboto3  # imported lazily so the gateway runs without AWS libraries configured

        if self._session is None:
            self._session = aioboto3.Session()
        return self._session.client("bedrock-runtime", region_name=self.region)

    def serves(self, model: str) -> bool:
        return model in self.models or model in self.models.values()

    def _model_id(self, model: str) -> str:
        return self.models.get(model, model)

    def _error(self, exc: Exception) -> ProviderError:
        code = (
            getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        )
        kind = ERROR_KINDS.get(code)
        if kind is None:
            name = exc.__class__.__name__
            if "Credentials" in name or "Token" in name:  # NoCredentialsError, ExpiredTokenException, ...
                kind = ErrorKind.AUTH
            else:
                kind = ErrorKind.TIMEOUT if "Timeout" in name else ErrorKind.UNAVAILABLE
        return ProviderError(
            kind, f"{self.name}: {code or exc.__class__.__name__}: {exc}", provider=self.name
        )

    async def complete(self, body: dict) -> dict:
        model = body.get("model", "")
        try:
            async with self._client_factory() as client:
                resp = await client.converse(**to_converse(body, self._model_id(model)))
        except Exception as exc:  # botocore raises many unrelated exception types
            raise self._error(exc) from exc
        return from_converse(resp, model)

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        model = body.get("model", "")
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> dict:
            choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
            out = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": choices,
            }
            if usage is not None:
                out["usage"] = usage
            return out

        tool_index: dict[int, int] = {}  # Converse content block index -> OpenAI tool call index
        try:
            async with self._client_factory() as client:
                resp = await client.converse_stream(**to_converse(body, self._model_id(model)))
                async for event in resp["stream"]:
                    if "messageStart" in event:
                        yield chunk({"role": "assistant", "content": ""})
                    elif "contentBlockStart" in event:
                        start = event["contentBlockStart"]
                        use = start.get("start", {}).get("toolUse")
                        if use:
                            idx = tool_index.setdefault(start["contentBlockIndex"], len(tool_index))
                            yield chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "id": use["toolUseId"],
                                            "type": "function",
                                            "function": {"name": use["name"], "arguments": ""},
                                        }
                                    ]
                                }
                            )
                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]
                        if "text" in delta["delta"]:
                            yield chunk({"content": delta["delta"]["text"]})
                        elif "toolUse" in delta["delta"]:
                            idx = tool_index.setdefault(delta["contentBlockIndex"], len(tool_index))
                            yield chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "function": {
                                                "arguments": delta["delta"]["toolUse"].get("input", "")
                                            },
                                        }
                                    ]
                                }
                            )
                    elif "messageStop" in event:
                        yield chunk({}, STOP_REASONS.get(event["messageStop"].get("stopReason"), "stop"))
                    elif "metadata" in event:
                        yield chunk(usage=_usage(event["metadata"].get("usage")))
        except Exception as exc:
            raise self._error(exc) from exc

    async def list_models(self) -> list[str]:
        return list(self.models)

    async def aclose(self) -> None:
        return None
