import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings, hash_key
from llm_gateway.logs import JSONFormatter
from llm_gateway.providers import ErrorKind, ProviderError, Registry
from llm_gateway.providers.bedrock import BedrockProvider, from_converse, to_converse
from llm_gateway.providers.openai_compat import OpenAICompatProvider

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class FakeProvider:
    """In-memory provider: canned completion, canned stream, or an error."""

    def __init__(self, name="fake", models=("*",), reply="hello", error=None, fail_after=None):
        self.name, self.models, self.reply, self.error, self.fail_after = (
            name,
            list(models),
            reply,
            error,
            fail_after,
        )
        self.bodies = []

    def serves(self, model):
        return "*" in self.models or model in self.models

    async def complete(self, body):
        self.bodies.append(body)
        if self.error:
            raise self.error
        return {
            "model": "upstream-name",
            "choices": [{"message": {"role": "assistant", "content": self.reply}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

    async def stream(self, body):
        self.bodies.append(body)
        if self.error and self.fail_after is None:
            raise self.error
        for i, ch in enumerate(self.reply):
            if self.fail_after is not None and i == self.fail_after:
                raise self.error
            yield {"model": "upstream-name", "choices": [{"index": 0, "delta": {"content": ch}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    async def list_models(self):
        return [m for m in self.models if m != "*"]

    async def aclose(self):
        pass


def gateway(*providers, **settings):
    settings.setdefault("retry", {"base_delay_s": 0.0})
    return TestClient(create_app(Settings(**settings), providers=list(providers)))


# --- routing and auth -------------------------------------------------------------------------


def test_registry_prefers_exact_model_match_over_wildcard():
    wildcard, exact = FakeProvider("ollama"), FakeProvider("bedrock", models=["nova-micro"])
    registry = Registry([wildcard, exact])
    assert registry.resolve("nova-micro") is exact
    assert registry.resolve("qwen") is wildcard
    assert Registry([exact]).resolve("qwen") is None


def test_requests_route_by_model_and_unknown_models_get_404():
    a, b = FakeProvider("a", models=["m-a"], reply="A"), FakeProvider("b", models=["m-b"], reply="B")
    with gateway(a, b) as gw:
        assert gw.post("/v1/chat/completions", json={"model": "m-b", "messages": []}).json()["choices"][0][
            "message"]["content"] == "B"  # fmt: skip
        resp = gw.post("/v1/chat/completions", json={"model": "nope", "messages": []})
        assert resp.status_code == 404 and resp.json()["error"]["code"] == "model_not_found"
        assert [m["id"] for m in gw.get("/v1/models").json()["data"]] == ["m-a", "m-b"]


def test_auth_required_accepts_plain_and_hashed_keys_only():
    p = FakeProvider()
    with gateway(
        p, require_auth=True, keys={"sk-dev": "dev"}, key_hashes={hash_key("sk-prod"): "prod"}
    ) as gw:
        body = {"model": "m", "messages": []}
        assert gw.post("/v1/chat/completions", json=body).status_code == 401
        assert (
            gw.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer bad"}).status_code
            == 401
        )
        for key in ("sk-dev", "sk-prod"):
            assert (
                gw.post(
                    "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
                ).status_code
                == 200
            )


def test_provider_errors_map_to_taxonomy_statuses():
    cases = [(ErrorKind.RATE_LIMITED, 429), (ErrorKind.OVERLOADED, 503), (ErrorKind.TIMEOUT, 504),
             (ErrorKind.UNAVAILABLE, 502), (ErrorKind.AUTH, 502)]  # fmt: skip
    for kind, status in cases:
        with gateway(FakeProvider(error=ProviderError(kind, "boom"))) as gw:
            for stream in (False, True):
                resp = gw.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": stream})
                assert resp.status_code == status
                assert resp.json()["error"]["code"] == f"upstream_{kind.value}"


def test_mid_stream_error_is_reported_in_band():
    p = FakeProvider(reply="abcdef", error=ProviderError(ErrorKind.OVERLOADED, "died"), fail_after=3)
    with gateway(p) as gw, gw.stream("POST", "/v1/chat/completions",
                                     json={"model": "m", "messages": [], "stream": True}) as resp:  # fmt: skip
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()
    assert '"content": "c"' in body and '"code": "upstream_overloaded"' in body and "[DONE]" not in body


def test_responses_carry_the_requested_model_name():
    with gateway(FakeProvider()) as gw:
        assert (
            gw.post("/v1/chat/completions", json={"model": "mine", "messages": []}).json()["model"] == "mine"
        )


# --- OpenAI-compatible adapter ----------------------------------------------------------------


async def test_openai_compat_maps_http_errors():
    async def handler(request):
        status = int(request.url.params.get("s", 0)) or json.loads(request.content).get("_status")
        return httpx.Response(status, json={"error": {"message": f"status {status}"}})

    for status, kind in [(429, ErrorKind.RATE_LIMITED), (503, ErrorKind.OVERLOADED), (401, ErrorKind.AUTH),
                         (422, ErrorKind.BAD_REQUEST)]:  # fmt: skip
        client = httpx.AsyncClient(base_url="http://u", transport=httpx.MockTransport(handler))
        p = OpenAICompatProvider("x", "", client=client)
        with pytest.raises(ProviderError) as info:
            await p.complete({"model": "m", "_status": status})
        assert info.value.kind is kind
        if kind is ErrorKind.BAD_REQUEST:
            assert info.value.status == 422 and info.value.message == "status 422"


# --- Bedrock adapter --------------------------------------------------------------------------


def test_to_converse_translates_messages_tools_and_params():
    body = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Pune?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "31C"},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": TOOLS,
        "max_tokens": 50,
        "temperature": 0.2,
        "stop": "END",
    }
    out = to_converse(body, "us.amazon.nova-micro-v1:0")
    assert out["system"] == [{"text": "Be brief."}]
    assert [m["role"] for m in out["messages"]] == [
        "user",
        "assistant",
        "user",
    ]  # tool result merged into user turn
    assert out["messages"][1]["content"][0]["toolUse"]["input"] == {"city": "Pune"}
    assert out["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "t1"
    assert out["messages"][2]["content"][1] == {"text": "Thanks"}
    assert out["inferenceConfig"] == {"maxTokens": 50, "temperature": 0.2, "stopSequences": ["END"]}
    assert out["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"


def test_from_converse_builds_openai_completion():
    resp = {
        "output": {
            "message": {
                "content": [
                    {"text": "Let me check."},
                    {"toolUse": {"toolUseId": "t9", "name": "get_weather", "input": {"city": "Goa"}}},
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 4},
    }
    out = from_converse(resp, "nova-micro")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Goa"}',
    }
    assert out["usage"]["total_tokens"] == 15 and out["usage"]["prompt_tokens_details"]["cached_tokens"] == 4


class FakeBedrockClient:
    def __init__(self, events=None, error=None):
        self.events, self.error, self.kwargs = events or [], error, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def converse(self, **kwargs):
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return {"output": {"message": {"content": [{"text": "hi"}]}}, "stopReason": "end_turn", "usage": {}}

    async def converse_stream(self, **kwargs):
        self.kwargs = kwargs

        async def gen():
            for e in self.events:
                yield e

        return {"stream": gen()}


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


async def test_bedrock_stream_yields_openai_chunks_with_tool_calls():
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Checking"}}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city":'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ' "Pune"}'}}}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 7}}},
    ]
    fake = FakeBedrockClient(events)
    p = BedrockProvider("bedrock", {"nova-micro": "us.amazon.nova-micro-v1:0"}, client_factory=lambda: fake)
    chunks = [
        c async for c in p.stream({"model": "nova-micro", "messages": [{"role": "user", "content": "hi"}]})
    ]
    assert fake.kwargs["modelId"] == "us.amazon.nova-micro-v1:0"
    from llm_gateway.signals import StreamAccumulator

    acc = StreamAccumulator()
    for c in chunks:
        acc.add(c)
    assert acc.message()["content"] == "Checking"
    assert acc.message()["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Pune"}',
    }
    assert acc.finish_reason == "tool_calls" and acc.usage["completion_tokens"] == 7


async def test_bedrock_errors_map_to_taxonomy():
    for code, kind in [("ThrottlingException", ErrorKind.RATE_LIMITED), ("ValidationException", ErrorKind.BAD_REQUEST),
                       ("ModelTimeoutException", ErrorKind.TIMEOUT), ("AccessDeniedException", ErrorKind.AUTH)]:  # fmt: skip
        fake = FakeBedrockClient(error=FakeClientError(code))
        p = BedrockProvider("bedrock", {"m": "id"}, client_factory=lambda f=fake: f)
        with pytest.raises(ProviderError) as info:
            await p.complete({"model": "m", "messages": []})
        assert info.value.kind is kind


# --- logging ----------------------------------------------------------------------------------


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("llm_gateway.access", logging.INFO, "", 0, "request", None, None)
    record.tenant, record.status = "math", 200
    line = json.loads(JSONFormatter().format(record))
    assert (
        line["msg"] == "request"
        and line["tenant"] == "math"
        and line["status"] == 200
        and line["level"] == "info"
    )
