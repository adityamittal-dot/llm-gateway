import json

import httpx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "qwen2.5:1.5b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
SSE_CHUNKS = [
    b'data: {"choices":[{"index":0,"delta":{"content":"h"}}]}\n\n',
    b'data: {"choices":[{"index":0,"delta":{"content":"i"},"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
]


def make_gateway(handler) -> TestClient:
    upstream = httpx.AsyncClient(base_url="http://upstream/v1", transport=httpx.MockTransport(handler))
    return TestClient(create_app(Settings(upstream_base_url="http://upstream/v1"), client=upstream))


def test_non_streaming_completion_is_forwarded_unchanged():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=COMPLETION)

    body = {"model": "qwen2.5:1.5b", "messages": [{"role": "user", "content": "hello"}], "temperature": 0}
    with make_gateway(handler) as gw:
        resp = gw.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    assert resp.json() == COMPLETION
    assert seen == {"path": "/v1/chat/completions", "body": body}


def test_streaming_completion_is_relayed_as_sse():
    async def chunks():
        for chunk in SSE_CHUNKS:
            yield chunk

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}
    with make_gateway(handler) as gw, gw.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert b"".join(resp.iter_bytes()) == b"".join(SSE_CHUNKS)


def test_upstream_error_status_and_body_pass_through():
    error = {"error": {"message": "model not found", "type": "invalid_request_error", "code": None}}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=error)

    body = {"model": "missing", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with make_gateway(handler) as gw:
        resp = gw.post("/v1/chat/completions", json=body)

    assert resp.status_code == 404
    assert resp.json() == error


def test_unreachable_upstream_returns_502():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_gateway(handler) as gw:
        resp = gw.post("/v1/chat/completions", json={"model": "m", "messages": []})

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_unavailable"


def test_upstream_timeout_returns_504():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with make_gateway(handler) as gw:
        resp = gw.post("/v1/chat/completions", json={"model": "m", "messages": []})

    assert resp.status_code == 504


def test_invalid_json_body_returns_400():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    with make_gateway(handler) as gw:
        resp = gw.post(
            "/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"}
        )

    assert resp.status_code == 400
