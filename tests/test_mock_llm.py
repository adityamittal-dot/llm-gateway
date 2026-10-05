import httpx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.mock_llm import create_mock_app
from llm_gateway.providers.openai_compat import OpenAICompatProvider

FAST = {"x-mock-ttft-ms": "0", "x-mock-tokens-per-s": "100000"}


def test_mock_replies_deterministically_and_reports_usage():
    with TestClient(create_mock_app()) as mock:
        body = {
            "model": "mock-small",
            "messages": [{"role": "user", "content": "hi there"}],
            "max_tokens": 10,
        }
        a = mock.post("/v1/chat/completions", json=body, headers=FAST).json()
        b = mock.post("/v1/chat/completions", json=body, headers=FAST).json()
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]
    assert a["usage"]["completion_tokens"] == 10 and a["choices"][0]["finish_reason"] == "length"


def test_mock_streams_tool_calls_and_injects_errors():
    with TestClient(create_mock_app()) as mock:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"required": ["q"]}}}]
        body = {
            "model": "m",
            "messages": [],
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        text = mock.post("/v1/chat/completions", json=body, headers=FAST).text
        assert '"name": "f"' in text and '"usage"' in text and text.endswith("data: [DONE]\n\n")
        err = mock.post("/v1/chat/completions", json=body, headers=FAST | {"x-mock-error-rate": "1"})
        assert err.status_code == 503


def test_gateway_against_mock_end_to_end():
    transport = httpx.ASGITransport(app=create_mock_app())
    upstream = httpx.AsyncClient(base_url="http://mock/v1", transport=transport, headers=FAST)
    provider = OpenAICompatProvider("mock", "", models=["mock-small"], client=upstream)
    with TestClient(create_app(Settings(), providers=[provider])) as gw:
        resp = gw.post(
            "/v1/chat/completions",
            json={"model": "mock-small", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"]
        streamed = gw.post(
            "/v1/chat/completions", json={"model": "mock-small", "messages": [], "stream": True}
        )
        assert streamed.text.endswith("data: [DONE]\n\n")
