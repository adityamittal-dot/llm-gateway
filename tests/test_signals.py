import json

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from llm_gateway import signals as sig
from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.signal_log import SignalLog

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def call(name: str, arguments: str) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def test_repetition_score_separates_loops_from_prose():
    assert sig.repetition_score("the quick brown fox jumps over the lazy dog today") == 0.0
    assert sig.repetition_score("I am stuck " * 30) > 0.8


def test_refusal_detection_only_matches_openings():
    assert sig.is_refusal("I'm sorry, but I can't help with that.")
    assert sig.is_refusal("I cannot provide that information.")
    assert not sig.is_refusal("The answer is 42. I'm sorry it took so long.")


def test_tool_call_validation():
    tools = [WEATHER_TOOL]
    assert sig.validate_tool_calls([call("get_weather", '{"city": "Pune"}')], tools)
    assert not sig.validate_tool_calls([call("get_weather", '{"unit": "C"}')], tools)  # missing required
    assert not sig.validate_tool_calls(
        [call("get_weather", '{"city": "Pune", "x": 1}')], tools
    )  # unknown key
    assert not sig.validate_tool_calls([call("get_time", "{}")], tools)  # undeclared tool
    assert not sig.validate_tool_calls([call("get_weather", "{city: Pune")], tools)  # broken JSON


def test_regenerate_tracker_window():
    tracker = sig.RegenerateTracker(window_s=60)
    assert not tracker.seen_recently("a", "h1", now=0)
    assert tracker.seen_recently("a", "h1", now=30)
    assert not tracker.seen_recently("b", "h1", now=31)  # other tenant
    assert not tracker.seen_recently("a", "h1", now=200)  # outside window


def test_stream_accumulator_rebuilds_text_and_split_tool_calls():
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": '{"ci'}}]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Pune"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 7}},
    ]
    raw = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
    acc = sig.StreamAccumulator()
    for i in range(0, len(raw), 13):  # arbitrary network chunking
        acc.feed(raw[i : i + 13])
    msg = acc.message()
    assert msg["content"] == "Hello"
    assert msg["tool_calls"] == [call("get_weather", '{"city": "Pune"}')]
    assert acc.finish_reason == "tool_calls"
    assert acc.usage == {"completion_tokens": 7}


def test_signal_log_round_trips_through_parquet(tmp_path):
    log = SignalLog(tmp_path)
    for i in range(3):
        log.append(sig.ResponseSignals(ts=1_760_000_000 + i, request_id=str(i), tenant="t", provider="p",
                                       model="m", stream=False, status_code=200, latency_s=0.5))  # fmt: skip
    log.flush()
    df = pd.read_parquet(tmp_path)
    assert list(df["request_id"]) == ["0", "1", "2"]
    assert df["ttft_s"].isna().all()


def test_gateway_records_signals_per_tenant(tmp_path):
    completion = {
        "choices": [
            {"message": {"role": "assistant", "content": "I'm sorry, I can't help."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 6},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion)

    upstream = httpx.AsyncClient(base_url="http://u/v1", transport=httpx.MockTransport(handler))
    settings = Settings(signal_dir=str(tmp_path), keys={"sk-math": "math"}, provider_name="ollama")
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(create_app(settings, client=upstream)) as gw:
        for _ in range(2):
            gw.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-math"})
        gw.post("/v1/chat/completions", json=body)  # no key -> anonymous

    df = pd.read_parquet(tmp_path).sort_values("ts")
    assert list(df["tenant"]) == ["math", "math", "anonymous"]
    assert list(df["regenerate"]) == [False, True, False]
    assert df["refusal"].all()
    assert (df["provider"] == "ollama").all()
    assert (df["completion_tokens"] == 6).all()
