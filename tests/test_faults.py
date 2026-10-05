import json

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.faults import Fault, FaultInjector, truncate_messages

BODY = {
    "model": "qwen-q8",
    "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "a" * 400},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "b" * 40},
    ],
    "temperature": 0,
    "max_tokens": 256,
}


def apply(name: str, **params):
    return FaultInjector([Fault(name, params=params)]).apply(BODY)


def test_each_fault_changes_the_request_as_documented():
    assert all(m["role"] != "system" for m in apply("drop_system").body["messages"])

    truncated = apply("truncate_context", tokens=20).body["messages"]
    assert sum(len(m["content"]) for m in truncated) <= 80
    assert truncated[-1]["content"] == "b" * 40

    sampled = apply("sampling", temperature=1.5).body
    assert sampled["temperature"] == 1.5 and sampled["top_p"] == 1.0

    assert apply("output_cap", max_tokens=16).body["max_tokens"] == 16

    sub = apply("quant_swap", model="qwen-q2")
    assert sub.body["model"] == "qwen-q2" and sub.original_model == "qwen-q8"

    assert apply("throttle", delay_s=0.2).chunk_delay_s == 0.2
    assert BODY["model"] == "qwen-q8" and BODY["messages"][0]["role"] == "system"  # input untouched


def test_probability_controls_how_often_a_fault_lands():
    injector = FaultInjector([Fault("drop_system", p=0.3)], seed=7)
    hits = sum(injector.apply(BODY).fault == "drop_system" for _ in range(2000))
    assert 500 < hits < 700


def test_invalid_faults_are_rejected():
    with pytest.raises(ValueError):
        Fault("melt_gpu")
    with pytest.raises(ValueError):
        Fault("model_substitution")  # needs a target model


def test_truncate_keeps_newest_and_cuts_oldest_kept_message():
    msgs = [{"role": "user", "content": "x" * 100}, {"role": "user", "content": "y" * 10}]
    out = truncate_messages(msgs, max_tokens=5)  # 20 chars
    assert out == [{"role": "user", "content": "x" * 10}, {"role": "user", "content": "y" * 10}]


def test_gateway_applies_faults_silently_and_labels_ground_truth(tmp_path):
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        seen.append(sent)
        return httpx.Response(200, json={"model": sent["model"], "choices": [{"message": {"content": "hi"}}]})

    upstream = httpx.AsyncClient(base_url="http://u/v1", transport=httpx.MockTransport(handler))
    settings = Settings(signal_dir=str(tmp_path), admin_key="adm")
    with TestClient(create_app(settings, client=upstream)) as gw:
        assert gw.get("/admin/faults", headers={"X-Admin-Key": "nope"}).status_code == 401
        gw.post("/v1/chat/completions", json=BODY)
        gw.put("/admin/faults", headers={"X-Admin-Key": "adm"},
               json=[{"name": "model_substitution", "params": {"model": "small"}}])  # fmt: skip
        resp = gw.post("/v1/chat/completions", json=BODY)

    assert [s["model"] for s in seen] == ["qwen-q8", "small"]
    assert resp.json()["model"] == "qwen-q8"  # the client cannot tell
    df = pd.read_parquet(tmp_path).sort_values("ts")
    assert list(df["truth_fault"].fillna("none")) == ["none", "model_substitution"]


def test_admin_endpoints_are_disabled_without_admin_key():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    upstream = httpx.AsyncClient(base_url="http://u/v1", transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(), client=upstream)) as gw:
        assert gw.get("/admin/faults").status_code == 404


def test_truncation_costs_non_string_content():
    image = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400}}],
    }
    out = truncate_messages([image, {"role": "user", "content": "short"}], max_tokens=10)
    assert out == [{"role": "user", "content": "short"}]
