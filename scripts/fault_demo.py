"""Manual check that every fault is visible: one live request per fault through the gateway.

Start the gateway with an admin key and Ollama running:
    ADMIN_KEY=dev uv run llm-gateway
then:
    uv run python scripts/fault_demo.py
"""

import time

import httpx

GATEWAY = "http://127.0.0.1:8000"
ADMIN = {"X-Admin-Key": "dev"}
MODEL = "qwen2.5:1.5b-instruct-q8_0"
BODY = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Always answer in French."},
        {"role": "user", "content": "My name is Asha and I live in Pune. I like chess."},
        {"role": "assistant", "content": "D'accord."},
        {"role": "user", "content": "In one sentence: what is my name, city and hobby?"},
    ],
    "temperature": 0,
    "max_tokens": 60,
}
CASES = [
    ("healthy", []),
    ("drop_system", [{"name": "drop_system"}]),
    ("truncate_context", [{"name": "truncate_context", "params": {"tokens": 15}}]),
    ("sampling", [{"name": "sampling", "params": {"temperature": 2.0}}]),
    ("output_cap", [{"name": "output_cap", "params": {"max_tokens": 6}}]),
    ("quant_swap", [{"name": "quant_swap", "params": {"model": "qwen2.5:1.5b-instruct-q2_K"}}]),
    ("model_substitution", [{"name": "model_substitution", "params": {"model": "llama3.2:1b-instruct-q8_0"}}]),
    ("throttle", [{"name": "throttle", "params": {"delay_s": 0.5}}]),
]

with httpx.Client(base_url=GATEWAY, timeout=300) as client:
    for label, faults in CASES:
        client.put("/admin/faults", headers=ADMIN, json=faults).raise_for_status()
        start = time.perf_counter()
        resp = client.post("/v1/chat/completions", json=BODY).json()
        choice = resp["choices"][0]
        text = (choice["message"]["content"] or "").replace("\n", " ")
        print(f"{label:<19} {time.perf_counter() - start:5.1f}s  model={resp['model']}  "
              f"finish={choice['finish_reason']:<6} {text[:90]}")  # fmt: skip
    client.put("/admin/faults", headers=ADMIN, json=[]).raise_for_status()
