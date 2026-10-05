"""Quality-breaker demo (day 17): a silent fault on the primary (answers cut to a few tokens, still
HTTP 200) is detected from passive signals and traffic moves to the backup; the HTTP circuit
breaker never notices. When the fault is removed, canary traffic shows recovery.

Runs locally on two mock LLMs (no GPU):  uv run python scripts/quality_breaker_demo.py
"""

import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import httpx
import yaml

GW, ADMIN = "http://127.0.0.1:8400", {"X-Admin-Key": "demo"}
FAULT = [{"name": "output_cap", "params": {"max_tokens": 3}, "provider": "primary"}]


def wait(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(url)


def burst(label: str, n: int) -> None:
    served, codes = Counter(), Counter()
    for i in range(n):
        r = httpx.post(f"{GW}/v1/chat/completions", timeout=30,
                       json={"model": "chat", "max_tokens": 48,
                             "messages": [{"role": "user", "content": f"question {i} {time.time()}"}]})  # fmt: skip
        served[r.headers.get("x-gateway-provider")] += 1
        codes[r.status_code] += 1
    q = httpx.get(f"{GW}/admin/quality", headers=ADMIN).json()["providers"].get("primary", {})
    b = httpx.get(f"{GW}/admin/breakers", headers=ADMIN).json().get("primary", {})
    print(f"{label:<34} served_by={dict(served)}  http={dict(codes)}  quality_level={q.get('level')}  "
          f"http_breaker={b.get('state')}")  # fmt: skip


def main() -> None:
    cfg = {
        "admin_key": "demo",
        "providers": [
            {"name": "primary", "type": "openai_compat", "base_url": "http://127.0.0.1:9401/v1", "models": ["mock-small"]},
            {"name": "backup", "type": "openai_compat", "base_url": "http://127.0.0.1:9402/v1", "models": ["mock-large"]},
        ],
        "routes": {"chat": [{"provider": "primary", "model": "mock-small"}, {"provider": "backup", "model": "mock-large"}]},
        "quality": {"warmup": 60, "canary_fraction": 0.2, "recovery_window": 30},
    }  # fmt: skip
    path = Path(tempfile.mkdtemp()) / "q.yaml"
    path.write_text(yaml.safe_dump(cfg))
    mock = [sys.executable, "-c", "import llm_gateway.mock_llm as m; m.main()"]
    fast = {"MOCK_TTFT_MS": "1", "MOCK_TOKENS_PER_S": "100000", "MOCK_REPLY_TOKENS": "40"}
    procs = [
        subprocess.Popen(mock, env=os.environ | fast | {"MOCK_PORT": "9401"}),
        subprocess.Popen(mock, env=os.environ | fast | {"MOCK_PORT": "9402"}),
        subprocess.Popen([sys.executable, "-c", "import llm_gateway; llm_gateway.main()"],
                         env=os.environ | {"GATEWAY_CONFIG": str(path), "GATEWAY_PORT": "8400", "LOG_LEVEL": "ERROR"}),
    ]  # fmt: skip
    try:
        for url in ("http://127.0.0.1:9401/healthz", "http://127.0.0.1:9402/healthz", f"{GW}/healthz"):
            wait(url)
        burst("1. warm-up (reference learned)", 80)
        burst("2. healthy traffic", 40)
        httpx.put(f"{GW}/admin/faults", headers=ADMIN, json=FAULT).raise_for_status()
        burst("3. silent fault on primary", 40)
        burst("4. fault continues", 40)
        httpx.put(f"{GW}/admin/faults", headers=ADMIN, json=[]).raise_for_status()
        burst("5. primary fixed (canaries)", 200)
        burst("6. after recovery", 40)
        events = httpx.get(f"{GW}/admin/quality", headers=ADMIN).json()["events"]
        print("level changes:", [(e["provider"], e["from"], e["to"]) for e in events])
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


if __name__ == "__main__":
    main()
