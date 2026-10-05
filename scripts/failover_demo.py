"""Chaos demo (day 12): Toxiproxy breaks the primary provider; the gateway fails over and the
breaker opens; when the primary heals, a half-open probe closes the breaker again.

Everything runs locally: two mock LLMs, toxiproxy-server in front of the primary, and the gateway.
    uv run python scripts/failover_demo.py        (needs toxiproxy-server on PATH)
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

GW, TOXI = "http://127.0.0.1:8200", "http://127.0.0.1:8474"
ADMIN = {"X-Admin-Key": "demo"}
procs: list[subprocess.Popen] = []


def spawn(cmd: list[str], env: dict | None = None) -> None:
    procs.append(subprocess.Popen(cmd, env=os.environ | (env or {}), stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL))  # fmt: skip


def wait_for(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


def burst(label: str, n: int = 15) -> None:
    served, codes = Counter(), Counter()
    started = time.perf_counter()
    for _ in range(n):
        r = httpx.post(f"{GW}/v1/chat/completions", timeout=30, headers={"Authorization": "Bearer sk-demo"},
                       json={"model": "chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})  # fmt: skip
        codes[r.status_code] += 1
        served[r.headers.get("x-gateway-provider", "-")] += 1
    breakers = httpx.get(f"{GW}/admin/breakers", headers=ADMIN).json()
    states = ", ".join(f"{p}={s['state']}" for p, s in breakers.items())
    print(f"{label:<38} served_by={dict(served)}  http={dict(codes)}  "
          f"{(time.perf_counter() - started) / n * 1000:5.0f} ms/req  breakers: {states}")  # fmt: skip


def main() -> None:
    cfg = {
        "require_auth": True,
        "admin_key": "demo",
        "keys": [{"key": "sk-demo", "tenant": "demo"}],
        "providers": [
            {"name": "primary", "type": "openai_compat", "base_url": "http://127.0.0.1:9101/v1", "models": ["mock-small"]},
            {"name": "backup", "type": "openai_compat", "base_url": "http://127.0.0.1:9002/v1", "models": ["mock-small"]},
        ],
        "routes": {"chat": [{"provider": "primary", "model": "mock-small"}, {"provider": "backup", "model": "mock-small"}]},
        "retry": {"max_attempts": 2, "base_delay_s": 0.05},
        "breaker": {"min_requests": 5, "failure_ratio": 0.5, "cooldown_s": 5},
    }  # fmt: skip
    cfg_path = Path(tempfile.mkdtemp()) / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    try:
        spawn(["toxiproxy-server", "-port", "8474"])
        for port in (9001, 9002):
            spawn([sys.executable, "-c", "import llm_gateway.mock_llm as m; m.main()"],
                  {"MOCK_PORT": str(port), "MOCK_TTFT_MS": "5", "MOCK_TOKENS_PER_S": "2000"})  # fmt: skip
        spawn([sys.executable, "-c", "import llm_gateway; llm_gateway.main()"],
              {"GATEWAY_CONFIG": str(cfg_path), "GATEWAY_PORT": "8200", "UPSTREAM_READ_TIMEOUT_S": "1",
               "LOG_LEVEL": "WARNING"})  # fmt: skip
        for url in (
            f"{TOXI}/version",
            "http://127.0.0.1:9001/healthz",
            "http://127.0.0.1:9002/healthz",
            f"{GW}/healthz",
        ):
            wait_for(url)
        httpx.post(f"{TOXI}/proxies", json={"name": "primary", "listen": "127.0.0.1:9101",
                                            "upstream": "127.0.0.1:9001"}).raise_for_status()  # fmt: skip

        burst("1. healthy")
        httpx.post(f"{TOXI}/proxies/primary/toxics", json={"name": "slow", "type": "latency", "stream": "downstream",
                                                           "attributes": {"latency": 5000}}).raise_for_status()  # fmt: skip
        burst("2. primary hangs (5 s latency toxic)")
        burst("3. still broken (breaker open: skip)")
        httpx.delete(f"{TOXI}/proxies/primary/toxics/slow").raise_for_status()
        burst("4. primary healed, cooldown running")
        time.sleep(5.5)
        burst("5. after cooldown: probe closes breaker")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)


if __name__ == "__main__":
    main()
