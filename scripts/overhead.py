"""Measure the gateway's own latency (README "Gateway overhead"): k6 against the mock LLM directly,
then through the gateway, same load. The mock answers instantly (no TTFT, very high tokens/s),
so the difference is the gateway: auth, routing, limits, signals, logging, metrics, serialisation.

    uv run python scripts/overhead.py [--vus 10] [--duration 30s]     (needs k6 on PATH)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import yaml


def wait(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(url)


def k6(base_url: str, vus: int, duration: str) -> dict:
    out = Path(tempfile.mkdtemp()) / "summary.json"
    env = os.environ | {"BASE_URL": base_url, "KEY": "sk-load", "VUS": str(vus), "DURATION": duration}
    subprocess.run(
        ["k6", "run", "--quiet", "--summary-export", str(out), "loadtest/overhead.js"], check=True, env=env
    )
    m = json.loads(out.read_text())["metrics"]
    d = m["http_req_duration"]
    return {
        "rps": round(m["http_reqs"]["rate"], 1),
        "p50_ms": round(d["p(50)"], 2),
        "p90_ms": round(d["p(90)"], 2),
        "p99_ms": round(d["p(99)"], 2),
        "failed_rate": m["http_req_failed"]["value"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vus", type=int, default=10)
    parser.add_argument("--duration", default="30s")
    args = parser.parse_args()
    cfg = Path(tempfile.mkdtemp()) / "gw.yaml"
    provider = {
        "name": "mock",
        "type": "openai_compat",
        "base_url": "http://127.0.0.1:9300/v1",
        "models": ["mock-small"],
    }
    cfg.write_text(yaml.safe_dump({"require_auth": True, "keys": [{"key": "sk-load", "tenant": "load"}],
                                   "providers": [provider]}))  # fmt: skip
    mock_env = os.environ | {"MOCK_TTFT_MS": "0", "MOCK_TOKENS_PER_S": "1000000", "MOCK_PORT": "9300"}
    gw_env = os.environ | {"GATEWAY_CONFIG": str(cfg), "GATEWAY_PORT": "8300", "LOG_LEVEL": "WARNING"}
    procs = [
        subprocess.Popen([sys.executable, "-c", "import llm_gateway.mock_llm as m; m.main()"], env=mock_env),
        subprocess.Popen([sys.executable, "-c", "import llm_gateway; llm_gateway.main()"], env=gw_env),
    ]
    try:
        wait("http://127.0.0.1:9300/healthz")
        wait("http://127.0.0.1:8300/healthz")
        direct = k6("http://127.0.0.1:9300", args.vus, args.duration)
        gateway = k6("http://127.0.0.1:8300", args.vus, args.duration)
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
    overhead = {k: round(gateway[k] - direct[k], 2) for k in ("p50_ms", "p90_ms", "p99_ms")}
    rows = {"direct (mock only)": direct, "through gateway": gateway, "gateway overhead": overhead}
    Path("results").mkdir(exist_ok=True)
    Path("results/overhead.json").write_text(
        json.dumps({"vus": args.vus, "duration": args.duration, **rows}, indent=2)
    )
    for name, r in rows.items():
        print(f"{name:<20} {r}")


if __name__ == "__main__":
    main()
