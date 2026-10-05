"""Record one signal log per experimental condition (RESEARCH.md §9, "record once, replay many").

For each condition a fresh gateway process is started with that condition's provider label and
fault, the tenant workloads are replayed through it as mixed traffic, and the gateway is stopped
so its signal log is flushed. A client-side sidecar keeps per-request ground truth (e.g. GSM8K
correctness) keyed by the gateway's X-Request-Id; the gateway itself never stores content.

    uv run python -m llm_gateway.research.record --conditions qwen_healthy,qwen_quant_swap --n 150

Finished conditions are skipped, so the command can be re-run after an interruption.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import yaml

from llm_gateway.research.workloads import Item, gsm8k_correct, load

QWEN = "qwen2.5:1.5b-instruct-q8_0"
LLAMA = "llama3.2:1b-instruct-q8_0"
TENANTS = ["math", "chat", "code", "tools"]


def condition(
    provider: str, model: str, fault: dict | None = None, shifted: tuple[str, ...] = (), providers=None
) -> dict:
    cond = {"provider": provider, "model": model, "faults": [fault] if fault else [], "shifted": shifted}
    return cond | ({"providers": providers} if providers else {})


# Amazon Bedrock (needs AWS credentials + model access): faults 1-5 only, quantization is not controllable.
NOVA = "nova-micro"
BEDROCK = [
    {
        "name": "bedrock-nova",
        "type": "bedrock",
        "region": "us-east-1",
        "models": {NOVA: "us.amazon.nova-micro-v1:0", "nova-lite": "us.amazon.nova-lite-v1:0"},
    }
]


# Provider A = Qwen 1.5B (faults injected here); provider B = Llama 3.2 1B (the control provider).
CONDITIONS = {
    "qwen_healthy": condition("ollama-qwen", QWEN),
    "qwen_healthy_2": condition("ollama-qwen", QWEN),  # independent healthy replicate for false alarms
    "llama_healthy": condition("ollama-llama", LLAMA),
    "qwen_drop_system": condition("ollama-qwen", QWEN, {"name": "drop_system"}),
    "qwen_truncate_context": condition(
        "ollama-qwen", QWEN, {"name": "truncate_context", "params": {"tokens": 32}}
    ),
    "qwen_sampling": condition("ollama-qwen", QWEN, {"name": "sampling", "params": {"temperature": 1.8}}),
    "qwen_output_cap": condition("ollama-qwen", QWEN, {"name": "output_cap", "params": {"max_tokens": 48}}),
    "qwen_quant_swap": condition(
        "ollama-qwen", QWEN, {"name": "quant_swap", "params": {"model": "qwen2.5:1.5b-instruct-q2_K"}}
    ),
    "qwen_model_substitution": condition(
        "ollama-qwen", QWEN, {"name": "model_substitution", "params": {"model": "qwen2.5:0.5b-instruct-q8_0"}}
    ),
    "qwen_throttle": condition("ollama-qwen", QWEN, {"name": "throttle", "params": {"delay_s": 0.03}}),
    # Traffic shift: the chat and math tenants change their own prompts; seen on both providers.
    "qwen_shifted": condition("ollama-qwen", QWEN, shifted=("chat", "math")),
    "llama_shifted": condition("ollama-llama", LLAMA, shifted=("chat", "math")),
    "nova_healthy": condition("bedrock-nova", NOVA, providers=BEDROCK),
    "nova_drop_system": condition("bedrock-nova", NOVA, {"name": "drop_system"}, providers=BEDROCK),
    "nova_truncate_context": condition(
        "bedrock-nova", NOVA, {"name": "truncate_context", "params": {"tokens": 32}}, providers=BEDROCK
    ),
    "nova_sampling": condition(
        "bedrock-nova", NOVA, {"name": "sampling", "params": {"temperature": 1.0}}, providers=BEDROCK
    ),
    "nova_output_cap": condition(
        "bedrock-nova", NOVA, {"name": "output_cap", "params": {"max_tokens": 48}}, providers=BEDROCK
    ),
    "nova_model_substitution": condition(
        "bedrock-nova",
        NOVA,
        {"name": "model_substitution", "params": {"model": "nova-lite"}},
        providers=BEDROCK,
    ),
}
LOCAL_CONDITIONS = [name for name, cond in CONDITIONS.items() if "providers" not in cond]


def stable_coin(key: str) -> bool:
    return hashlib.sha256(key.encode()).digest()[0] % 2 == 0


def interleave(items: list[Item], seed: int) -> list[Item]:
    """Mixed traffic: shuffle so tenants arrive interleaved, deterministically per seed."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


async def send(client: httpx.AsyncClient, item: Item, model: str, key: str, replicate: str) -> dict:
    body = {**item.body, "model": model}
    stream = stable_coin(item.item_id)  # half the traffic streams, so TTFT is observed
    if stream:
        body |= {"stream": True, "stream_options": {"include_usage": True}}
    headers = {"Authorization": f"Bearer {key}"}
    started = time.time()
    text, status, request_id = "", 0, None
    try:
        if stream:
            async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
                status, request_id = resp.status_code, resp.headers.get("x-request-id")
                async for line in resp.aiter_lines():
                    if line.startswith("data: {"):
                        for choice in json.loads(line[6:]).get("choices") or []:
                            text += (choice.get("delta") or {}).get("content") or ""
        else:
            resp = await client.post("/v1/chat/completions", json=body, headers=headers)
            status, request_id = resp.status_code, resp.headers.get("x-request-id")
            if status == 200:
                text = resp.json()["choices"][0]["message"].get("content") or ""
    except httpx.HTTPError as exc:
        text = f"<client error {exc.__class__.__name__}>"
    return {
        "request_id": request_id,
        "tenant": item.tenant,
        "item_id": item.item_id,
        "replicate": replicate,
        "status": status,
        "client_latency_s": time.time() - started,
        "correct": gsm8k_correct(text, item.answer) if item.answer else None,
    }


def start_gateway(out: Path, cond: dict, port: int) -> subprocess.Popen:
    keys = [{"key": f"sk-{t}", "tenant": t} for t in TENANTS]
    cfg = out / "gateway.yaml"
    cfg.write_text(
        yaml.safe_dump({"keys": keys, "faults": cond["faults"], "providers": cond.get("providers", [])})
    )
    env = os.environ | {
        "GATEWAY_CONFIG": str(cfg),
        "SIGNAL_DIR": str(out / "signals"),
        "UPSTREAM_PROVIDER_NAME": cond["provider"],
        "GATEWAY_PORT": str(port),
    }
    log = open(out / "gateway.log", "w")  # noqa: SIM115 - closed with the process
    proc = subprocess.Popen([sys.executable, "-c", "import llm_gateway; llm_gateway.main()"], env=env,
                            stdout=log, stderr=subprocess.STDOUT)  # fmt: skip
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"gateway did not start; see {out / 'gateway.log'}")


async def replay(port: int, items: list[Item], model: str, concurrency: int, replicate: str) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0
    timeout = httpx.Timeout(600, connect=5)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=timeout) as client:

        async def one(item: Item) -> dict:
            nonlocal done
            async with sem:
                row = await send(client, item, model, f"sk-{item.tenant}", replicate)
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(items)}", flush=True)
            return row

        return await asyncio.gather(*(one(i) for i in items))


def record(name: str, n: int, root: Path, concurrency: int, port: int) -> None:
    out = root / name
    if (out / "DONE").exists():
        print(f"skip {name} (done)")
        return
    out.mkdir(parents=True, exist_ok=True)
    cond = CONDITIONS[name]
    items = interleave(
        load(TENANTS, n, shifted=cond["shifted"]), seed=int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    )
    print(f"record {name}: {len(items)} requests on {cond['model']}", flush=True)
    started = time.time()
    proc = start_gateway(out, cond, port)
    try:
        rows = asyncio.run(replay(port, items, cond["model"], concurrency, name))
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    pd.DataFrame(rows).to_parquet(out / "client.parquet")
    meta = {"condition": name, **cond, "n_per_tenant": n, "requests": len(rows),
            "seconds": round(time.time() - started, 1), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=list))
    (out / "DONE").touch()
    print(f"  done in {meta['seconds']} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--conditions", default=",".join(LOCAL_CONDITIONS), help="comma-separated names (default: local ones)"
    )
    parser.add_argument("--n", type=int, default=150, help="requests per tenant")
    parser.add_argument("--out", type=Path, default=Path("data/recordings"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    for name in args.conditions.split(","):
        record(name.strip(), args.n, args.out, args.concurrency, args.port)


if __name__ == "__main__":
    main()
