"""Mock OpenAI-compatible LLM server for development, tests, load tests and chaos demos.

No GPU, no spend. Replies are deterministic per prompt; latency, throughput and error rate are
configurable by environment variable or per request via headers:

    MOCK_TTFT_MS=50  MOCK_TOKENS_PER_S=200  MOCK_ERROR_RATE=0.0  MOCK_ERROR_STATUS=503

    uv run llm-gateway-mock            # serves http://127.0.0.1:9000/v1
"""

import asyncio
import hashlib
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

WORDS = ["the", "gateway", "routes", "each", "request", "to", "a", "healthy", "provider", "and", "records", "what", "it", "observed", "so", "that", "silent", "quality", "changes", "can", "be", "detected", "early", "and", "traffic", "moved", "safely"]  # fmt: skip


def _cfg(request: Request, name: str, default: float) -> float:
    header = request.headers.get(f"x-mock-{name.replace('_', '-')}")
    return float(header if header is not None else os.environ.get(f"MOCK_{name.upper()}", default))


def reply_text(body: dict, n_tokens: int) -> str:
    seed = int(hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


def create_mock_app() -> FastAPI:
    app = FastAPI(title="Mock LLM")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "mock-small", "object": "model"},
                                           {"id": "mock-large", "object": "model"}]}  # fmt: skip

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if random.random() < _cfg(request, "error_rate", 0.0):
            status = int(_cfg(request, "error_status", 503))
            return JSONResponse(
                {"error": {"message": "mock failure", "type": "server_error"}}, status_code=status
            )
        ttft = _cfg(request, "ttft_ms", 50) / 1000
        tps = max(_cfg(request, "tokens_per_s", 200), 1)
        n = min(int(body.get("max_tokens") or 64), int(_cfg(request, "reply_tokens", 32)))
        model = body.get("model", "mock-small")
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in body.get("messages") or [])
        tools = body.get("tools") or []
        tool_call = None
        if tools:
            fn = tools[0]["function"]
            args = {k: "x" for k in (fn.get("parameters") or {}).get("required", [])}
            tool_call = {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                         "function": {"name": fn["name"], "arguments": json.dumps(args)}}  # fmt: skip
        text = "" if tool_call else reply_text(body, n)
        finish = "tool_calls" if tool_call else ("length" if n == body.get("max_tokens") else "stop")
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        if not body.get("stream"):
            await asyncio.sleep(ttft + n / tps)
            message = {"role": "assistant", "content": text or None}
            if tool_call:
                message["tool_calls"] = [tool_call]
            return {"id": cid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}  # fmt: skip

        async def events():
            def chunk(delta, finish_reason=None, **extra):
                data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}  # fmt: skip
                return f"data: {json.dumps(data)}\n\n"

            await asyncio.sleep(ttft)
            yield chunk({"role": "assistant", "content": ""})
            if tool_call:
                yield chunk({"tool_calls": [{"index": 0, **tool_call}]})
            for word in text.split():
                await asyncio.sleep(1 / tps)
                yield chunk({"content": word + " "})
            yield chunk({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_mock_app(), host=os.environ.get("MOCK_HOST", "127.0.0.1"),
                port=int(os.environ.get("MOCK_PORT", "9000")), log_level="warning")  # fmt: skip
