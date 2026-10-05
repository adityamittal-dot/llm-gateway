"""OpenAI-compatible gateway.

Requests to /v1/chat/completions are forwarded to one OpenAI-compatible upstream (Ollama by
default). Streaming responses are relayed chunk by chunk. Every response produces one row
of passive quality signals (see signals.py).
"""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_gateway import signals as sig
from llm_gateway.config import Settings
from llm_gateway.signal_log import SignalLog


def openai_error(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


def build_client(settings: Settings) -> httpx.AsyncClient:
    headers = {}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    timeout = httpx.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s)
    return httpx.AsyncClient(base_url=settings.upstream_base_url, headers=headers, timeout=timeout)


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


def create_app(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or build_client(settings)
        app.state.signal_log = SignalLog(settings.signal_dir)
        app.state.signal_log.start()
        app.state.regenerates = sig.RegenerateTracker()
        try:
            yield
        finally:
            await app.state.signal_log.stop()
            await app.state.client.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

    def tenant_for(request: Request) -> str:
        return settings.keys.get(bearer_token(request) or "", "anonymous")

    def record(request: Request, **kwargs) -> sig.ResponseSignals:
        signals = sig.extract(provider=settings.provider_name, **kwargs)
        request.app.state.signal_log.append(signals)
        return signals

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        try:
            upstream = await request.app.state.client.get("/models")
        except httpx.HTTPError as exc:
            return upstream_unavailable(exc)
        return Response(upstream.content, upstream.status_code, media_type="application/json")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return openai_error(400, "Request body must be JSON.", "invalid_request_error")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object.", "invalid_request_error")

        tenant = tenant_for(request)
        common = {
            "request_id": uuid.uuid4().hex,
            "tenant": tenant,
            "body": body,
            "started": time.time(),
            "regenerate": request.app.state.regenerates.seen_recently(tenant, sig.prompt_hash(body)),
        }

        http: httpx.AsyncClient = request.app.state.client
        try:
            upstream = await http.send(
                http.build_request("POST", "/chat/completions", json=body), stream=True
            )
        except httpx.HTTPError as exc:
            error = upstream_unavailable(exc)
            record(request, **common, status_code=error.status_code, finished=time.time(),
                   first_token_at=None, message=None, finish_reason=None, usage=None)  # fmt: skip
            return error

        # Errors and non-streaming results are returned whole, with the upstream status code.
        if not body.get("stream") or upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            finished = time.time()
            message, finish_reason, usage = None, None, None
            if upstream.status_code == 200:
                try:
                    payload = upstream.json()
                    choice = (payload.get("choices") or [{}])[0]
                    message, finish_reason, usage = (
                        choice.get("message"),
                        choice.get("finish_reason"),
                        payload.get("usage"),
                    )
                except ValueError:
                    pass
            record(request, **common, status_code=upstream.status_code, finished=finished,
                   first_token_at=None, message=message, finish_reason=finish_reason, usage=usage)  # fmt: skip
            media_type = upstream.headers.get("content-type", "application/json")
            return Response(content, upstream.status_code, media_type=media_type)

        async def relay() -> AsyncIterator[bytes]:
            acc = sig.StreamAccumulator()
            first_token_at = None
            try:
                async for chunk in upstream.aiter_bytes():
                    if first_token_at is None:
                        first_token_at = time.time()
                    acc.feed(chunk)
                    yield chunk
            finally:
                await upstream.aclose()
                record(request, **common, status_code=200, finished=time.time(), first_token_at=first_token_at,
                       message=acc.message(), finish_reason=acc.finish_reason, usage=acc.usage)  # fmt: skip

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def upstream_unavailable(exc: httpx.HTTPError) -> JSONResponse:
    if isinstance(exc, httpx.TimeoutException):
        return openai_error(504, "Upstream provider timed out.", "upstream_error", "upstream_timeout")
    return openai_error(
        502,
        f"Upstream provider unavailable: {exc.__class__.__name__}.",
        "upstream_error",
        "upstream_unavailable",
    )
