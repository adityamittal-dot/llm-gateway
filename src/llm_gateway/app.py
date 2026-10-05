"""OpenAI-compatible passthrough gateway (Phase 0, day 1).

Requests to /v1/chat/completions are forwarded unchanged to one OpenAI-compatible
upstream (Ollama by default). Streaming responses are relayed chunk by chunk.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_gateway.config import Settings


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


def create_app(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or build_client(settings)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

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

        http: httpx.AsyncClient = request.app.state.client
        try:
            upstream = await http.send(
                http.build_request("POST", "/chat/completions", json=body), stream=True
            )
        except httpx.HTTPError as exc:
            return upstream_unavailable(exc)

        # Errors and non-streaming results are returned whole, with the upstream status code.
        if not body.get("stream") or upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            media_type = upstream.headers.get("content-type", "application/json")
            return Response(content, upstream.status_code, media_type=media_type)

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

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
