"""OpenAI-compatible gateway.

POST /v1/chat/completions is authenticated by API key, routed to the provider that serves the
requested model, and returned in OpenAI format (streaming or not). Every response produces one
row of passive quality signals (signals.py) and one JSON access-log line.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_gateway import signals as sig
from llm_gateway.config import Settings, hash_key
from llm_gateway.faults import FaultInjector, parse_faults
from llm_gateway.providers import ErrorKind, Provider, ProviderError, Registry, build_provider
from llm_gateway.providers.openai_compat import OpenAICompatProvider
from llm_gateway.signal_log import SignalLog

log = logging.getLogger("llm_gateway.access")


def openai_error(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


def provider_error_response(err: ProviderError, headers: dict | None = None) -> JSONResponse:
    if err.kind is ErrorKind.BAD_REQUEST:
        resp = openai_error(err.status, err.message, "invalid_request_error")
    else:
        resp = openai_error(err.status, err.message, "upstream_error", f"upstream_{err.kind.value}")
    resp.headers.update(headers or {})
    return resp


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


def sse(data: dict | str) -> bytes:
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n".encode()


def create_app(
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    providers: list[Provider] | None = None,
) -> FastAPI:
    """`client` (tests) wraps a ready httpx client as the single provider; `providers` replaces config."""
    settings = settings or Settings.from_env()
    tenants = settings.tenant_index()

    def make_providers() -> list[Provider]:
        if providers is not None:
            return providers
        if client is not None:
            return [OpenAICompatProvider(settings.provider_name, "", client=client)]
        return [
            build_provider(c, settings.connect_timeout_s, settings.read_timeout_s)
            for c in settings.provider_configs()
        ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = Registry(make_providers())
        app.state.signal_log = SignalLog(settings.signal_dir)
        app.state.signal_log.start()
        app.state.regenerates = sig.RegenerateTracker()
        app.state.faults = FaultInjector(parse_faults(settings.faults))
        try:
            yield
        finally:
            await app.state.signal_log.stop()
            await app.state.registry.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

    def require_admin(key: str | None) -> None:
        if not settings.admin_key:
            raise HTTPException(404)
        if key != settings.admin_key:
            raise HTTPException(401, "invalid admin key")

    @app.get("/admin/faults")
    async def get_faults(request: Request, x_admin_key: str | None = Header(None)) -> list[dict]:
        require_admin(x_admin_key)
        return [vars(f) for f in request.app.state.faults.faults]

    @app.put("/admin/faults")
    async def put_faults(request: Request, x_admin_key: str | None = Header(None)) -> list[dict]:
        require_admin(x_admin_key)
        try:
            faults = parse_faults(await request.json())
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        request.app.state.faults.set(faults)
        return [vars(f) for f in faults]

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        models = await request.app.state.registry.list_models()
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "gateway"} for m in models],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        token = bearer_token(request)
        tenant = tenants.get(hash_key(token)) if token else None
        if tenant is None and settings.require_auth:
            return openai_error(401, "Missing or invalid API key.", "authentication_error", "invalid_api_key")
        tenant = tenant or "anonymous"

        try:
            body = await request.json()
        except ValueError:
            return openai_error(400, "Request body must be JSON.", "invalid_request_error")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object.", "invalid_request_error")
        model = str(body.get("model") or "")
        provider: Provider | None = request.app.state.registry.resolve(model)
        if provider is None:
            return openai_error(
                404,
                f"Model {model!r} is not served by this gateway.",
                "invalid_request_error",
                "model_not_found",
            )

        request_id = uuid.uuid4().hex
        headers = {"X-Request-Id": request_id}
        applied = request.app.state.faults.apply(body)
        common = {
            "request_id": request_id,
            "tenant": tenant,
            "body": body,
            "started": time.time(),
            "regenerate": request.app.state.regenerates.seen_recently(tenant, sig.prompt_hash(body)),
            "truth_fault": applied.fault,
        }

        def record(status: int, first_token_at=None, message=None, finish_reason=None, usage=None) -> None:
            signals = sig.extract(
                provider=provider.name,
                status_code=status,
                finished=time.time(),
                first_token_at=first_token_at,
                message=message,
                finish_reason=finish_reason,
                usage=usage,
                **common,
            )
            request.app.state.signal_log.append(signals)
            log.info(
                "request",
                extra={
                    "request_id": request_id,
                    "tenant": tenant,
                    "provider": provider.name,
                    "model": model,
                    "stream": signals.stream,
                    "status": status,
                    "latency_ms": round(signals.latency_s * 1000, 1),
                    "prompt_tokens": signals.prompt_tokens,
                    "completion_tokens": signals.completion_tokens,
                    "finish_reason": finish_reason,
                },
            )

        if not body.get("stream"):
            try:
                result = await provider.complete(applied.body)
            except ProviderError as err:
                record(err.status)
                return provider_error_response(err, headers)
            result["model"] = model  # a substituted model stays invisible to the client
            choice = (result.get("choices") or [{}])[0]
            usage = result.get("usage") or {}
            if applied.chunk_delay_s:
                await asyncio.sleep(applied.chunk_delay_s * (usage.get("completion_tokens") or 0) / 10)
            record(200, None, choice.get("message"), choice.get("finish_reason"), usage)
            return JSONResponse(result, headers=headers)

        # Streaming: fetch the first chunk before answering, so an upstream error that happens
        # before any token is returned as a proper HTTP error (and can trigger failover).
        chunks = provider.stream(applied.body)
        try:
            first = await anext(chunks)
        except StopAsyncIteration:
            first = None
        except ProviderError as err:
            record(err.status)
            return provider_error_response(err, headers)

        async def relay() -> AsyncIterator[bytes]:
            acc = sig.StreamAccumulator()
            first_token_at = time.time()
            pending = [first] if first is not None else []
            try:
                while pending:
                    chunk = pending.pop()
                    chunk["model"] = model
                    acc.add(chunk)
                    if applied.chunk_delay_s:
                        await asyncio.sleep(applied.chunk_delay_s)
                    yield sse(chunk)
                    try:
                        pending.append(await anext(chunks))
                    except StopAsyncIteration:
                        pass
                yield sse("[DONE]")
            except ProviderError as err:  # mid-stream: too late to fail over; tell the client in-band
                error = {
                    "message": err.message,
                    "type": "upstream_error",
                    "code": f"upstream_{err.kind.value}",
                }
                yield sse({"error": error})
            finally:
                record(200, first_token_at, acc.message(), acc.finish_reason, acc.usage)

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"} | headers,
        )

    return app
