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
from redis.asyncio import from_url as redis_from_url

from llm_gateway import signals as sig
from llm_gateway.breaker import BreakerConfig, CircuitBreaker, MemoryStore, RedisStore
from llm_gateway.config import Settings, hash_key
from llm_gateway.faults import FaultInjector, parse_faults
from llm_gateway.pricing import Pricing
from llm_gateway.providers import ErrorKind, Provider, ProviderError, Registry, build_provider
from llm_gateway.providers.openai_compat import OpenAICompatProvider
from llm_gateway.ratelimit import Limiter, MemoryLimitStore, RedisLimitStore, estimate_tokens
from llm_gateway.retries import Retrier, RetryPolicy
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
    redis_client=None,
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
        app.state.registry = Registry(make_providers(), settings.routes)
        app.state.redis = redis_client or (redis_from_url(settings.redis_url) if settings.redis_url else None)
        store = RedisStore(app.state.redis) if app.state.redis is not None else MemoryStore()
        app.state.breaker = CircuitBreaker(store, BreakerConfig.from_config(settings.breaker))
        limit_store = RedisLimitStore(app.state.redis) if app.state.redis is not None else MemoryLimitStore()
        app.state.limiter = Limiter(limit_store, settings.tenants, settings.teams, settings.orgs)
        app.state.pricing = Pricing.from_config(settings.prices)
        app.state.signal_log = SignalLog(settings.signal_dir)
        app.state.signal_log.start()
        app.state.regenerates = sig.RegenerateTracker()
        app.state.faults = FaultInjector(parse_faults(settings.faults))
        app.state.retrier = Retrier(RetryPolicy.from_config(settings.retry))
        try:
            yield
        finally:
            await app.state.signal_log.stop()
            await app.state.registry.aclose()
            if app.state.redis is not None and redis_client is None:
                await app.state.redis.aclose()

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

    @app.get("/admin/breakers")
    async def get_breakers(request: Request, x_admin_key: str | None = Header(None)) -> dict:
        require_admin(x_admin_key)
        return await request.app.state.breaker.snapshot(list(request.app.state.registry.providers))

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
        candidates = request.app.state.registry.candidates(model)
        if not candidates:
            return openai_error(
                404,
                f"Model {model!r} is not served by this gateway.",
                "invalid_request_error",
                "model_not_found",
            )

        limiter: Limiter = request.app.state.limiter
        if scope := await limiter.over_budget(tenant):
            return openai_error(
                429, f"Monthly budget exhausted for {scope}.", "budget_exceeded", "budget_exceeded"
            )
        reservation, wait_ms = await limiter.reserve(tenant, estimate_tokens(body))
        if reservation is None:
            if wait_ms < 0:
                return openai_error(429, "Request is larger than the token-per-minute limit.", "rate_limit_exceeded",
                                    "request_too_large")  # fmt: skip
            resp = openai_error(429, "Rate limit exceeded.", "rate_limit_exceeded", "rate_limit_exceeded")
            resp.headers["Retry-After"] = str(max(1, -(-wait_ms // 1000)))
            return resp

        request_id = uuid.uuid4().hex
        headers = {"X-Request-Id": request_id}
        common = {
            "request_id": request_id,
            "tenant": tenant,
            "body": body,
            "started": time.time(),
            "regenerate": request.app.state.regenerates.seen_recently(tenant, sig.prompt_hash(body)),
        }
        breaker: CircuitBreaker = request.app.state.breaker
        retrier: Retrier = request.app.state.retrier
        attempts: list[dict] = []  # one entry per provider tried, for logs and the signal row

        async def record(provider: Provider, applied, status: int, first_token_at=None, message=None,
                         finish_reason=None, usage=None) -> None:  # fmt: skip
            signals = sig.extract(
                provider=provider.name,
                status_code=status,
                finished=time.time(),
                first_token_at=first_token_at,
                message=message,
                finish_reason=finish_reason,
                usage=usage,
                truth_fault=applied.fault if applied else None,
                **common,
            )
            if len(attempts) > 1 or any(a["outcome"] != "ok" for a in attempts):
                signals.extra["attempts"] = attempts
            request.app.state.signal_log.append(signals)
            # Settle the token reservation (refund the unused estimate) and charge the spend.
            actual = (usage or {}).get("total_tokens") or (
                ((usage or {}).get("prompt_tokens") or 0) + ((usage or {}).get("completion_tokens") or 0)
            )
            await limiter.settle(reservation, actual if usage else 0)
            target = applied.body.get("model", model) if applied else model
            cost = request.app.state.pricing.cost(target, usage)
            for scope, spent, soft in await limiter.add_spend(tenant, cost):
                log.warning("soft_budget_exceeded", extra={"scope": scope, "spent_usd": round(spent, 6),
                                                           "soft_budget_usd": soft})  # fmt: skip
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
                    "failovers": max(0, len(attempts) - 1),
                    "cost_usd": round(cost, 8),
                },
            )

        def on_retry_for(provider: Provider):
            def on_retry(attempt: int, err: ProviderError, delay: float) -> None:
                log.warning("retry", extra={"request_id": request_id, "provider": provider.name, "attempt": attempt,
                                            "error_kind": err.kind.value, "delay_ms": round(delay * 1000)})  # fmt: skip

            return on_retry

        async def open_stream(provider: Provider, send_body: dict):
            stream = provider.stream(send_body)
            try:
                return stream, await anext(stream)
            except StopAsyncIteration:
                return stream, None

        # Walk the fallback chain: skip providers whose breaker is open, fail over on provider
        # errors, never on a bad request (it would fail everywhere). Streams can only fail over
        # before the first chunk, which is why the first chunk is fetched here.
        last_error: ProviderError | None = None
        for provider, target in candidates:
            if not await breaker.allow(provider.name):
                attempts.append({"provider": provider.name, "outcome": "breaker_open"})
                continue
            applied = request.app.state.faults.apply({**body, "model": target})
            try:
                if body.get("stream"):
                    opened = await retrier.call(
                        provider.name,
                        lambda p=provider, b=applied.body: open_stream(p, b),
                        on_retry_for(provider),
                    )
                else:
                    result = await retrier.call(
                        provider.name,
                        lambda p=provider, b=applied.body: p.complete(b),
                        on_retry_for(provider),
                    )
            except ProviderError as err:
                await breaker.record(provider.name, err.kind)
                attempts.append({"provider": provider.name, "outcome": err.kind.value})
                last_error = err
                if err.kind is ErrorKind.BAD_REQUEST:
                    break
                log.warning("failover", extra={"request_id": request_id, "provider": provider.name,
                                               "error_kind": err.kind.value})  # fmt: skip
                continue
            await breaker.record(provider.name, None)
            attempts.append({"provider": provider.name, "outcome": "ok"})
            break
        else:
            provider = candidates[-1][0]
            applied = None
            if last_error is None:  # every candidate was skipped by an open breaker
                last_error = ProviderError(
                    ErrorKind.OVERLOADED, "All providers for this model are unavailable."
                )
            await record(provider, applied, last_error.status)
            return provider_error_response(last_error, headers)
        if attempts[-1]["outcome"] != "ok":  # bad request: stop without failover
            await record(provider, applied, last_error.status)
            return provider_error_response(last_error, headers)
        headers["X-Gateway-Provider"] = provider.name

        if not body.get("stream"):
            result["model"] = model  # the client sees the model it asked for, whoever answered
            choice = (result.get("choices") or [{}])[0]
            usage = result.get("usage") or {}
            if applied.chunk_delay_s:
                await asyncio.sleep(applied.chunk_delay_s * (usage.get("completion_tokens") or 0) / 10)
            await record(
                provider, applied, 200, None, choice.get("message"), choice.get("finish_reason"), usage
            )
            return JSONResponse(result, headers=headers)

        chunks, first = opened

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
                await breaker.record(provider.name, err.kind)
                error = {
                    "message": err.message,
                    "type": "upstream_error",
                    "code": f"upstream_{err.kind.value}",
                }
                yield sse({"error": error})
            finally:
                await record(
                    provider, applied, 200, first_token_at, acc.message(), acc.finish_reason, acc.usage
                )

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"} | headers,
        )

    return app
