"""Adapter for any OpenAI-compatible API: Ollama, vLLM, Groq, Gemini's OpenAI endpoint, OpenAI itself."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_gateway.providers.base import ErrorKind, ProviderError, kind_for_status


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        models: list[str] | None = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.name = name
        self.models = models or ["*"]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
        )

    def serves(self, model: str) -> bool:
        return "*" in self.models or model in self.models

    def _error(self, exc: Exception) -> ProviderError:
        kind = ErrorKind.TIMEOUT if isinstance(exc, httpx.TimeoutException) else ErrorKind.UNAVAILABLE
        return ProviderError(kind, f"{self.name}: {exc.__class__.__name__}", provider=self.name)

    def _status_error(self, status: int, content: bytes) -> ProviderError:
        try:
            message = json.loads(content)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = content.decode(errors="replace")[:500] or f"HTTP {status}"
        kind = kind_for_status(status)
        # Request errors keep the provider's status (400/404/422); upstream faults use the taxonomy's.
        return ProviderError(kind, message, status if kind is ErrorKind.BAD_REQUEST else None, self.name)

    async def complete(self, body: dict) -> dict:
        body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        try:
            resp = await self.client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        if resp.status_code != 200:
            raise self._status_error(resp.status_code, resp.content)
        return resp.json()

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        body = {**body, "stream": True}
        body.setdefault("stream_options", {"include_usage": True})
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    raise self._status_error(resp.status_code, await resp.aread())
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except ValueError:
                        continue
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc

    async def list_models(self) -> list[str]:
        if "*" not in self.models:
            return list(self.models)
        try:
            resp = await self.client.get("/models")
            return [m["id"] for m in resp.json().get("data", [])] if resp.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return []

    async def aclose(self) -> None:
        await self.client.aclose()
