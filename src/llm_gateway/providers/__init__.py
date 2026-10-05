"""Provider registry: builds adapters from config and resolves a request's model to a provider."""

import os

from llm_gateway.providers.base import ErrorKind, Provider, ProviderError
from llm_gateway.providers.bedrock import BedrockProvider
from llm_gateway.providers.openai_compat import OpenAICompatProvider

__all__ = ["ErrorKind", "Provider", "ProviderError", "Registry", "build_provider"]


def build_provider(cfg: dict, connect_timeout_s: float = 5.0, read_timeout_s: float = 300.0) -> Provider:
    kind = cfg.get("type", "openai_compat")
    if kind == "openai_compat":
        api_key = cfg.get("api_key") or (
            os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None
        )
        return OpenAICompatProvider(
            cfg["name"],
            cfg["base_url"],
            api_key,
            list(cfg.get("models") or ["*"]),
            connect_timeout_s,
            read_timeout_s,
        )
    if kind == "bedrock":
        models = cfg.get("models") or {}
        if isinstance(models, list):
            models = {m: m for m in models}
        return BedrockProvider(cfg["name"], models, cfg.get("region"))
    raise ValueError(f"unknown provider type {kind!r}")


class Registry:
    def __init__(self, providers: list[Provider], routes: dict | None = None):
        self.providers = {p.name: p for p in providers}
        self.routes = dict(routes or {})

    def candidates(self, model: str) -> list[tuple[Provider, str]]:
        """Ordered (provider, provider-side model) pairs to try for a requested model or route alias."""
        if model not in self.routes:
            provider = self.resolve(model)
            return [(provider, model)] if provider else []
        out = []
        for entry in self.routes[model]:
            if isinstance(entry, dict):
                provider, target = self.providers.get(entry["provider"]), entry["model"]
            else:
                provider, target = self.resolve(entry), entry
            if provider is not None:
                out.append((provider, target))
        return out

    def resolve(self, model: str) -> Provider | None:
        """First configured provider that serves the model; wildcard providers come last."""
        exact = [
            p for p in self.providers.values() if p.serves(model) and "*" not in getattr(p, "models", [])
        ]
        wildcard = [p for p in self.providers.values() if p.serves(model)]
        return (exact or wildcard or [None])[0]

    def get(self, name: str) -> Provider | None:
        return self.providers.get(name)

    async def list_models(self) -> list[str]:
        seen: list[str] = list(self.routes)
        for p in self.providers.values():
            seen += [m for m in await p.list_models() if m not in seen]
        return seen

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
