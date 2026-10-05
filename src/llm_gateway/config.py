"""Gateway settings, read from environment variables and an optional YAML file (GATEWAY_CONFIG).

Example YAML:

    require_auth: true
    keys:
      - {key_hash: <sha256 of the key>, tenant: acme}   # preferred: no plaintext keys on disk
      - {key: sk-dev, tenant: dev}                       # convenient for local development
    providers:
      - {name: ollama, type: openai_compat, base_url: "http://localhost:11434/v1", models: ["*"]}
      - {name: groq, type: openai_compat, base_url: "https://api.groq.com/openai/v1",
         api_key_env: GROQ_API_KEY, models: [llama-3.1-8b-instant]}
      - {name: bedrock, type: bedrock, region: us-east-1, models: {nova-micro: us.amazon.nova-micro-v1:0}}

Without a `providers` list, one OpenAI-compatible provider is built from UPSTREAM_BASE_URL.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class Settings:
    # Default single upstream (used when no providers are configured); a local Ollama server.
    upstream_base_url: str = "http://localhost:11434/v1"
    # Sent as the upstream bearer token when set; the client's own key is never forwarded.
    upstream_api_key: str | None = None
    provider_name: str = "ollama"
    connect_timeout_s: float = 5.0
    # Long read timeout: completions on small local GPUs can take minutes.
    read_timeout_s: float = 300.0
    # Directory for per-response signal Parquet files; None disables signal logging.
    signal_dir: str | None = None
    # Plaintext API key -> tenant (development and tests).
    keys: dict[str, str] = field(default_factory=dict)
    # sha256(API key) -> tenant.
    key_hashes: dict[str, str] = field(default_factory=dict)
    # Reject requests without a known key (401). Off by default for local experiments.
    require_auth: bool = False
    # Provider configs, see module docstring.
    providers: list[dict] = field(default_factory=list)
    # Research fault injection (faults.py); each entry is {name, p, params}.
    faults: list[dict] = field(default_factory=list)
    # Enables /admin endpoints when set (sent as the X-Admin-Key header).
    admin_key: str | None = None
    log_level: str = "INFO"
    # Retry policy overrides (retries.RetryPolicy fields).
    retry: dict = field(default_factory=dict)
    # Circuit breaker overrides (breaker.BreakerConfig fields).
    breaker: dict = field(default_factory=dict)
    # Model alias -> ordered fallback chain. Entries are model names (served by whichever provider
    # serves them) or {provider: name, model: id} to pin a provider.
    routes: dict = field(default_factory=dict)
    # Shared state (breaker, rate limits, caches). Without it, state is per process.
    redis_url: str | None = None
    # Limits and budgets per tenant (+ its team and org): {rpm, tpm, budget_usd, soft_budget_usd, team, org}.
    tenants: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)
    orgs: dict = field(default_factory=dict)
    # USD per million tokens: {model: {input_per_mtok, output_per_mtok, cached_input_per_mtok}}.
    prices: dict = field(default_factory=dict)
    # Postgres (postgresql+asyncpg://...): usage ledger and versioned prices. Optional.
    database_url: str | None = None
    # Response cache (cache.CacheConfig fields).
    cache: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        file_cfg = load_yaml(os.environ.get("GATEWAY_CONFIG"))
        keys = file_cfg.get("keys") or []
        return cls(
            upstream_base_url=os.environ.get("UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.environ.get("UPSTREAM_API_KEY") or None,
            provider_name=os.environ.get("UPSTREAM_PROVIDER_NAME", cls.provider_name),
            connect_timeout_s=float(os.environ.get("UPSTREAM_CONNECT_TIMEOUT_S", cls.connect_timeout_s)),
            read_timeout_s=float(os.environ.get("UPSTREAM_READ_TIMEOUT_S", cls.read_timeout_s)),
            signal_dir=os.environ.get("SIGNAL_DIR") or file_cfg.get("signal_dir"),
            keys={str(k["key"]): str(k["tenant"]) for k in keys if "key" in k},
            key_hashes={str(k["key_hash"]): str(k["tenant"]) for k in keys if "key_hash" in k},
            require_auth=env_bool("REQUIRE_AUTH", bool(file_cfg.get("require_auth", False))),
            providers=list(file_cfg.get("providers") or []),
            faults=list(file_cfg.get("faults") or []),
            admin_key=os.environ.get("ADMIN_KEY") or file_cfg.get("admin_key"),
            log_level=os.environ.get("LOG_LEVEL", file_cfg.get("log_level", cls.log_level)),
            retry=dict(file_cfg.get("retry") or {}),
            breaker=dict(file_cfg.get("breaker") or {}),
            routes=dict(file_cfg.get("routes") or {}),
            redis_url=os.environ.get("REDIS_URL") or file_cfg.get("redis_url"),
            tenants=dict(file_cfg.get("tenants") or {}),
            teams=dict(file_cfg.get("teams") or {}),
            orgs=dict(file_cfg.get("orgs") or {}),
            prices=dict(file_cfg.get("prices") or {}),
            database_url=os.environ.get("DATABASE_URL") or file_cfg.get("database_url"),
            cache=dict(file_cfg.get("cache") or {}),
        )

    def provider_configs(self) -> list[dict]:
        if self.providers:
            return self.providers
        default = {"name": self.provider_name, "type": "openai_compat", "base_url": self.upstream_base_url}
        return [default | {"api_key": self.upstream_api_key, "models": ["*"]}]

    def tenant_index(self) -> dict[str, str]:
        """sha256(key) -> tenant for every configured key."""
        return {hash_key(k): t for k, t in self.keys.items()} | dict(self.key_hashes)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}
