"""Gateway settings, read from environment variables and an optional YAML file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    # OpenAI-compatible upstream; defaults to a local Ollama server.
    upstream_base_url: str = "http://localhost:11434/v1"
    # Sent as the upstream bearer token when set; the client's own key is never forwarded.
    upstream_api_key: str | None = None
    provider_name: str = "ollama"
    connect_timeout_s: float = 5.0
    # Long read timeout: completions on small local GPUs can take minutes.
    read_timeout_s: float = 300.0
    # Directory for per-response signal Parquet files; None disables signal logging.
    signal_dir: str | None = None
    # API key -> tenant name.
    keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        file_cfg = load_yaml(os.environ.get("GATEWAY_CONFIG"))
        return cls(
            upstream_base_url=os.environ.get("UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.environ.get("UPSTREAM_API_KEY") or None,
            provider_name=os.environ.get("UPSTREAM_PROVIDER_NAME", cls.provider_name),
            connect_timeout_s=float(os.environ.get("UPSTREAM_CONNECT_TIMEOUT_S", cls.connect_timeout_s)),
            read_timeout_s=float(os.environ.get("UPSTREAM_READ_TIMEOUT_S", cls.read_timeout_s)),
            signal_dir=os.environ.get("SIGNAL_DIR") or file_cfg.get("signal_dir"),
            keys={str(k["key"]): str(k["tenant"]) for k in file_cfg.get("keys", [])},
        )


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}
