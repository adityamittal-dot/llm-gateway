"""Gateway settings, read from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # OpenAI-compatible upstream; defaults to a local Ollama server.
    upstream_base_url: str = "http://localhost:11434/v1"
    # Sent as the upstream bearer token when set; the client's own key is never forwarded.
    upstream_api_key: str | None = None
    connect_timeout_s: float = 5.0
    # Long read timeout: completions on small local GPUs can take minutes.
    read_timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            upstream_base_url=os.environ.get("UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.environ.get("UPSTREAM_API_KEY") or None,
            connect_timeout_s=float(os.environ.get("UPSTREAM_CONNECT_TIMEOUT_S", cls.connect_timeout_s)),
            read_timeout_s=float(os.environ.get("UPSTREAM_READ_TIMEOUT_S", cls.read_timeout_s)),
        )
