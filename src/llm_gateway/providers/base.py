"""Provider adapter interface.

The gateway's internal format is the OpenAI Chat Completions format, as plain dicts: requests
are OpenAI request bodies, `complete` returns a `chat.completion` object and `stream` yields
`chat.completion.chunk` objects. Each adapter translates to and from its provider's API and
maps provider errors onto one taxonomy (ErrorKind), which drives retry and failover decisions.
"""

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol


class ErrorKind(StrEnum):
    RATE_LIMITED = "rate_limited"  # 429: retry later / fail over
    OVERLOADED = "overloaded"  # 5xx, provider busy: fail over
    TIMEOUT = "timeout"  # no answer in time: fail over
    UNAVAILABLE = "unavailable"  # cannot connect: fail over
    BAD_REQUEST = "bad_request"  # 4xx caused by the request: fails everywhere, never fail over
    AUTH = "auth"  # gateway's provider credentials rejected: fail over, alert

    @property
    def retryable(self) -> bool:
        return self not in (ErrorKind.BAD_REQUEST,)

    @property
    def http_status(self) -> int:
        return {
            ErrorKind.RATE_LIMITED: 429,
            ErrorKind.OVERLOADED: 503,
            ErrorKind.TIMEOUT: 504,
            ErrorKind.UNAVAILABLE: 502,
            ErrorKind.BAD_REQUEST: 400,
            ErrorKind.AUTH: 502,  # the client's request was fine; the gateway's upstream key was not
        }[self]


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, status: int | None = None, provider: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status or kind.http_status
        self.provider = provider


class Provider(Protocol):
    name: str

    def serves(self, model: str) -> bool: ...

    async def complete(self, body: dict) -> dict: ...

    def stream(self, body: dict) -> AsyncIterator[dict]: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


def kind_for_status(status: int) -> ErrorKind:
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if status in (401, 403):
        return ErrorKind.AUTH
    if status in (408, 504):
        return ErrorKind.TIMEOUT
    if status >= 500:
        return ErrorKind.OVERLOADED
    return ErrorKind.BAD_REQUEST
