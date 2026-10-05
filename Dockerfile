# Gateway image (also runs the mock LLM with a different command).
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY README.md ./
RUN uv sync --frozen --no-dev

RUN useradd --system --uid 10001 gateway && mkdir -p /app/data && chown gateway /app/data
USER gateway
ENV PATH="/app/.venv/bin:$PATH" GATEWAY_HOST=0.0.0.0 GATEWAY_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["llm-gateway"]
