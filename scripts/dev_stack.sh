#!/usr/bin/env bash
# Local stack without Docker: Valkey (Redis-compatible), embedded Postgres, mock LLM and the gateway.
# Mirrors docker-compose.yml for machines where Docker isn't available. Ctrl-C stops everything.
#   Needs: valkey-server (or redis-server) on PATH; Postgres comes from the `pgserver` dev dependency.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=data/devstack && mkdir -p "$RUN"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

redis_bin="$(command -v valkey-server || command -v redis-server)"
"$redis_bin" --port 6379 --save "" --appendonly no --dir "$RUN" >"$RUN/redis.log" 2>&1 &
pids+=($!)

# pgserver starts Postgres in a data dir and prints a connection URI.
uv run python - <<'PY' >"$RUN/pg.log" 2>&1 &
import pgserver, time, pathlib
srv = pgserver.get_server(pathlib.Path("data/devstack/pg").resolve(), cleanup_mode="stop")
srv.psql("SELECT 1;")
try:
    srv.psql("CREATE DATABASE gateway;")
except Exception:
    pass
print(srv.get_uri("gateway"), flush=True)
while True:
    time.sleep(3600)
PY
pids+=($!)

MOCK_PORT=9000 uv run llm-gateway-mock >"$RUN/mock.log" 2>&1 &
pids+=($!)

for _ in $(seq 60); do grep -q postgresql "$RUN/pg.log" 2>/dev/null && break; sleep 0.5; done
pg_uri="$(grep -m1 postgresql "$RUN/pg.log")"
export DATABASE_URL="${pg_uri/postgresql:/postgresql+asyncpg:}"
export REDIS_URL=redis://127.0.0.1:6379/0 GATEWAY_CONFIG=config/dev.yaml
echo "redis: $REDIS_URL"
echo "postgres: $DATABASE_URL"
echo "gateway: http://127.0.0.1:8000  (key sk-dev-local, model mock-small)"
uv run llm-gateway
