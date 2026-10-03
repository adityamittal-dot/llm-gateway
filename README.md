# LLM Gateway

A self-built, provider-agnostic gateway that routes, caches, secures and meters every LLM call an application makes — instead of apps hitting Bedrock, Anthropic or OpenAI directly. Apps keep using the OpenAI (or Anthropic) SDK they already have and only change `base_url`.

Built to demonstrate real distributed-systems infra skills (token-aware rate limiting, semantic caching with measured correctness, circuit breaking, multi-provider failover, cost/latency observability), not just "call an LLM API". Positioned as **the gateway for agent workloads**: the place where runaway agent loops, model spend and cache correctness are controlled.

## Problem statement

Every team adding an LLM feature eventually hits the same problems:

- **Cost is invisible until it isn't.** No visibility into cost-per-request, per-tenant or per-agent-run until the bill spikes.
- **Vendor lock-in / outage risk.** A single provider dependency means a single point of failure.
- **No caching for non-deterministic responses.** Exact-match caching barely works for natural language; most teams cache nothing and re-pay for near-identical prompts.
- **No per-tenant limits.** One noisy client — or one agent stuck in a loop — can starve everyone else or blow the budget alone.

This project solves these by building the gateway layer that should sit between any app and any LLM provider.

## Architecture

```
Client (OpenAI / Anthropic SDK, base_url = gateway)
  |
  v
CloudFront (optional) -> ALB            <- TLS, long idle timeout for streaming
  |
  v
ECS Fargate (N tasks): Uvicorn -> FastAPI (async, SSE streaming)
  |
  +-- Auth: API key (hashed) -> tenant / team / key
  +-- Rate limit + budget check (Redis token bucket, token-aware)
  +-- Cache: exact-match hash -> semantic (vector) lookup, tenant-scoped
  |     +-- HIT  -> return cached response
  |     +-- MISS -> Router
  |
  +-- Router (fallback chains, cost/latency rules, complexity routing)
  |     |
  |     +-- Circuit breaker (state shared in Redis)
  |           |
  |           +-- BedrockAdapter        (Converse API: Claude, Llama, Mistral, Nova)
  |           +-- AnthropicAdapter      (direct API)
  |           +-- OpenAIAdapter / AzureOpenAIAdapter
  |           +-- GeminiAdapter
  |           +-- OpenAICompatAdapter   (vLLM, Ollama, Groq, Together...)
  |
  +-- Usage event -> SQS / Firehose -> S3 + Athena (off the hot path)

State:  Redis       - rate limits, caches, breaker state
        Postgres    - tenants, keys, budgets, routing config, price table, usage ledger
Telemetry: OpenTelemetry (GenAI semantic conventions) -> X-Ray / CloudWatch, Sentry for errors
```

### Why not API Gateway in the hot path

API Gateway HTTP APIs cap integrations at 30s and buffer responses, which breaks token streaming and long completions. Auth and rate limiting live inside the gateway instead (as LiteLLM and Portkey do). Cognito is only used for the admin dashboard login.

## Provider-agnostic by design

Bedrock is one adapter, not the architecture. Every provider implements the same interface:

```python
class ProviderAdapter(Protocol):
    async def complete(self, req: ChatRequest) -> ChatResponse: ...
    def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
    def classify_error(self, exc: Exception) -> ErrorKind: ...  # rate_limited | overloaded | bad_request | auth | timeout
```

Adapters translate the unified request to the provider's format and back, normalise errors into one taxonomy (which drives retry/failover decisions) and report token usage for cost accounting.

Public API surface:

| Endpoint | Compatible with |
|---|---|
| `POST /v1/chat/completions` | OpenAI SDKs (streaming + non-streaming, tools) |
| `POST /v1/messages` | Anthropic SDKs |
| `POST /v1/embeddings` | OpenAI SDKs |
| `GET /v1/models` | Lists models the calling key may use |
| `/admin/*` | Tenants, keys, budgets, routing rules, usage reports |

## Data model (Postgres)

Postgres is the system of record; Redis only holds hot, rebuildable state (rate-limit counters, caches, breaker state).

| Table | Purpose |
|---|---|
| `orgs`, `teams` | Tenant hierarchy |
| `api_keys` | Hashed key, owner (org/team), allowed models, rate limits, status |
| `budgets` | Soft/hard spend limits per org, team, key or session; period and reset rules |
| `routing_rules` | Fallback chains, model aliases, complexity-routing and shadow-traffic config |
| `model_prices` | Versioned price per model/region (input, output, cached tokens), effective-from date |
| `usage_ledger` | One row per request: tenant, key, session, model, provider, tokens, cost, latency, cache status, failover flag. **Partitioned by month** so old data can be detached and archived to S3 cheaply |

Design rules:
- The ledger is append-only; budgets are checked against Redis counters on the hot path and reconciled from the ledger.
- Writes to the ledger happen off the request path (background batch insert or via SQS), so a slow database never slows a completion.
- Config reads (keys, routing rules, prices) are cached in-process with a short TTL and invalidated on admin changes.
- `pgvector` is a possible alternative store for the semantic cache, but Redis is the default because cache lookups sit on the hot path.

## The real engineering problems

### 1. Semantic cache correctness
- **Tenant-scoped.** A shared cache can serve one customer's answer to another; that's a data leak, not a miss.
- **Key = model + system prompt + params (temperature, tools) + full conversation**, not just the last user message.
- **Two layers:** exact-match hash first (free, zero false positives), then embedding similarity.
- **Opt-in per request** (`x-cache: semantic`); skipped by default for tool calls and high temperature.
- **Measured:** a labelled set of prompt pairs (same intent / different intent) is used to tune the similarity threshold and report precision/recall in CI. The dashboard shows hit rate, estimated false-hit rate and dollars saved.
- The embedding call adds latency on a miss; the net latency/cost effect is reported, not assumed.

### 2. Failover and circuit breaking
- Fail over on `429`, `5xx` and timeouts. **Never on `400`** — a bad request fails everywhere.
- Breaker state lives in Redis and is shared by all Fargate tasks.
- **Streams can only fail over before the first token.** A time-to-first-token deadline makes that fast.
- Retries use exponential backoff with jitter and a per-request retry budget to prevent retry storms.

### 3. Token-aware multi-tenant rate limiting
- Requests-per-minute alone is meaningless when one request can be 100 or 100k tokens.
- Token bucket in Redis (atomic Lua script): reserve an **estimated** token count up front, reconcile with actual usage (reported at the end of a stream).
- Hierarchical limits and budgets: **org -> team -> key**, soft (alert) and hard (reject or downgrade) limits.

### 4. Cost accounting
- Versioned price table per model and region, with cached-token discounts handled separately.
- Every request writes to a usage ledger, so per-tenant bills can be rebuilt from it.

### 5. Gateway overhead
- The gateway's own p50/p99 latency (excluding the provider, cache off) is measured with k6 and published in this README. It is the first number anyone evaluating a gateway asks for.

## Differentiating features

The basics (unified API, fallback, keys, logs) are table stakes — LiteLLM, Portkey, Kong, Cloudflare AI Gateway, Bifrost, OpenRouter and Helicone all do them. This project goes deep on agent workloads and cost:

| Feature | What it does |
|---|---|
| **Agent cost controls** | Budgets per session / agent run (`x-session-id`), runaway-loop detection (repeated identical tool calls), auto-downgrade to a cheaper model near budget |
| **Measured semantic cache** | Hit rate, estimated false-hit rate and dollars saved, backed by a labelled eval set |
| **Complexity routing** | A small classifier sends easy prompts to a cheap model (Haiku) and hard ones to a strong one (Sonnet); quality checked with sampled LLM-as-judge |
| **Shadow traffic / model A/B** | Mirror X% of traffic to a candidate model and compare cost, latency and quality |
| **"What-if" cost replay** | "Last month on model Y would have cost $Z" from the usage ledger |
| **Prompt-caching passthrough** | Supports Anthropic `cache_control` / Bedrock prompt caching and reports the savings |
| **Guardrails** | PII redaction, prompt-injection detection, optional Bedrock Guardrails |
| **Tool-call / MCP governance** | Allow-lists and audit log for the tools agents may call |
| **Request hedging** | Fire a second request if the first is slow past p95; take whichever returns first |
| **Batch routing** | Requests marked async go to provider batch APIs (~50% cheaper) |

## Tech stack

### Compute & routing
| Layer | Choice |
|---|---|
| Container runtime | ECS Fargate |
| App server | Uvicorn (ASGI) |
| Framework | FastAPI (async, Pydantic v2 models, SSE streaming via `StreamingResponse`) |
| HTTP client | `httpx` (async, connection pooling) |
| Load balancer | Application Load Balancer (idle timeout raised for streaming), optional CloudFront in front |
| Auth | API keys (hashed, per tenant/team) in-app; Cognito for the admin dashboard |

### Caching & state
| Layer | Choice |
|---|---|
| Rate limits, breaker state, exact cache | Redis (ElastiCache) |
| Semantic cache | Redis vector search (ElastiCache Valkey/MemoryDB with vector search; Redis Stack locally) |
| Redis client | `redis-py` asyncio |
| Embedding model | Amazon Titan Embeddings via Bedrock (pluggable) |
| Primary database | PostgreSQL 16+ on RDS — system of record for tenants, keys, budgets, routing config, price table and usage ledger |
| DB access | SQLAlchemy 2.0 async on `asyncpg`, Alembic migrations |
| Connection pooling | RDS Proxy (many Fargate tasks x pool size would otherwise exhaust Postgres connections) |
| Analytics | SQS/Firehose -> S3 -> Athena |

### Model backends
| Layer | Choice |
|---|---|
| Bedrock | Converse / ConverseStream API via `aioboto3` (one format across Bedrock models, cross-region inference profiles) |
| Others | Anthropic, OpenAI, Azure OpenAI, Gemini, any OpenAI-compatible server (vLLM, Ollama) |
| Dev primary | Claude Haiku on Bedrock |
| Fallback | Second Bedrock model or Anthropic direct (stretch: self-hosted vLLM, demo-only) |

### Observability
| Layer | Choice |
|---|---|
| Tracing & metrics | OpenTelemetry with GenAI semantic conventions -> X-Ray / CloudWatch (ADOT collector sidecar) |
| Logs | Structured JSON to CloudWatch Logs (metadata by default; prompt logging opt-in with redaction and retention) |
| Error tracking | Sentry (FastAPI integration), custom events for breaker trips |
| Dashboards & alerts | CloudWatch Dashboards + Alarms, Sentry alerts |

### Local dev / testing
| Layer | Choice |
|---|---|
| Local stack | Docker Compose: gateway + Redis Stack + Postgres + **mock LLM server** (no Bedrock spend in dev) |
| Tests | pytest + pytest-asyncio, fake providers, contract tests per adapter |
| Chaos | Toxiproxy to inject latency/errors and prove failover |
| Load | k6 (gateway overhead, breaker behaviour under failure) |
| Cache eval | Labelled prompt-pair set, precision/recall gate in CI |
| Lint/type | ruff, mypy |
| CI/CD | GitHub Actions: test -> build image -> push to ECR -> `terraform apply` -> ECS deploy |

## Infrastructure as code (Terraform)

All AWS infrastructure is Terraform; application code ships as a container image that Terraform references by tag.

```
infra/
  bootstrap/            # one-time: S3 state bucket (native S3 locking), GitHub OIDC role
  modules/
    networking/         # VPC, public/private subnets, NAT (or VPC endpoints), security groups
    ecr/                # image repository + lifecycle policy
    ecs-service/        # cluster, task definition (app + ADOT sidecar), service, autoscaling, IAM task role
    alb/                # ALB, listeners, target group, ACM cert, idle timeout
    cache/              # ElastiCache / MemoryDB, subnet group, parameter group
    database/           # RDS Postgres, RDS Proxy, credentials in Secrets Manager
    bedrock-access/     # IAM policies scoped to allowed model ARNs / inference profiles
    secrets/            # Secrets Manager entries for provider API keys (values set out of band)
    analytics/          # SQS/Firehose, S3 bucket, Glue table, Athena workgroup
    observability/      # CloudWatch dashboards, alarms, log groups + retention, AWS Budgets alerts
    auth/               # Cognito user pool for the admin dashboard
  envs/
    dev/                # small instances, scale-to-zero friendly
    demo/
```

- **Remote state** in S3 with native lockfile locking (`use_lockfile = true`; DynamoDB locking is deprecated); one state per environment.
- **GitHub Actions authenticates via OIDC** — no long-lived AWS keys in CI.
- **Secret values never live in Terraform**; it creates the Secrets Manager entries, values are set out of band.
- `terraform destroy` on `envs/dev` is the cost-control habit; AWS Budgets alerts are themselves Terraform-managed.
- CI runs `terraform fmt -check`, `validate`, `tflint` and `plan` on PRs; `apply` only from `main`.

## Roadmap

Each phase ends with something demoable.

### Phase 1 — Core path
- FastAPI service with OpenAI-compatible `/v1/chat/completions`, Bedrock adapter (Converse), SSE streaming
- API-key auth, structured logs, Docker Compose local stack with a mock LLM server
- **Done when:** the OpenAI Python SDK works against the gateway with only `base_url` changed

### Phase 2 — Reliability
- Error taxonomy, retries with jitter, Redis-shared circuit breaker
- Second adapter (Anthropic direct or a fallback Bedrock model), fallback chains
- **Done when:** a Toxiproxy-broken primary triggers automatic failover and the breaker opens

### Phase 3 — Cost control
- Token-bucket rate limits in Redis, hierarchical budgets, price table, usage ledger

### Phase 4 — Caching
- Exact-match then semantic cache, tenant-scoped, opt-in
- Labelled eval set with published precision/recall

### Phase 5 — Observability
- OpenTelemetry traces/metrics, dashboard: cost, cache hit rate, $ saved, failovers, gateway overhead
- k6 results published in this README

### Phase 6 — Differentiators
- Agent cost controls (session budgets, loop detection) first, then complexity routing and shadow traffic

### Phase 7 — Infra & polish
- Terraform modules and environments, GitHub Actions CI/CD with OIDC
- Remaining adapters (OpenAI, Gemini, OpenAI-compatible), `/v1/messages` endpoint, admin API

## Cost notes

- Never run the dev environment against real Bedrock by default — use the mock LLM server; switch to Haiku only for integration tests and demos.
- Self-hosted GPU inference is the biggest silent cost risk (~$380/month if a g4dn.xlarge is left running) — keep it a stretch goal, spun up for demo recording, then torn down.
- ECS Fargate + ALB + ElastiCache + RDS have no meaningful free tier — expect roughly $30-60/month while running; tear down `envs/dev` when not active. NAT gateways are a hidden cost; prefer VPC endpoints in dev.
- AWS Budgets alerts at $10 and $25 are provisioned before anything else.

## Request path (end to end)

Client -> ALB -> Fargate (FastAPI) -> API-key auth -> token-aware rate limit and budget check (Redis) -> exact then semantic cache (Redis) -> hit: return cached response / miss: router -> circuit breaker -> provider adapter (Bedrock, Anthropic, OpenAI, ...) with fallback -> streamed response -> usage event to the ledger, with OpenTelemetry tracing every hop and Sentry capturing exceptions and failover events.
