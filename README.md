# LLM Gateway

A self-built, provider-agnostic gateway that routes, caches, secures and meters every LLM call an application makes — instead of apps hitting Bedrock, Anthropic or OpenAI directly. Apps keep using the OpenAI (or Anthropic) SDK they already have and only change `base_url`.

Built to demonstrate real distributed-systems infra skills (token-aware rate limiting, semantic caching with measured correctness, circuit breaking, multi-provider failover, cost/latency observability), not just "call an LLM API". The headline feature is a **quality circuit breaker**: the gateway notices when a provider starts returning worse answers with `200 OK`, works out whether the provider or the traffic changed, and fails over with a bounded false-alarm rate.

> **Status (6 Oct 2026):** days 1–17 of the [3-week plan](PLAN.md) are built, reviewed and merged
> (94 tests). The research go/no-go is **GO** ([results/memo.md](results/memo.md)). Not started yet:
> days 18–21 (Terraform, CI/CD, AWS deploy, final polish). See [What's left](#whats-left).
> Sections from "Architecture" down describe the target design; the table below says what exists today.

## What works today

| Area | Implemented | Where |
|---|---|---|
| API | OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming, tools), `GET /v1/models` | `app.py` |
| Providers | OpenAI-compatible (Ollama, vLLM, Groq, Gemini, OpenAI), Amazon Bedrock Converse (text, tools, images as data URLs) | `providers/` |
| Auth | API keys stored as SHA-256 hashes, tenant per key, `require_auth` | `config.py` |
| Reliability | Error taxonomy, retries with full jitter + retry budget, Redis-shared circuit breaker (half-open probe), fallback chains (`routes:`) | `retries.py`, `breaker.py` |
| Cost control | Token-aware rate limits (Redis Lua token buckets, key → team → org), monthly budgets, price table, per-request cost | `ratelimit.py`, `pricing.py` |
| System of record | Postgres schema + Alembic, usage ledger partitioned by month, batched off-path writes, DB prices | `db.py`, `migrations/` |
| Caching | Exact + semantic cache, tenant-scoped, opt-in (`x-cache`), SSE replay of hits | `cache.py` |
| Observability | OpenTelemetry metrics at `/metrics`, GenAI-convention traces (OTLP), Grafana dashboard, k6 overhead benchmark | `telemetry.py`, `observability/`, `loadtest/` |
| **Quality circuit breaker** | Passive quality signals per response, online conformal e-detector per provider, graded actions (alert → shift → open + canary), recovery | `signals.py`, `quality.py` |
| Agent controls | Session budgets, iteration caps and tool-call loop detection (`X-Session-Id`); cache-break diagnosis | `sessions.py`, `promptcache.py` |
| Research harness | Fault injector (7 faults), tenant workloads, recorder, detectors, attribution, ablation, report | `faults.py`, `research/` |
| Local stack | Mock LLM, Docker Compose (gateway, mock, Redis Stack, Postgres, Prometheus, Grafana), no-Docker dev stack | `mock_llm.py`, `docker-compose.yml`, `scripts/dev_stack.sh` |

## Key results

- **Quality breaker (research, [memo](results/memo.md)):** zero false alarms in 600,000 healthy test
  requests at the theoretical threshold; every injected quality fault detected at full severity (a
  quantization swap within 6 requests); tuned CUSUM/threshold baselines exceeded the false-alarm budget
  2.2–2.4× on unseen prompts; the HTTP-error breaker caught nothing. Provider-vs-traffic attribution
  needs input-side evidence: 0% false blame under traffic shifts once added (0.89 correct overall).
- **Failover demo:** Toxiproxy breaks the primary → backup serves every request, breaker opens, half-open probe closes it ([results/failover_demo.txt](results/failover_demo.txt)).
- **Quality-breaker demo:** a silent fault (HTTP 200, truncated answers) moves traffic to the backup and recovers via canaries; the HTTP breaker never moves ([results/quality_breaker_demo.txt](results/quality_breaker_demo.txt)).
- **Gateway overhead (k6):** 1.3 ms p50 / 2.1 ms p99 per request unloaded; one worker saturates near 500 req/s ([§5](#5-gateway-overhead)).
- **Semantic cache:** unsafe with a fixed threshold on adversarial near-duplicates (98% false hits on PAWS) — stays opt-in ([results/cache_eval.md](results/cache_eval.md)).

## Quickstart

```bash
uv sync
uv run pytest                                  # 94 tests; Redis/Postgres tests use a local valkey-server / embedded Postgres

# Local stack without Docker: Valkey + embedded Postgres + mock LLM + gateway (key sk-dev-local, model mock-small)
scripts/dev_stack.sh
# With Docker: docker compose up --build   (adds Prometheus :9090 and Grafana :3000)

# Demos (no GPU needed)
uv run python scripts/failover_demo.py         # needs toxiproxy-server on PATH
uv run python scripts/quality_breaker_demo.py
uv run python scripts/overhead.py              # needs k6 on PATH

# Real local models: Ollama on http://localhost:11434
ollama pull qwen2.5:1.5b-instruct-q8_0
uv run llm-gateway                             # http://127.0.0.1:8000, config via GATEWAY_CONFIG (see config/gateway.example.yaml)
uv run python scripts/smoke.py                 # stock OpenAI SDK, only base_url changed
```

Research pipeline: `scripts/get_datasets.sh` → `python -m llm_gateway.research.record` (≈3.5 h on a GTX 1650)
→ `python -m llm_gateway.research.experiments {baselines,detectors,attribution,ablation}` → `python -m llm_gateway.research.report`.

## What's left

| Item | Status |
|---|---|
| Day 18–19: Terraform (networking, ECR, ECS, ALB, ElastiCache, RDS, IAM) + GitHub Actions CI/CD with OIDC | not started |
| Day 20: deploy the demo to AWS on credits, then `terraform destroy` | not started; needs your AWS credentials |
| Day 21: README results polish, architecture diagram | not started |
| Port input-side shift detection into the online breaker (`quality.py` uses output signals only) | open — the memo shows it is required for correct attribution |
| Combined pooled + per-tenant detector; Bedrock replication of faults 1–5; active-probing baseline | research follow-ups |
| Deferred from the 3-week scope | `/v1/messages`, `/v1/embeddings`, Anthropic/Azure/Gemini-native adapters, admin UI + Cognito, SQS/Athena analytics, RDS Proxy, shadow traffic, guardrails, MCP governance |

**Needs you:** `aws configure` → `scripts/aws_budget_alerts.sh <email>` (confirm the subscription emails);
Bedrock model access for Amazon Nova; optional `GROQ_API_KEY` / `GEMINI_API_KEY`; optional Docker
access (`sudo usermod -aG docker $USER`).

## Problem statement

Every team adding an LLM feature eventually hits the same problems:

- **Cost is invisible until it isn't.** No visibility into cost-per-request, per-tenant or per-agent-run until the bill spikes.
- **Vendor lock-in / outage risk.** A single provider dependency means a single point of failure.
- **No caching for non-deterministic responses.** Exact-match caching barely works for natural language; most teams cache nothing and re-pay for near-identical prompts.
- **No per-tenant limits.** One noisy client — or one agent stuck in a loop — can starve everyone else or blow the budget alone.
- **Silent degradation.** Providers ship quantization, kernel, routing and sampling changes without a version bump. Responses still return `200 OK`, so uptime checks and HTTP-error circuit breakers see nothing while answer quality, tool-call validity or refusal rates move.

This project solves these by building the gateway layer that should sit between any app and any LLM provider.

## Positioning

This is a crowded category, and most of the obvious features already exist:

| Already shipped by others | Where |
|---|---|
| Unified OpenAI-compatible API, fallbacks, virtual keys, spend logs | LiteLLM, Portkey, Bifrost, Kong, Envoy AI Gateway, Cloudflare/Vercel AI Gateway, OpenRouter, Helicone |
| Per-session agent budgets and iteration caps | LiteLLM (`max_budget_per_session`), agentgateway (hierarchical budgets with reserve-then-settle) |
| Semantic caching | Portkey, Bifrost, Kong, Zuplo, TrueFoundry; with error guarantees in research (vCache) |
| Auto-injected prompt-cache breakpoints | LiteLLM, Bifrost |
| Complexity / cost routing | Not Diamond, Martian, several gateways |

So this gateway does not compete on breadth. It covers those basics well enough to be usable and puts its depth into one thing none of them do. Their circuit breakers trip on `429`/`5xx`/timeouts. This one also trips on **statistical evidence that a provider's output quality changed**, measured passively from production traffic. Because it sees many tenants and many providers, it can tell *"the provider changed"* apart from *"your traffic changed"*. A single app's own monitoring cannot separate the two.

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
  +-- Router (fallback chains, cost/latency rules)
  |     |
  |     +-- Circuit breaker (HTTP errors + quality signals, state shared in Redis)
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
| `routing_rules` | Fallback chains, model aliases, shadow-traffic and quality-breaker config |
| `model_prices` | Versioned price per model/region (input, output, cached tokens), effective-from date |
| `usage_ledger` | One row per request: tenant, key, session, model, provider, tokens, cost, latency, cache status, failover flag, quality signals (finish reason, tool-call validity, refusal). **Partitioned by month** so old data can be detached and archived to S3 cheaply |

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
- **Result so far:** on 100 adversarial PAWS pairs no fixed threshold reaches 95% precision (98% false hits at 0.92) — see [results/cache_eval.md](results/cache_eval.md). Implemented with brute-force cosine over capped buckets in plain Redis rather than a vector index.
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
- **Measured** (`scripts/overhead.py`: k6 against an instant mock LLM directly vs through the gateway; one uvicorn worker on a laptop CPU, auth + routing + limits + metrics on, 30 s per run):

  | load | p50 overhead | p90 | p99 | gateway throughput |
  |---|---|---|---|---|
  | 1 concurrent client | **1.3 ms** | 1.7 ms | 2.1 ms | 579 req/s |
  | 10 concurrent clients | 15.4 ms | 37.1 ms | 71.6 ms | 473 req/s (CPU-bound single worker; the extra is queueing) |

  Per-request cost is ~1–2 ms; under load one Python worker saturates near 500 req/s, so capacity comes from more workers/tasks (results in `results/overhead_*.json`).

### 6. Quality circuit breaker (the differentiator)
- **Passive signals only, no extra model calls.** Every response already carries cheap quality proxies:
  - tool-call JSON/schema validity
  - `finish_reason` mix (length cut-offs, content filters)
  - refusal and empty-response rate
  - output-length distribution
  - client regenerate/retry rate (same session, same prompt prefix)
  - agent loop rate
  - TTFT and tokens/sec fingerprint
  - prompt-cache read ratio
- **Per (provider, model, region) stream**, so a degraded Bedrock region doesn't condemn the same model on Anthropic direct.
- **Attribution: "provider changed" vs "traffic changed".**
  - If one tenant's metrics move on every provider, its traffic changed.
  - If many tenants' metrics move on one provider at once, the provider changed.
  - This is a difference-in-differences test across tenants and providers. A single-app monitor cannot run it.
- **Anytime-valid sequential tests** (e-processes / CUSUM) so the dashboard can be checked continuously without inflating false alarms. The false-failover rate is a configured bound, not a hope.
- **Actions are graded:** alert → shift a share of traffic to the fallback → full open. Recovery is probed with a small canary share.
- **Privacy:** only aggregate per-response metadata is pooled across tenants, never prompt or response content.

## Differentiating features

The basics (unified API, fallback, keys, logs, session budgets) are table stakes (see [Positioning](#positioning)). They are built here because the gateway needs them, not because they are new.

| Feature | What it does | New? |
|---|---|---|
| **Quality circuit breaker** | Detects silent provider degradation from passive traffic signals, attributes it to provider vs traffic, fails over with a bounded false-alarm rate | **Yes:** the research track below |
| **Cache-break diagnosis** | Hashes each prompt block; when a key's prompt-cache read ratio drops, names the block that changed (timestamp in the system prompt, reordered tools, a new tool) and the deploy that introduced it | Mostly new; gateways report hit rate but not *why* it fell |
| **Measured semantic cache** | Hit rate, estimated false-hit rate and dollars saved, backed by a labelled eval set | Partly; error-bounded caching exists in research |
| **Agent cost controls** | Budgets per session (`x-session-id`), loop detection from repeated tool calls, auto-downgrade near budget | No (LiteLLM, agentgateway); built as table stakes |
| **Prompt-caching passthrough** | Anthropic `cache_control`, Bedrock `cachePoint`, OpenAI breakpoints; reports savings | No |
| **Shadow traffic / model A/B** | Mirror X% of traffic to a candidate model and compare cost, latency and quality | No |
| **Cost repricing** | "Last month's tokens at model Y's prices": an estimate for single-turn traffic only. It is *not* valid for agents: a different model takes different steps (see "The Replay Gap", arXiv 2608.08239) | No |
| **Guardrails, tool/MCP allow-lists, batch routing** | Standard features, later phases | No |

Dropped from the earlier plan: **complexity routing** (Not Diamond, Martian and many papers cover it) and **request hedging** (it doubles spend, which conflicts with the cost goals).

## Research track

The quality circuit breaker is designed so it can also be written up as a paper.

**Question.** Using only passive, content-free metadata from production traffic, how quickly can a multi-tenant gateway detect a silent provider-side degradation? Can it attribute that degradation to the provider rather than to a shift in traffic, at a user-chosen false-alarm rate?

**Why it is open (as of Oct 2026):**
- **Endpoint monitoring uses active probes.** Fixed prompt sets are sampled on a schedule ("Behavioral Fingerprints for LLM Endpoint Stability", arXiv 2603.19022; "Log Probability Tracking of LLM APIs", arXiv 2512.03816). That costs money, detects model *identity* change rather than impact on your traffic, and is not wired into failover.
- **Agent failure monitors work per episode, not per provider.** "Real-Time Detection and Repair of LLM Agent Failures" (arXiv 2608.02464) also reports that monitors need a per-deployment healthy baseline, which this design gets for free from the fleet.
- **Gateway circuit breakers trip only on HTTP errors.**
- **Nobody uses the gateway's position to separate provider drift from traffic drift.**

**Planned evaluation:**
- **Fault injection on open models served with vLLM behind the gateway.** Faults are modelled on real incident classes:
  - quantization swap
  - top-k/top-p sampling bug
  - context truncation / long-context misrouting
  - dropped system prompt
  - tokenizer/template mismatch
  - throttled decode
- **Traffic:** public chat and agent datasets, split into synthetic "tenants".
- **Metrics:**
  - detection delay at fixed false-alarm rates
  - attribution accuracy when a traffic shift and a provider fault happen together
  - user-visible bad responses avoided by failover
  - comparison with active probing at equal dollar cost

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

Status legend: ✅ done · 🟡 partly done · ⬜ not started. Details per day in [PLAN.md](PLAN.md).

### Phase 0 — Research sprint (days 1–11) ✅
- Local-only: a minimal proxy, per-response quality signals, a fault proxy, and detectors. Uses local Ollama models, Amazon Bedrock on AWS credits, and free-tier APIs; no always-on AWS infrastructure.
- **Done when:** the day-11 go/no-go memo exists. Research details and publishing route: [RESEARCH.md](RESEARCH.md)
- **All phases run on a 3-week schedule (5–25 Oct 2026), each at MVP depth:** see [PLAN.md](PLAN.md) for the day-by-day plan and what is deferred.

### Phase 1 — Core path ✅
- FastAPI service with OpenAI-compatible `/v1/chat/completions`, Bedrock adapter (Converse), SSE streaming
- API-key auth, structured logs, Docker Compose local stack with a mock LLM server
- **Done when:** the OpenAI Python SDK works against the gateway with only `base_url` changed

### Phase 2 — Reliability ✅
- Error taxonomy, retries with jitter, Redis-shared circuit breaker
- Second adapter (Anthropic direct or a fallback Bedrock model), fallback chains
- **Done when:** a Toxiproxy-broken primary triggers automatic failover and the breaker opens

### Phase 3 — Cost control ✅
- Token-bucket rate limits in Redis, hierarchical budgets, price table, usage ledger

### Phase 4 — Caching ✅ (eval published; result: fixed thresholds unsafe on adversarial pairs)
- Exact-match then semantic cache, tenant-scoped, opt-in
- Labelled eval set with published precision/recall

### Phase 5 — Observability ✅
- OpenTelemetry traces/metrics, dashboard: cost, cache hit rate, $ saved, failovers, gateway overhead
- k6 results published in this README

### Phase 6 — Quality circuit breaker and research 🟡 (breaker live; input-side attribution not yet online; shadow traffic deferred; paper not started)
- Per-response quality signals recorded in the ledger (from Phase 3 onward, so baselines exist early)
- Sequential detectors per (provider, model, region), cross-tenant attribution test, graded breaker actions
- vLLM fault-injection harness and evaluation; paper draft
- Cache-break diagnosis, agent session budgets and loop detection, shadow traffic
- **Done when:** an injected quantization swap on one provider is detected and failed over within a published detection delay, at the configured false-alarm rate, while a simultaneous tenant traffic shift is *not* blamed on the provider

### Phase 7 — Infra & polish ⬜
- Terraform modules and environments, GitHub Actions CI/CD with OIDC
- Remaining adapters (OpenAI, Gemini, OpenAI-compatible), `/v1/messages` endpoint, admin API

## Cost notes

- Until Phase 0 shows a solid result, spending is limited to AWS credits, Claude and free-tier models (see [RESEARCH.md §9](RESEARCH.md#9-constraints-and-resources-phase-0-before-anything-solid)).

- Never run the dev environment against real Bedrock by default — use the mock LLM server; switch to Haiku only for integration tests and demos.
- Self-hosted GPU inference is the biggest silent cost risk (~$380/month if a g4dn.xlarge is left running) — keep it a stretch goal, spun up for demo recording, then torn down.
- ECS Fargate + ALB + ElastiCache + RDS have no meaningful free tier — expect roughly $30-60/month while running; tear down `envs/dev` when not active. NAT gateways are a hidden cost; prefer VPC endpoints in dev.
- AWS Budgets alerts at every $5 of spend (credits excluded) are provisioned before anything else: `scripts/aws_budget_alerts.sh`.

## Request path (end to end)

Client -> ALB -> Fargate (FastAPI) -> API-key auth -> token-aware rate limit and budget check (Redis) -> exact then semantic cache (Redis) -> hit: return cached response / miss: router -> circuit breaker -> provider adapter (Bedrock, Anthropic, OpenAI, ...) with fallback -> streamed response -> quality signals feed the quality breaker -> usage event to the ledger, with OpenTelemetry tracing every hop and Sentry capturing exceptions and failover events.

_Last updated: 2026-10-10._
