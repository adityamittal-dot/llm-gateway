# LLM Gateway

A self-built AWS-based gateway that routes, caches, and secures every LLM call an application makes — instead of hitting Bedrock/OpenAI directly. Built to demonstrate real distributed-systems infra skills (rate limiting, semantic caching, circuit breaking, multi-backend failover, cost/latency observability), not just "call an LLM API."

## Problem statement

Every team adding an LLM feature eventually hits the same problems:

- **Cost is invisible until it isn't.** No visibility into cost-per-request until the bill spikes.
- **Vendor lock-in / outage risk.** A single provider dependency means a single point of failure.
- **No caching for non-deterministic responses.** Exact-match caching doesn't work for natural language; most teams cache nothing and re-pay for near-identical prompts.
- **No per-tenant rate limiting.** One noisy client can starve everyone else or blow the budget alone.

This project solves all four by building the gateway layer that should sit between any app and any LLM provider.

## Architecture

```
Client
  |
  v
API Gateway (HTTP API) + Cognito authorizer   <- auth, per-key rate limiting
  |
  v
ALB
  |
  v
ECS Fargate task
  |
  +-- nginx (sidecar, public-facing)
  |     |
  |     v
  +-- Uvicorn -> Django (ASGI, async views)
        |
        v
      Redis semantic cache check (ElastiCache, RediSearch)
        |
        +-- HIT  -> return cached response
        |
        +-- MISS -> Circuit breaker
                      |
                      +-- Bedrock (primary: Claude Haiku/Sonnet)
                      |
                      +-- Fallback (secondary Bedrock model,
                          stretch: self-hosted EC2 GPU model)
```

Every hop is instrumented:
- **X-Ray** traces the full chain (ALB -> Fargate -> Redis -> Bedrock)
- **CloudWatch custom metrics** capture cost-per-token, cache hit rate, failover count
- **Sentry** captures exceptions and custom circuit-breaker-trip events

## What we're solving (the real engineering problems)

1. **LLM cost as the new cloud cost problem** — semantic caching gives a measurable, defensible cost reduction story ("cut redundant model calls by X%").
2. **Vendor lock-in and outages** — multi-backend routing with circuit breaker logic means one provider's outage doesn't take the product down.
3. **Rate limiting under multi-tenancy** — per-API-key sliding-window quotas via DynamoDB TTL counters.
4. **Cache correctness for non-deterministic systems** — deciding what "close enough" means for two prompts to share a cached answer, and what happens when that's wrong.

## Tech stack

### Compute & routing
| Layer | Choice |
|---|---|
| Container runtime | ECS Fargate |
| In-task reverse proxy | nginx (sidecar, proxies to Uvicorn over Unix socket) |
| App server | Uvicorn (ASGI) |
| Framework | Django (ASGI, async views) + Django REST Framework |
| Load balancer | Application Load Balancer (ALB) |
| API ingress / auth | API Gateway (HTTP API) + Cognito authorizer |

### Caching
| Layer | Choice |
|---|---|
| Cache store | ElastiCache for Redis (RediSearch for vector similarity) |
| Redis client | `redis-py` async interface |
| Embedding model | Amazon Titan Embeddings via Bedrock |

### Model backends
| Layer | Choice |
|---|---|
| Primary | Bedrock — Claude Haiku (dev) / Sonnet (demo) |
| Fallback | Second Bedrock model (stretch: self-hosted EC2 GPU, demo-only) |
| Bedrock client | `aioboto3` |

### State & data
| Layer | Choice |
|---|---|
| Request/rate-limit state | DynamoDB (TTL-based sliding window counters) |
| ORM (optional) | Django ORM + Postgres, only if admin/migrations for request logs are wanted |

### Observability
| Layer | Choice |
|---|---|
| Metrics/dashboards | CloudWatch custom metrics + Dashboards |
| Distributed tracing | AWS X-Ray |
| Logs | CloudWatch Logs (structured JSON, Logs Insights) |
| Error tracking | Sentry (Django/ASGI integration) |
| Alerting | Sentry issue alerts + CloudWatch Alarms |

### Infra as code
| Layer | Choice |
|---|---|
| IaC | Terraform, modular (`networking`, `ecs-service`, `cache`, `api-gateway`, `observability`) |
| State | Remote state in S3 + DynamoDB lock table |

### Local dev / testing
| Layer | Choice |
|---|---|
| Local Redis | Docker Compose |
| Load testing | k6 or Artillery (prove circuit breaker trips under failure) |

## Roadmap

### MVP (week 1-2)
- API Gateway with Cognito auth + basic per-key rate limiting
- Single route straight to Bedrock, no cache/router yet
- CloudWatch logging of latency and token counts

### Core build (week 3-4)
- Redis semantic cache (embed prompt, cosine-similarity lookup)
- Circuit breaker around the Bedrock call (fail over after N consecutive failures)
- Second backend for failover proof (start with a cheaper Bedrock model, not GPU EC2)

### Polish for resume/demo (week 5+)
- Dashboard: cache hit rate over time, cost saved by caching, failover events triggered
- Load test proving the circuit breaker trips under simulated failure
- Terraform module set, with README arguing the design tradeoffs
- Sentry wired to capture circuit-breaker-trip events as custom tagged messages

## Cost notes

- Avoid OpenSearch Serverless-style always-on minimums where possible.
- SageMaker/self-hosted GPU inference is the biggest silent cost risk (~$380/month if left running on g4dn.xlarge) — keep the fallback backend as a second Bedrock model for the MVP, add GPU-hosted fallback only as a stretch goal spun up for demo recording, then torn down.
- ECS Fargate + ALB + ElastiCache have no meaningful free tier — expect roughly $15-30/month if development habits (tear down when not active) are followed.
- Set AWS Budgets alerts at $10 and $25 before provisioning anything.

## Request path (end to end)

Client -> API Gateway (Cognito auth, rate limit) -> ALB -> ECS Fargate task -> nginx -> Uvicorn -> Django (ASGI) -> Redis semantic cache check -> hit: return cached response / miss: circuit breaker -> Bedrock primary or fallback -> response, with X-Ray tracing the full hop chain, CloudWatch capturing cost/latency metrics, and Sentry capturing exceptions and failover events.
