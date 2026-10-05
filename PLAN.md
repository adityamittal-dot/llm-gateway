# 3-week build plan (Mon 5 Oct → Sun 25 Oct 2026)

Goal: every roadmap phase (0–7) reaches a working, demoable **MVP** in 21 days, with the research result first.

- **Research:** the quality-breaker go/no-go comes on **day 11**. The paper itself is written after the 3 weeks ([RESEARCH.md §11](RESEARCH.md#11-after-the-sprint-if-go-roughly-810-more-weeks-to-submission)).
- **Pace:** assumes roughly 4–6 focused hours a day, with Claude writing most of the code. Long model runs happen overnight.
- **Budget:** AWS credits, Claude and free-tier models only. AWS infrastructure is deployed for **one demo window in week 3** and then destroyed. Budget alerts fire every $5 (`scripts/aws_budget_alerts.sh`).

## What "all phases" means here (MVP scope)

| Phase | In the 3 weeks | Deferred to "later" |
|---|---|---|
| 0 Research | Signals, fault proxy, recordings, detectors, attribution test, go/no-go memo | Paper writing, scale-up on AWS GPUs, active-probing baseline |
| 1 Core path | OpenAI-compatible API with streaming; Ollama, OpenAI-compatible (Groq/Gemini free tiers) and Bedrock adapters; API keys; structured logs; Docker Compose | `/v1/messages`, `/v1/embeddings` |
| 2 Reliability | Error taxonomy, retries with jitter, Redis circuit breaker, fallback chains, Toxiproxy failover demo | Request hedging (dropped) |
| 3 Cost control | Redis token bucket, org/team/key budgets, price table, Postgres usage ledger (Alembic) | Downgrade-on-budget, ledger reconciliation job |
| 4 Caching | Exact cache + tenant-scoped semantic cache (Redis Stack, local embeddings), small labelled eval | CI precision/recall gate, Titan embeddings |
| 5 Observability | OpenTelemetry traces/metrics, local Grafana dashboard, k6 gateway-overhead numbers | CloudWatch dashboards, Sentry |
| 6 Differentiators | Quality breaker wired into the router (from Phase 0), cache-break diagnosis, session budgets + loop detection | Shadow traffic, guardrails, MCP governance, batch routing |
| 7 Infra & polish | Terraform: networking (public subnets, no NAT), ECR, ECS Fargate, ALB, ElastiCache, RDS; GitHub Actions CI (lint, test, build) + OIDC deploy; one live demo, then `terraform destroy` | RDS Proxy, Cognito/admin UI, SQS/Firehose/Athena analytics, CloudFront, `envs/demo` |

## Week 1 (5–11 Oct): research pipeline + core path

| Day | Date | Work | Done when |
|---|---|---|---|
| 1 ✅ | Mon 5 | Ollama (user service, GTX 1650), Qwen2.5 1.5B q8_0 + q2_K; uv/FastAPI scaffold; passthrough `/v1/chat/completions` with streaming; AWS CLI; budget-alert script | OpenAI SDK works via `base_url` (`scripts/smoke.py`); 6 tests pass |
| 2 | Tue 6 | Signal extractor (RESEARCH §6) → Parquet log per response with tenant, provider, timestamp; tenant from API key | Every response produces a signal row |
| 3 | Wed 7 | Fault proxy: faults 1–5 and 7 as config-toggled middleware; fault 6 via Ollama tag swap | Each fault visible in a manual test |
| 4 | Thu 8 | Tenant replayer (GSM8K, BFCL, Dolly, MBPP); **start overnight healthy recordings** | 500 requests per tenant recorded |
| 5 | Fri 9 | Provider adapter interface; Bedrock (Converse, Amazon Nova) and OpenAI-compatible (Groq/Gemini free tier) adapters; API-key auth; JSON logs | Same replayer runs against Bedrock and a free tier |
| 6 | Sat 10 | Docker Compose (gateway, Redis Stack, Postgres, mock LLM); **overnight faulty recordings** | `docker compose up` serves completions |
| 7 | Sun 11 | Phase 2 start: error taxonomy, retries with jitter and a retry budget; Bedrock recordings for faults 1–5 | Data card for all recordings committed |

## Week 2 (12–18 Oct): research result + reliability + cost control

| Day | Date | Work | Done when |
|---|---|---|---|
| 8 | Mon 12 | Splicer + evaluation harness (ARL₀, CADD); baselines: HTTP breaker, fixed thresholds, CUSUM | First delay-vs-false-alarm numbers |
| 9 | Tue 13 | Main detector: per-stream e-processes, pooling across tenants | Delay curve vs baselines |
| 10 | Wed 14 | Attribution experiment (tenant shift / provider fault / both) + signal ablation | Attribution matrix and ablation heatmap |
| 11 | Thu 15 | **Go/no-go memo** with the 3 plots (`results/memo.md`) | Decision recorded |
| 12 | Fri 16 | Redis circuit breaker (HTTP errors), fallback chains, Toxiproxy failover demo | Broken primary → automatic failover, breaker opens |
| 13 | Sat 17 | Token-aware rate limiting (Lua token bucket: reserve `max_tokens`, settle on actual), hierarchical budgets | Over-limit key gets 429; budget exhausted → rejected |
| 14 | Sun 18 | Postgres schema + Alembic (orgs, teams, keys, budgets, routing rules, prices, monthly-partitioned ledger); async batched ledger writes | Every request lands in the ledger with cost |

## Week 3 (19–25 Oct): caching, observability, differentiators, AWS

| Day | Date | Work | Done when |
|---|---|---|---|
| 15 | Mon 19 | Exact + semantic cache (tenant-scoped, opt-in header, SSE replay of cached hits); local `nomic-embed-text` embeddings; 100-pair labelled eval | Hit rate and false-hit rate reported |
| 16 | Tue 20 | OpenTelemetry traces/metrics, Grafana dashboard (cost, cache, failovers, breaker state); k6 overhead run | Gateway p50/p99 overhead published in README |
| 17 | Wed 21 | Quality breaker in the router (graded actions + canary recovery); cache-break diagnosis; session budgets + loop detection | Live demo: injected fault → breaker shifts traffic |
| 18 | Thu 22 | Terraform modules: bootstrap (state bucket, GitHub OIDC), networking, ECR, ECS service, ALB | `terraform plan` clean for `envs/dev` |
| 19 | Fri 23 | Terraform: ElastiCache, RDS, secrets, Bedrock IAM, log groups; GitHub Actions: lint/test/build on PR, deploy from `main` | CI green; image pushed to ECR |
| 20 | Sat 24 | **Deploy to AWS (credits)**, run the demo, record numbers and a screen capture, then `terraform destroy` | Demo recorded; AWS back to ~$0/day |
| 21 | Sun 25 | README results section, architecture diagram, clean-up, buffer for slips | Repo presentable |

## Rules for staying on schedule

- **Research comes first.** If the detectors slip, Phases 2–3 shrink, not the day-11 memo.
- **Slipping a day:** drop from the end of that week's "deferred" column, never skip tests.
- **AWS:** nothing runs on AWS before day 20 except Bedrock calls. After the demo, run `terraform destroy` the same day.
- **Daily:** commit and push at the end of each day, with a one-line status in the commit message.

## If day 11 is "no-go"

Keep the schedule. Phase 6 ships the quality breaker as a product feature with thresholds instead of guarantees, and cache-break diagnosis becomes the headline. The paper track stops.
