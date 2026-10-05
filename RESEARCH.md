# Research plan: the quality circuit breaker

Working title: **"Silent Failures Return 200: Detecting and Attributing Provider Degradation at the LLM Gateway"**

This file holds the research topic, the plan, the research sprint, and the route to publication. The README describes the full gateway product. This file covers only the part that becomes a paper.

---

## 1. The idea in one paragraph

Hosted LLM providers sometimes get worse silently. A quantization change, a sampling bug, a context-routing error or a dropped system prompt still returns `200 OK`. Every LLM gateway on the market trips its circuit breaker only on HTTP errors and timeouts, so none of them notice. Apps find out from user complaints, hours later.

This project detects such degradations **passively**, from metadata the gateway already sees on every response. It needs no extra model calls and does not read prompt or response content. The gateway sits between many tenants and many providers, so it can also **attribute** a change: *the provider changed* or *this tenant's traffic changed*. A single app monitoring itself cannot separate the two. Detection uses anytime-valid sequential tests, so the false-alarm rate is a configured bound. When there is enough evidence, the gateway shifts traffic to a fallback provider.

## 2. Motivation

- **Real incidents.** Anthropic published a postmortem (Sept 2025) of three infrastructure bugs that degraded Claude's output quality for weeks: a context-window routing error, output corruption from a misconfiguration, and an approximate top-k miscompilation. Requests kept succeeding. Similar silent changes (quantization, inference-engine and kernel updates) are documented across hosted endpoints (see 2603.19022 below).
- **Startup pain.** Small teams have no ML-ops staff watching output quality. A gateway feature that says *"provider X degraded at 14:05; 30% of your traffic moved to Y"* sells directly.
- **Gap in products.** LiteLLM, Portkey, Bifrost, Kong, agentgateway and the rest fail over on `429`/`5xx`/timeouts only.

## 3. Related work and why this is still open (checked Oct 2026)

| Work | What it does | Gap this project fills |
|---|---|---|
| Behavioral Fingerprints for LLM Endpoint Stability (arXiv 2603.19022) | Active probing: fixed prompt set sampled on a schedule, energy-distance tests | Costs money per probe; detects model *identity* change rather than the effect on your traffic; no failover |
| Log Probability Tracking of LLM APIs (arXiv 2512.03816) | Cheap active audits via logprobs | Active, needs logprobs (many APIs don't expose them), no attribution or failover |
| Real-Time Detection and Repair of LLM Agent Failures (arXiv 2608.02464) | Per-episode monitors from step telemetry | Per episode, not per provider; reports that monitors need a per-deployment healthy baseline, which a fleet provides for free |
| Who Drifted: the System or the Judge? (arXiv 2606.15474) | Anytime-valid attribution between system and LLM judge | Attribution idea is related, but the setting differs (eval pipelines, not provider routing) |
| FailureAtlas (arXiv 2607.17525), ContinuityBench (arXiv 2607.15899) | Gateway failure taxonomy; stateful failover | Neither detects silent quality loss |
| The Replay Gap (arXiv 2608.08239) | Shows log replay mis-scores agent model switches | Methodology warning: evaluate with live traffic, not stitched logs |
| Gateways (LiteLLM, Portkey, Bifrost, Kong, agentgateway) | HTTP-error circuit breakers | No quality signal |

**The claim of novelty:**
1. **Passive** (no probe cost), **content-free** signals.
2. **Cross-tenant, cross-provider attribution**: provider drift vs traffic drift.
3. **Anytime-valid** false-alarm control, wired into **failover**.

Re-run the literature search on arXiv and Semantic Scholar before writing (section 11). New papers appear weekly.

## 4. Research questions and hypotheses

- **RQ1:** How fast can passive, content-free signals detect common provider-side faults at a fixed false-alarm rate?
  - **H1:** At least 3 of the 6 fault types in section 7 are detected within a few hundred requests at under 1 false alarm per 10k healthy requests.
- **RQ2:** Can pooling across tenants, compared against a control provider, separate provider faults from tenant traffic shifts?
  - **H2:** With 4 or more tenants, attribution accuracy is clearly above per-tenant monitoring.
- **RQ3:** How does passive detection compare with active probing at equal dollar cost?
  - **H3:** Passive detection is faster for faults that affect real traffic (e.g. tool-call breakage) and costs nothing extra.
- **RQ4:** Which signals catch which faults? (ablation)

## 5. Formalization (draft)

- **Streams.** For provider *p* (a provider/model/region triple) and tenant *k*, each response yields a signal vector *x* (section 6).
- **Null H0.** For every tenant *k*, the distribution of *x* on *p* is unchanged.
- **Provider fault at unknown τ.** The distribution changes on *p* for all (or most) tenants from τ onward.
- **Traffic shift (the confound).** Tenant *k*'s input distribution changes, so *x* moves on *every* provider that tenant uses.
- **Detector.**
  1. A per-(p, k) sequential statistic: CUSUM as the simple baseline, an e-process via testing-by-betting as the main method.
  2. Combine evidence across tenants on *p*, e.g. by averaging e-values.
  3. Run a difference test against a control provider *q* that the same tenants also use.
  4. Raise a provider alarm only when evidence is provider-wide **and** absent on the control.
- **Guarantee.** By Ville's inequality, P(any false alarm ever) ≤ α under H0.
- **Metrics.** Mean time to false alarm (ARL₀), and conditional average detection delay (CADD, in requests and in minutes).

## 6. Passive signals (all available per response, no extra model calls)

| Signal | Catches |
|---|---|
| Tool-call validity (parses as JSON, matches the declared schema) | Template/format bugs, quantization damage |
| `finish_reason` mix (`stop`/`length`/`content_filter`) | Truncation, runaway generation |
| Output length distribution | Sampling bugs, truncation, degenerate loops |
| Empty or refusal response rate | Safety-layer or template changes |
| Repetition score (repeated n-grams in the output, computed locally) | Sampling bugs, quantization |
| Client regenerate rate (same session and prompt prefix re-sent) | User-perceived quality drop |
| TTFT and tokens/sec fingerprint | Hardware, quantization or engine changes |
| Prompt-cache read ratio (where reported) | Routing changes |

**Content-free rule:** signals are computed on the fly. Only the numbers are stored and pooled across tenants, never text.

## 7. Fault catalog (what we inject)

Faults are injected by a small **fault proxy** that sits between the gateway and a provider. Faults 1–5 work on any provider, real API or local. Fault 6 needs a model we control.

| # | Fault | How it is injected | Real-world analogue |
|---|---|---|---|
| 1 | Dropped system prompt | Proxy removes the system message | Template/routing bug |
| 2 | Context truncation | Proxy keeps only the last N tokens of the conversation | Context-window misrouting |
| 3 | Sampling corruption | Proxy overrides temperature/top-p (e.g. temperature 1.8) | Top-k/top-p miscompilation |
| 4 | Output cap | Proxy lowers `max_tokens` | Silent limit change |
| 5 | Model substitution | Proxy sends the request to a smaller model under the same name | Silent downgrade |
| 6 | Quantization swap | Local Ollama model switched from `q8_0` to `q2_K`/`q3_K` | Quantization change |
| 7 | Throttling | Proxy adds delay per token | Capacity change (latency-only; checks the detector isn't fooled) |

Each fault comes in 2–3 severities (e.g. probability of applying the fault: 10%, 30%, 100%).

## 8. Workloads ("tenants")

Each tenant is a public dataset with a distinct traffic shape. Check every licence before use.

| Tenant | Dataset (candidates) | Why |
|---|---|---|
| Math | GSM8K | Ground-truth answers, so the true quality drop is measurable |
| Tool use | BFCL (Berkeley Function Calling Leaderboard) | Tool-call validity |
| Chat | Dolly-15k or WildChat | General traffic, length and refusal signals |
| Code | HumanEval or MBPP prompts | Distinct length and format profile |

A **traffic shift** is simulated by changing one tenant's prompt template or mix partway through a run.

## 9. Constraints and resources (phase 0: before anything "solid")

Spending is limited to **AWS credits + Claude + free-tier models** until results are promising.

| Resource | Used for | Cost |
|---|---|---|
| **Local Ollama on the laptop GPU** (GTX 1650, 4 GB) | Main provider for experiments. Small models (Qwen2.5 0.5B/1.5B, Llama 3.2 1B) in several quantizations; the only place fault 6 is possible | Free |
| **Amazon Bedrock (AWS credits)** | A real hosted provider for realism: Amazon Nova Micro/Lite or another cheap model | A few dollars at this scale. Check that your credits cover the model you pick: some credit types exclude Marketplace-billed third-party models, so first-party Amazon Nova is the safest choice |
| **Free-tier APIs** (Gemini free tier, Groq free tier, OpenRouter free models) | A third provider later, the LLM-judge baseline, and a small passive-monitoring study | Free, but rate-limited; free-tier prompts may be used for training, so send only public datasets |
| **Claude (Claude Code)** | Writing code, the detector math, and drafts of the paper | Existing subscription; not used as an experiment provider |
| Docker, Python + uv, SQLite/Parquet | Local stack | Free |

**Not used in phase 0:** ECS, RDS, ElastiCache, ALB, NAT or any always-on AWS infrastructure. The full AWS deployment from the README comes after the research result.

**AWS guardrails (day 1):** AWS Budgets alerts at $5 and $20, and Bedrock model access enabled only for the one model in use.

### The key trick: record once, replay many

Generating responses is the expensive part, so each (provider, fault, severity, tenant) combination is generated **once** and logged with its signals. Detector experiments then **splice** recorded streams offline: healthy responses up to a random τ, faulty ones after. This gives thousands of trials at zero model cost.

This is valid here (unlike the agent-routing case in the Replay Gap) because each request is independent: the fault doesn't change which requests arrive next. Some live end-to-end runs confirm the replay results.

## 10. The research sprint (days 1–11 of the 3-week plan)

The research work is part of the 3-week build plan in [PLAN.md](PLAN.md):

| Days | Research work |
|---|---|
| 1 ✅ | Ollama + quantized models, passthrough gateway (`src/llm_gateway`), smoke test |
| 2–3 | Signal extractor; fault proxy (faults 1–7) |
| 4–7 | Tenant replayer; healthy and faulty recordings (overnight on the local GPU), then Bedrock recordings for faults 1–5 |
| 8–10 | Splicer + evaluation harness; baselines; e-process detector with pooling; attribution experiment; signal ablation |
| 11 | Go/no-go memo with 3 plots in `results/memo.md` |

### Go/no-go criteria (day 11)

- **Go** if at least 3 fault types are detected well before 1,000 requests at under 1 false alarm per 10k healthy requests, **and** the attribution test beats per-tenant monitoring. Then continue with section 11.
- **Partial** if detection works but attribution doesn't. A detection-only paper is still a workshop paper; attribution becomes future work.
- **No-go** if signals don't separate faulty from healthy traffic. Ship the breaker and cache-break diagnosis as product features, and drop the paper track.

## 11. After the sprint (if "go"): roughly 8–10 more weeks to submission

1. **Literature re-check** (arXiv, Semantic Scholar, Connected Papers) and a related-work draft.
2. **Scale up experiments.** Spend AWS credits on one GPU instance only when the local GPU becomes the bottleneck: one `g5`/`g6`-class spot instance running vLLM, for a few hours at a time, started and stopped by script. GPU instance quotas often start at 0 on new accounts, so request a quota increase early; it can take days.
3. **Active-probing baseline at equal cost**, and an LLM-judge baseline via a free-tier API.
4. **Optional real-world study:** passive monitoring of real providers through the gateway for a few weeks, with anomalies lined up against public status-page incidents.
5. **Write the paper** (section 12 outline). Get feedback from a mentor or coauthor (see section 13).
6. **Release** code, recordings and the fault proxy (GitHub + Zenodo DOI).

## 12. Paper outline

1. **Introduction:** incident story → gap (HTTP-only breakers, costly active probes) → contributions (passive signals; cross-tenant attribution with anytime-valid control; breaker system + fault benchmark).
2. **Background and related work:** section 3.
3. **Problem formulation:** section 5.
4. **Method:** signals, per-stream e-processes, pooling, attribution test, breaker policy.
5. **Implementation:** gateway, fault proxy, overhead per request.
6. **Evaluation:** RQ1–RQ4; baselines; live vs replay agreement.
7. **Limitations:** small open models as stand-ins for frontier APIs; synthetic tenants; metadata misses subtle reasoning-quality loss; the attacker/adversarial case is out of scope.
8. **Conclusion.**

**Figure 1** (sketch it first): over time, a fault starts; the HTTP breaker stays green, the quality breaker fires, and traffic shifts.

## 13. Publishing route and costs

1. **Preprint on arXiv.** Free.
   - Category `cs.DC` or `cs.LG`.
   - First-time submitters usually need an **endorsement**: a coauthor or mentor already published in that category.
   - Check the target venue's preprint/anonymity policy first.
2. **Workshop first** (4–6 pages): ML-systems or reliability workshops at NeurIPS, ICML or ICLR, or EuroMLSys. Submission is free at reputable CS venues, and review is usually double-blind (anonymize the repo link). Prefer **non-archival** workshops so the work can grow into a conference paper.
3. **Conference later** (extended version): MLSys, SoCC, Middleware, USENIX ATC. Check current deadlines.
4. **Find a mentor early:** email one or two researchers in ML systems or sequential testing a one-page summary plus the day-11 plots.

| Item | Cost |
|---|---|
| arXiv, Overleaf, Zotero, GitHub, Zenodo, OpenReview | Free |
| Submission | Free |
| Phase 0 experiments (local GPU + Bedrock credits + free tiers) | ~$0 out of pocket |
| Scale-up GPU time (AWS credits) | Covered by credits if they apply to EC2 |
| Registration, only if accepted | ~$100–1,000+; virtual and student rates are lower |
| Travel, if presenting in person | Varies; travel grants and student-volunteer programs exist |

**Avoid:** venues that charge to publish quickly or email unsolicited invitations (predatory), paid "publication assistance", and open-access journal fees. None of them are needed in CS.

## 14. Risks

| Risk | Mitigation |
|---|---|
| Small local models behave unlike frontier APIs | Repeat the core result on Bedrock with faults 1–5; state the limitation |
| Signals too noisy at low traffic | Report the minimum traffic needed; pooling across tenants is the point |
| Someone publishes the same idea first | Re-check literature at day 11 and before writing; lean on attribution, which is the hardest part to copy |
| Free-tier rate limits | Use free tiers only for baselines and small studies; main data comes from the local GPU |
| AWS credits don't cover a chosen model or instance | Check the credit terms first; default to Amazon Nova on Bedrock and spot instances |
| Scope creep into the full gateway | Phase 0 builds only proxy + signals + fault proxy + detectors. The AWS stack waits |

## 15. References (starting set)

- vCache: Verified Semantic Prompt Caching, arXiv 2502.03771
- Behavioral Fingerprints for LLM Endpoint Stability and Identity, arXiv 2603.19022
- Log Probability Tracking of LLM APIs, arXiv 2512.03816
- Real-Time Detection and Repair of LLM Agent Failures, arXiv 2608.02464
- Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines, arXiv 2606.15474
- FailureAtlas: A Taxonomy of Failure Modes in Multi-Provider LLM Serving Infrastructure, arXiv 2607.17525
- ContinuityBench: Stateful Failover in Multi-Provider LLM Routing, arXiv 2607.15899
- The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the Wrong World, arXiv 2608.08239
- Sequential statistical inference for LLMs: representation, validity, and monitoring, arXiv 2606.07624
- Anthropic, "A postmortem of three recent issues" (Sept 2025)
- Background on anytime-valid inference: testing by betting / e-processes (Ramdas, Grünwald, Vovk, Shafer et al.); CUSUM (Page, 1954)
