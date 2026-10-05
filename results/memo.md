# Day-11 go/no-go memo: the quality circuit breaker

**Decision: GO**, with one design change (input-side shift detection) and clear scope limits.

Data: 12 recorded conditions, 12,000 requests through the gateway on a laptop GPU (Qwen2.5-1.5B q8 as
provider A, Llama-3.2-1B q8 as control provider B), 4 tenants from public datasets (GSM8K, Dolly,
MBPP, BFCL). See `datacard.md`. Detectors were evaluated on spliced streams ("record once, replay
many") with held-out prompts: calibration and test never share a prompt. Full tables in
`summary.md`; figures in `figures/`.

## 1. Do passive signals catch silent faults? (RQ1) — yes, for the faults that matter

| severity → | 100% of requests faulty | 30% faulty |
|---|---|---|
| quant swap (q8 → q2_K) | 6 requests | 45 |
| output cap | 20 | 315 |
| dropped system prompt | 25 | 534 |
| context truncation | 32 | 1,248 (68% detected) |
| sampling corruption (T=1.8) | 210 | weak (18%) |
| model substitution (1.5B → 0.5B) | 103 | weak (10%) |
| throttle (latency only) | not seen (by design) | not seen |

Median detection delay in requests for the pooled conformal e-detector at its *theoretical* threshold
(c = 10,000, no tuning). Every quality fault at full severity was detected in 100% of runs.

- **False alarms: zero in 600,000 healthy test requests** for the e-detector (target: < 1 per 10,000).
- CUSUM and fixed thresholds are faster on paper, but calibrated to the same target on reference
  prompts they realise **ARL₀ ≈ 4,200–4,500 on new prompts — 2.2–2.4× the allowed false-alarm rate**,
  and 7–17% of their runs alarm before the fault even starts. The e-detector's guarantee held; theirs
  did not generalise. This is the core practical argument for the anytime-valid approach.
- The HTTP-error breaker every gateway ships today detected **none** of the faults: every faulty
  response was a 200.
- Ground truth confirms these faults hurt users: GSM8K accuracy fell from 61% to 4% (quant swap),
  3% (output cap), 18% (truncation); tool calls collapsed from 99.6% to 0.4% under the quant swap.

**H1 met:** ≥3 fault types detected within a few hundred requests (quant swap 45, output cap 315,
system prompt 534 at 30% severity) at under 1 false alarm per 10k requests.

## 2. Can the gateway tell "provider changed" from "your traffic changed"? (RQ2) — yes, with inputs

Share of runs in which provider A was blamed (correct: 0% for *none* and *traffic shift*, 100% for
*provider fault* and *both*):

| rule | none | provider fault | traffic shift | both | correct overall |
|---|---|---|---|---|---|
| pooled detector, no control | 0% | 78% | 100% | 100% | 0.70 |
| per-tenant detectors | 0% | 89% | 100% | 100% | 0.72 |
| cross-provider, outputs only (original design) | 0% | 78% | **99%** | 100% | 0.70 |
| **cross-provider + input-side shift detection** | 0% | 79% | **0%** | 77% | **0.89** |

- **The original design failed**, and the data shows why: the traffic shift was *provider-specific*.
  Changing the chat tenant's system prompt moved Qwen's outputs strongly but Llama's barely, so the
  shift never fired on both providers and was blamed on Qwen. With output signals alone, a traffic
  change that only one provider reacts to is indistinguishable from a provider fault.
- **The fix is cheap and content-free:** a traffic shift changes what tenants *send*; a provider
  fault never does. An input-side detector on the client's prompt size (measured by the gateway
  before any provider sees the request) flags shifted tenants and removes them from the provider's
  evidence. False blame under a pure traffic shift went from 99% to **0%**, and provider faults are
  still caught (100% at full severity).
- **H2 met with the modified method** (0.89 vs 0.72 for per-tenant monitoring).

## 3. Which signals catch which faults? (RQ4, `figures/ablation.png`)

- Output length + `finish_reason` carry most of the signal: alone they catch 4 of 6 quality faults reliably (98–100%) and sampling/model substitution partially (50–68%).
- Tool-call validity and repetition are what expose the quantization swap.
- Refusal/empty rates caught nothing here (small models rarely refuse); keep them for safety-layer faults.
- Latency alone is weak and **confounded by load**: the two healthy Qwen runs differ in median TTFT
  (0.26 s vs 0.12 s) only because other traffic overlapped the first. Latency stays out of the
  quality statistic and gets its own breaker.

## 4. What did not work / limits

- **Pooling across tenants did not speed up detection** (the H2 sub-hypothesis): when a fault hits
  some tenants harder (truncation, model substitution), healthy tenants' e-values (mean slightly
  below 1 with finite calibration sets) dilute the pooled statistic; per-tenant detectors were faster
  there. Next step: run both and alarm on either (split the false-alarm budget).
- **10% severity faults are mostly missed** within 3,000 requests by every method that respects the
  false-alarm target. Detecting rare faults needs more traffic or richer signals.
- Small local models stand in for frontier APIs; the Bedrock replication (faults 1–5) is ready but
  needs AWS credentials. Tenants are synthetic (public datasets); prompt pools are finite, which
  slightly overstates persistent shifts.
- Active probing at equal cost (RQ3/H3) is not tested yet.
- Separate negative result: a fixed-threshold **semantic cache** is unsafe on adversarial
  near-duplicates (98% false hits at the default threshold on PAWS; `cache_eval.md`).

## 5. Next steps (if continuing the paper track)

1. Port input-side shift detection into the online breaker (`quality.py` currently uses outputs only).
2. Combined pooled + per-tenant alarm; re-run the delay tables.
3. Bedrock replication of faults 1–5; active-probing baseline at equal dollar cost.
4. Write the workshop paper around: zero false alarms where tuned baselines exceed the budget 2.4×,
   and attribution that only works once input-side evidence is added.
