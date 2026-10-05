# Semantic cache eval (PAWS, adversarial)

100 balanced pairs from PAWS `labeled_final/test` (50 paraphrases, 50 non-paraphrases with almost
identical wording, e.g. swapped subject/object), embedded with local `nomic-embed-text`.
Full sweep: `cache_eval.csv`. Reproduce: `uv run python -m llm_gateway.research.cache_eval`.

| | value |
|---|---|
| mean cosine, same intent | 0.987 |
| mean cosine, different intent | 0.975 |
| false-hit rate at the default threshold 0.92 | **98%** |
| best precision (threshold 0.99) | 65% (recall 60%) |
| lowest threshold with precision ≥ 95% | **none** |

**Conclusion.** On adversarial near-duplicates, a single embedding-similarity threshold cannot
separate "same question" from "different question with the same words": a fixed-threshold semantic
cache would serve wrong answers. This matches vCache's argument that static thresholds give no
correctness guarantee. Consequences for the gateway:
- semantic caching stays **opt-in** (`x-cache: semantic`), exact caching is the safe default;
- the threshold must be set per workload from that workload's own labelled pairs, and this eval is
  the tool for it;
- a per-entry, error-bounded threshold (vCache-style) or a cross-encoder verification step is the
  natural next step before enabling semantic caching broadly.

PAWS is a worst case; typical traffic with genuinely different questions separates far better, but
that is exactly the traffic where a wrong cache hit is hardest to notice.
