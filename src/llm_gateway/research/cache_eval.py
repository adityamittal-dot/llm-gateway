"""Semantic-cache correctness eval (README "Measured semantic cache").

A labelled set of question pairs (same intent / different intent) is embedded with the
gateway's embedding model; for each similarity threshold we report the precision of a "hit"
(would the cached answer be right?), recall (how many reusable answers we catch) and the
false-hit rate. PAWS is adversarial: its non-paraphrases share almost all words ("A beats B" vs
"B beats A"), so this is a worst case for embedding similarity.

    uv run python -m llm_gateway.research.cache_eval [--pairs 100]
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from llm_gateway.cache import Embedder
from llm_gateway.research.workloads import DATA


def load_pairs(n: int, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(DATA / "paws_test.parquet")
    pos = df[df.label == 1].sample(n // 2, random_state=seed)
    neg = df[df.label == 0].sample(n - n // 2, random_state=seed)
    return pd.concat([pos, neg]).reset_index(drop=True)


def sweep(sims: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        hit = sims >= t
        tp, fp = int((hit & (labels == 1)).sum()), int((hit & (labels == 0)).sum())
        rows.append({
            "threshold": round(float(t), 3),
            "hits": int(hit.sum()),
            "precision": tp / hit.sum() if hit.sum() else float("nan"),
            "recall": tp / (labels == 1).sum(),
            "false_hit_rate": fp / (labels == 0).sum(),  # share of different-intent pairs wrongly served
        })  # fmt: skip
    return pd.DataFrame(rows)


async def main_async(args) -> None:
    pairs = load_pairs(args.pairs)
    embedder = Embedder(args.base_url, args.model)
    try:
        a = await embedder.embed(list(pairs.sentence1))
        b = await embedder.embed(list(pairs.sentence2))
    finally:
        await embedder.aclose()
    sims = (a * b).sum(axis=1)
    labels = pairs.label.to_numpy()
    table = sweep(sims, labels, np.round(np.arange(0.80, 0.995, 0.01), 3))
    safe = table[(table.precision >= 0.95) & (table.hits > 0)]
    out = Path("results")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "cache_eval.csv", index=False)
    pairs.assign(similarity=sims).to_csv(out / "cache_eval_pairs.csv", index=False)
    print(f"{len(pairs)} PAWS pairs ({labels.sum()} same intent), embeddings: {args.model}")
    print(
        f"similarity same-intent mean={sims[labels == 1].mean():.3f}  different-intent mean={sims[labels == 0].mean():.3f}"
    )
    print(table.round(3).to_string(index=False))
    print("lowest threshold with precision >= 0.95:", safe.threshold.min() if len(safe) else "none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="nomic-embed-text")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
