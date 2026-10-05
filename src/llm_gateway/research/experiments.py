"""Research experiments (RESEARCH.md §10), run on the recorded conditions.

    uv run python -m llm_gateway.research.experiments baselines   # day 8
    uv run python -m llm_gateway.research.experiments detectors   # day 9
    uv run python -m llm_gateway.research.experiments attribution # day 10
    uv run python -m llm_gateway.research.experiments ablation    # day 10

Results are written to results/*.csv. All randomness is seeded.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from llm_gateway.research import harness as H

RESULTS = Path("results")
PROVIDER, HEALTHY, HEALTHY_TEST = "ollama-qwen", "qwen_healthy", "qwen_healthy_2"
FAULTS = [
    "drop_system",
    "truncate_context",
    "sampling",
    "output_cap",
    "quant_swap",
    "model_substitution",
    "throttle",
]
SEVERITIES = [0.1, 0.3, 1.0]
TARGET_ARL = 10_000  # "under 1 false alarm per 10k healthy requests"
TAU, HORIZON = 500, 3000


def stable_seed(*parts) -> int:
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def conditions_present(table: H.Table) -> list[str]:
    present = set(table.rows.condition)
    return [f for f in FAULTS if f"qwen_{f}" in present]


def healthy_paths(table, detector, n, length, condition, folds, seed) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [detector(table, H.make_stream(table, rng, length, condition, folds=folds)) for _ in range(n)]


def evaluate(table: H.Table, detectors: dict, args) -> pd.DataFrame:
    """Calibrate each detector to TARGET_ARL on reference-fold healthy streams, then test on held-out prompts."""
    rows = []
    for name, (detector, lo, hi, fixed) in detectors.items():
        if fixed is None:
            cal = healthy_paths(table, detector, args.n_cal, args.length, HEALTHY, (0,), seed=1)
            h = H.calibrate(cal, TARGET_ARL, lo, hi)
        else:
            h = fixed
        test = healthy_paths(table, detector, args.n_cal, args.length, HEALTHY_TEST, (1,), seed=2)
        arl0 = H.arl(test, h)
        print(f"{name:<22} threshold={h:9.3f}  test ARL0={arl0:10.0f}", flush=True)
        for fault in conditions_present(table):
            for p in SEVERITIES:
                rng = np.random.default_rng(stable_seed(name, fault, p))
                paths = [
                    detector(
                        table, H.make_stream(table, rng, TAU + HORIZON, HEALTHY_TEST, f"qwen_{fault}", TAU, p)
                    )
                    for _ in range(args.reps)
                ]
                d = H.detection(paths, h, TAU)
                rows.append({"detector": name, "fault": fault, "severity": p, "threshold": h, "test_arl0": arl0,
                             **asdict(d)})  # fmt: skip
    return pd.DataFrame(rows)


def baselines(table: H.Table, args) -> pd.DataFrame:
    detectors = {
        # name: (path function, calibration bracket lo, hi, fixed threshold or None)
        "http_breaker": (H.http_path, 0, 1, 0.2),
        "fixed_threshold": (H.threshold_path, 0.0, 5.0, None),
        "cusum": (H.cusum_path, 0.0, 500.0, None),
    }
    return evaluate(table, detectors, args)


EXPERIMENTS = {"baselines": baselines}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--recordings", default="data/recordings")
    parser.add_argument("--reps", type=int, default=60, help="streams per (fault, severity)")
    parser.add_argument("--n-cal", type=int, default=30, help="healthy streams for calibration and ARL0")
    parser.add_argument("--length", type=int, default=20_000, help="healthy stream length")
    args = parser.parse_args()

    df = H.load(args.recordings)
    table, _ = H.build_table(df, PROVIDER, HEALTHY)
    print(f"{len(table.rows):,} rows for {PROVIDER}; conditions: {sorted(set(table.rows.condition))}")
    out = EXPERIMENTS[args.experiment](table, args)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / f"{args.experiment}.csv", index=False)
    (RESULTS / f"{args.experiment}.json").write_text(json.dumps(vars(args), indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        cols = [
            c
            for c in (
                "detector",
                "fault",
                "severity",
                "detected",
                "median_delay",
                "false_alarm_before_change",
            )
            if c in out
        ]
        print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
