"""Day-11 report: figures and summary tables from results/*.csv.

    uv run python -m llm_gateway.research.report
writes results/figures/{delays,attribution,ablation}.png and results/summary.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
FIG = RESULTS / "figures"
ORDER = [
    "output_cap",
    "quant_swap",
    "truncate_context",
    "drop_system",
    "sampling",
    "model_substitution",
    "throttle",
]


def detectors_table() -> pd.DataFrame:
    frames = [pd.read_csv(RESULTS / f) for f in ("baselines.csv", "detectors.csv") if (RESULTS / f).exists()]
    df = pd.concat(frames).drop_duplicates(["detector", "fault", "severity"])
    return df[df.detector.isin(["http_breaker", "fixed_threshold", "cusum", "e_pooled", "e_per_tenant"])]


def plot_delays(df: pd.DataFrame) -> None:
    dets = ["e_pooled", "e_per_tenant", "cusum", "fixed_threshold", "http_breaker"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, sev in zip(axes, (0.1, 0.3, 1.0), strict=True):
        sub = df[df.severity == sev]
        x = np.arange(len(ORDER))
        for i, det in enumerate(dets):
            vals = [sub[(sub.detector == det) & (sub.fault == f)].median_delay.mean() for f in ORDER]
            det_rate = [sub[(sub.detector == det) & (sub.fault == f)].detected.mean() for f in ORDER]
            vals = [
                v if r >= 0.5 else np.nan for v, r in zip(vals, det_rate, strict=True)
            ]  # undetected -> gap
            ax.bar(x + (i - 2) * 0.16, vals, 0.16, label=det)
        ax.set_xticks(x, ORDER, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"fault severity p = {sev}")
    axes[0].set_ylabel("median detection delay (requests, log)\nmissing bar = detected in <50% of runs")
    axes[0].legend(fontsize=8)
    fig.suptitle("Detection delay after a silent fault (all detectors at ARL₀ target 10,000)")
    fig.tight_layout()
    fig.savefig(FIG / "delays.png", dpi=130)


def plot_attribution(df: pd.DataFrame) -> pd.DataFrame:
    table = df[df.method != "ours_shift_flag"].groupby(["method", "scenario"]).blamed.mean().unstack()
    methods = [m for m in ("pooled", "per_tenant", "ours", "ours_input") if m in table.index]
    table = table[["none", "provider_fault", "traffic_shift", "both"]].loc[methods]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(table.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4), ["none", "provider\nfault", "traffic\nshift", "both"])
    labels = {
        "pooled": "pooled (no control)",
        "per_tenant": "per-tenant",
        "ours": "cross-provider (outputs)",
        "ours_input": "cross-provider + input shift",
    }
    ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
    for (i, j), v in np.ndenumerate(table.values):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < 0.6 else "black")
    ax.set_title("How often provider A is blamed\n(correct: 0% for none/shift, 100% for fault/both)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "attribution.png", dpi=130)
    return table


def plot_ablation(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="signals", columns="fault", values="detected")
    table = table[[f for f in ORDER if f in table.columns]]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(table.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    for (i, j), v in np.ndenumerate(table.values):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_title("Detection rate by signal group (severity 0.3, within 3,000 requests)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=130)
    return table


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    det = detectors_table()
    plot_delays(det)
    lines = ["# Results summary (generated)\n"]
    arl = det.groupby("detector").test_arl0.first()
    lines += [
        "## False alarms on held-out healthy prompts (target ARL₀ = 10,000)\n",
        arl.to_frame().to_markdown(),
        "",
    ]
    for sev in (0.1, 0.3, 1.0):
        sub = det[det.severity == sev]
        t = sub.pivot_table(index="fault", columns="detector", values="median_delay").reindex(ORDER)
        r = sub.pivot_table(index="fault", columns="detector", values="detected").reindex(ORDER)
        cell = t.copy().astype(object)
        for f in t.index:
            for d in t.columns:
                delay, rate = t.loc[f, d], r.loc[f, d]
                cell.loc[f, d] = (
                    "–" if pd.isna(rate) else f"{'—' if pd.isna(delay) else int(delay)} / {rate:.0%}"
                )
        lines += [f"## Severity {sev}: median delay (requests) / detection rate\n", cell.to_markdown(), ""]
    if (RESULTS / "attribution.csv").exists():
        attr = pd.read_csv(RESULTS / "attribution.csv")
        lines += [
            "## Attribution: share of runs blaming provider A\n",
            plot_attribution(attr).round(2).to_markdown(),
            "",
        ]
        acc = attr[attr.method != "ours_shift_flag"].groupby("method").correct.mean()
        lines += ["Overall correct blame decisions:\n", acc.round(3).to_frame().to_markdown(), ""]
    if (RESULTS / "ablation.csv").exists():
        lines += ["## Ablation: detection rate by signal group\n",
                  plot_ablation(pd.read_csv(RESULTS / "ablation.csv")).round(2).to_markdown(), ""]  # fmt: skip
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
