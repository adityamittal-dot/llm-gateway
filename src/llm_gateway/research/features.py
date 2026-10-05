"""Per-response features and per-tenant reference models (RESEARCH.md §5–6).

Each signal row becomes a small feature vector. A reference model is fitted per (provider,
tenant) on known-healthy traffic: Bernoulli rates for binary features and Gaussians on
transformed continuous features. A response's anomaly score is its negative log-likelihood
under the reference model of its own tenant, so tenants with very different traffic shapes
(long math answers vs short tool calls) share one scale.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Feature groups (used for ablation). "latency" is kept separate from the content-derived
# quality signals because it also moves with load, not only with provider changes.
GROUPS: dict[str, list[str]] = {
    "length": ["log_out_tokens", "finish_length"],
    "tools": ["tool_called", "tool_valid"],
    "refusal": ["refusal", "empty"],
    "repetition": ["repetition"],
    "latency": ["log_ttft", "log_tps"],
}
QUALITY = [f for g in ("length", "tools", "refusal", "repetition") for f in GROUPS[g]]
ALL = QUALITY + GROUPS["latency"]
INPUT = ["log_prompt_tokens"]
BINARY = {"finish_length", "tool_called", "tool_valid", "refusal", "empty"}


def fold(item_id: str, k: int = 2) -> int:
    """Deterministic item fold: 0 = reference (fit + calibrate), 1 = held-out test prompts."""
    return hashlib.sha256(item_id.encode()).digest()[0] % k


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Signal rows -> feature columns (NaN where a feature does not apply)."""
    out = pd.DataFrame(index=df.index)
    out["log_out_tokens"] = np.log1p(df["completion_tokens"].astype(float))
    out["finish_length"] = (df["finish_reason"] == "length").astype(float)
    tools = df["tools_offered"].astype(bool)
    out["tool_called"] = np.where(tools, (df["tool_calls"] > 0).astype(float), np.nan)
    valid = df["tool_call_valid"].astype("float")  # None -> NaN
    out["tool_valid"] = np.where(tools & (df["tool_calls"] > 0), valid, np.nan)
    out["refusal"] = df["refusal"].astype(float)
    out["empty"] = df["empty"].astype(float)
    out["repetition"] = df["repetition"].astype(float)
    out["log_ttft"] = np.log(df["ttft_s"].astype(float).clip(lower=1e-3))
    out["log_tps"] = np.log(df["tokens_per_s"].astype(float).clip(lower=1e-2))
    # Input side (what the tenant sent, not what the provider returned): used to recognise traffic shifts.
    prompt = df["prompt_tokens"] if "prompt_tokens" in df else pd.Series(np.nan, index=df.index)
    out["log_prompt_tokens"] = np.log1p(prompt.astype(float))
    return out


@dataclass
class TenantModel:
    rate: dict[str, float] = field(default_factory=dict)  # binary features: P(x=1)
    mean: dict[str, float] = field(default_factory=dict)  # continuous features
    std: dict[str, float] = field(default_factory=dict)


class ReferenceModel:
    """Per-tenant reference distributions fitted on healthy traffic."""

    def __init__(self, features: list[str]):
        self.features = features
        self.tenants: dict[str, TenantModel] = {}

    def fit(self, feats: pd.DataFrame, tenants: pd.Series) -> "ReferenceModel":
        for tenant, rows in feats.groupby(tenants.values):
            model = TenantModel()
            for f in self.features:
                x = rows[f].dropna()
                if f in BINARY:
                    model.rate[f] = (x.sum() + 0.5) / (len(x) + 1.0)  # Jeffreys-smoothed
                elif len(x):
                    model.mean[f] = float(x.mean())
                    model.std[f] = float(max(x.std(ddof=1) if len(x) > 1 else 0.0, 0.05))
            self.tenants[str(tenant)] = model
        return self

    def nll(self, feats: pd.DataFrame, tenants: pd.Series) -> np.ndarray:
        """Negative log-likelihood of each row under its tenant's reference model."""
        score = np.zeros(len(feats))
        tenant_arr = tenants.to_numpy()
        for tenant, model in self.tenants.items():
            mask = tenant_arr == tenant
            if not mask.any():
                continue
            rows = feats.loc[mask]
            s = np.zeros(mask.sum())
            for f in self.features:
                x = rows[f].to_numpy(dtype=float)
                ok = ~np.isnan(x)
                if f in BINARY:
                    p = model.rate[f]
                    s[ok] += -np.log(np.where(x[ok] > 0.5, p, 1 - p))
                elif f in model.mean:
                    z = (x[ok] - model.mean[f]) / model.std[f]
                    s[ok] += 0.5 * np.minimum(z**2, 100.0)  # clip: one absurd value can't dominate
            score[mask] = s
        return score

    def z(self, feats: pd.DataFrame, tenants: pd.Series) -> pd.DataFrame:
        """Per-feature standardised deviation from the tenant reference (for threshold/CUSUM baselines)."""
        out = pd.DataFrame(np.nan, index=feats.index, columns=self.features)
        tenant_arr = tenants.to_numpy()
        for tenant, model in self.tenants.items():
            mask = tenant_arr == tenant
            for f in self.features:
                x = feats.loc[mask, f].to_numpy(dtype=float)
                if f in BINARY:
                    p = model.rate[f]
                    out.loc[mask, f] = (x - p) / np.sqrt(p * (1 - p))
                elif f in model.mean:
                    out.loc[mask, f] = (x - model.mean[f]) / model.std[f]
        return out.clip(-5, 5)
