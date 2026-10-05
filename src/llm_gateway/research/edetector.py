"""Anytime-valid change detection with conformal e-values (RESEARCH.md §5).

1. Score: a response's negative log-likelihood under its tenant's reference model.
2. Conformal p-value against out-of-sample reference scores of the same tenant (cross-fitted on
   the reference fold, so calibration scores are not optimistic). Under no change, p ~ U(0,1).
3. e-value: the mixture betting function e(p) = ∫₀¹ ε p^(ε-1) dε = (1 - p + p ln p) / (p ln² p),
   which has mean exactly 1 when p is uniform and grows large when responses become unusual.
4. Shiryaev–Roberts e-detector: R_t = (R_{t-1} + 1) · e_t, alarm when R_t >= c. Under the null
   E[R_t] = t, which gives an average run length to false alarm of at least c (Shin, Ramdas &
   Rinaldo, "E-detectors", 2022). With c = 10,000 that is the "1 false alarm per 10k requests"
   target, without tuning on healthy data.

Pooling: one detector per provider over all tenants' traffic (evidence accumulates across
tenants). Per-tenant monitoring: one detector per tenant with threshold c·K (union bound).
"""

import hashlib

import numpy as np
import pandas as pd

from llm_gateway.research.features import ReferenceModel
from llm_gateway.research.harness import Table


def mixture_e(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(p)
    return (1 - p + p * lp) / (p * lp**2)


def conformal_pvalues(table: Table, features: list[str], healthy: str, seed: int = 0) -> np.ndarray:
    """p-value of every row in `table` against cross-fitted reference scores of its tenant."""
    rows, feats = table.rows, table.feats
    ref_mask = ((rows.condition == healthy) & (rows.fold == 0)).to_numpy()
    ref_idx = np.flatnonzero(ref_mask)
    half = rows.item_id.map(hash_half).to_numpy()

    # Cross-fit: score each reference half with a model fitted on the other half.
    cal_scores = np.empty(len(ref_idx))
    for side in (0, 1):
        fit_idx = ref_idx[half[ref_idx] == side]
        score_idx = ref_idx[half[ref_idx] != side]
        model = ReferenceModel(features).fit(feats.iloc[fit_idx], rows.tenant.iloc[fit_idx])
        cal_scores[np.isin(ref_idx, score_idx)] = model.nll(
            feats.iloc[score_idx], rows.tenant.iloc[score_idx]
        )

    full = ReferenceModel(features).fit(feats.iloc[ref_idx], rows.tenant.iloc[ref_idx])
    scores = full.nll(feats, rows.tenant)
    rng = np.random.default_rng(seed)
    u = rng.random(len(rows))
    p = np.ones(len(rows))
    tenants = rows.tenant.to_numpy()
    ref_tenants = tenants[ref_idx]
    for tenant in np.unique(tenants):
        cal = np.sort(cal_scores[ref_tenants == tenant])
        mask = tenants == tenant
        s = scores[mask]
        greater = len(cal) - np.searchsorted(cal, s, side="right")
        equal = np.searchsorted(cal, s, side="right") - np.searchsorted(cal, s, side="left")
        p[mask] = (greater + u[mask] * (equal + 1)) / (len(cal) + 1)
    return p


def hash_half(item_id: str) -> int:
    return hashlib.sha256(("half:" + item_id).encode()).digest()[0] % 2


def sr_path(e: np.ndarray) -> np.ndarray:
    """Shiryaev–Roberts statistic path in log space (returned as log R_t)."""
    out = np.empty(len(e))
    log_r = -np.inf
    log_e = np.log(np.maximum(e, 1e-300))
    for t in range(len(e)):
        log_r = np.logaddexp(log_r, 0.0) + log_e[t]  # R = (R + 1) * e
        out[t] = log_r
    return out


class EDetector:
    """Callable detector for the harness: stream row indices -> statistic path (log R)."""

    def __init__(self, pvalues: np.ndarray, pooled: bool = True, n_tenants: int = 4):
        self.e = mixture_e(pvalues)
        self.pooled = pooled
        self.n_tenants = n_tenants

    def __call__(self, table: Table, idx: np.ndarray) -> np.ndarray:
        e = self.e[idx]
        if self.pooled:
            return sr_path(e)
        # Per-tenant detectors, each only updated by its own tenant's requests; report the max,
        # shifted by log K so one threshold (log c) gives the union-bound per-tenant threshold c·K.
        tenant = table.tenant_idx[idx]
        out = np.full(len(idx), -np.inf)
        for k in np.unique(tenant):
            mask = tenant == k
            path = np.full(len(idx), np.nan)
            path[mask] = sr_path(e[mask])
            out = np.fmax(out, pd.Series(path).ffill().fillna(-np.inf).to_numpy())
        return out - np.log(self.n_tenants)
