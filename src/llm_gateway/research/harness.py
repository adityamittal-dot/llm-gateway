"""Evaluation harness: splice recorded responses into streams and score detectors (RESEARCH.md §9–10).

Streams are built from recorded rows ("record once, replay many"): before the change point τ
every request comes from the provider's healthy test pool; after τ each request is faulty with
probability p (fault severity) and healthy otherwise. Tenants arrive interleaved, uniformly.
Requests are independent, so splicing is valid here (unlike agent trajectories).

Items are split two ways by a hash of the prompt id: fold 0 is the reference (fits the
per-tenant model and calibrates thresholds/p-values), fold 1 is the held-out test pool. Test
streams therefore always contain prompts the detectors never saw. Resampling a finite pool
turns its small prompt-driven mean offset into a persistent shift, so realised false-alarm
rates on test streams can exceed the calibrated target; both are reported.

Every detector maps a stream to a statistic path (higher = more evidence of change). Without
restarts the path does not depend on the threshold, so the first alarm for any threshold h is
the first t with path[t] >= h, which makes calibration cheap.

Metrics: ARL0 (average run length to a false alarm on healthy traffic, censored-exponential
estimate) and detection delay after τ (in provider requests).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llm_gateway.research.datacard import load_all
from llm_gateway.research.features import ALL, QUALITY, ReferenceModel, featurize, fold

TENANTS = ["math", "chat", "code", "tools"]


@dataclass
class Table:
    """Precomputed per-row arrays for every recorded row of one provider."""

    rows: pd.DataFrame  # condition, tenant, item fold, ...
    status: np.ndarray
    z: np.ndarray  # per-feature standardised deviation (rows x features)
    features: list[str]
    tenant_idx: np.ndarray
    feats: pd.DataFrame  # raw feature columns
    ref: ReferenceModel

    def pool(self, condition: str, tenant: str, folds=(1,)) -> np.ndarray:
        r = self.rows
        return np.flatnonzero((r.condition == condition) & (r.tenant == tenant) & r.fold.isin(folds))


def build_table(
    df: pd.DataFrame, provider: str, healthy: str, features: list[str] = ALL
) -> tuple[Table, ReferenceModel]:
    """Fit the reference model on fold 0 of the healthy run and featurise every row of `provider`."""
    rows = df[df.provider == provider].reset_index(drop=True).copy()
    rows["fold"] = rows.item_id.map(fold)
    feats = featurize(rows)
    fit = (rows.condition == healthy) & (rows.fold == 0)
    ref = ReferenceModel(features).fit(feats[fit], rows.tenant[fit])
    table = Table(
        rows=rows,
        status=rows.status_code.to_numpy(),
        z=ref.z(feats, rows.tenant).to_numpy(dtype=float),
        features=features,
        tenant_idx=rows.tenant.map({t: i for i, t in enumerate(TENANTS)}).to_numpy(),
        feats=feats,
        ref=ref,
    )
    return table, ref


def make_stream(
    table: Table,
    rng: np.random.Generator,
    length: int,
    healthy: str,
    fault: str | None = None,
    tau: int = 0,
    p: float = 1.0,
    tenants: list[str] = TENANTS,
    folds=(1,),
    overrides: dict[str, str] | None = None,
) -> np.ndarray:
    """Row indices of one spliced stream. `overrides` maps tenant -> condition (e.g. a traffic shift)."""
    overrides = overrides or {}
    healthy_pools = [table.pool(overrides.get(t, healthy), t, folds) for t in tenants]
    fault_pools = [table.pool(fault, t, folds) for t in tenants] if fault else None
    tenant = rng.integers(len(tenants), size=length)
    faulty = (np.arange(length) >= tau) & (rng.random(length) < p) if fault else np.zeros(length, bool)
    idx = np.empty(length, dtype=int)
    for k in range(len(tenants)):
        for pools, flag in ((healthy_pools, False), (fault_pools, True)):
            if pools is None:
                continue
            sel = (tenant == k) & (faulty == flag)
            if sel.any():
                idx[sel] = rng.choice(pools[k], size=sel.sum())
    return idx


# --- baseline detectors: stream row indices -> statistic path ---------------------------------


def http_path(table: Table, idx: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling HTTP error rate: the circuit breaker every gateway already has."""
    err = (table.status[idx] != 200).astype(float)
    return pd.Series(err).rolling(window, min_periods=window).mean().fillna(0).to_numpy()


def threshold_path(table: Table, idx: np.ndarray, window: int = 100, features=QUALITY) -> np.ndarray:
    """Max over features of |rolling mean of z| (fixed-threshold monitoring of each signal)."""
    cols = [table.features.index(f) for f in features]
    z = pd.DataFrame(table.z[idx][:, cols])
    rolled = z.rolling(window, min_periods=window // 2).mean().abs()
    return rolled.max(axis=1).fillna(0).to_numpy()


def cusum_path(table: Table, idx: np.ndarray, drift: float = 0.5, features=QUALITY) -> np.ndarray:
    """Max over features of two-sided CUSUM statistics on z (Page 1954); missing values are skipped."""
    cols = [table.features.index(f) for f in features]
    z = table.z[idx][:, cols]
    hi = np.zeros(len(cols))
    lo = np.zeros(len(cols))
    out = np.empty(len(idx))
    for t in range(len(idx)):
        zt = z[t]
        ok = ~np.isnan(zt)
        hi[ok] = np.maximum(0, hi[ok] + zt[ok] - drift)
        lo[ok] = np.maximum(0, lo[ok] - zt[ok] - drift)
        out[t] = max(hi.max(), lo.max())
    return out


# --- metrics ----------------------------------------------------------------------------------


def first_alarm(path: np.ndarray, h: float) -> int | None:
    hits = np.flatnonzero(path >= h)
    return int(hits[0]) if len(hits) else None


def arl(paths: list[np.ndarray], h: float) -> float:
    """Censored-exponential ARL estimate: total observed requests / number of false alarms."""
    alarms, observed = 0, 0
    for path in paths:
        t = first_alarm(path, h)
        if t is None:
            observed += len(path)
        else:
            alarms += 1
            observed += t + 1
    return observed / alarms if alarms else float("inf")


def calibrate(paths: list[np.ndarray], target_arl: float, lo: float, hi: float, iters: int = 40) -> float:
    """Smallest threshold whose empirical ARL0 reaches the target (bisection; ARL is monotone in h)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if arl(paths, mid) >= target_arl:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class Detection:
    detected: float  # fraction of streams with an alarm in [tau, tau + horizon)
    false_alarm_before_change: float
    median_delay: float
    mean_delay: float


def detection(paths: list[np.ndarray], h: float, tau: int) -> Detection:
    delays, early, missed = [], 0, 0
    for path in paths:
        t = first_alarm(path, h)
        if t is None:
            missed += 1
        elif t < tau:
            early += 1
        else:
            delays.append(t - tau)
    n = len(paths)
    return Detection(
        detected=len(delays) / n,
        false_alarm_before_change=early / n,
        median_delay=float(np.median(delays)) if delays else float("nan"),
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
    )


Detector = Callable[[Table, np.ndarray], np.ndarray]


def load(recordings: str | Path = "data/recordings") -> pd.DataFrame:
    return load_all(Path(recordings))
