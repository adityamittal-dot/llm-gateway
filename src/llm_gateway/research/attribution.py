"""Provider-vs-traffic attribution (RESEARCH.md §4 RQ2, §5).

Two providers serve the same tenants: provider A (faults injected) and a control provider B.
After the change point τ one of four scenarios holds:

    none            nothing changes
    provider_fault  every tenant on A gets faulty responses (with probability p)
    traffic_shift   some tenants change their own prompts, on both providers
    both            A is faulty for the unshifted tenants while the shifted tenants shift on both

Decision rules after a fixed horizon (did we blame provider A?):

    pooled      blame A if A's pooled e-detector over all tenants fires (no control provider)
    per_tenant  blame A if any of A's per-tenant e-detectors fires
    ours        tenants whose detector fires on BOTH providers are flagged as traffic shifts and
                excluded; blame A only if A's pooled detector over the remaining tenants fires
                while B's pooled detector over the same tenants does not
    ours_input  like `ours`, but a tenant also counts as shifted when its *inputs* changed (an
                input-side detector on prompt length fires on either provider): a provider fault
                never changes what tenants send, while a traffic shift always does, even when only
                one provider's outputs react to it
"""

from dataclasses import dataclass

import numpy as np

from llm_gateway.research import harness as H
from llm_gateway.research.edetector import sr_path

SCENARIOS = ("none", "provider_fault", "traffic_shift", "both")
SHIFTED_TENANTS = ("chat", "math")


@dataclass
class Side:
    """One provider in the experiment."""

    table: H.Table
    e: np.ndarray  # e-value per row
    healthy: str  # healthy test condition
    shifted: str  # traffic-shift condition
    fault: str | None = None  # fault condition (provider A only)
    e_in: np.ndarray | None = None  # input-side e-value per row (prompt length)


def post_conditions(side: Side, scenario: str, is_a: bool) -> dict[str, str]:
    """Condition each tenant's requests come from after τ."""
    out = {}
    for tenant in H.TENANTS:
        shifted = scenario in ("traffic_shift", "both") and tenant in SHIFTED_TENANTS
        faulty = is_a and scenario in ("provider_fault", "both") and not shifted
        out[tenant] = side.shifted if shifted else side.fault if faulty else side.healthy
    return out


def stream(side: Side, rng, length: int, tau: int, post: dict[str, str], p: float) -> np.ndarray:
    """Row indices: healthy before τ; after τ each request comes from its tenant's post condition
    with probability p (fault severity) for fault conditions, always for traffic shifts."""
    tenant = rng.integers(len(H.TENANTS), size=length)
    idx = np.empty(length, dtype=int)
    after = np.arange(length) >= tau
    for k, name in enumerate(H.TENANTS):
        healthy_pool = side.table.pool(side.healthy, name)
        cond = post[name]
        prob = p if cond == side.fault else 1.0
        use_post = after & (rng.random(length) < prob) & (cond != side.healthy)
        for pool, flag in ((healthy_pool, False), (side.table.pool(cond, name), True)):
            sel = (tenant == k) & (use_post == flag)
            if sel.any():
                idx[sel] = rng.choice(pool, size=sel.sum())
    return idx


def fired(e: np.ndarray, log_c: float) -> bool:
    return bool(len(e)) and bool((sr_path(e) >= log_c).any())


def decide(a: Side, b: Side, idx_a: np.ndarray, idx_b: np.ndarray, log_c: float) -> dict[str, bool]:
    k = len(H.TENANTS)
    ta, tb = a.table.tenant_idx[idx_a], b.table.tenant_idx[idx_b]
    ea, eb = a.e[idx_a], b.e[idx_b]
    per_tenant_a = [fired(ea[ta == j], log_c + np.log(k)) for j in range(k)]
    per_tenant_b = [fired(eb[tb == j], log_c + np.log(k)) for j in range(k)]
    shifted = [per_tenant_a[j] and per_tenant_b[j] for j in range(k)]
    keep_a = ~np.isin(ta, np.flatnonzero(shifted))
    keep_b = ~np.isin(tb, np.flatnonzero(shifted))
    ours = fired(ea[keep_a], log_c) and not fired(eb[keep_b], log_c)
    out = {
        "pooled": fired(ea, log_c),
        "per_tenant": any(per_tenant_a),
        "ours": ours,
        "ours_flags_shift": any(shifted),
    }
    if a.e_in is not None and b.e_in is not None:
        ia, ib = a.e_in[idx_a], b.e_in[idx_b]
        input_shift = [fired(ia[ta == j], log_c + np.log(2 * k)) or fired(ib[tb == j], log_c + np.log(2 * k))
                       for j in range(k)]  # fmt: skip
        excluded = np.flatnonzero([shifted[j] or input_shift[j] for j in range(k)])
        keep_a, keep_b = ~np.isin(ta, excluded), ~np.isin(tb, excluded)
        out["ours_input"] = fired(ea[keep_a], log_c) and not fired(eb[keep_b], log_c)
    return out


def run(
    a: Side, b: Side, reps: int, tau: int, horizon: int, p: float, log_c: float, seed: int = 0
) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        rng = np.random.default_rng([seed, SCENARIOS.index(scenario), int(p * 100)])
        for _ in range(reps):
            idx_a = stream(a, rng, tau + horizon, tau, post_conditions(a, scenario, True), p)
            idx_b = stream(b, rng, tau + horizon, tau, post_conditions(b, scenario, False), p)
            truth = scenario in ("provider_fault", "both")
            for method, blamed in decide(a, b, idx_a, idx_b, log_c).items():
                if method == "ours_flags_shift":
                    rows.append({"scenario": scenario, "severity": p, "method": "ours_shift_flag", "blamed": blamed,
                                 "correct": blamed == (scenario in ("traffic_shift", "both"))})  # fmt: skip
                    continue
                rows.append({"scenario": scenario, "severity": p, "method": method, "blamed": blamed,
                             "correct": blamed == truth})  # fmt: skip
    return rows
