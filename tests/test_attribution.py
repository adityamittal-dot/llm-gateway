from types import SimpleNamespace

import numpy as np

from llm_gateway.research import attribution as A

LOG_C = float(np.log(1e4))


def side(e_by_tenant: list[float], n: int = 400):
    """Fake provider: n requests per tenant, each tenant with a constant e-value."""
    tenant_idx = np.repeat(np.arange(4), n)
    table = SimpleNamespace(tenant_idx=tenant_idx)
    e = np.repeat(np.array(e_by_tenant, dtype=float), n)
    return A.Side(table=table, e=e, healthy="h", shifted="s", fault="f"), np.arange(4 * n)


def test_post_conditions_per_scenario():
    a = A.Side(table=None, e=None, healthy="h", shifted="s", fault="f")
    assert set(A.post_conditions(a, "none", True).values()) == {"h"}
    assert set(A.post_conditions(a, "provider_fault", True).values()) == {"f"}
    assert set(A.post_conditions(a, "provider_fault", False).values()) == {"h"}  # control stays healthy
    both = A.post_conditions(a, "both", True)
    assert both["chat"] == both["math"] == "s" and both["code"] == both["tools"] == "f"


def test_decisions():
    quiet, loud = 0.9, 1.5  # e-values below 1 never accumulate; above 1 they fire quickly
    # Provider fault on A only: everyone blames A.
    (a, ia), (b, ib) = side([loud] * 4), side([quiet] * 4)
    d = A.decide(a, b, ia, ib, LOG_C)
    assert d["pooled"] and d["per_tenant"] and d["ours"] and not d["ours_flags_shift"]
    # Traffic shift on tenants 0 and 1, visible on both providers: only `ours` avoids blaming A.
    (a, ia), (b, ib) = side([loud, loud, quiet, quiet]), side([loud, loud, quiet, quiet])
    d = A.decide(a, b, ia, ib, LOG_C)
    assert d["pooled"] and d["per_tenant"] and not d["ours"] and d["ours_flags_shift"]
    # Both: shift on 0,1 everywhere plus a fault on A's other tenants: `ours` still blames A.
    (a, ia), (b, ib) = side([loud] * 4), side([loud, loud, quiet, quiet])
    d = A.decide(a, b, ia, ib, LOG_C)
    assert d["ours"] and d["ours_flags_shift"]
