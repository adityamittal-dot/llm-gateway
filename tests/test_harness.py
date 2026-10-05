import numpy as np
import pandas as pd

from llm_gateway.research import harness as H
from llm_gateway.research.features import featurize


def fake_recordings(seed=0, n_items=300) -> pd.DataFrame:
    """Two conditions for one provider: healthy, and a fault that truncates every answer."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, short in (("healthy", False), ("healthy_2", False), ("fault", True)):
        for tenant in H.TENANTS:
            for i in range(n_items):
                tokens = int(rng.integers(5, 15)) if short else int(rng.normal(120, 20))
                rows.append({
                    "condition": condition, "provider": "p", "tenant": tenant, "item_id": f"{tenant}-{i}",
                    "status_code": 200, "completion_tokens": tokens, "finish_reason": "length" if short else "stop",
                    "tools_offered": tenant == "tools", "tool_calls": int(tenant == "tools"),
                    "tool_call_valid": True if tenant == "tools" else None, "refusal": False, "empty": False,
                    "repetition": 0.0, "ttft_s": 0.1, "tokens_per_s": 80.0,
                })  # fmt: skip
    return pd.DataFrame(rows)


def test_featurize_handles_tools_and_missing_values():
    feats = featurize(fake_recordings(n_items=3))
    assert (
        feats.loc[feats.index[0], "tool_called"] != feats.loc[feats.index[0], "tool_called"]
    )  # NaN for non-tool
    assert set(feats["finish_length"].unique()) <= {0.0, 1.0}


def test_stream_respects_change_point_and_severity():
    table, _ = H.build_table(fake_recordings(), "p", "healthy")
    rng = np.random.default_rng(0)
    idx = H.make_stream(table, rng, 4000, "healthy_2", "fault", tau=1000, p=0.3)
    cond = table.rows.condition.to_numpy()[idx]
    assert (cond[:1000] == "healthy_2").all()
    assert 0.25 < (cond[1000:] == "fault").mean() < 0.35
    assert (table.rows.fold.to_numpy()[idx] == 1).all()  # test fold only


def test_cusum_detects_a_strong_fault_and_http_breaker_does_not():
    table, _ = H.build_table(fake_recordings(), "p", "healthy")
    rng = np.random.default_rng(1)
    healthy = [H.cusum_path(table, H.make_stream(table, rng, 5000, "healthy", folds=(0,))) for _ in range(20)]
    h = H.calibrate(healthy, 5000, 0, 500)
    faulty = [
        H.cusum_path(table, H.make_stream(table, rng, 1000, "healthy_2", "fault", 200, 1.0))
        for _ in range(30)
    ]
    d = H.detection(faulty, h, 200)
    assert d.detected + d.false_alarm_before_change == 1.0  # nothing is missed
    assert d.detected > 0.6 and d.median_delay < 20  # small synthetic pools also cause some early alarms
    http = [
        H.http_path(table, H.make_stream(table, rng, 1500, "healthy_2", "fault", 500, 1.0)) for _ in range(3)
    ]
    assert H.detection(http, 0.2, 500).detected == 0.0


def test_arl_estimate_and_calibration_are_monotone():
    paths = [np.arange(100.0), np.arange(100.0) * 2]
    assert H.arl(paths, 10) < H.arl(paths, 50) < H.arl(paths, 150)
    assert H.arl(paths, 1e9) == float("inf")
    h = H.calibrate(paths, 60, 0, 1000)
    assert H.arl(paths, h) >= 60
