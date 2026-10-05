import numpy as np
from test_harness import fake_recordings

from llm_gateway.research import harness as H
from llm_gateway.research.edetector import EDetector, conformal_pvalues, mixture_e, sr_path
from llm_gateway.research.features import QUALITY


def test_mixture_e_value_has_mean_one_under_uniform_p():
    # E[e(U)] = ∫ e(p) dp; substitute p = exp(-x) and integrate on a log grid (the mass sits in a
    # thin 1/(p ln²p) tail near 0 that Monte Carlo barely reaches). Tail beyond x=X is ≈ 1/X.
    big_x = 25.0  # stay above the 1e-12 clip; for large x, e(p)·p ≈ 1/x², so the tail is ≈ 1/X
    x = np.geomspace(1e-9, big_x, 400_000)
    p = np.exp(-x)
    integral = np.trapezoid(mixture_e(p) * p, x) + 1 / big_x
    assert abs(integral - 1) < 0.01
    assert mixture_e(np.array([1e-3]))[0] > 10 and mixture_e(np.array([0.9]))[0] < 1


def test_shiryaev_roberts_respects_its_false_alarm_bound_on_null_data():
    rng = np.random.default_rng(1)
    c = 200.0
    paths = [sr_path(mixture_e(rng.random(4000))) for _ in range(200)]
    assert H.arl(paths, np.log(c)) >= c  # ARL0 >= c
    assert all(np.isfinite(p).all() for p in paths)


def test_conformal_pvalues_are_roughly_uniform_on_healthy_and_small_on_faulty_rows():
    table, _ = H.build_table(fake_recordings(), "p", "healthy")
    p = conformal_pvalues(table, QUALITY, "healthy")
    rows = table.rows
    healthy_test = p[((rows.condition == "healthy_2") & (rows.fold == 1)).to_numpy()]
    faulty = p[(rows.condition == "fault").to_numpy()]
    assert 0.4 < healthy_test.mean() < 0.6
    assert faulty.mean() < 0.05


def test_pooled_detector_detects_faster_than_per_tenant():
    table, _ = H.build_table(fake_recordings(), "p", "healthy")
    p = conformal_pvalues(table, QUALITY, "healthy")
    rng = np.random.default_rng(3)
    streams = [H.make_stream(table, rng, 900, "healthy_2", "fault", 300, 0.3) for _ in range(20)]
    pooled = H.detection([EDetector(p)(table, s) for s in streams], np.log(1e4), 300)
    per_tenant = H.detection([EDetector(p, pooled=False)(table, s) for s in streams], np.log(1e4), 300)
    assert pooled.median_delay <= per_tenant.median_delay
