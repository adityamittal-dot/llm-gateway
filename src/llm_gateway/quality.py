"""Quality circuit breaker: the online version of the research detector (RESEARCH.md §5).

Per provider, the gateway learns what healthy responses look like for each tenant during a warm-up
(the first `warmup` responses per tenant: first half fits the reference, second half calibrates
p-values). After that every response is scored:

    features -> NLL under the tenant reference -> conformal p-value -> e-value e(p)
    pooled Shiryaev–Roberts per provider: R = (R + 1) · e

and the provider's level is set from R (ARL₀ ≥ threshold under no change):

    0 healthy    R < alert
    1 alert      R ≥ alert          log a warning; traffic unchanged
    2 shift      R ≥ shift          `shift_fraction` of requests go to the next provider in the chain
    3 open       R ≥ open           only `canary_fraction` of requests still reach the provider

Recovery: while a provider is degraded, its responses (including canaries) keep feeding a
recovery window; once `recovery_window` consecutive responses have mean p-value ≥ 0.4 (what
healthy traffic gives), R resets and the level returns to 0.

Attribution (online, simplified): a per-(provider, tenant) detector also runs. A tenant whose
detector fires on two or more providers is treated as a traffic change, not a provider fault, and
its responses stop feeding the pooled provider statistic.

State is per gateway process: each Fargate task sees a sample of the traffic and decides alone.
"""

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

FEATURES_BINARY = ("finish_length", "tool_called", "tool_valid", "refusal", "empty")
FEATURES_CONT = ("log_out_tokens", "repetition")


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool = True
    warmup: int = 100  # healthy responses per (provider, tenant) before monitoring starts
    alert: float = 1e3
    shift: float = 1e4  # the "1 false alarm per 10k requests" operating point
    open: float = 1e6
    shift_fraction: float = 0.5
    canary_fraction: float = 0.05
    recovery_window: int = 50

    @classmethod
    def from_config(cls, cfg: dict | None) -> "QualityConfig":
        cfg = cfg or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


def features(signals) -> dict[str, float | None]:
    """Per-response features from a ResponseSignals row (same definitions as research/features.py)."""
    tools = bool(signals.tools_offered)
    return {
        "log_out_tokens": math.log1p(signals.completion_tokens)
        if signals.completion_tokens is not None
        else None,
        "finish_length": float(signals.finish_reason == "length"),
        "tool_called": float(signals.tool_calls > 0) if tools else None,
        "tool_valid": float(bool(signals.tool_call_valid)) if tools and signals.tool_calls > 0 else None,
        "refusal": float(signals.refusal),
        "empty": float(signals.empty),
        "repetition": float(signals.repetition),
    }


def mixture_e(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    lp = math.log(p)
    return (1 - p + p * lp) / (p * lp * lp)


@dataclass
class TenantReference:
    rows: list[dict] = field(default_factory=list)
    rate: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    calibration: list[float] = field(default_factory=list)  # sorted NLL scores of held-out warm-up rows

    @property
    def ready(self) -> bool:
        return bool(self.calibration)

    def fit(self, rows: list[dict]) -> None:
        for f in FEATURES_BINARY:
            xs = [r[f] for r in rows if r[f] is not None]
            self.rate[f] = (sum(xs) + 0.5) / (len(xs) + 1.0)
        for f in FEATURES_CONT:
            xs = [r[f] for r in rows if r[f] is not None]
            if xs:
                m = sum(xs) / len(xs)
                var = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
                self.mean[f], self.std[f] = m, max(math.sqrt(var), 0.05)

    def nll(self, row: dict) -> float:
        s = 0.0
        for f in FEATURES_BINARY:
            if row[f] is not None:
                p = self.rate[f]
                s -= math.log(p if row[f] > 0.5 else 1 - p)
        for f in FEATURES_CONT:
            if row[f] is not None and f in self.mean:
                z = (row[f] - self.mean[f]) / self.std[f]
                s += 0.5 * min(z * z, 100.0)
        return s

    def pvalue(self, score: float, u: float) -> float:
        cal = self.calibration
        lo, hi = _bisect(cal, score, left=True), _bisect(cal, score, left=False)
        greater, equal = len(cal) - hi, hi - lo
        return (greater + u * (equal + 1)) / (len(cal) + 1)


def _bisect(xs: list[float], x: float, left: bool) -> int:
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x or (not left and xs[mid] == x):
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass
class ProviderState:
    log_r: float = -math.inf
    level: int = 0
    observed: int = 0
    since: float = 0.0
    recent_p: deque = field(default_factory=lambda: deque(maxlen=200))


class QualityMonitor:
    def __init__(self, config: QualityConfig | None = None, seed: int | None = None, clock=time.time):
        self.config = config or QualityConfig()
        self.refs: dict[tuple[str, str], TenantReference] = defaultdict(TenantReference)
        self.providers: dict[str, ProviderState] = defaultdict(ProviderState)
        self.tenant_log_r: dict[tuple[str, str], float] = defaultdict(lambda: -math.inf)
        self.rng = random.Random(seed)
        self.clock = clock
        self.events: deque = deque(maxlen=100)

    def _levels(self, log_r: float) -> int:
        c = self.config
        return (
            3
            if log_r >= math.log(c.open)
            else 2
            if log_r >= math.log(c.shift)
            else 1
            if log_r >= math.log(c.alert)
            else 0
        )

    def shifted_tenants(self) -> set[str]:
        """Tenants whose own detector fires on at least two providers: a traffic change, not a fault."""
        firing = defaultdict(int)
        for (_, tenant), log_r in self.tenant_log_r.items():
            if log_r >= math.log(self.config.shift) + math.log(4):
                firing[tenant] += 1
        return {t for t, n in firing.items() if n >= 2}

    def observe(self, provider: str, tenant: str, signals) -> None:
        if not self.config.enabled or signals.status_code != 200:
            return
        row = features(signals)
        ref = self.refs[(provider, tenant)]
        if not ref.ready:
            ref.rows.append(row)
            if len(ref.rows) >= self.config.warmup:
                half = len(ref.rows) // 2
                ref.fit(ref.rows[:half])
                ref.calibration = sorted(ref.nll(r) for r in ref.rows[half:])
                ref.rows = []
            return
        p = ref.pvalue(ref.nll(row), self.rng.random())
        log_e = math.log(mixture_e(p))
        key = (provider, tenant)
        self.tenant_log_r[key] = _logaddexp(self.tenant_log_r[key], 0.0) + log_e
        state = self.providers[provider]
        state.observed += 1
        state.recent_p.append(p)
        if tenant not in self.shifted_tenants():
            state.log_r = _logaddexp(state.log_r, 0.0) + log_e
        new_level = max(state.level, self._levels(state.log_r))  # escalate on evidence...
        window = list(state.recent_p)[-self.config.recovery_window :]
        if (
            state.level > 0
            and len(window) == self.config.recovery_window
            and sum(window) / len(window) >= 0.4
        ):
            new_level, state.log_r = 0, -math.inf  # ...recover only on a healthy-looking window
            for k in [k for k in self.tenant_log_r if k[0] == provider]:
                self.tenant_log_r[k] = -math.inf
        if new_level != state.level:
            self.events.append({"ts": self.clock(), "provider": provider, "from": state.level, "to": new_level,
                                "log_r": round(state.log_r, 2) if state.log_r > -math.inf else None})  # fmt: skip
            state.level, state.since = new_level, self.clock()
            state.recent_p.clear()

    def admit(self, provider: str, has_fallback: bool) -> bool:
        """Should this request go to `provider` (False = try the next one in the chain)?"""
        level = self.providers[provider].level if provider in self.providers else 0
        if level < 2 or not has_fallback:
            return True
        if level == 2:
            return self.rng.random() >= self.config.shift_fraction
        return self.rng.random() < self.config.canary_fraction

    def snapshot(self) -> dict:
        return {
            "providers": {p: {"level": s.level, "log_r": round(s.log_r, 2) if s.log_r > -math.inf else None,
                              "observed": s.observed, "since": s.since} for p, s in self.providers.items()},
            "warming_up": [f"{p}/{t}" for (p, t), r in self.refs.items() if not r.ready],
            "shifted_tenants": sorted(self.shifted_tenants()),
            "events": list(self.events),
        }  # fmt: skip


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
