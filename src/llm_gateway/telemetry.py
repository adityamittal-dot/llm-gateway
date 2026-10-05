"""OpenTelemetry metrics and traces.

Metrics are recorded through the OTel API and exposed in Prometheus format at /metrics (scraped by
Prometheus locally, or by the ADOT collector on AWS). Traces follow the OpenTelemetry GenAI
semantic conventions (gen_ai.* attributes) and are exported over OTLP/HTTP when
OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise tracing is a no-op.
"""

import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

_configured = False


def setup(service_name: str = "llm-gateway") -> None:
    """Install global meter/tracer providers once per process."""
    global _configured
    if _configured:
        return
    resource = Resource.create({"service.name": service_name})
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[PrometheusMetricReader()]))
    tracer_provider = TracerProvider(resource=resource)
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    _configured = True


def prometheus_payload() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


class Metrics:
    """Gateway instruments (names follow Prometheus conventions after export)."""

    def __init__(self):
        meter = metrics.get_meter("llm_gateway")
        self.requests = meter.create_counter("gateway_requests", description="Completed requests")
        self.tokens = meter.create_counter("gateway_tokens", description="Tokens by direction")
        self.cost = meter.create_counter("gateway_cost_usd", description="Provider spend in USD")
        self.saved = meter.create_counter(
            "gateway_cache_saved_usd", description="Spend avoided by cache hits"
        )
        self.failovers = meter.create_counter("gateway_failovers", description="Requests moved to a fallback")
        self.rate_limited = meter.create_counter(
            "gateway_rate_limited", description="Requests rejected by limits"
        )
        self.latency = meter.create_histogram(
            "gateway_request_duration", unit="s", description="End-to-end latency"
        )
        self.ttft = meter.create_histogram(
            "gateway_time_to_first_token", unit="s", description="Streaming TTFT"
        )
        self.breaker_states: dict[str, int] = {}
        meter.create_observable_gauge(
            "gateway_breaker_open",
            callbacks=[self._breaker_cb],
            description="1 when the provider breaker is open",
        )

    def _breaker_cb(self, _options):
        return [metrics.Observation(v, {"provider": p}) for p, v in self.breaker_states.items()]

    def request(self, *, tenant: str, provider: str, model: str, status: int, cache: str, latency_s: float,
                ttft_s: float | None, prompt_tokens: int | None, completion_tokens: int | None, cost_usd: float,
                failovers: int) -> None:  # fmt: skip
        attrs = {
            "tenant": tenant,
            "provider": provider,
            "model": model,
            "status": str(status),
            "cache": cache,
        }
        self.requests.add(1, attrs)
        self.latency.record(latency_s, {"provider": provider, "cache": cache})
        if ttft_s is not None:
            self.ttft.record(ttft_s, {"provider": provider})
        if prompt_tokens:
            self.tokens.add(prompt_tokens, {"provider": provider, "direction": "input"})
        if completion_tokens:
            self.tokens.add(completion_tokens, {"provider": provider, "direction": "output"})
        if cost_usd:
            self.cost.add(cost_usd, {"tenant": tenant, "provider": provider})
        if failovers:
            self.failovers.add(failovers, {"model": model})


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Process-wide instruments (OTel instruments must not be created twice per process)."""
    global _metrics
    if _metrics is None:
        setup()
        _metrics = Metrics()
    return _metrics


tracer = trace.get_tracer("llm_gateway")
