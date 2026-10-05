from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_providers import FakeProvider

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.telemetry import get_metrics


def test_metrics_and_genai_spans():
    get_metrics()  # installs the global providers
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    settings = Settings(
        keys={"sk-t": "telemetry-tenant"}, prices={"m": {"input_per_mtok": 0, "output_per_mtok": 5}}
    )
    with TestClient(create_app(settings, providers=[FakeProvider("fake", models=["m"])])) as gw:
        gw.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"Authorization": "Bearer sk-t"},
        )
        text = gw.get("/metrics").text
    assert "gateway_requests_total{" in text and 'tenant="telemetry-tenant"' in text
    assert "gateway_request_duration_seconds_bucket" in text
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root, attempt = spans["chat m"], spans["provider fake"]
    assert attempt.parent.span_id == root.context.span_id
    assert root.attributes["gen_ai.system"] == "fake" and root.attributes["gen_ai.usage.output_tokens"] == 5


def test_root_span_ends_when_the_handler_crashes():
    from opentelemetry.sdk.trace import TracerProvider  # noqa: F401  (global provider set up above)

    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

    class Exploding(FakeProvider):
        async def complete(self, body):
            raise RuntimeError("boom")

    app = create_app(Settings(), providers=[Exploding("x", models=["m"])])
    with TestClient(app, raise_server_exceptions=False) as gw:
        assert gw.post("/v1/chat/completions", json={"model": "m", "messages": []}).status_code == 500
    span = next(s for s in exporter.get_finished_spans() if s.name == "chat m")
    assert span.events[0].name == "exception"
