"""
OpenTelemetry setup for the speech-generator harness.

Exports traces + metrics to an OTEL collector (default: localhost:4317).
Falls back to console export when OTEL_EXPORTER_OTLP_ENDPOINT is not set.

Environment variables:
  OTEL_EXPORTER_OTLP_ENDPOINT  e.g. http://localhost:4317  (gRPC)
  OTEL_SERVICE_NAME            defaults to "speech-generator"
  OTEL_ENVIRONMENT             defaults to "development"
"""

import os
from contextvars import ContextVar

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from otel_exporter import SelfMetricExporter, SelfSpanExporter

SERVICE_NAME  = os.environ.get("OTEL_SERVICE_NAME", "speech-generator")
ENVIRONMENT   = os.environ.get("OTEL_ENVIRONMENT", "development")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
SELF_ENDPOINT = os.environ.get("OTEL_SELF_ENDPOINT", "http://localhost:8000")

_resource = Resource.create({
    "service.name": SERVICE_NAME,
    "deployment.environment": ENVIRONMENT,
})

# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

def _build_tracer_provider() -> TracerProvider:
    provider = TracerProvider(resource=_resource)
    if OTLP_ENDPOINT:
        exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
        print(f"[otel] tracing → OTLP at {OTLP_ENDPOINT}")
    else:
        exporter = SelfSpanExporter(base_url=SELF_ENDPOINT)
        print(f"[otel] tracing → self API at {SELF_ENDPOINT}/telemetry/spans")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider

_tracer_provider = _build_tracer_provider()
trace.set_tracer_provider(_tracer_provider)
tracer = trace.get_tracer(SERVICE_NAME)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _build_meter_provider() -> MeterProvider:
    if OTLP_ENDPOINT:
        exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    else:
        exporter = SelfMetricExporter(base_url=SELF_ENDPOINT)
        print(f"[otel] metrics  → self API at {SELF_ENDPOINT}/telemetry/metrics")
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15_000)
    return MeterProvider(resource=_resource, metric_readers=[reader])

_meter_provider = _build_meter_provider()
metrics.set_meter_provider(_meter_provider)
meter = metrics.get_meter(SERVICE_NAME)

# ── Instruments ─────────────────────────────────────────────────────────────
run_counter = meter.create_counter(
    "speech.runs.total",
    description="Total number of generation runs",
)
blocked_counter = meter.create_counter(
    "speech.runs.blocked",
    description="Runs blocked by guardrails or post-generation filter",
)
token_counter = meter.create_counter(
    "speech.tokens.total",
    description="Total LLM tokens consumed",
    unit="tokens",
)
corpus_query_counter = meter.create_counter(
    "speech.corpus.queries",
    description="Total policy corpus tool calls",
)
iteration_histogram = meter.create_histogram(
    "speech.iterations",
    description="Number of loop iterations per run",
    unit="iterations",
)
latency_histogram = meter.create_histogram(
    "speech.latency_ms",
    description="End-to-end run latency",
    unit="ms",
)
confidence_histogram = meter.create_histogram(
    "speech.confidence",
    description="Final confidence score distribution",
)


# ---------------------------------------------------------------------------
# Run-scoped context  — set once per run, read by any child span
# ---------------------------------------------------------------------------

_current_run_id: ContextVar[str] = ContextVar("run_id", default="")

def set_run_id(run_id: str):
    """Call at the start of each run. All spans created in this context inherit it."""
    _current_run_id.set(run_id)

def get_run_id() -> str:
    return _current_run_id.get()

def stamp_span(span, **extra):
    """Stamp the active span with run_id + any extra attributes."""
    run_id = get_run_id()
    if run_id:
        span.set_attribute("run.id", run_id)
    for k, v in extra.items():
        span.set_attribute(k, v)


# ---------------------------------------------------------------------------
# Shutdown helper (call on app exit)
# ---------------------------------------------------------------------------

def shutdown():
    _tracer_provider.shutdown()
    _meter_provider.shutdown()
