"""
Custom OpenTelemetry exporters that POST spans and metrics
to our own FastAPI /telemetry/* endpoints.

Replaces the OTLP/console exporters in telemetry.py when
OTEL_EXPORTER_OTLP_ENDPOINT is not set.
"""

import json
import logging
import urllib.request
from typing import Sequence

from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

# Default: post back to ourselves. Override with OTEL_SELF_ENDPOINT env var.
DEFAULT_BASE_URL = "http://localhost:8000"


def _post(url: str, payload: dict) -> bool:
    """Fire-and-forget JSON POST. Returns True on success."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status < 300
    except Exception as exc:
        logger.warning("[otel_exporter] POST to %s failed: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Span exporter
# ---------------------------------------------------------------------------

def _span_to_dict(span: ReadableSpan) -> dict:
    ctx = span.context
    return {
        "trace_id": format(ctx.trace_id, "032x") if ctx else None,
        "span_id": format(ctx.span_id, "016x") if ctx else None,
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "name": span.name,
        "start_time_ms": (span.start_time or 0) // 1_000_000,
        "end_time_ms": (span.end_time or 0) // 1_000_000,
        "duration_ms": ((span.end_time or 0) - (span.start_time or 0)) // 1_000_000,
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
    }


class SelfSpanExporter(SpanExporter):
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.url = f"{base_url.rstrip('/')}/telemetry/spans"

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        payload = {"spans": [_span_to_dict(s) for s in spans]}
        ok = _post(self.url, payload)
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Metric exporter
# ---------------------------------------------------------------------------

def _metric_to_dict(metric) -> dict:
    points = []
    for dp in (metric.data.data_points or []):
        point = {
            "attributes": dict(dp.attributes or {}),
            "time_ms": (dp.time_unix_nano or 0) // 1_000_000,
        }
        # Counter / Gauge
        if hasattr(dp, "value"):
            point["value"] = dp.value
        # Histogram
        if hasattr(dp, "sum"):
            point["sum"] = dp.sum
            point["count"] = dp.count
        points.append(point)

    return {
        "name": metric.name,
        "description": metric.description,
        "unit": metric.unit,
        "type": type(metric.data).__name__,
        "data_points": points,
    }


class SelfMetricExporter(MetricExporter):
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.url = f"{base_url.rstrip('/')}/telemetry/metrics"
        self._preferred_temporality = {}
        self._preferred_aggregation = {}

    def export(self, metrics_data: MetricsData, **_) -> MetricExportResult:
        metrics_list = []
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    metrics_list.append(_metric_to_dict(metric))

        if not metrics_list:
            return MetricExportResult.SUCCESS

        ok = _post(self.url, {"metrics": metrics_list})
        return MetricExportResult.SUCCESS if ok else MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, **kwargs) -> None:
        pass
