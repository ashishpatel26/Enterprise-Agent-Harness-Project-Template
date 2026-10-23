"""OpenTelemetry tracing helpers.

`span()` is safe to call whether or not tracing is configured: without an SDK provider,
OpenTelemetry's API returns no-op spans. `configure_tracing()` installs the SDK and an
OTLP exporter when enabled in settings.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

_tracer = trace.get_tracer("agent-harness")

_Primitive = str | bool | int | float


def _attr(value: Any) -> _Primitive:
    return value if isinstance(value, str | bool | int | float) else str(value)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    attrs = {f"harness.{k}": _attr(v) for k, v in attributes.items() if v is not None}
    with _tracer.start_as_current_span(name, attributes=attrs) as s:
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def configure_tracing(service_name: str, endpoint: str | None) -> None:
    """Install the OTel SDK with a batch OTLP/gRPC exporter. Idempotent per process."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
