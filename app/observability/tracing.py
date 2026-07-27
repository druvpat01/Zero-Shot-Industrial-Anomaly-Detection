"""Distributed tracing: where the milliseconds in a request actually went.

The gap this fills
==================
The metrics in :mod:`app.observability.metrics` say a request took 380 ms. They
cannot say *why*. A histogram observation is one number, and by the time it is
recorded the four things that produced it — decoding a 4 MB PNG, running the
guard, the model's forward pass, encoding the heatmap back to PNG — have been
summed into a single value with no way to take it apart again. That is exactly
the question asked when latency regresses, and answering it by adding four more
histograms multiplies the cardinality of every dashboard for a question that is
per-request rather than aggregate.

A trace is the right shape for it. One span per request, four child spans named
for the pipeline stages, each carrying its own duration and attributes. The
aggregate view stays in Prometheus and the "what happened in *this* request"
view lives here, which is the division of labour the two tools were designed for.

The four stages, and why those four
===================================
=====================  ===================================================
Span                   Covers
=====================  ===================================================
``preprocess``         base64 decode and PNG/JPEG decode to a NumPy frame.
                       Scales with the *payload*, not the model — this is
                       the span that explains a slow request from a client
                       that started sending 4K frames.
``guard``              :meth:`~app.guardrails.quality.FrameGuard.validate`.
                       Measured at 1.3 ms (256x256) to 23 ms (900x900) —
                       it scales with the frame, so the span is where a
                       jump in it gets attributed to a resolution change
                       rather than to the model.
``model_inference``    The forward pass, and nothing else. Cold model load
                       is deliberately *outside* it — see below.
``postprocess``        Heatmap colourisation and PNG encode. Scales with
                       the frame's resolution, like ``preprocess``.
=====================  ===================================================

Cold model load gets its own span (``model_load``) rather than being folded into
``model_inference``, for the same reason ``latency_ms`` excludes it in
:mod:`app.serving.main`: a 30-second WinCLIP load and a 200 ms forward pass in
one span produces a trace where the interesting structure is invisible under the
outlier, and a p95 that is a fact about startup rather than about serving.

Exporters
=========
``ConsoleSpanExporter`` by default, so tracing produces something visible with
zero infrastructure — which is the state this repo is normally read in.
``OTEL_EXPORTER_OTLP_ENDPOINT`` switches to OTLP over HTTP when a collector
exists. The standard ``OTEL_TRACES_EXPORTER`` variable (``console``/``otlp``/
``none``) overrides both, and ``none`` is the one to set on a load test, where a
span per request printed to stderr costs more than the thing being measured.

The OTLP exporter is a separate distribution (``opentelemetry-exporter-otlp-
proto-http``) that the SDK does not pull in. If an endpoint is configured and
that package is missing, :func:`configure_tracing` logs an error and falls back
to the console rather than raising: a missing *telemetry* dependency must not be
the reason an inspection line stops answering.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from app.observability.logging_config import get_logger

__all__ = [
    "NO_TRACE_ID",
    "STAGES",
    "STAGE_GUARD",
    "STAGE_MODEL_INFERENCE",
    "STAGE_MODEL_LOAD",
    "STAGE_POSTPROCESS",
    "STAGE_PREPROCESS",
    "configure_tracing",
    "current_trace_id",
    "get_tracer",
    "is_configured",
    "shutdown_tracing",
    "stage_span",
]

log = get_logger(__name__)

#: Instrumentation scope. Shows up on every span as ``otel.scope.name`` and is
#: how a collector tells this app's spans from a library's.
_INSTRUMENTATION_NAME = "app.serving"

#: Default ``service.name``. Overridable with the standard ``OTEL_SERVICE_NAME``.
_DEFAULT_SERVICE_NAME = "defect-detection"

_ENDPOINT_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"
_EXPORTER_VAR = "OTEL_TRACES_EXPORTER"
_SERVICE_NAME_VAR = "OTEL_SERVICE_NAME"

#: ``trace_id`` for a log record emitted with no span in scope. Mirrors
#: :data:`~app.observability.logging_config.NO_REQUEST_ID`: a sentinel keeps
#: every record the same shape.
NO_TRACE_ID = "-"

#: Stage span names. Constants rather than string literals at the call sites,
#: because a dashboard and a trace query both select on them and a typo produces
#: a span that simply never matches anything.
STAGE_PREPROCESS = "preprocess"
STAGE_GUARD = "guard"
STAGE_MODEL_LOAD = "model_load"
STAGE_MODEL_INFERENCE = "model_inference"
STAGE_POSTPROCESS = "postprocess"

#: The pipeline stages in the order a request passes through them.
STAGES: tuple[str, ...] = (
    STAGE_PREPROCESS,
    STAGE_GUARD,
    STAGE_MODEL_LOAD,
    STAGE_MODEL_INFERENCE,
    STAGE_POSTPROCESS,
)

_configure_lock = threading.Lock()
_configured = False
_provider: TracerProvider | None = None


def _resolve_exporter_choice() -> str:
    """Decide which exporter to build: ``"otlp"``, ``"console"`` or ``"none"``.

    ``OTEL_TRACES_EXPORTER`` wins if set — it is the variable the OpenTelemetry
    spec defines for this, so an operator who already knows OTel can turn tracing
    off without reading this file. Otherwise the presence of an endpoint implies
    OTLP, and the absence of one implies the console.
    """
    explicit = os.getenv(_EXPORTER_VAR, "").strip().lower()
    if explicit in ("otlp", "console", "none"):
        return explicit
    if explicit:
        log.warning(
            "unknown_traces_exporter",
            variable=_EXPORTER_VAR,
            value=explicit,
            expected=["otlp", "console", "none"],
            action="ignoring; falling back to endpoint detection",
        )
    return "otlp" if os.getenv(_ENDPOINT_VAR, "").strip() else "console"


def _build_span_processor(choice: str) -> Any | None:
    """Build the span processor for ``choice``, or ``None`` to export nothing.

    The pairing of processor to exporter is deliberate:

    * **OTLP gets a** :class:`BatchSpanProcessor`. Spans are queued and flushed
      on a background thread, so a slow or unreachable collector adds nothing to
      request latency. This is the only safe choice for anything doing network
      I/O on the request path.
    * **Console gets a** :class:`SimpleSpanProcessor`. Synchronous and
      unbatched, which would be indefensible over a network but is right for a
      terminal: spans appear interleaved with the log lines they belong to,
      in order, instead of arriving in a clump five seconds later.
    """
    if choice == "none":
        return None

    if choice == "otlp":
        endpoint = os.getenv(_ENDPOINT_VAR, "").strip()
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.error(
                "otlp_exporter_unavailable",
                endpoint=endpoint,
                fix="pip install opentelemetry-exporter-otlp-proto-http",
                action="falling back to the console exporter",
            )
            return SimpleSpanProcessor(ConsoleSpanExporter())
        # An empty endpoint lets the exporter apply its own default
        # (http://localhost:4318/v1/traces) rather than being handed "".
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        return BatchSpanProcessor(exporter)

    return SimpleSpanProcessor(ConsoleSpanExporter())


def configure_tracing(*, force: bool = False) -> str:
    """Install a global :class:`TracerProvider`. Idempotent.

    Called once from the FastAPI lifespan in :mod:`app.serving.main`, alongside
    :func:`~app.observability.logging_config.configure_logging`.

    Args:
        force: Rebuild even if tracing is already configured. Off by default:
            :func:`opentelemetry.trace.set_tracer_provider` refuses to replace an
            existing provider and logs an error when asked twice, and a test
            importing the app after a script configured it should be a no-op.

    Returns:
        The exporter that was installed — ``"otlp"``, ``"console"`` or
        ``"none"`` — so the caller can log what it got.
    """
    global _configured, _provider  # noqa: PLW0603 - module-level provider handle

    with _configure_lock:
        if _configured and not force:
            return _resolve_exporter_choice()

        choice = _resolve_exporter_choice()
        resource = Resource.create(
            {
                "service.name": os.getenv(_SERVICE_NAME_VAR, "").strip() or _DEFAULT_SERVICE_NAME,
                "service.version": "0.1.0",
            },
        )
        provider = TracerProvider(resource=resource)

        processor = _build_span_processor(choice)
        if processor is not None:
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _provider = provider
        _configured = True

    return choice


def shutdown_tracing() -> None:
    """Flush and stop the provider. Called from the FastAPI lifespan on shutdown.

    Without this a :class:`BatchSpanProcessor` can lose whatever is still in its
    queue when the process exits — which is precisely the tail of spans around a
    crash, the ones worth having.
    """
    global _configured, _provider  # noqa: PLW0603 - module-level provider handle

    with _configure_lock:
        if _provider is not None:
            _provider.shutdown()
        _provider = None
        _configured = False


def is_configured() -> bool:
    """Whether :func:`configure_tracing` has installed a provider in this process."""
    return _configured


def get_tracer() -> trace.Tracer:
    """The tracer every span in this app is created from."""
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def current_trace_id() -> str:
    """The active trace id as 32 lowercase hex characters, or :data:`NO_TRACE_ID`.

    This is what gets bound into the log context, and it is the join key between
    the two systems: given a log line, the trace id finds the trace; given a slow
    trace, it finds every log line the request emitted. Returns the sentinel when
    no span is in scope — startup, a CLI script — rather than the all-zero id an
    invalid span context reports, which reads like a real value.
    """
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return NO_TRACE_ID
    return format(context.trace_id, "032x")


@contextmanager
def stage_span(name: str, **attributes: Any) -> Iterator[Span]:
    """Time one pipeline stage as a child span, recording its duration and any error.

    On success the span is closed with an explicit ``duration_ms`` attribute.
    OpenTelemetry already records start and end timestamps, so this is
    redundant to a backend that computes the difference — but it makes a span
    readable *as printed* by :class:`ConsoleSpanExporter`, which is the default
    exporter here and shows two ``time.time_ns()`` integers rather than an
    elapsed time.

    On failure the exception is recorded on the span (type, message, stack) and
    the span's status is set to ``ERROR``, **and then re-raised unchanged**. This
    context manager observes; it never handles. A ``GuardError`` swallowed here
    would turn a 422 into a silently successful request, which is the single
    worst thing an instrumentation layer can do.

    Args:
        name: One of :data:`STAGES`.
        **attributes: Span attributes. ``None`` values are dropped — OTel rejects
            a null attribute value, and an optional field is better absent.

    Yields:
        The active :class:`~opentelemetry.trace.Span`, so a caller can attach
        attributes it only learns partway through the stage.

    Example:
        >>> with stage_span(STAGE_MODEL_INFERENCE, model="patchcore") as span:  # doctest: +SKIP
        ...     output = model.predict(frame)
        ...     span.set_attribute("anomaly_score", output.anomaly_score)
    """
    started = time.perf_counter()
    with get_tracer().start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            span.set_attribute("duration_ms", round((time.perf_counter() - started) * 1000.0, 3))
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise
        span.set_attribute("duration_ms", round((time.perf_counter() - started) * 1000.0, 3))
        span.set_status(Status(StatusCode.OK))


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """Attach ``attributes`` to ``span``, skipping ``None`` values."""
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
