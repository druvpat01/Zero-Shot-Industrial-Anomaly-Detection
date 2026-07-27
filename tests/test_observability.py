"""Tests for the observability layer: metrics, structured logs and request context.

What is actually being tested here
----------------------------------
That the instrumentation *fires*, not that Prometheus and structlog work.
Those libraries have their own suites; what is untested until this file exists
is the wiring — whether a real request through the real app moves the real
counter, and whether the fields a dashboard and an on-call engineer select on
are actually present on the records they select from.

That framing decides the shape of every test below. Each one drives the app
through :class:`~fastapi.testclient.TestClient` and then asserts on the
*observable* output: a counter read back out of the registry, a rendered
``/metrics`` body, a captured log event. None of them reach into the module
internals to check that a function was called, because "was ``.inc()`` called"
is a fact about this code's structure and "did the counter move" is a fact about
what a scrape will return, and only the second one is what production depends
on.

The two counter styles
----------------------
Counters are asserted as a *delta*, never as an absolute. The metric registry is
a process-wide singleton that every other test in the suite also writes to, and
a test that asserts ``images_processed_total == 1`` passes alone and fails the
moment ``tests/test_api.py`` runs first. :func:`_count` reads the current value
(treating a not-yet-created series as zero, exactly as Prometheus's own
``increase()`` does) and each test compares before and after.

Artifact-gated tests
--------------------
The success-path tests need a model on disk and skip when there is none,
matching the convention in ``tests/test_api.py``. The rejection-path tests do
not: a frame that fails the guard is refused before any model is fetched, so
they exercise the full request path — middleware, span, counter, log record —
with nothing installed. That is deliberate coverage rather than a happy
accident, and it is why the counter test that matters most is the one that needs
no checkpoint.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.models.onnx_runner import onnx_artifact_path
from app.observability.logging_config import (
    NO_REQUEST_ID,
    configure_logging,
    get_logger,
    resolve_log_format,
)
from app.observability.metrics import (
    GUARD_LATENCY_BUCKETS,
    INFERENCE_LATENCY_BUCKETS,
    RESULT_REJECTED,
    RESULTS,
    backend_kind,
    get_metric_value,
    record_image_processed,
)
from app.observability.middleware import REQUEST_ID_HEADER, TRACE_ID_HEADER
from app.observability.tracing import NO_TRACE_ID, current_trace_id
from app.serving.auth import API_KEY_HEADER, AuthConfig, get_auth_config
from app.serving.main import app

CATEGORY = "bottle"

#: The ONNX export is used for the success-path tests rather than the PyTorch
#: checkpoint: it is a ~30 ms forward pass against PatchCore's ~150 ms and a
#: 230 MB load, and this file is testing instrumentation, not the model.
BACKEND = "onnx_patchcore"

OPERATOR_KEY = "test-observability-key"

ONNX_PATCHCORE: Path = onnx_artifact_path("patchcore", "fp32")

requires_onnx = pytest.mark.skipif(
    not ONNX_PATCHCORE.is_file(),
    reason=f"{ONNX_PATCHCORE} not found; run `python scripts/export_onnx.py`",
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _png_b64(image: np.ndarray) -> str:
    """PNG-encode a BGR array and base64 it, exactly as an API client would."""
    ok, buffer = cv2.imencode(".png", image)
    assert ok, "failed to encode the test image"
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _sharp_frame() -> np.ndarray:
    """Uniform noise: the highest Laplacian variance available, so it always passes."""
    return np.random.default_rng(1).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def _blurred_frame() -> np.ndarray:
    """Sharp noise put through a heavy Gaussian blur: a fouled or defocused lens.

    Mirrors ``tests/test_api.py``'s fixture of the same name, on purpose — the
    guard is what both files are delegating to, and the observability test should
    be provoked by the same input that provokes the API test.
    """
    noise = np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 20)


def _predict_body(image_b64: str, *, backend: str = BACKEND, category: str = CATEGORY) -> dict[str, str]:
    return {"category": category, "model_backend": backend, "image_b64": image_b64}


def _count(name: str, **labels: str) -> float:
    """Current value of a counter series, with an absent series read as zero.

    A counter's first observation *creates* its series, so before the first
    request ``images_processed_total{...}`` genuinely does not exist and
    :func:`get_metric_value` returns ``None``. Treating that as zero is what
    Prometheus's own ``increase()`` does and it is what makes a before/after
    delta well-defined on a cold registry.
    """
    value = get_metric_value(name, **labels)
    return 0.0 if value is None else value


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client over the real app, authenticated with an injected operator key.

    Used as a context manager so the FastAPI lifespan runs — which is what calls
    :func:`~app.observability.logging_config.configure_logging` and
    :func:`~app.observability.tracing.configure_tracing`. A ``TestClient`` built
    without ``with`` never fires startup, and every test here would then be
    exercising an app that was never instrumented.
    """
    app.dependency_overrides[get_auth_config] = lambda: AuthConfig(operator_keys=(OPERATOR_KEY,))
    with TestClient(app, headers={API_KEY_HEADER: OPERATOR_KEY}) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_auth_config, None)


@pytest.fixture
def restore_logging():
    """Put the logging configuration back after a test has changed it.

    :func:`configure_logging` mutates the root logger, which is process-wide.
    Without this, a test that switches to JSON leaves every later test — in this
    file and every other — rendering JSON, and the failure surfaces somewhere
    unrelated.
    """
    yield
    configure_logging(log_format="console", force=True)


@pytest.fixture
def json_logs(capsys, restore_logging):
    """Capture emitted log records by parsing the rendered JSON off stderr.

    Deliberately *not* :func:`structlog.testing.capture_logs`, which is the
    obvious tool and the wrong one here. It works by swapping the entire
    processor chain for a single capturing processor, so ``merge_contextvars``
    never runs and the captured dicts are missing exactly the fields these tests
    exist to assert on — ``request_id``, ``trace_id``, ``model_backend``,
    ``category``. A test using it would pass while the shipped log records
    carried none of them.

    Parsing rendered output instead tests the whole chain end to end: the
    contextvars really were merged, the renderer really did emit valid JSON, and
    the field names really are the ones a log shipper will index. Configuring
    inside the fixture is what makes it work under pytest's capture — the
    handler binds to the ``sys.stderr`` pytest has already replaced.

    Returns:
        A callable draining everything logged since the last drain, as a list of
        parsed records.
    """
    configure_logging(log_format="json", force=True)
    capsys.readouterr()  # discard anything buffered before the test's own output

    def drain() -> list[dict]:
        records = []
        for line in capsys.readouterr().err.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue  # a stray non-JSON line from a library writing to stderr
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return records

    return drain


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200_text_plain(client: TestClient) -> None:
    """The scrape target answers, in the format a scraper content-negotiates on.

    The content type is asserted as a prefix rather than an equality: the full
    value carries a version parameter (``text/plain; version=1.0.0;
    charset=utf-8``) that prometheus_client owns and has changed across
    releases. Pinning it here would turn a routine dependency bump into a
    failing test about nothing.
    """
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_endpoint_needs_no_credential() -> None:
    """A Prometheus server holds no API key, so ``/metrics`` must not require one.

    Asserted with a bare client carrying no ``X-API-Key`` at all — the fixture
    above sends one on every request, which would hide a regression that
    accidentally gated this endpoint.
    """
    with TestClient(app) as anonymous:
        assert anonymous.get("/metrics").status_code == 200


def test_metrics_body_declares_every_metric(client: TestClient) -> None:
    """All five application metrics are present in the exposition from process start.

    They are registered at import rather than on first use, so a dashboard shows
    a flat zero instead of "no data" before the first request — which is the
    difference between "the service is idle" and "the scrape is broken".
    """
    body = client.get("/metrics").text

    for metric in (
        "images_processed_total",
        "inference_latency_seconds",
        "guard_check_latency_seconds",
        "guard_rejections_total",
        "models_loaded_count",
    ):
        assert f"# HELP {metric}" in body or f"# TYPE {metric}" in body, f"{metric} missing from /metrics"


def test_inference_histogram_uses_the_specified_buckets() -> None:
    """The inference histogram's bucket boundaries are exactly the ones specified.

    Bucket edges are not an implementation detail. They are the resolution limit
    of every percentile a dashboard computes from this histogram, and unlike a
    query they cannot be changed retroactively for data already scraped — a p95
    is only ever as precise as the bucket it lands in.

    Asserted against the module constant rather than against rendered output,
    because a labelled histogram has no series at all until its first
    observation: on a cold registry there is nothing in ``/metrics`` to check.
    """
    assert INFERENCE_LATENCY_BUCKETS == (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def test_guard_histogram_brackets_the_measured_range() -> None:
    """The guard's buckets span the range the guard was actually measured in.

    Measured: 1.3 ms on a 256x256 frame, 23 ms on a real 900x900 MVTec frame —
    the cost is O(H*W), so it tracks the frame rather than being fixed. The
    buckets must therefore resolve *below* the cheapest observed case and extend
    well *above* the most expensive one, since a larger camera is the obvious way
    to exceed it.

    This test is a correction, not a formality. The guard was documented as
    costing "microseconds" and the first buckets were chosen to match, putting
    their resolution between 100 µs and 1 ms where nothing ever landed and
    capping at 100 ms. Pinning both ends here is what stops that drifting back.
    """
    assert min(GUARD_LATENCY_BUCKETS) <= 0.001, "must resolve below the 1.3 ms small-frame case"
    assert max(GUARD_LATENCY_BUCKETS) >= 0.25, "must not dump a large frame into +Inf"
    # Still finer-grained at the bottom than the inference histogram, which is
    # the reason it does not simply reuse those buckets.
    assert min(GUARD_LATENCY_BUCKETS) < min(INFERENCE_LATENCY_BUCKETS)


def test_guard_histogram_buckets_are_rendered(client: TestClient) -> None:
    """The guard histogram's buckets reach the exposition.

    Unlike the inference histogram this one is unlabelled, so its series exists
    from process start and can be asserted against the rendered body directly.
    """
    body = client.get("/metrics").text

    for edge in ("0.001", "0.01", "0.1", "0.5"):
        assert f'guard_check_latency_seconds_bucket{{le="{edge}"}}' in body, f"bucket {edge} missing"


# ---------------------------------------------------------------------------
# images_processed_total
# ---------------------------------------------------------------------------


def test_predict_increments_images_processed_total_on_rejection(client: TestClient) -> None:
    """One ``/predict`` call moves ``images_processed_total`` by exactly one.

    The guard-rejection path is used because it needs no model on disk, so this
    — the counter assertion the spec asks for — runs everywhere rather than
    skipping on a machine without a checkpoint. A rejected frame still consumed
    a request and still reached an outcome, which is precisely why
    ``rejected_by_guard`` is a ``result`` value rather than an absence.
    """
    labels = {"model": BACKEND, "category": CATEGORY, "result": RESULT_REJECTED}
    before = _count("images_processed_total", **labels)

    response = client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))

    assert response.status_code == 422
    assert response.json()["reason"] == "blurry"
    assert _count("images_processed_total", **labels) == before + 1


@requires_onnx
def test_predict_increments_images_processed_total_on_success(client: TestClient) -> None:
    """A scored frame is counted once, under the verdict it received.

    The counter is keyed by the *resolved* model name, so this also pins the
    attribution rule: a frame scored by an ONNX graph is counted against
    ``onnx_patchcore``, never against the backend the caller happened to name.
    """
    response = client.post("/predict", json=_predict_body(_png_b64(_sharp_frame())))
    assert response.status_code == 200, response.json()

    model_name = response.json()["model_name"]
    result = "defective" if response.json()["is_defective"] else "normal"
    labels = {"model": model_name, "category": CATEGORY, "result": result}
    before = _count("images_processed_total", **labels)

    again = client.post("/predict", json=_predict_body(_png_b64(_sharp_frame())))

    assert again.status_code == 200
    assert _count("images_processed_total", **labels) == before + 1


@requires_onnx
def test_predict_observes_inference_latency(client: TestClient) -> None:
    """A scored frame adds one observation to the latency histogram.

    The ``backend`` label is asserted alongside ``model`` because it is derived
    rather than passed — :func:`~app.observability.metrics.backend_kind` reads it
    off the model name — and a derivation that silently stopped working would
    leave every ONNX observation filed under ``pytorch``.
    """
    labels = {"model": BACKEND, "backend": "onnx"}
    before = _count("inference_latency_seconds_count", **labels)

    response = client.post("/predict", json=_predict_body(_png_b64(_sharp_frame())))

    assert response.status_code == 200
    assert _count("inference_latency_seconds_count", **labels) == before + 1


def test_guard_rejection_increments_guard_rejections_total(client: TestClient) -> None:
    """A refused frame is counted against the check that refused it.

    Labelled by ``reason``, which is what makes this a camera-health signal
    rather than a generic error count: a rising ``blurry`` rate is a fouling
    lens, and it is a different operational problem from a rising ``too_small``.
    """
    before = _count("guard_rejections_total", reason="blurry")

    client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))

    assert _count("guard_rejections_total", reason="blurry") == before + 1


def test_guard_check_latency_is_observed(client: TestClient) -> None:
    """The guard times itself on every call, pass or fail.

    The histogram exists to keep this module's central claim honest — that the
    guard costs a small fraction of a forward pass, which is the whole reason it
    runs first — so an unobserved guard would leave that claim unfalsifiable.
    It is not a hypothetical risk: the claim in the docstrings was "microseconds"
    until this histogram measured 1.3-23 ms and the prose was corrected.
    """
    before = _count("guard_check_latency_seconds_count")

    client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))

    assert _count("guard_check_latency_seconds_count") > before


def test_record_image_processed_rejects_an_unknown_result() -> None:
    """A typo'd ``result`` fails loudly instead of creating a permanent dead series.

    Prometheus would accept ``"defectiv"`` and hold that series forever, queried
    by nothing. Failing here costs one test and saves a dashboard that silently
    under-counts.
    """
    with pytest.raises(ValueError, match="Unknown result"):
        record_image_processed(model="patchcore", category=CATEGORY, result="defectiv")

    assert "defectiv" not in RESULTS


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("patchcore", "pytorch"),
        ("efficientad", "pytorch"),
        ("winclip", "pytorch"),
        ("onnx_patchcore", "onnx"),
        ("onnx_efficientad", "onnx"),
    ],
)
def test_backend_kind_classifies_every_backend(model_name: str, expected: str) -> None:
    """Every backend name in the schema classifies to a runtime, none to a default."""
    assert backend_kind(model_name) == expected


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def test_guard_rejection_log_record_contains_reason(client: TestClient, json_logs) -> None:
    """The rejection log record carries ``reason`` — the field the spec asks for.

    The guard's raw metrics are asserted alongside the reason because they are
    the drift-monitoring payload — watching ``laplacian_variance`` sag across a
    shift is how a fouling lens is caught *before* it starts rejecting frames —
    and they are spread as top-level fields rather than nested so a log backend
    can aggregate on them directly.
    """
    response = client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))
    records = json_logs()

    assert response.status_code == 422

    rejections = [record for record in records if record.get("event") == "guard_rejected"]
    assert rejections, f"no guard_rejected record in {[r.get('event') for r in records]}"

    record = rejections[0]
    assert record["source"] == "api"
    assert "reason" in record
    assert record["reason"] == "blurry"
    assert record["level"] == "warning"
    # The guard metrics, flattened onto the record.
    assert "laplacian_variance" in record
    assert record["laplacian_variance"] < 50.0
    # Bound by /predict before the guard ran, so the record is attributable.
    assert record["model_backend"] == BACKEND
    assert record["category"] == CATEGORY


def test_log_records_carry_request_and_trace_ids(client: TestClient, json_logs) -> None:
    """Every record emitted inside a request carries the ids that join the systems.

    ``request_id`` is what a caller quotes when reporting a failure; ``trace_id``
    is what opens the trace for the same request. Both are bound by the
    middleware, so they appear on records the handlers never touched — which is
    the entire reason for binding them in a contextvar rather than passing them
    down as arguments.
    """
    client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))
    records = json_logs()

    inside_request = [record for record in records if record.get("event") == "guard_rejected"]
    assert inside_request

    record = inside_request[0]
    assert record["request_id"] != NO_REQUEST_ID
    assert len(record["trace_id"]) == 32
    int(record["trace_id"], 16)  # raises if it is not hex


def test_request_id_is_echoed_on_the_response(client: TestClient) -> None:
    """The response carries the id, so a caller can quote it without reading logs."""
    response = client.get("/health")

    assert response.headers[REQUEST_ID_HEADER]
    assert response.headers[TRACE_ID_HEADER]


def test_inbound_request_id_is_reused(client: TestClient) -> None:
    """A caller's own id is adopted rather than replaced.

    This is what lets an inspection line correlate its frame number with this
    service's logs without anybody matching timestamps by hand.
    """
    response = client.get("/health", headers={REQUEST_ID_HEADER: "line-7-frame-41823"})

    assert response.headers[REQUEST_ID_HEADER] == "line-7-frame-41823"


def test_inbound_request_id_is_sanitised(client: TestClient) -> None:
    """A newline in an inbound id cannot forge a log record.

    The id is written into every log line the request emits. An unsanitised
    newline in one is a log-injection primitive handed over by the caller, so
    non-printable characters are stripped rather than trusted.
    """
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc\r\ndef"})

    echoed = response.headers[REQUEST_ID_HEADER]
    assert "\n" not in echoed
    assert "\r" not in echoed


def test_records_outside_a_request_get_the_sentinel_id(json_logs) -> None:
    """A record with no request in scope still has a ``request_id`` field.

    A sentinel rather than an absent key, so a log query can select on the field
    without an "or missing" clause — and so the shape of a record does not depend
    on where it was emitted from.
    """
    get_logger(__name__).info("test_event_outside_request")
    records = json_logs()

    record = next(r for r in records if r.get("event") == "test_event_outside_request")
    assert record["request_id"] == NO_REQUEST_ID


def test_json_format_emits_the_four_mandated_fields(json_logs) -> None:
    """``LOG_FORMAT=json`` produces records a log shipper can parse.

    The four fields every record must carry — ``timestamp``, ``level``,
    ``request_id``, and the applicable context — are asserted on the *rendered*
    output, because what matters is the bytes on the stream rather than the
    contents of an intermediate dict.
    """
    get_logger("test").warning("json_render_check", model_backend="patchcore", category="bottle")
    records = json_logs()

    record = next(r for r in records if r.get("event") == "json_render_check")

    assert record["level"] == "warning"
    assert record["model_backend"] == "patchcore"
    assert record["category"] == "bottle"
    assert record["request_id"] == NO_REQUEST_ID
    # UTC with an explicit marker: naive local timestamps are how a log becomes
    # unreadable the first time it is opened on a machine in another timezone.
    assert record["timestamp"].endswith("Z")


def test_stdlib_log_calls_get_the_same_treatment(json_logs) -> None:
    """A plain ``logging`` call renders identically to a structlog one.

    This is the property that made the migration ~200 lines of configuration
    instead of a rewrite of every module: the model and evaluation packages log
    prose through the standard library, and those records pick up the same
    timestamp, level and request context on their way through the shared
    formatter. If this breaks, half the codebase silently stops being
    structured.
    """
    logging.getLogger("test.stdlib.bridge").warning("bridged %s", "message")
    records = json_logs()

    record = next(r for r in records if r.get("event") == "bridged message")
    assert record["level"] == "warning"
    assert record["logger"] == "test.stdlib.bridge"
    assert record["request_id"] == NO_REQUEST_ID
    assert "timestamp" in record


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("json", "json"),
        ("JSON", "json"),
        ("  console  ", "console"),
        ("", "console"),
        ("yaml", "console"),
    ],
)
def test_log_format_falls_back_rather_than_raising(raw: str, expected: str) -> None:
    """An unrecognised ``LOG_FORMAT`` degrades to console instead of failing startup.

    A typo in a deployment's environment must not be the reason a server refuses
    to boot: a server logging in the wrong *format* is still a server that logs.
    """
    assert resolve_log_format(raw) == expected


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def test_current_trace_id_outside_a_span_is_the_sentinel() -> None:
    """No span in scope reports the sentinel, not an all-zero id.

    OpenTelemetry's invalid span context reports a trace id of 32 zeroes, which
    reads like a real value in a log line and would join to nothing.
    """
    assert current_trace_id() == NO_TRACE_ID


@requires_onnx
def test_predict_emits_a_span_for_every_pipeline_stage(client: TestClient) -> None:
    """A scored frame produces the four named stage spans beneath its request span.

    Collected with an in-memory exporter attached to the live provider, so this
    tests the spans the app actually emits rather than spans constructed by the
    test. ``model_load`` is asserted as *optional*: it only appears when the
    backend was cold, and by the time this test runs another test has very
    likely warmed it — asserting on it would make the test order-dependent.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    provider.add_span_processor(processor)
    try:
        response = client.post("/predict", json=_predict_body(_png_b64(_sharp_frame())))
        assert response.status_code == 200
        names = {span.name for span in exporter.get_finished_spans()}
    finally:
        processor.shutdown()

    assert {"preprocess", "guard", "model_inference", "postprocess"} <= names
    assert any(name.startswith("POST /predict") for name in names)


def test_guard_span_records_the_rejection_reason(client: TestClient) -> None:
    """A refused frame's ``guard`` span carries the failing check as an attribute.

    This is what makes "find every rejected frame in the last hour" a trace query
    rather than a log grep.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)
    try:
        client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))
        guard_spans = [span for span in exporter.get_finished_spans() if span.name == "guard"]
    finally:
        processor.shutdown()

    assert guard_spans
    attributes = guard_spans[-1].attributes
    assert attributes["guard.passed"] is False
    assert attributes["guard.reason"] == "blurry"


def test_metrics_endpoint_is_not_traced(client: TestClient) -> None:
    """A scrape produces no span.

    Prometheus hits ``/metrics`` every 15 seconds forever. A trace backend full
    of identical scrape spans is one nobody opens, so the path is excluded — but
    it still gets a request id, because a *failing* scrape is worth finding.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)
    try:
        response = client.get("/metrics")
        spans = exporter.get_finished_spans()
    finally:
        processor.shutdown()

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]
    assert not [span for span in spans if "metrics" in span.name]


# ---------------------------------------------------------------------------
# models_loaded_count
# ---------------------------------------------------------------------------


@requires_onnx
def test_models_loaded_gauge_tracks_the_registry(client: TestClient) -> None:
    """The gauge reports what is actually resident, not a running inc/dec tally.

    Set to an absolute value from inside the registry's two size-changing
    methods, so it cannot drift the way a paired increment/decrement would the
    first time a load raised between them.
    """
    from app.serving.model_registry import get_registry

    response = client.post("/predict", json=_predict_body(_png_b64(_sharp_frame())))
    assert response.status_code == 200

    assert get_metric_value("models_loaded_count") == float(len(get_registry().loaded_keys()))
