"""The FastAPI application: six endpoints over everything the rest of the app builds.

Run it with::

    uvicorn app.serving.main:app --host 0.0.0.0 --port 8000    # or: make serve

and read the generated contract at ``/docs``.

What this module is, and is not
===============================
It is the wiring. Every hard part lives somewhere else — quality checks in
:mod:`app.guardrails`, scoring in :mod:`app.models`, metrics, drift and
calibration in :mod:`app.evaluation`, base64 plumbing in
:mod:`app.serving.imaging`, model lifetime in :mod:`app.serving.model_registry` —
and the handlers below are short functions that call them in the right order and
translate failures into status codes. That is on purpose: the interesting
decisions in a serving layer are about *sequencing and failure*, and those are
hard to see in a route handler that also resizes images.

The request path, and why it is ordered this way
===============================================
``POST /predict`` does four things, and the order is the design:

1. **Decode**, and refuse anything that is not an image (422). Cheapest check,
   and it needs no model.
2. **Guard**, and refuse a frame not worth scoring (422). Cheap relative to a
   forward pass, and, critically, *before* the model is fetched. A blurry frame
   submitted against a backend with no checkpoint returns "blurry", not
   "model_not_ready": the caller's problem is reported ahead of the server's,
   and a cold 30-second model load is never paid for a frame that was going to
   be rejected anyway.
3. **Fetch the model** from the registry, loading it if this is the first
   request for it (503 if no artifact can serve it).
4. **Score**, render the heatmap, and answer.

Step 2 is a deliberate duplicate: every ``AnomalyModel.predict`` runs the same
guard internally, so the frame is validated twice. The second pass is not free —
measured at 1.3 ms on a 256x256 frame and 23 ms on a full-resolution 900x900 one,
because the guard's cost scales with pixel count — but against a forward pass of
several hundred milliseconds it is worth paying. Running the guard *here* is what
lets the API return a structured 422 with the failing reason and short-circuit
before the registry is touched, while the model's own call keeps the guarantee
true for every caller of the model layer, not just this one.

Errors
======
Every failure gets an exception handler, and every one returns a small JSON
object with a machine-readable ``detail`` slug rather than a Python exception:

===============================  ======  ====================================================
Condition                        Status  Body
===============================  ======  ====================================================
Undecodable ``image_b64``        422     ``{"detail": "invalid_image", "reason": ...}``
Frame fails the quality guard    422     ``{"detail": "guard_failed", "reason": "blurry"}``
Schema/validation failure        422     ``{"detail": "invalid_request", "errors": [...]}``
Unusable calibration set         422     ``{"detail": "invalid_calibration_set", "reason": ...}``
Threshold outside ``[0, 1]``     422     ``{"detail": "threshold_out_of_range", ...}``
No artifact for the backend      503     ``{"detail": "model_not_ready", "backend": ...}``
Anything else                    500     ``{"detail": "internal_error"}``
===============================  ======  ====================================================

The 500 is the one that matters for security: the traceback goes to the server
log via ``log.error(..., exc_info=True)`` and *only* there. Leaking it to the client would
hand out filesystem paths, library versions and enough stack detail to fingerprint
the deployment. The validation handler is overridden for a related reason —
pydantic's default error payload echoes the offending input back, which for a
multi-megabyte base64 frame is a response nobody wants and a log line nobody can
read.

Sync handlers, on purpose
=========================
The handlers are ``def``, not ``async def``. Inference is CPU-bound blocking work
(a PyTorch forward pass holds the GIL for tens to hundreds of milliseconds), and
an ``async def`` handler runs *on the event loop*, so one in-flight prediction
would stall every other connection including the health check. Declaring them
sync hands them to Starlette's thread pool, where blocking is exactly what is
expected. This is also why the registry's cache is locked.

Who may call what
=================
Every endpoint but ``/health`` is gated by an API key (:mod:`app.serving.auth`),
and the split follows what a call *costs* and what it *changes* rather than only
what it reveals:

===============  ============  =======================================================
Endpoint         Role          Why
===============  ============  =======================================================
``/health``      *(none)*      A liveness probe cannot hold a credential. Answers
                               nothing an anonymous caller could not learn by
                               observing that the port is open.
``/predict``     ``viewer``    The line's own traffic. One frame, bounded cost.
``/models``      ``operator``  Enumerates artifacts and filesystem paths.
``/drift``       ``operator``  Reports the score distribution over the customer's
                               production traffic — the same class of disclosure
                               that gates ``/benchmark``, at a fraction of the cost.
``/calibrate``   ``operator``  Changes how the service grades parts, for every
                               caller, until restart. The only *write* in the API.
``/benchmark``   ``operator``  Minutes of CPU per call, and returns metrics over
                               the customer's test split.
===============  ============  =======================================================

``/benchmark`` and ``/calibrate`` additionally write to the audit trail
(:mod:`app.observability.audit_log`) — a separate append-only file recording who
ran what and what they got, which survives the log-level filtering an application
log is subject to. They are audited for opposite reasons: the benchmark because
of what the caller *learns*, the calibration because of what it *changes*.
``docs/security.md`` has the reasoning, and is honest about what this buys and
what it does not.

Reliability, and why ``/drift`` and ``/calibrate`` are a pair
============================================================
Everything above assumes the model's scores still mean what they meant when its
threshold was chosen. :mod:`app.evaluation.drift` is what checks that assumption:
``/predict`` feeds every score into a rolling window per ``(model, category)``,
and ``/drift`` compares that window against a reference distribution with a
two-sample KS test. When it fires, the cheapest fix is usually not a retrain but
a new operating point — which is ``/calibrate``, and which also re-installs the
reference the next drift check runs against. One endpoint notices the problem,
the other resolves the common case, and they share the calibration set.

What a request emits
====================
Everything this service reports about itself is set up here and produced on the
path below. One request produces:

* **Log records**, structured, every one carrying ``request_id`` and
  ``trace_id`` from :class:`~app.observability.middleware.ObservabilityMiddleware`
  and — once ``/predict`` knows them — ``model_backend`` and ``category``.
* **A trace**: a root span per request with ``preprocess``, ``guard``,
  ``model_load``, ``model_inference`` and ``postprocess`` beneath it.
* **Metrics**: a counter increment and a latency observation, scraped from
  ``GET /metrics``.

The three are joined on the same words. A spike on a Grafana panel filters to a
``model`` and a ``category``; those are the same two field names on the log
records; those records carry the ``trace_id`` that opens the trace. That
correspondence is deliberate and it is the only reason three systems are less
work than one.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.data import DataModule
from app.evaluation import BenchmarkRunner, evaluate_threshold, find_optimal_threshold
from app.guardrails import GuardError, guard
from app.models.base import AnomalyModel
from app.observability.audit_log import OUTCOME_OK, record_benchmark, record_calibration
from app.observability.logging_config import bind_log_context, configure_logging, get_logger
from app.observability.metrics import (
    CONTENT_TYPE_LATEST,
    RESULT_DEFECTIVE,
    RESULT_NORMAL,
    RESULT_REJECTED,
    observe_inference,
    record_image_processed,
    render_latest,
)
from app.observability.middleware import ObservabilityMiddleware
from app.observability.tracing import (
    STAGE_GUARD,
    STAGE_MODEL_INFERENCE,
    STAGE_MODEL_LOAD,
    STAGE_POSTPROCESS,
    STAGE_PREPROCESS,
    configure_tracing,
    shutdown_tracing,
    stage_span,
)
from app.serving.auth import Principal, require_operator, require_viewer
from app.serving.imaging import InvalidImageError, decode_image_b64, encode_heatmap_png_b64
from app.serving.model_registry import (
    ModelNotReadyError,
    ModelRegistry,
    ThresholdOutOfRangeError,
    get_registry,
)
from app.serving.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    CalibrationRequest,
    CalibrationResponse,
    DriftStatus,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
)

__all__ = ["DatasetNotAvailableError", "InvalidCalibrationSetError", "app"]

log = get_logger(__name__)

#: A model load slower than this gets its own log line. Not an error — a cold
#: WinCLIP legitimately takes tens of seconds — but it is the single best
#: explanation for a request that looked inexplicably slow end to end, and
#: ``latency_ms`` deliberately excludes it.
_SLOW_LOAD_SECONDS = 1.0

#: Scoring slower than this is logged at WARNING with the measured duration.
#: Chosen against what the backends actually do on CPU rather than picked round:
#: an ONNX graph runs in ~30 ms and PatchCore in ~150 ms, so half a second is
#: comfortably outside normal for those two and something to look at. WinCLIP
#: exceeds it on every request by design — it is the zero-shot backend, not the
#: fast one — which is exactly why the warning names the model, and why this is a
#: WARNING and not an alert.
_SLOW_INFERENCE_SECONDS = 0.5

#: Starlette renamed ``HTTP_422_UNPROCESSABLE_ENTITY`` to ``..._CONTENT`` (RFC
#: 9110's wording) and deprecated the old spelling, which would otherwise emit a
#: warning on *every* rejected request. Resolved once, here, so the app is
#: warning-free on either version; the short-circuit means the deprecated name is
#: only touched on installs where the new one does not exist.
_HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or status.HTTP_422_UNPROCESSABLE_ENTITY


class DatasetNotAvailableError(RuntimeError):
    """Raised when ``POST /benchmark`` is asked for a category with no data on disk.

    Same shape of problem as :class:`~app.serving.model_registry.ModelNotReadyError`
    and the same 503: the request is fine, the server is missing an artifact, and
    running the download script fixes it.
    """

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(detail)


class InvalidCalibrationSetError(ValueError):
    """Raised when ``POST /calibrate`` is handed data no threshold can be fitted to.

    The schema already rejects a malformed *request* — wrong types, a label
    outside ``{0, 1}``, fewer than two samples. What it cannot check is whether
    the numbers form a usable calibration set: whether both classes are present,
    whether every score is finite. Those are semantic, they are only knowable by
    looking at the whole list, and :func:`app.evaluation.calibration.find_optimal_threshold`
    raises :class:`ValueError` for each with a message written to be read by an
    operator. This wrapper carries that message to a 422 instead of letting it
    become a 500.

    422 rather than 400 to match every other rejected body in this API, and
    ``reason`` carries the explanation because the caller can act on it — unlike
    the internal errors, whose detail is deliberately withheld.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure observability once, before the first request, and flush on the way out.

    This is the *only* place :func:`configure_logging` and
    :func:`configure_tracing` are called on the serving path. Both are
    idempotent, so a test importing the app after a script already configured
    logging gets a no-op rather than a second handler on the root logger — but
    keeping the call in one place is what makes "when does logging start being
    structured" a question with an answer.

    Startup only touches configuration: no model is loaded here, and that is the
    contract the whole lazy-loading design in
    :mod:`app.serving.model_registry` rests on. A container orchestrator gets a
    ready signal in milliseconds instead of waiting out a 30-second CLIP load and
    killing the pod for failing its probe.

    Shutdown flushes the tracer. A :class:`~opentelemetry.sdk.trace.export.BatchSpanProcessor`
    holds spans in a queue, and without an explicit shutdown the last batch is
    lost on exit — which is precisely the tail of spans around a crash, the ones
    most worth having.
    """
    log_format = configure_logging()
    exporter = configure_tracing()
    log.info(
        "service_starting",
        service="defect-detection",
        version="0.1.0",
        log_format=log_format,
        trace_exporter=exporter,
    )
    try:
        yield
    finally:
        log.info("service_stopping")
        shutdown_tracing()


app = FastAPI(
    title="Zero-Shot Industrial Defect Detection",
    version="0.1.0",
    summary="Pixel-level anomaly detection and segmentation for industrial QA.",
    description=(
        "Scores inspection frames for defects and returns a pixel-level heatmap, "
        "using PatchCore, EfficientAD or zero-shot WinCLIP — the PyTorch wrappers "
        "or their exported ONNX graphs. Models load lazily on first use, so "
        "`/health` answers immediately at startup.\n\n"
        "Authenticate with an `X-API-Key` header. `/predict` needs a **viewer** key; "
        "`/models` and `/benchmark` need an **operator** key; `/health` and `/metrics` "
        "are open so liveness probes and Prometheus scrapes work."
    ),
    lifespan=lifespan,
)

#: Outermost application middleware: every request gets an id, a root span and a
#: log context before any handler or exception handler runs. Registered here
#: rather than per-route because its whole value is that nothing escapes it —
#: including the 404s and 500s that no route produced.
app.add_middleware(ObservabilityMiddleware)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(InvalidImageError)
def _handle_invalid_image(request: Request, exc: InvalidImageError) -> JSONResponse:
    """422: ``image_b64`` was not a decodable image."""
    log.info("request_rejected", detail="invalid_image", path=request.url.path, reason=exc.reason)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "invalid_image", "reason": exc.reason},
    )


@app.exception_handler(GuardError)
def _handle_guard_error(request: Request, exc: GuardError) -> JSONResponse:
    """422: the frame decoded fine but is not worth scoring.

    Translation only — the ``guard_rejected`` record is emitted by the caller
    that raised, not here, and that split is worth explaining because the obvious
    arrangement is the broken one.

    This app's handlers are synchronous, so Starlette runs them in a thread pool.
    A :class:`~contextvars.ContextVar` bound *inside* that worker — which is
    where ``/predict`` binds ``model_backend`` and ``category`` — propagates
    downward but not back out: by the time an exception has unwound to this
    handler, execution is on the event loop again and those bindings are gone. A
    log line written here would therefore be missing exactly the two fields that
    make a rejection attributable, and would be missing them *silently*.

    So the rejection is logged where the context is live: in :func:`predict` for
    the API path, in :meth:`app.models.base.AnomalyModel._check_frame` for every
    other caller of the model layer. Between them every raise site is covered,
    and no frame is logged twice.
    """
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "guard_failed", "reason": exc.reason},
    )


@app.exception_handler(ModelNotReadyError)
def _handle_model_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    """503: the backend has no artifact that can serve this category."""
    log.error(
        "request_rejected",
        detail="model_not_ready",
        path=request.url.path,
        backend=exc.backend,
        category=exc.category,
        reason=exc.detail,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "model_not_ready", "backend": exc.backend, "reason": exc.detail},
    )


@app.exception_handler(DatasetNotAvailableError)
def _handle_dataset_not_available(request: Request, exc: DatasetNotAvailableError) -> JSONResponse:
    """503: the benchmark was asked for a category whose test split is not present."""
    log.error(
        "request_rejected",
        detail="dataset_not_available",
        path=request.url.path,
        category=exc.category,
        reason=exc.detail,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "dataset_not_available", "category": exc.category, "reason": exc.detail},
    )


@app.exception_handler(InvalidCalibrationSetError)
def _handle_invalid_calibration_set(request: Request, exc: InvalidCalibrationSetError) -> JSONResponse:
    """422: the calibration set is well-formed JSON but cannot produce a threshold."""
    log.warning("request_rejected", detail="invalid_calibration_set", path=request.url.path, reason=exc.reason)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "invalid_calibration_set", "reason": exc.reason},
    )


@app.exception_handler(ThresholdOutOfRangeError)
def _handle_threshold_out_of_range(request: Request, exc: ThresholdOutOfRangeError) -> JSONResponse:
    """422: the fitted threshold does not fit on the model's score scale.

    Logged at WARNING rather than ERROR: nothing is broken. An uncalibrated
    backend emitting raw distances is a documented state, and asking it for an
    operating point on the ``[0, 1]`` scale is a request that cannot be granted
    rather than a server that has failed.
    """
    log.warning(
        "request_rejected",
        detail="threshold_out_of_range",
        path=request.url.path,
        backend=exc.backend,
        category=exc.category,
        threshold=exc.threshold,
        reason=exc.detail,
    )
    return JSONResponse(
        status_code=_HTTP_422,
        content={
            "detail": "threshold_out_of_range",
            "backend": exc.backend,
            "threshold": exc.threshold,
            "reason": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422: the body did not match the schema — reported without echoing it back.

    FastAPI's default handler serialises ``exc.errors()`` verbatim, and each
    entry carries an ``input`` key holding the value that failed. For
    ``image_b64`` that is the entire submitted frame, reflected into the error
    response and every log line that touches it. This keeps the location, the
    message and the error type — everything a caller needs to fix their request
    — and drops the payload.
    """
    errors = [
        {"loc": [str(part) for part in error.get("loc", ())], "msg": error.get("msg", ""), "type": error.get("type", "")}
        for error in exc.errors()
    ]
    log.info("request_rejected", detail="invalid_request", path=request.url.path, errors=errors)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "invalid_request", "errors": errors},
    )


@app.exception_handler(Exception)
def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """500: anything not accounted for above.

    The traceback goes to the log and nowhere else — see the module docstring.
    The client gets a slug and nothing to fingerprint the deployment with. It
    also gets an ``X-Request-Id`` header from the middleware, which is the whole
    point of the id: a caller reporting "my request failed" hands over an opaque
    string that finds the traceback, without the traceback ever being sent to
    them.
    """
    log.error(
        "request_failed",
        detail="internal_error",
        path=request.url.path,
        error=type(exc).__name__,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal_error"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    """Liveness probe. Loads nothing, touches no disk, needs no credential.

    This is the contract that makes lazy loading work: a container orchestrator
    can get a ready signal in milliseconds after startup, instead of waiting out
    a model load and killing the pod for failing its probe. ``models_loaded``
    reports what happens to be resident *now*, which is empty until the first
    ``/predict``.

    Unauthenticated on purpose, and it is the *only* such endpoint. A kubelet
    probe cannot hold an API key, and a health check that can fail closed on a
    credential problem is a health check that will eventually restart a perfectly
    healthy pod. What it discloses is bounded to match: liveness and the names of
    resident models, which is strictly less than an anonymous caller learns from
    the port being open at all.
    """
    return HealthResponse(status="ok", models_loaded=registry.loaded_keys())


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape target. Unauthenticated, like ``/health``.

    Returns the text exposition format described in
    :mod:`app.observability.metrics`, with the content type prometheus_client
    declares — the version parameter in it is content-negotiated by the scraper,
    so it is re-exported rather than hardcoded.

    **Open on purpose, and worth being explicit about what that discloses.** A
    Prometheus server scrapes on a schedule and holds no credential; giving it
    one means a secret in the scrape config of every environment, which is a
    worse problem than the one it solves. What an anonymous caller learns from
    this endpoint is real but bounded: request volume, the defect rate, which
    categories and backends are configured, and the process's memory. That is
    strictly more than ``/health`` gives away and strictly less than ``/models``,
    which is why ``/models`` is gated and this is not. In a deployment the
    correct control is a network one — bind the metrics port to the cluster's
    internal network, or scrape through a sidecar — and ``docs/security.md``
    says so rather than pretending the endpoint is harmless.

    Excluded from the OpenAPI schema: it is not part of the API's contract with
    an inspection line, and its response is not JSON.
    """
    return Response(content=render_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", response_model=list[ModelInfo], tags=["ops"])
def list_models(
    category: str | None = Query(
        default=None,
        description="Category to report availability for. Defaults to the configured DEFAULT_CATEGORY.",
        examples=["bottle"],
    ),
    principal: Principal = Depends(require_operator),
    registry: ModelRegistry = Depends(get_registry),
) -> list[ModelInfo]:
    """List every backend and whether it can serve ``category`` right now.

    Availability is answered from the filesystem — is there a checkpoint, is
    there an exported graph — so this endpoint stays as cheap as ``/health`` and
    can be polled. ``detail`` explains the interesting cases: which file is
    missing, or that a backend would be served by an ONNX fallback rather than
    the checkpoint the caller might assume.

    **Operator only**, despite being cheap. Cost is not the reason this one is
    gated: ``detail`` and ``artifact`` name absolute filesystem paths and the
    exact set of trained categories, which is a free map of the deployment for
    anyone probing it. That is an operator's business, not a viewer's.
    """
    resolved = category or registry.config.category
    log.debug("models_listed", category=resolved, caller=principal.key_id, role=principal.role)
    return [ModelInfo(**row) for row in registry.describe(resolved)]


@app.post("/predict", response_model=InferenceResponse, tags=["inference"])
def predict(
    request: InferenceRequest,
    principal: Principal = Depends(require_viewer),
    registry: ModelRegistry = Depends(get_registry),
) -> InferenceResponse:
    """Score one frame and return its anomaly score plus a pixel-level heatmap.

    See the module docstring for why the steps happen in this order. In short:
    the cheap rejections (undecodable payload, unusable frame) come first and
    need no model, so a bad request never triggers a model load.

    **Viewer role.** This is the endpoint the inspection line itself calls, and a
    line operator's key must open it. One known consequence: the first request
    for a cold backend pays that backend's load, so a viewer can indirectly cause
    a multi-second, multi-hundred-megabyte model load even though loading is
    nominally an operator concern. Gating it would mean a viewer key that only
    works once somebody else has warmed the process, which is worse. The mismatch
    is real and ``docs/security.md`` names it rather than papering over it.

    Args:
        request: The frame and the backend that should score it.

    Returns:
        The score, the defect verdict, a base64 PNG heatmap at the submitted
        frame's resolution, and the server-side latency.

    Raises:
        InvalidImageError: 422 — ``image_b64`` is not a decodable image.
        GuardError: 422 — the frame failed a quality check.
        ModelNotReadyError: 503 — no artifact can serve this backend/category.
    """
    started = time.perf_counter()

    # Bound before anything can fail, so every record emitted from here to the
    # end of this function is attributable to a model and a category. These are
    # the same two words as the `model` and `category` labels in
    # app.observability.metrics, on purpose: a spike on a dashboard and the log
    # lines explaining it are then selected with the same query terms.
    #
    # The binding reaches everything this handler calls, but *not* the exception
    # handlers above — this runs in Starlette's thread pool and a contextvar set
    # here does not survive the unwind back to the event loop. That is why the
    # rejection below is logged here rather than there.
    bind_log_context(model_backend=request.model_backend, category=request.category)

    with stage_span(STAGE_PREPROCESS, payload_chars=len(request.image_b64)) as span:
        frame = decode_image_b64(request.image_b64)
        span.set_attribute("frame.height", int(frame.shape[0]))
        span.set_attribute("frame.width", int(frame.shape[1]))

    with stage_span(STAGE_GUARD) as span:
        # FrameGuard times itself into guard_check_latency_seconds and counts its
        # own rejections — see app.guardrails.quality. This span adds the
        # per-request view of the same event, and the reason as an attribute so a
        # trace search can find every rejected frame.
        verdict = guard.validate(frame)
        span.set_attribute("guard.passed", verdict.passed)
        if verdict.reason is not None:
            span.set_attribute("guard.reason", verdict.reason)

    if not verdict.passed:
        # Both the count and the log line happen here rather than in the
        # exception handler, and for the same reason: this is the last point that
        # still has the request's log context and both labels. See
        # `_handle_guard_error` for why the handler cannot have them.
        #
        # The requested backend is used, not a resolved model name: no model ran,
        # and inventing one would put a fictitious row in the model breakdown.
        record_image_processed(
            model=request.model_backend,
            category=request.category,
            result=RESULT_REJECTED,
        )
        # The guard's metrics are spread as top-level fields rather than nested
        # under a `metrics` key, so a log backend can aggregate and alert on
        # `laplacian_variance` as a number instead of parsing it out of a blob.
        # They are drift-monitoring telemetry and stay server-side: watching that
        # value sag across a shift is how a fouling lens is caught *before* it
        # starts rejecting frames, which is not something the caller of a single
        # request can act on beyond the `reason` they are already given.
        log.warning("guard_rejected", reason=verdict.reason, source="api", **verdict.metrics)
        raise GuardError(verdict)

    # Timed separately and subtracted below: a cold load is a one-off startup
    # cost, and folding it into the reported latency would put a 30-second
    # outlier in the same series as the 150 ms steady state, ruining every
    # percentile computed from it. It gets its own span for the same reason.
    load_started = time.perf_counter()
    with stage_span(STAGE_MODEL_LOAD, backend=request.model_backend, category=request.category) as span:
        model = registry.get_model(request.model_backend, request.category)
        span.set_attribute("model_name", model.model_name)
        # False on all but the first request for a backend. Recorded so a trace
        # showing an anomalous duration can be dismissed in one glance.
        span.set_attribute("cold_load", time.perf_counter() - load_started > _SLOW_LOAD_SECONDS)
    load_elapsed = time.perf_counter() - load_started
    if load_elapsed > _SLOW_LOAD_SECONDS:
        log.info(
            "model_cold_loaded",
            backend=request.model_backend,
            model_name=model.model_name,
            duration_seconds=round(load_elapsed, 3),
            note="excluded from the reported latency",
        )

    with stage_span(STAGE_MODEL_INFERENCE, model_name=model.model_name) as span:
        inference_started = time.perf_counter()
        # BGR because decode_image_b64 hands back OpenCV's channel order. Passing
        # the default "rgb" here would swap the channels under an ImageNet- or
        # CLIP-pretrained backbone: no error, no shape change, quietly worse scores.
        output = model.predict(frame, color_order="bgr")
        inference_elapsed = time.perf_counter() - inference_started
        span.set_attribute("anomaly_score", float(output.anomaly_score))
        span.set_attribute("is_defective", bool(output.is_defective))

    # The forward pass alone, excluding decode and heatmap encoding — the two
    # ends of the request that scale with the payload rather than with the model.
    # A histogram that mixed them could not answer "did the model get slower",
    # which is the only question it is asked.
    observe_inference(model=output.model_name, seconds=inference_elapsed)

    if inference_elapsed > _SLOW_INFERENCE_SECONDS:
        log.warning(
            "slow_inference",
            model_name=output.model_name,
            duration_seconds=round(inference_elapsed, 3),
            threshold_seconds=_SLOW_INFERENCE_SECONDS,
            frame_height=int(frame.shape[0]),
            frame_width=int(frame.shape[1]),
        )

    with stage_span(STAGE_POSTPROCESS, calibrated=model.is_calibrated):
        heatmap_b64 = encode_heatmap_png_b64(output.anomaly_map, calibrated=model.is_calibrated)

    latency_ms = (time.perf_counter() - started - load_elapsed) * 1000.0

    # Attributed to the *resolved* model name, so an ONNX fallback is counted as
    # `onnx_patchcore`. The requested backend is what the caller asked for; this
    # is what actually scored the frame, and a defect rate attributed to the
    # wrong one is worse than no defect rate.
    record_image_processed(
        model=output.model_name,
        category=request.category,
        result=RESULT_DEFECTIVE if output.is_defective else RESULT_NORMAL,
    )

    # Into the rolling window this model's drift is judged from. Keyed by the
    # resolved model name for the same reason the counter above is: an ONNX graph
    # scores a little differently from the checkpoint it was exported from, and a
    # window that pooled the two would report that difference as drift.
    #
    # Deliberately *after* the verdict and outside the timed section: it is an
    # O(1) deque append behind an uncontended lock, but it is observability, and
    # nothing observational belongs inside the number a latency SLO is read from.
    # It is also unconditional — a monitor that only ran when someone had
    # configured a reference would have no history the first time anyone asked.
    registry.monitor_for(output.model_name, request.category).record_score(float(output.anomaly_score))

    # The caller's hashed identity, not their key — see app.serving.auth.
    # /predict is high-volume, so it gets a log line rather than an audit entry;
    # the audit trail is reserved for the expensive, privacy-relevant call.
    log.info(
        "frame_scored",
        model_name=output.model_name,
        caller=principal.key_id,
        role=principal.role,
        anomaly_score=round(float(output.anomaly_score), 4),
        is_defective=bool(output.is_defective),
        latency_ms=round(latency_ms, 2),
        inference_seconds=round(inference_elapsed, 4),
    )

    return InferenceResponse(
        anomaly_score=output.anomaly_score,
        is_defective=output.is_defective,
        # From the output, not from the request: an ONNX fallback reports what
        # actually scored the frame, so a caller can explain a shift in numbers.
        model_name=output.model_name,
        anomaly_map_b64=heatmap_b64,
        latency_ms=latency_ms,
        guard_passed=True,
        guard_reason=None,
    )


@app.get("/drift", response_model=list[DriftStatus], tags=["evaluation"])
def drift(
    principal: Principal = Depends(require_operator),
    registry: ModelRegistry = Depends(get_registry),
) -> list[DriftStatus]:
    """Report every active monitor's score distribution and its drift verdict.

    One row per ``(model, category)`` that has scored at least one frame since
    startup, sorted so two consecutive polls are diffable. Each row carries the
    KS verdict *and* the window's percentiles, and the two are meant to be read
    together: the p-value says whether the distribution moved, the percentiles
    say by how much, and only the second answers whether anybody should care.
    :mod:`app.evaluation.drift` argues this at length, along with what a
    production system does when a row comes back ``drifted``.

    ``drifted: false, p_value: null`` is the common state of a fresh process and
    is **not** a clean bill of health — it means no verdict was available,
    because nothing has established what normal looks like. ``POST /calibrate``
    does that.

    As cheap as ``/health``: a KS test over at most a few hundred floats plus
    five percentiles, per monitor. Nothing is loaded, no frame is scored, and it
    is safe to poll on a dashboard refresh.

    **Operator only.** Not for cost — this is one of the cheapest endpoints in
    the service — but for disclosure. The summary describes the distribution of
    anomaly scores over the customer's *production* traffic: how often parts come
    close to the threshold, how heavy the defect tail is, and, tracked over time,
    when a line's quality changed. That is the same class of information that
    gates ``/benchmark``, arrived at from live frames rather than a test split.
    """
    statuses: list[DriftStatus] = []
    for (model_name, category), monitor in sorted(registry.monitors().items()):
        drifted, reason = monitor.is_drifted()
        # Two KS computations per monitor (this and the one inside is_drifted),
        # each microseconds over a bounded window. Caching the p-value on the
        # monitor would make it stateful to save nothing measurable.
        p_value = monitor.ks_p_value()
        summary = monitor.get_summary()

        if drifted:
            # WARNING and not INFO: this is the line that should reach an alert
            # rule. It carries the summary spread as top-level fields for the
            # same reason the guard's metrics are — `p50` as a number is
            # something a log backend can graph and threshold, whereas the same
            # value inside a nested object is a string somebody has to parse.
            log.warning(
                "drift_detected",
                model_name=model_name,
                category=category,
                reason=reason,
                p_value=round(p_value, 6) if p_value is not None else None,
                ks_threshold=monitor.threshold,
                reference_size=monitor.reference_size,
                **{key: value for key, value in summary.items() if value is not None},
            )

        statuses.append(
            DriftStatus(
                model_name=model_name,
                category=category,
                drifted=drifted,
                reason=reason,
                p_value=p_value,
                threshold=monitor.threshold,
                window_size=monitor.window_size,
                reference_size=monitor.reference_size,
                min_samples=monitor.min_samples,
                summary=summary,
            ),
        )

    log.info(
        "drift_reported",
        monitors=len(statuses),
        drifted=sum(status.drifted for status in statuses),
        caller=principal.key_id,
        role=principal.role,
    )
    return statuses


@app.post("/calibrate", response_model=CalibrationResponse, tags=["evaluation"])
def calibrate(
    request: CalibrationRequest,
    principal: Principal = Depends(require_operator),
    registry: ModelRegistry = Depends(get_registry),
) -> CalibrationResponse:
    """Fit a decision threshold to a labelled score set and apply it in memory.

    The *operating point*: the score at or above which this service calls a part
    defective. It ships as ``ANOMALY_THRESHOLD=0.5``, which is a guess;
    :func:`~app.evaluation.calibration.find_optimal_threshold` replaces it with a
    number fitted to real labelled scores, and this endpoint installs it without
    a redeploy. See :mod:`app.evaluation.calibration` for which metric to
    maximise and why F1 is the default.

    **The calibration set must be held out.** Scores from images the model was
    fitted on produce a threshold that looks excellent here and generalises
    poorly, and nothing in this endpoint can detect that — ``metric_value`` is
    computed on the submitted data, so it is an upper bound on production
    performance rather than an estimate of it.

    By default this also installs the submitted scores as the drift monitor's
    reference distribution, because the calibration set *is* the new definition
    of normal: the threshold and the baseline the next drift check runs against
    should move together, and letting them diverge is how a service ends up
    alerting against a distribution nobody chose. Pass
    ``set_drift_reference: false`` to fit the threshold alone.

    **In memory only.** The change is lost on restart, and applies to this
    process — under ``uvicorn --workers N`` it applies to the *one* worker that
    served the request, which is a real limitation and the reason a threshold
    worth keeping belongs in ``ANOMALY_THRESHOLD``. It does survive a model
    eviction and reload within the process; see
    :meth:`~app.serving.model_registry.ModelRegistry.set_threshold`.

    **Operator role, and audited.** This is the only endpoint that changes how
    the service grades parts, for every subsequent caller, and a change that
    silently moves a scrap rate needs to be attributable long after the
    application log has rotated. Every call — successful or not — appends an
    entry to ``results/audit.jsonl``.

    Args:
        request: The labelled scores, the model they came from, and the metric to
            maximise.

    Returns:
        The new threshold, the one it replaced, and what it achieves on the
        submitted set.

    Raises:
        InvalidCalibrationSetError: 422 — the samples cannot produce a
            threshold (one class only, a non-finite score).
        ThresholdOutOfRangeError: 422 — the fitted value is outside ``[0, 1]``,
            which means the model is uncalibrated and emits raw distances.
        ModelNotReadyError: 503 — no artifact can serve this backend/category.
    """
    started = time.perf_counter()
    bind_log_context(model_backend=request.model_backend, category=request.category)

    try:
        response = _run_calibration(request, registry)
    except Exception as exc:
        # Audited before the exception handler turns this into a 422 or a 503. A
        # rejected calibration is an operator trying to move the threshold and
        # being refused, which is exactly as worth knowing as a successful one —
        # repeated failures here are somebody working from a bad calibration set.
        record_calibration(
            caller=principal.key_id,
            role=principal.role,
            category=request.category,
            model_name=request.model_backend,
            duration_seconds=time.perf_counter() - started,
            metrics={"samples": len(request.samples), "metric": request.metric},
            outcome=f"failed:{type(exc).__name__}",
        )
        raise

    record_calibration(
        caller=principal.key_id,
        role=principal.role,
        category=request.category,
        model_name=response.model_name,
        duration_seconds=time.perf_counter() - started,
        metrics={
            "metric": response.metric,
            "threshold": response.threshold,
            "previous_threshold": response.previous_threshold,
            "metric_value": response.metric_value,
            "samples": response.samples,
            "positives": response.positives,
        },
        outcome=OUTCOME_OK,
    )
    return response


def _run_calibration(request: CalibrationRequest, registry: ModelRegistry) -> CalibrationResponse:
    """Fit the threshold, install it, and refresh the drift reference.

    Split out of the handler so the audit bookkeeping around it stays legible,
    matching :func:`_run_benchmark`.

    The order matters and mirrors ``/predict``'s: the *cheap* rejection comes
    first. Fitting the threshold is milliseconds and needs no model, so an
    unusable calibration set is refused before the registry is touched — a caller
    who sends 200 all-normal samples gets ``invalid_calibration_set``, not a
    thirty-second wait for a cold model load that was never going to be used.
    """
    scores = [sample.score for sample in request.samples]
    labels = [sample.label for sample in request.samples]

    try:
        threshold = find_optimal_threshold(scores, labels, metric=request.metric)
        metric_value = evaluate_threshold(scores, labels, threshold, metric=request.metric)
    except ValueError as exc:
        # The messages from app.evaluation.calibration are written for an
        # operator to read, so they are carried through to the 422 body rather
        # than replaced with a slug. Nothing about the caller's own data is
        # secret from the caller.
        raise InvalidCalibrationSetError(str(exc)) from exc

    # Resolves the ONNX fallback, and 503s here if nothing can serve the pair.
    model = registry.get_model(request.model_backend, request.category)
    previous = registry.set_threshold(request.model_backend, request.category, threshold)

    monitor = registry.monitor_for(model.model_name, request.category)
    if request.set_drift_reference:
        monitor.set_reference(scores)

    log.info(
        "calibration_applied",
        model_name=model.model_name,
        metric=request.metric,
        threshold=round(threshold, 6),
        previous_threshold=round(previous, 6),
        metric_value=round(metric_value, 6),
        samples=len(scores),
        positives=int(sum(labels)),
        drift_reference_size=monitor.reference_size,
    )

    return CalibrationResponse(
        model_name=model.model_name,
        category=request.category,
        metric=request.metric,
        threshold=threshold,
        previous_threshold=previous,
        metric_value=metric_value,
        samples=len(scores),
        positives=int(sum(labels)),
        drift_reference_size=monitor.reference_size,
    )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["evaluation"])
def benchmark(
    request: BenchmarkRequest,
    principal: Principal = Depends(require_operator),
    registry: ModelRegistry = Depends(get_registry),
) -> BenchmarkResponse:
    """Score every requested backend over a category's full test split.

    **This is an offline evaluation endpoint, not a real-time one.** It runs
    every model over every test image and computes image-AUROC, pixel-AUROC,
    AU-PRO and F1 — on ``bottle`` that is 83 images per backend, which is
    seconds for the ONNX graphs, roughly a minute for PatchCore and several
    minutes for WinCLIP. The request holds a thread-pool worker for its whole
    duration and there is no timeout that will save a caller who expected
    otherwise. Use it from a script, a CI job or a dashboard refresh; never from
    a request path a user is waiting on, and never as a health check.

    **Operator role, and audited.** Everything in the paragraph above is also a
    description of a denial-of-service primitive, which is the first reason this
    is the most restricted endpoint in the service; the second is that its
    response describes the customer's test data (how many images, how many
    defective, how separable) to whoever asks. So every call — successful or not
    — appends an entry to ``results/audit.jsonl`` naming the hashed caller, what
    they asked for, what it cost and what they received. See
    :mod:`app.observability.audit_log` and ``docs/security.md``.

    A JSON report is also written to
    ``results/benchmark_<category>_<timestamp>.json`` as a side effect, matching
    what ``scripts/run_benchmark.py`` produces — runs accumulate rather than
    overwrite, so an API-triggered run leaves the same dated trail as a CLI one.

    Args:
        request: The category and the backends to compare.

    Returns:
        ``{model_name: metrics}`` for every backend that ran.

    Raises:
        ModelNotReadyError: 503 — one of the backends has no artifact.
        DatasetNotAvailableError: 503 — the category's test split is not on disk.
    """
    started = time.perf_counter()
    try:
        results = _run_benchmark(request, registry)
    except Exception as exc:
        # Audited before the exception handler turns this into a 503 or a 500.
        # A failed benchmark still consumed the CPU up to the point it failed,
        # and "this key repeatedly triggers minute-long failures" is precisely
        # the pattern an audit trail exists to make visible.
        record_benchmark(
            caller=principal.key_id,
            role=principal.role,
            category=request.category,
            models=request.model_backends,
            duration_seconds=time.perf_counter() - started,
            metrics={},
            outcome=f"failed:{type(exc).__name__}",
        )
        raise

    record_benchmark(
        caller=principal.key_id,
        role=principal.role,
        category=request.category,
        models=request.model_backends,
        duration_seconds=time.perf_counter() - started,
        metrics=results,
        outcome=OUTCOME_OK,
    )
    return BenchmarkResponse(results=results)


def _run_benchmark(request: BenchmarkRequest, registry: ModelRegistry) -> dict[str, dict]:
    """The benchmark itself: resolve the backends, load the split, score it.

    Split out of the handler so the audit bookkeeping around it stays legible —
    the handler is then "time it, run it, record it, whichever way it goes" and
    this is the part that does the work.
    """
    config = registry.config.with_overrides(category=request.category)

    # Loaded before the dataset so a missing checkpoint fails fast, rather than
    # after the test split has been read into memory.
    models: dict[str, AnomalyModel] = {}
    for backend in request.model_backends:
        model = registry.get_model(backend, request.category)
        if model.model_name in models:
            # Only reachable when a PyTorch backend fell back to ONNX and the
            # caller also asked for that ONNX backend by name. BenchmarkRunner
            # keys results by model_name and rejects duplicates, so collapse
            # them here — running the identical graph twice would produce two
            # identical rows at twice the cost.
            log.warning(
                "benchmark_backend_collapsed",
                backend=backend,
                model_name=model.model_name,
                reason="already in this run; scoring it once",
            )
            continue
        models[model.model_name] = model

    datamodule = DataModule(
        category=request.category,
        image_size=config.image_size,
        batch_size=config.batch_size,
        root=config.data_root,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    try:
        datamodule.setup()
    except FileNotFoundError as exc:
        raise DatasetNotAvailableError(request.category, str(exc)) from exc

    log.info(
        "benchmark_started",
        models=list(models),
        category=request.category,
        note="slow by design; see the endpoint docstring",
    )
    started = time.perf_counter()
    results = BenchmarkRunner(list(models.values()), datamodule, results_dir=config.results_dir).run()
    log.info(
        "benchmark_finished",
        models=list(models),
        category=request.category,
        duration_seconds=round(time.perf_counter() - started, 3),
    )

    return results
