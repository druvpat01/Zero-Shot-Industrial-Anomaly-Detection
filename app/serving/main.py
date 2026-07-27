"""The FastAPI application: four endpoints over everything the rest of the app builds.

Run it with::

    uvicorn app.serving.main:app --host 0.0.0.0 --port 8000    # or: make serve

and read the generated contract at ``/docs``.

What this module is, and is not
===============================
It is the wiring. Every hard part lives somewhere else — quality checks in
:mod:`app.guardrails`, scoring in :mod:`app.models`, metrics in
:mod:`app.evaluation`, base64 plumbing in :mod:`app.serving.imaging`, model
lifetime in :mod:`app.serving.model_registry` — and the handlers below are four
short functions that call them in the right order and translate failures into
status codes. That is on purpose: the interesting decisions in a serving layer
are about *sequencing and failure*, and those are hard to see in a route handler
that also resizes images.

The request path, and why it is ordered this way
===============================================
``POST /predict`` does four things, and the order is the design:

1. **Decode**, and refuse anything that is not an image (422). Cheapest check,
   and it needs no model.
2. **Guard**, and refuse a frame not worth scoring (422). Also cheap —
   microseconds — and, critically, *before* the model is fetched. A blurry frame
   submitted against a backend with no checkpoint returns "blurry", not
   "model_not_ready": the caller's problem is reported ahead of the server's,
   and a cold 30-second model load is never paid for a frame that was going to
   be rejected anyway.
3. **Fetch the model** from the registry, loading it if this is the first
   request for it (503 if no artifact can serve it).
4. **Score**, render the heatmap, and answer.

Step 2 is a deliberate duplicate: every ``AnomalyModel.predict`` runs the same
guard internally, so the frame is validated twice. That costs about a
millisecond and it is worth it — running the guard here is what lets the API
return a *structured* 422 with the failing reason and short-circuit before the
registry is touched, while the model's own call keeps the guarantee true for
every caller of the model layer, not just this one.

Errors
======
Four exception handlers cover the four ways a request fails, and every one
returns a small JSON object with a machine-readable ``detail`` slug rather than
a Python exception:

===============================  ======  ====================================================
Condition                        Status  Body
===============================  ======  ====================================================
Undecodable ``image_b64``        422     ``{"detail": "invalid_image", "reason": ...}``
Frame fails the quality guard    422     ``{"detail": "guard_failed", "reason": "blurry"}``
Schema/validation failure        422     ``{"detail": "invalid_request", "errors": [...]}``
No artifact for the backend      503     ``{"detail": "model_not_ready", "backend": ...}``
Anything else                    500     ``{"detail": "internal_error"}``
===============================  ======  ====================================================

The 500 is the one that matters for security: the traceback goes to the server
log via ``logger.exception`` and *only* there. Leaking it to the client would
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
"""

from __future__ import annotations

import logging
import time

from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.data import DataModule
from app.evaluation import BenchmarkRunner
from app.guardrails import GuardError, guard
from app.models.base import AnomalyModel
from app.serving.imaging import InvalidImageError, decode_image_b64, encode_heatmap_png_b64
from app.serving.model_registry import ModelNotReadyError, ModelRegistry, get_registry
from app.serving.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
)

__all__ = ["DatasetNotAvailableError", "app"]

logger = logging.getLogger(__name__)

#: A model load slower than this gets its own log line. Not an error — a cold
#: WinCLIP legitimately takes tens of seconds — but it is the single best
#: explanation for a request that looked inexplicably slow end to end, and
#: ``latency_ms`` deliberately excludes it.
_SLOW_LOAD_SECONDS = 1.0

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


app = FastAPI(
    title="Zero-Shot Industrial Defect Detection",
    version="0.1.0",
    summary="Pixel-level anomaly detection and segmentation for industrial QA.",
    description=(
        "Scores inspection frames for defects and returns a pixel-level heatmap, "
        "using PatchCore, EfficientAD or zero-shot WinCLIP — the PyTorch wrappers "
        "or their exported ONNX graphs. Models load lazily on first use, so "
        "`/health` answers immediately at startup."
    ),
)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(InvalidImageError)
def _handle_invalid_image(request: Request, exc: InvalidImageError) -> JSONResponse:
    """422: ``image_b64`` was not a decodable image."""
    logger.info("Rejected %s: invalid image (%s)", request.url.path, exc.reason)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "invalid_image", "reason": exc.reason},
    )


@app.exception_handler(GuardError)
def _handle_guard_error(request: Request, exc: GuardError) -> JSONResponse:
    """422: the frame decoded fine but is not worth scoring.

    The guard's metrics go to the log rather than the response. They are drift-
    monitoring telemetry — watching ``laplacian_variance`` sag over a shift is
    how a fouling lens is caught before it starts rejecting frames — and that is
    a server-side concern, not something the caller of a single request can act
    on beyond the ``reason``.
    """
    logger.warning("Rejected %s: guard_failed reason=%s metrics=%s", request.url.path, exc.reason, exc.metrics)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "guard_failed", "reason": exc.reason},
    )


@app.exception_handler(ModelNotReadyError)
def _handle_model_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    """503: the backend has no artifact that can serve this category."""
    logger.error("Rejected %s: model_not_ready backend=%s (%s)", request.url.path, exc.backend, exc.detail)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "model_not_ready", "backend": exc.backend, "reason": exc.detail},
    )


@app.exception_handler(DatasetNotAvailableError)
def _handle_dataset_not_available(request: Request, exc: DatasetNotAvailableError) -> JSONResponse:
    """503: the benchmark was asked for a category whose test split is not present."""
    logger.error("Rejected %s: dataset_not_available category=%s", request.url.path, exc.category)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "dataset_not_available", "category": exc.category, "reason": exc.detail},
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
    logger.info("Rejected %s: invalid_request %s", request.url.path, errors)
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": "invalid_request", "errors": errors},
    )


@app.exception_handler(Exception)
def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """500: anything not accounted for above.

    The traceback goes to the log and nowhere else — see the module docstring.
    The client gets a slug and nothing to fingerprint the deployment with.
    """
    logger.exception("Unhandled error serving %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal_error"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    """Liveness probe. Loads nothing and touches no disk.

    This is the contract that makes lazy loading work: a container orchestrator
    can get a ready signal in milliseconds after startup, instead of waiting out
    a model load and killing the pod for failing its probe. ``models_loaded``
    reports what happens to be resident *now*, which is empty until the first
    ``/predict``.
    """
    return HealthResponse(status="ok", models_loaded=registry.loaded_keys())


@app.get("/models", response_model=list[ModelInfo], tags=["ops"])
def list_models(
    category: str | None = Query(
        default=None,
        description="Category to report availability for. Defaults to the configured DEFAULT_CATEGORY.",
        examples=["bottle"],
    ),
    registry: ModelRegistry = Depends(get_registry),
) -> list[ModelInfo]:
    """List every backend and whether it can serve ``category`` right now.

    Availability is answered from the filesystem — is there a checkpoint, is
    there an exported graph — so this endpoint stays as cheap as ``/health`` and
    can be polled. ``detail`` explains the interesting cases: which file is
    missing, or that a backend would be served by an ONNX fallback rather than
    the checkpoint the caller might assume.
    """
    resolved = category or registry.config.category
    return [ModelInfo(**row) for row in registry.describe(resolved)]


@app.post("/predict", response_model=InferenceResponse, tags=["inference"])
def predict(
    request: InferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> InferenceResponse:
    """Score one frame and return its anomaly score plus a pixel-level heatmap.

    See the module docstring for why the steps happen in this order. In short:
    the cheap rejections (undecodable payload, unusable frame) come first and
    need no model, so a bad request never triggers a model load.

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

    frame = decode_image_b64(request.image_b64)

    verdict = guard.validate(frame)
    if not verdict.passed:
        raise GuardError(verdict)

    # Timed separately and subtracted below: a cold load is a one-off startup
    # cost, and folding it into the reported latency would put a 30-second
    # outlier in the same series as the 150 ms steady state, ruining every
    # percentile computed from it. (Step 10 observes the two as separate
    # Prometheus metrics for the same reason; this is where the histogram
    # observation for `latency_ms` will go.)
    load_started = time.perf_counter()
    model = registry.get_model(request.model_backend, request.category)
    load_elapsed = time.perf_counter() - load_started
    if load_elapsed > _SLOW_LOAD_SECONDS:
        logger.info(
            "Cold-loaded %r for %r in %.1fs; this is excluded from the reported latency.",
            request.model_backend,
            request.category,
            load_elapsed,
        )

    # BGR because decode_image_b64 hands back OpenCV's channel order. Passing
    # the default "rgb" here would swap the channels under an ImageNet- or
    # CLIP-pretrained backbone: no error, no shape change, quietly worse scores.
    output = model.predict(frame, color_order="bgr")
    heatmap_b64 = encode_heatmap_png_b64(output.anomaly_map, calibrated=model.is_calibrated)

    latency_ms = (time.perf_counter() - started - load_elapsed) * 1000.0
    logger.info(
        "Scored %s/%s: score=%.4f defective=%s in %.1fms",
        request.model_backend,
        request.category,
        output.anomaly_score,
        output.is_defective,
        latency_ms,
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


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["evaluation"])
def benchmark(
    request: BenchmarkRequest,
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
            logger.warning(
                "Backend %r resolved to %r, which is already in this run; scoring it once.",
                backend,
                model.model_name,
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

    logger.info(
        "Benchmarking %s on %r (this is slow by design; see the endpoint docstring)",
        list(models),
        request.category,
    )
    started = time.perf_counter()
    results = BenchmarkRunner(list(models.values()), datamodule, results_dir=config.results_dir).run()
    logger.info("Benchmark of %r finished in %.1fs", request.category, time.perf_counter() - started)

    return BenchmarkResponse(results=results)
