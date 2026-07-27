"""Prometheus metrics: the five numbers this service is actually judged on.

Why these five
==============
A metric is a commitment — every label combination is a time series held in
memory forever, in this process and in the scrape target's storage — so the set
is small and each entry answers a question somebody genuinely asks about an
inspection line:

===============================  =========================================================
Metric                           The question it answers
===============================  =========================================================
``images_processed_total``       How much traffic, and what is the defect rate? Split by
                                 model and category so a rate change can be attributed.
``inference_latency_seconds``    Is the line keeping up? A histogram, so p50/p95/p99 are
                                 computed at query time rather than baked in here.
``guard_check_latency_seconds``  Is the guard itself cheap? The argument for running it
                                 ahead of the model is that it costs a small fraction of a
                                 forward pass; this is the number that keeps that claim
                                 honest, and it has already corrected it once — see below.
``guard_rejections_total``       Is the *camera* healthy? A rising ``blurry`` rate is a
                                 fouling lens, and it shows up here hours before it shows
                                 up as bad predictions.
``models_loaded_count``          How much of the process's memory is resident weights, and
                                 did a deployment quietly start loading five backends?
===============================  =========================================================

The label sets, and the cardinality argument
============================================
``model`` and ``category`` are bounded by the deployment (five backends, a
handful of categories); ``result`` and ``reason`` are closed enumerations
declared below. Nothing here is labelled with anything a *caller* controls —
no request id, no image identifier, no key id. That is the rule a metrics module
lives or dies by: one unbounded label turns a scrape target into a memory leak,
and the natural place to leak is exactly the per-request identifier that feels so
useful. Those belong in a log line (:mod:`app.observability.logging_config`) and
a span (:mod:`app.observability.tracing`), which is where they are.

``result`` is deliberately three-valued rather than a boolean. "Defective" and
"normal" are model verdicts; ``rejected_by_guard`` is not a verdict at all — it
is the model declining to answer — and folding it into "normal" would make a
fouling lens read as a factory that has stopped producing defects. That
distinction is the entire point of :mod:`app.guardrails`, and a metric that
erases it would hide the failure the guard exists to surface.

Two histograms, two bucket layouts
==================================
``inference_latency_seconds`` uses the buckets in the spec — 10 ms to 5 s — which
straddle what this project actually measures end to end: tens of milliseconds for
an ONNX graph on a small frame up to a second or more for a full-resolution MVTec
frame on CPU.

``guard_check_latency_seconds`` gets its own lower set, 0.5 ms to 0.5 s, and the
reason is worth recording because **this metric falsified the assumption it was
built to confirm**. The guard was documented throughout this codebase as costing
"microseconds"; the first real traffic through it measured 1.3 ms on a 256x256
frame and 23 ms on a real 900x900 MVTec frame. The cost is dominated by the
Laplacian variance and the two exposure reductions, all of which are O(H*W) over
a ``float64`` copy — so it scales with the frame, and a 900x900 frame is 12x the
pixels of a 256x256 one.

That does not overturn the design: 23 ms against a ~1 s forward pass is still
around 2% of the request, and refusing a bad frame that cheaply is still the
right trade. But the *number* was wrong by two orders of magnitude and the
original buckets were chosen to match it — they topped out at 100 ms and put
their resolution between 100 µs and 1 ms, where nothing has ever landed. The
buckets below reflect the measurement instead of the assumption, which is the
entire point of having taken it.

Scraping under multiple workers
===============================
``uvicorn --workers N`` forks N processes, each with its own counters, and a
scrape hits whichever one the OS routes it to — so a counter appears to jump
around at random. :func:`render_latest` handles this when
``PROMETHEUS_MULTIPROC_DIR`` is set, aggregating every worker's shared-memory
files into one exposition. The variable is not set by default because the
single-worker case is what ``make serve`` runs and the directory has to be
cleared between restarts; ``docker/prometheus.yml`` documents the multi-worker
setup.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.core import REGISTRY

__all__ = [
    "CONTENT_TYPE_LATEST",
    "GUARD_LATENCY_BUCKETS",
    "INFERENCE_LATENCY_BUCKETS",
    "RESULT_DEFECTIVE",
    "RESULT_NORMAL",
    "RESULT_REJECTED",
    "RESULTS",
    "backend_kind",
    "get_metric_value",
    "guard_check_latency_seconds",
    "guard_rejections_total",
    "images_processed_total",
    "inference_latency_seconds",
    "models_loaded_count",
    "observe_guard_check",
    "observe_inference",
    "record_guard_rejection",
    "record_image_processed",
    "render_latest",
    "set_models_loaded",
]

# ``CONTENT_TYPE_LATEST`` is re-exported rather than hardcoded: it carries a
# version parameter (``text/plain; version=0.0.4``) that a scraper
# content-negotiates on, and pinning that string here would silently break the
# day prometheus_client bumps the exposition format.

#: ``result`` label values for :data:`images_processed_total`. See the module
#: docstring for why a rejection is not folded into "normal".
RESULT_DEFECTIVE = "defective"
RESULT_NORMAL = "normal"
RESULT_REJECTED = "rejected_by_guard"

#: The closed set, for validation and for anyone building a dashboard query.
RESULTS: tuple[str, ...] = (RESULT_DEFECTIVE, RESULT_NORMAL, RESULT_REJECTED)

#: Latency buckets for model inference, in seconds.
INFERENCE_LATENCY_BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

#: Latency buckets for the guard, in seconds: 0.5 ms to 0.5 s. Set from measured
#: values (1.3 ms at 256x256, 23 ms at 900x900) rather than from the assumption
#: the guard was originally documented under — see the module docstring. The top
#: two buckets exist because the cost scales with the frame, and a 4K camera is
#: 20x the pixels of the MVTec frames these numbers came from.
GUARD_LATENCY_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
)

#: Environment variable prometheus_client uses to coordinate forked workers.
_MULTIPROC_VAR = "PROMETHEUS_MULTIPROC_DIR"


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------
#
# Registered against the default REGISTRY at import, which also carries
# prometheus_client's process and GC collectors — resident memory and open file
# descriptors come free and are worth having next to the application metrics.
# Module-level construction means the series exist from process start, so a
# dashboard shows a flat zero rather than "no data" before the first request.

images_processed_total = Counter(
    "images_processed_total",
    "Frames that reached a verdict, by model, category and outcome.",
    labelnames=("model", "category", "result"),
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Wall-clock time for one model forward pass, excluding cold model load.",
    labelnames=("model", "backend"),
    buckets=INFERENCE_LATENCY_BUCKETS,
)

guard_check_latency_seconds = Histogram(
    "guard_check_latency_seconds",
    "Wall-clock time for one FrameGuard.validate() call.",
    buckets=GUARD_LATENCY_BUCKETS,
)

guard_rejections_total = Counter(
    "guard_rejections_total",
    "Frames refused by the input-quality guard, by failing check.",
    labelnames=("reason",),
)

models_loaded_count = Gauge(
    "models_loaded_count",
    "Models currently resident in the process's registry cache.",
)


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------
#
# Call sites go through these rather than touching the collectors directly. The
# label *order* is then written down once, here, instead of at every call site
# where a silent transposition of `model` and `category` would produce two
# plausible-looking series that are both wrong.


def backend_kind(model_name: str) -> str:
    """Classify a resolved model name as its execution backend.

    ``inference_latency_seconds`` is labelled by *runtime* as well as by model
    because the comparison that matters operationally is PyTorch against its own
    ONNX export — the same weights, a 3-5x latency difference — and a single
    ``model`` label cannot express it.

    Args:
        model_name: A resolved :attr:`~app.models.base.AnomalyModel.model_name`,
            e.g. ``"patchcore"`` or ``"onnx_patchcore"``.

    Returns:
        ``"onnx"`` or ``"pytorch"``.
    """
    return "onnx" if model_name.startswith("onnx_") else "pytorch"


def record_image_processed(*, model: str, category: str, result: str) -> None:
    """Count one frame that reached an outcome.

    Args:
        model: Resolved model name — what actually scored the frame, so an ONNX
            fallback is counted as ``onnx_patchcore``. For a guard rejection no
            model ran, so this is the *requested* backend.
        category: The requested category.
        result: One of :data:`RESULTS`.

    Raises:
        ValueError: If ``result`` is outside :data:`RESULTS`. Prometheus would
            happily accept a typo and create a permanent fourth series that no
            dashboard queries; failing loudly in a test is much cheaper.
    """
    if result not in RESULTS:
        msg = f"Unknown result {result!r}; expected one of {RESULTS}."
        raise ValueError(msg)
    images_processed_total.labels(model=model, category=category, result=result).inc()


def observe_inference(*, model: str, seconds: float) -> None:
    """Record one inference duration, deriving the ``backend`` label from ``model``."""
    inference_latency_seconds.labels(model=model, backend=backend_kind(model)).observe(seconds)


def record_guard_rejection(reason: str) -> None:
    """Count one frame refused by the guard, by the check that failed."""
    guard_rejections_total.labels(reason=reason).inc()


def set_models_loaded(count: int) -> None:
    """Publish how many models are resident.

    A :class:`~prometheus_client.Gauge` set to an absolute value rather than
    incremented, because the registry already knows the true count and an
    inc/dec pair would drift the moment a load raised between the two.
    """
    models_loaded_count.set(count)


@contextmanager
def observe_guard_check() -> Iterator[None]:
    """Time a :meth:`~app.guardrails.quality.FrameGuard.validate` call.

    A context manager rather than a decorator so it wraps the measurement and
    nothing else — in particular it does not swallow, and must not swallow, the
    exception a malformed frame raises. ``try/finally`` means a rejection is
    still timed, which is the case you most want in the histogram: it is the
    fast path, and if rejections ever became *slow* that would be worth seeing.

    Example:
        >>> with observe_guard_check():            # doctest: +SKIP
        ...     result = guard.validate(frame)
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        guard_check_latency_seconds.observe(time.perf_counter() - started)


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------


def render_latest() -> bytes:
    """Render the current metrics in Prometheus text exposition format.

    Backs ``GET /metrics``. When ``PROMETHEUS_MULTIPROC_DIR`` is set, every
    forked worker's samples are aggregated into a throwaway registry first — see
    the module docstring for why that matters under ``--workers N``.

    Returns:
        UTF-8 bytes to be served with :data:`CONTENT_TYPE_LATEST`.
    """
    multiproc_dir = os.getenv(_MULTIPROC_VAR, "").strip()
    if multiproc_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
        return generate_latest(registry)
    return generate_latest(REGISTRY)


def get_metric_value(name: str, **labels: str) -> float | None:
    """Read one sample's current value, or ``None`` if that series does not exist yet.

    A thin wrapper over :meth:`CollectorRegistry.get_sample_value` that exists
    for the tests: asserting "this counter went up by one" needs a before and an
    after, and a counter's first observation creates the series, so the *before*
    is legitimately ``None`` rather than ``0``. Callers should treat ``None`` as
    zero, which is what Prometheus's own ``increase()`` does.

    Args:
        name: Sample name — note the ``_total`` suffix on a counter is part of
            the sample name (``images_processed_total``), while a histogram
            exposes ``<name>_count``, ``<name>_sum`` and ``<name>_bucket``.
        **labels: The full label set identifying the series.

    Returns:
        The value, or ``None`` if nothing has been recorded for those labels.
    """
    return REGISTRY.get_sample_value(name, labels or None)
