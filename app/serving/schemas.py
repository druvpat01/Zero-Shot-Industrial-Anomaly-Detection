"""The API's wire contract: every request and response body, as pydantic v2 models.

Why these live in their own module
----------------------------------
The route handlers in :mod:`app.serving.main` are thin — decode, guard, score,
encode — and everything *shaped* is here. That split is what makes the API
self-documenting: FastAPI derives the OpenAPI schema at ``/docs`` from these
classes, so the field descriptions below are the published documentation, not a
copy of it that can drift.

The models also carry the boundary's validation. ``model_backend`` is a
:data:`ModelBackend` literal rather than a bare ``str``, so an unknown backend is
rejected by pydantic with a 422 before any handler code runs and before the model
registry is asked to load anything. :class:`BenchmarkRequest` validates its
backend list the same way, by hand, because a ``list[Literal[...]]`` produces a
markedly worse error message for a list of typos than a single explicit check.

One deliberate omission: nothing here validates that ``image_b64`` decodes to an
image. That is :func:`app.serving.imaging.decode_image_b64`'s job, because a
pydantic validation failure echoes the offending input back in its error payload
— which for a multi-megabyte base64 frame is a response body nobody wants and a
log line nobody can read.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CALIBRATION_METRICS",
    "MODEL_BACKENDS",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "CalibrationMetric",
    "CalibrationRequest",
    "CalibrationResponse",
    "CalibrationSample",
    "DriftStatus",
    "HealthResponse",
    "InferenceRequest",
    "InferenceResponse",
    "ModelBackend",
    "ModelInfo",
]

#: The backends a request may name. The ``onnx_*`` entries are the exported
#: graphs served through :class:`~app.models.onnx_runner.ONNXRunner`; the bare
#: names are the PyTorch wrappers. ``winclip`` has no ONNX twin because
#: anomalib's WinCLIP is not exportable (its sliding-window pooling reaches into
#: the CLIP module's internals), and it needs none: it is the zero-shot backend,
#: not the low-latency one.
ModelBackend = Literal[
    "patchcore",
    "efficientad",
    "winclip",
    "onnx_patchcore",
    "onnx_efficientad",
]

#: The same names as a tuple, for runtime checks and error messages. Derived from
#: the ``Literal`` rather than repeated, so the two cannot drift apart.
MODEL_BACKENDS: tuple[str, ...] = get_args(ModelBackend)

#: What ``POST /calibrate`` may be asked to maximise. Mirrors
#: :data:`app.evaluation.calibration.CALIBRATION_METRICS` — spelled out here
#: rather than imported, so the wire contract stays readable without following an
#: import, and pinned to the implementation by a test in ``tests/test_drift.py``
#: so the two cannot drift apart unnoticed.
CalibrationMetric = Literal[
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
]

#: The same names as a tuple, derived from the ``Literal``.
CALIBRATION_METRICS: tuple[str, ...] = get_args(CalibrationMetric)


class InferenceRequest(BaseModel):
    """One frame to score, and which model should score it."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description="MVTec-style object category, e.g. 'bottle'. Selects the checkpoint "
        "for the trained backends and the prompt noun for WinCLIP.",
        examples=["bottle"],
        min_length=1,
    )
    model_backend: ModelBackend = Field(
        description="Which anomaly model serves this frame.",
        examples=["patchcore"],
    )
    image_b64: str = Field(
        description="Base64-encoded image *file* bytes (PNG, JPEG, ... — anything "
        "OpenCV can decode), not a raw pixel buffer. A `data:image/png;base64,` "
        "prefix is accepted and stripped.",
        examples=["iVBORw0KGgoAAAANSUhEUgAA..."],
    )


class InferenceResponse(BaseModel):
    """The verdict on one frame.

    ``anomaly_score`` and the heatmap share a scale. For a calibrated backend
    that scale is ``[0, 1]`` with ``0.5`` the fitted decision boundary; an
    uncalibrated one emits raw distances, and the heatmap is then normalized per
    frame for display (see :func:`app.serving.imaging.encode_heatmap_png_b64`).
    """

    model_config = ConfigDict(extra="forbid")

    anomaly_score: float = Field(
        description="Image-level anomaly score. Calibrated backends emit [0, 1].",
        examples=[0.83],
    )
    is_defective: bool = Field(
        description="Whether the score cleared the configured ANOMALY_THRESHOLD.",
    )
    model_name: str = Field(
        description="The backend that actually served the frame. Not always the one "
        "requested: a PyTorch backend with no checkpoint falls back to its ONNX "
        "export, and says so here.",
        examples=["patchcore"],
    )
    anomaly_map_b64: str = Field(
        description="Base64-encoded PNG of the pixel-level heatmap (JET colormap), at "
        "the resolution of the submitted frame, so it overlays without rescaling.",
    )
    latency_ms: float = Field(
        description="Server-side time to serve this frame: decode + quality guard + "
        "inference + heatmap encoding. Excludes one-off model loading, which is "
        "cold-start cost rather than per-frame cost.",
        examples=[152.4],
    )
    guard_passed: bool = Field(
        description="Whether the frame cleared the input-quality guard. Always true on "
        "a 200 — a failing frame is never scored, and returns 422 instead.",
    )
    guard_reason: str | None = Field(
        default=None,
        description="Why the guard rejected the frame, on the 422 path. Null on a 200.",
        examples=[None],
    )


class BenchmarkRequest(BaseModel):
    """Which models to compare, on which category's test split."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description="Category whose test split every requested model is scored on.",
        examples=["bottle"],
        min_length=1,
    )
    model_backends: list[str] = Field(
        description=f"Backends to compare. Each must be one of {list(MODEL_BACKENDS)}.",
        examples=[["patchcore", "winclip"]],
        min_length=1,
    )

    @field_validator("model_backends")
    @classmethod
    def _known_backends(cls, value: list[str]) -> list[str]:
        """Reject unknown names here rather than deep inside the registry.

        Typed as ``list[str]`` with a hand-rolled check instead of
        ``list[ModelBackend]``: pydantic reports a bad literal in a list as one
        error *per allowed value per bad element*, which for two typos across
        five backends is ten error entries the caller has to read to find two
        mistakes.
        """
        unknown = [name for name in value if name not in MODEL_BACKENDS]
        if unknown:
            msg = f"unknown backend(s) {unknown}; choose from {list(MODEL_BACKENDS)}"
            raise ValueError(msg)
        return value


class BenchmarkResponse(BaseModel):
    """Per-model metrics from one benchmark run.

    ``results`` is keyed by the model name that produced each row (which is the
    backend name, so a fallback to ONNX is visible), and each value is the metric
    dict from :meth:`~app.evaluation.benchmark.BenchmarkResult.as_dict`: the four
    headline metrics plus the counts and timing behind them. It is left as an
    open ``dict`` rather than a typed model so a new metric shows up in the API
    the moment it shows up in the benchmark, without a schema change in two
    places.
    """

    model_config = ConfigDict(extra="forbid")

    results: dict[str, dict] = Field(
        description="model name -> metrics (image_auroc, pixel_auroc, au_pro, best_f1, "
        "counts, elapsed_seconds, ...).",
    )


class HealthResponse(BaseModel):
    """Liveness, and what the process currently holds in memory.

    Deliberately cheap: answering this must never load a model, so
    ``models_loaded`` reports what is already resident rather than what could be
    served. Use ``GET /models`` for the latter.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="'ok' when the process can serve requests.", examples=["ok"])
    models_loaded: list[str] = Field(
        description="Currently resident models as '<backend>:<category>' keys. Empty "
        "right after startup, because loading is lazy.",
        examples=[["patchcore:bottle"]],
    )


class ModelInfo(BaseModel):
    """One backend's availability, as reported by ``GET /models``.

    ``available`` is answered from the filesystem — does the checkpoint or the
    exported graph exist — so the question can be asked without paying to load
    anything, which is the whole point of the endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(description="Backend name, as accepted by /predict.", examples=["patchcore"])
    available: bool = Field(description="Whether this backend could serve the category right now.")
    loaded: bool = Field(description="Whether it is already resident in this process.")
    artifact: str | None = Field(
        default=None,
        description="Path to the checkpoint or .onnx file that would be served. Null for "
        "WinCLIP running pure zero-shot, which has no artifact by construction.",
    )
    detail: str | None = Field(
        default=None,
        description="Why the backend is unavailable, or how it would be served (e.g. an "
        "ONNX fallback). Null when there is nothing to explain.",
    )


class DriftStatus(BaseModel):
    """One model's score-distribution health, as reported by ``GET /drift``.

    Read ``drifted`` and ``summary`` together, never separately. ``drifted`` is a
    hypothesis test and answers only *did the distribution move*; the percentiles
    answer *by how much*, and with a large enough window the test will eventually
    flag a shift far too small to change a single verdict. See
    :mod:`app.evaluation.drift`.

    ``drifted=false`` with ``p_value=null`` is a third state and not a clean bill
    of health: it means no verdict was available, because no reference has been
    set or a window is still filling.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        description="The resolved model whose scores these are — an ONNX fallback is "
        "monitored separately from the PyTorch backend it stood in for.",
        examples=["patchcore"],
    )
    category: str = Field(description="The category these scores were produced for.", examples=["bottle"])
    drifted: bool = Field(
        description="Whether the KS p-value fell below `threshold`. False also when there "
        "is not yet enough data to say — check `p_value` for null to tell them apart.",
    )
    reason: str | None = Field(
        default=None,
        description="'ks_drift' when drift was detected, else null.",
        examples=["ks_drift"],
    )
    p_value: float | None = Field(
        default=None,
        description="Two-sample Kolmogorov-Smirnov p-value against the reference window. "
        "Null when no verdict is available: no reference set, or a window below the "
        "minimum sample count.",
        examples=[0.42],
    )
    threshold: float = Field(
        description="p-value at or below which drift is declared. From KS_DRIFT_THRESHOLD.",
        examples=[0.05],
    )
    window_size: int = Field(description="Capacity of the rolling window.", examples=[500])
    reference_size: int = Field(
        description="Scores in the reference distribution. Zero until POST /calibrate or an "
        "explicit set_reference call establishes one.",
        examples=[100],
    )
    min_samples: int = Field(
        description="Scores each window needs before a verdict is offered. Reported so a null "
        "`p_value` is explicable from the response alone: compare it against `reference_size` "
        "and `summary.count` to see which window is still filling.",
        examples=[30],
    )
    # The `int` member looks redundant next to `float` — PEP 484 already accepts an
    # int wherever a float is declared — but it is load-bearing here. Without it
    # pydantic coerces `count` to a float and the response reads `"count": 20.0`,
    # which is a sample size rendered as a measurement. Smart-union mode matches
    # the exact type, so `20` stays `20` and `1.0` stays `1.0`.
    summary: dict[str, float | int | None] = Field(
        description="Descriptive statistics for the current window: count, mean, std, p10, "
        "p50, p90. The five statistics are null on an empty window. Left as an open dict "
        "rather than a typed model so a new statistic reaches the API the moment it reaches "
        "the monitor, matching BenchmarkResponse.",
        examples=[{"count": 20, "mean": 0.31, "std": 0.12, "p10": 0.18, "p50": 0.28, "p90": 0.52}],
    )


class CalibrationSample(BaseModel):
    """One labelled point in a calibration set: what the model said, and the truth."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        description="The anomaly score this frame received, as returned by /predict.",
        examples=[0.83],
    )
    label: int = Field(
        description="Ground truth: 1 defective, 0 normal.",
        examples=[1],
        ge=0,
        le=1,
    )


class CalibrationRequest(BaseModel):
    """A labelled score set, and the model whose operating point it should fit."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description="Category whose threshold is being calibrated.",
        examples=["bottle"],
        min_length=1,
    )
    model_backend: ModelBackend = Field(
        description="Backend whose decision threshold is updated.",
        examples=["patchcore"],
    )
    samples: list[CalibrationSample] = Field(
        description="The calibration set: scores this model produced, with ground-truth "
        "labels. Must contain both classes — a threshold fitted to one class is not an "
        "operating point. Held out from training, or the fitted threshold is optimistic.",
        min_length=2,
    )
    metric: CalibrationMetric = Field(
        default="f1",
        description=f"What to maximise. One of {list(CALIBRATION_METRICS)}. 'f1' balances "
        "false alarms against missed defects; 'precision' and 'recall' alone are degenerate "
        "at the extremes — see app.evaluation.calibration.",
        examples=["f1"],
    )
    set_drift_reference: bool = Field(
        default=True,
        description="Also install these scores as the drift monitor's reference distribution. "
        "On by default because the calibration set *is* the new definition of normal: the "
        "threshold and the baseline it was fitted against should move together.",
    )


class CalibrationResponse(BaseModel):
    """The fitted operating point, and what it achieves on the submitted set.

    ``metric_value`` is the number that decides whether the calibration was worth
    applying. A best-achievable F1 of 0.42 is still the best available, and still
    says the model cannot be rescued by moving its threshold.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        description="The model that was updated. Not always the requested backend: an ONNX "
        "fallback is calibrated under its own name.",
        examples=["patchcore"],
    )
    category: str = Field(description="The category that was calibrated.", examples=["bottle"])
    metric: str = Field(description="The metric that was maximised.", examples=["f1"])
    threshold: float = Field(
        description="The new decision threshold. A frame is defective when score >= this.",
        examples=[0.42],
    )
    previous_threshold: float = Field(
        description="The threshold this replaced, so the change is visible in one response.",
        examples=[0.5],
    )
    metric_value: float = Field(
        description="What `metric` achieves at `threshold` on the submitted set. Fitted on "
        "this data, so it is an upper bound on production performance, not an estimate of it.",
        examples=[0.94],
    )
    samples: int = Field(description="Calibration points received.", examples=[100])
    positives: int = Field(description="How many of them were labelled defective.", examples=[40])
    drift_reference_size: int = Field(
        description="Scores now in the drift monitor's reference window. Equals `samples` when "
        "`set_drift_reference` was true; unchanged otherwise.",
        examples=[100],
    )
