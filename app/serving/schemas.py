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
    "MODEL_BACKENDS",
    "BenchmarkRequest",
    "BenchmarkResponse",
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
