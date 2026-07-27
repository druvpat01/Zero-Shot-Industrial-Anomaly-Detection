"""Serving layer: the HTTP surface over the model, guardrail and evaluation layers.

The ASGI application itself lives at :mod:`app.serving.main` and is *not*
re-exported here. Importing a package should not have the side effect of
constructing a FastAPI app and registering its routes, and keeping the app out
of ``__init__`` means ``from app.serving import InferenceRequest`` stays a cheap
import of a few pydantic models.

Start the server with::

    uvicorn app.serving.main:app --host 0.0.0.0 --port 8000    # or: make serve

Layout:

* :mod:`app.serving.schemas` — the wire contract, as pydantic v2 models.
* :mod:`app.serving.imaging` — base64 payload to frame, heatmap to base64 PNG.
* :mod:`app.serving.model_registry` — lazy, cached, warmed model loading.
* :mod:`app.serving.main` — the app, its four endpoints, and its error handling.
"""

from app.serving.imaging import InvalidImageError, decode_image_b64, encode_heatmap_png_b64
from app.serving.model_registry import ModelNotReadyError, ModelRegistry, get_registry
from app.serving.schemas import (
    MODEL_BACKENDS,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelBackend,
    ModelInfo,
)

__all__ = [
    "MODEL_BACKENDS",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "HealthResponse",
    "InferenceRequest",
    "InferenceResponse",
    "InvalidImageError",
    "ModelBackend",
    "ModelInfo",
    "ModelNotReadyError",
    "ModelRegistry",
    "decode_image_b64",
    "encode_heatmap_png_b64",
    "get_registry",
]
