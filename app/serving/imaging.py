"""Base64 in, frame out; heatmap in, base64 PNG out.

The API speaks JSON, and the model layer speaks NumPy. This module is the entire
translation, kept out of :mod:`app.serving.main` so the route handlers read as
the four steps they are (decode, guard, score, encode) rather than as image
plumbing with routing sprinkled through it.

Two decisions here are worth more than their line count.

**Decoding never trusts the payload.** A request body is attacker-controlled, so
every failure mode — non-base64 characters, valid base64 that is not an image, a
truncated file, a 400 MB PNG bomb — funnels into one :class:`InvalidImageError`
and one 422. The size cap is checked on the *decoded* bytes, before OpenCV is
handed anything, because ``cv2.imdecode`` allocates the full raster and a
decompression bomb is precisely a small payload that does not stay small.

**Heatmap rendering keeps the model's scale when the model has one.** A
calibrated backend's anomaly map is already on the ``[0, 1]`` scale its score
lives on, so it is rendered against a *fixed* ``[0, 1]`` ramp: a clean part comes
back uniformly cool, a defective one has a hot spot, and the two images are
directly comparable. Min-max normalizing every frame — the obvious thing, and
the wrong thing — would stretch the sensor noise on a flawless part across the
full colormap and hand an operator a picture of a defect that is not there. For
an *un*calibrated backend there is no fixed scale to keep (scores are unbounded
nearest-neighbour distances), so per-frame normalization is the only option and
:func:`encode_heatmap_png_b64` says so through its ``calibrated`` argument
rather than guessing from the data.
"""

from __future__ import annotations

import base64
import binascii
import os
import re

import cv2
import numpy as np

__all__ = ["MAX_IMAGE_BYTES", "InvalidImageError", "decode_image_b64", "encode_heatmap_png_b64"]

#: Ceiling on the *decoded* size of a submitted image, in bytes. Generous enough
#: for a full-resolution industrial frame (an uncompressed 4096x4096 RGB TIFF is
#: ~50 MB; MVTec's 900x900 PNGs are ~700 KB) and small enough that a request
#: cannot exhaust the process. Override with ``MAX_IMAGE_BYTES``.
MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", "").strip() or 25 * 1024 * 1024)

#: ``data:image/png;base64,`` and friends. Browsers and ``canvas.toDataURL()``
#: produce this prefix, and stripping it costs one regex versus every JS client
#: having to remember to.
_DATA_URI_PREFIX = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

#: Whitespace inside a base64 payload — MIME-style line wrapping at 76 columns,
#: or JSON pretty-printing. Harmless, and removed before strict validation so it
#: is not mistaken for corruption.
_WHITESPACE = re.compile(r"\s+")

#: Below this spread, an uncalibrated anomaly map is flat and min-max
#: normalization would amplify pure float noise into a full-range image.
_FLAT_MAP_EPSILON = 1e-12


class InvalidImageError(ValueError):
    """Raised when a request's ``image_b64`` is not a decodable image.

    Handled in :mod:`app.serving.main` as a 422 carrying ``{"detail":
    "invalid_image"}``. The :attr:`reason` is included alongside it — it names
    which of the several ways this can fail actually happened, which is the
    difference between a caller fixing their encoding in a minute and guessing
    for an afternoon.

    Attributes:
        reason: Short machine-readable slug: ``"not_base64"``, ``"empty"``,
            ``"too_large"`` or ``"undecodable"``.
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def decode_image_b64(payload: str) -> np.ndarray:
    """Decode a base64 image payload into an ``(H, W, 3)`` BGR ``uint8`` frame.

    Args:
        payload: Base64-encoded image *file* bytes (PNG, JPEG, BMP, TIFF — the
            format is whatever OpenCV's decoder recognises from the magic bytes,
            not something the caller declares). A ``data:image/...;base64,``
            prefix and any whitespace are stripped first.

    Returns:
        The decoded frame in **BGR** channel order — OpenCV's convention, and
        therefore what must be passed to ``predict(..., color_order="bgr")``.
        Grayscale and alpha inputs are normalized to 3-channel BGR here, so
        every frame reaching the guard and the model has the same shape.

    Raises:
        InvalidImageError: On anything that is not a decodable image within the
            size cap. Never propagates ``binascii``/OpenCV errors, whose messages
            are implementation detail and occasionally echo the input.
    """
    cleaned = _WHITESPACE.sub("", _DATA_URI_PREFIX.sub("", payload.strip()))
    if not cleaned:
        msg = "image_b64 is empty."
        raise InvalidImageError("empty", msg)

    try:
        # validate=True so a payload with characters outside the base64 alphabet
        # is rejected outright. Without it, Python silently discards them and
        # happily decodes garbage into a shorter byte string, which then fails
        # later as "undecodable" — a true statement about the wrong problem.
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = "image_b64 is not valid base64."
        raise InvalidImageError("not_base64", msg) from exc

    if not raw:
        msg = "image_b64 decoded to zero bytes."
        raise InvalidImageError("empty", msg)
    if len(raw) > MAX_IMAGE_BYTES:
        # Checked before cv2.imdecode: that call allocates the full decompressed
        # raster, so the cap has to bite while the data is still compressed.
        msg = f"Decoded image is {len(raw)} bytes; the limit is {MAX_IMAGE_BYTES}."
        raise InvalidImageError("too_large", msg)

    # IMREAD_COLOR normalizes grayscale and alpha inputs to 3-channel BGR, so
    # everything downstream sees one shape. cv2 returns None rather than raising
    # when the bytes are not an image it knows.
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        msg = "image_b64 decoded, but the bytes are not an image OpenCV can read."
        raise InvalidImageError("undecodable", msg)
    return frame


def encode_heatmap_png_b64(anomaly_map: np.ndarray, *, calibrated: bool) -> str:
    """Render a pixel-level anomaly map as a base64-encoded colormapped PNG.

    Args:
        anomaly_map: ``(H, W)`` float array at the submitted frame's resolution,
            straight off :attr:`~app.models.base.ModelOutput.anomaly_map`.
        calibrated: Whether the producing model's scores are normalized to
            ``[0, 1]`` (:attr:`~app.models.base.AnomalyModel.is_calibrated`).
            ``True`` renders against a fixed ``[0, 1]`` ramp, keeping frames
            comparable to each other and to ``anomaly_score``; ``False`` falls
            back to per-frame min-max, which is the only thing an unbounded
            distance scale admits. See the module docstring for why this is not
            a detail.

    Returns:
        Base64 ASCII of a PNG the caller can drop straight into an ``<img>`` tag
        or overlay on the frame they submitted.
    """
    array = np.asarray(anomaly_map, dtype=np.float32)
    # A NaN would silently become 0 in the uint8 cast, painting a cold hole
    # exactly where the model failed to produce a number. Make it the hot end
    # instead: a rendering artifact an operator can see beats one they cannot.
    array = np.nan_to_num(array, nan=1.0, posinf=1.0, neginf=0.0)

    if calibrated:
        scaled = np.clip(array, 0.0, 1.0)
    else:
        low, high = float(array.min()), float(array.max())
        span = high - low
        scaled = (array - low) / span if span > _FLAT_MAP_EPSILON else np.zeros_like(array)

    gray = np.rint(scaled * 255.0).astype(np.uint8)
    # applyColorMap emits BGR, which is exactly what imencode expects, so the
    # colours survive the round trip without a conversion in between.
    coloured = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    ok, buffer = cv2.imencode(".png", coloured)
    if not ok:  # pragma: no cover - PNG encoding of a valid uint8 raster does not fail
        msg = f"Failed to PNG-encode a {gray.shape} anomaly map."
        raise RuntimeError(msg)
    return base64.b64encode(buffer.tobytes()).decode("ascii")
