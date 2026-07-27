"""Input-quality gate for raw frames, ahead of any anomaly model.

Why this exists
===============
The three model wrappers (:mod:`app.models.patchcore`,
:mod:`app.models.efficientad`, :mod:`app.models.winclip`) will happily score
anything shaped like an image. That is exactly the problem on a real inspection
line. A camera drops a frame and hands back a black rectangle; a lens picks up a
film of coolant and everything goes soft; a light dies and the part vanishes
into shadow. None of these raise — they produce a *plausible* array, and the
model turns it into a *plausible* anomaly score. A soft frame reads as "no
sharp defect edges, looks fine"; a black frame reads as "uniform, looks fine".
The failure is silent, and a silent wrong answer on an inspection line is worse
than a loud refusal: it passes scrap downstream, or halts a good line chasing a
defect that is really just dirt on the glass.

:class:`FrameGuard` is the loud refusal. It runs four cheap, model-independent
checks on the raw frame and returns a :class:`GuardResult` saying whether the
frame is worth scoring and, if not, why. The model wrappers call it before they
preprocess, so a bad frame is rejected in a few milliseconds instead of being
scored in hundreds of them and quietly corrupting the numbers a performance
claim rests on. (What "a few" means is measured, not assumed — see "What this
module reports about itself" below.)

The checks
==========
* **Blur** — Laplacian variance below :attr:`GuardConfig.blur_threshold`. The
  Laplacian is a second-derivative (edge) operator; the variance of its
  response is the standard, cheap focus measure. A sharp frame has crisp edges
  and therefore a wide spread of Laplacian responses; an out-of-focus or
  dirty-lens frame has been low-pass filtered and its response collapses toward
  zero. See :meth:`FrameGuard._laplacian_variance` for the one subtlety that
  matters — the measure is not scale-free, so it is taken at a fixed working
  resolution.
* **Exposure** — more than :attr:`GuardConfig.exposure_fraction` of pixels
  pinned to the floor (``< dark_level``) or the ceiling (``> bright_level``).
  This is what catches a dead camera (all black) or a blown-out light (all
  white): both destroy the signal by clipping, and no model can recover a defect
  from a region that is uniformly 0 or 255.
* **Minimum resolution** — either side below
  :attr:`GuardConfig.min_resolution`. Too few pixels to carry a defect, and a
  strong hint the frame is a thumbnail, an error tile or a truncated read.
* **Aspect ratio** — width/height outside
  ``[1/max_aspect_ratio, max_aspect_ratio]``. Catches a transposed buffer, a
  single-row/column read, or a stride bug that reshapes a frame into a sliver.

Every threshold is configurable from the environment (see
:class:`GuardConfig`), so a line with a different camera, resolution or lighting
budget can retune the gate without a code change.

What this module reports about itself
=====================================
:meth:`FrameGuard.validate` is instrumented directly rather than at its call
sites, because it has several: the API handler in :mod:`app.serving.main`, every
model wrapper via :meth:`app.models.base.AnomalyModel._check_frame`, and the
demo script. Two Prometheus series come out of it (see
:mod:`app.observability.metrics`):

* ``guard_check_latency_seconds`` — the honesty check on this module's central
  claim, and it has already corrected it. The guard was documented here as
  costing "microseconds"; measured, it costs **1.3 ms on a 256x256 frame and
  23 ms on a real 900x900 MVTec frame**. Every check is O(H*W) over a ``float64``
  copy, so the cost scales with the frame rather than being a fixed overhead.
  The design still holds — 23 ms against a ~1 s forward pass is ~2% of the
  request, and rejecting a bad frame that cheaply is still clearly worth it —
  but the figure is two orders of magnitude off what was written, and the
  histogram is what caught it.
* ``guard_rejections_total{reason}`` — the operationally interesting one. A
  rising ``blurry`` rate is a lens acquiring a film of coolant, and it is visible
  here hours before anybody notices the predictions have gone soft.

Rejections are counted here but *not logged* here. The log line belongs at the
two places the :class:`GuardError` is raised — :meth:`app.models.base.AnomalyModel._check_frame`
for the model path and ``predict`` in :mod:`app.serving.main` for the API path —
because those are the places that hold the request context (``request_id``,
``model_backend``, ``category``) that makes the line worth reading. Logging here
as well would double every rejection; logging in the API's *exception handler*
instead would silently lose that context, for a reason worth reading in
:func:`app.serving.main._handle_guard_error`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Callable, TypeVar

import cv2
import numpy as np

from app.observability.metrics import observe_guard_check, record_guard_rejection

__all__ = ["GuardConfig", "GuardError", "GuardResult", "FrameGuard"]

_T = TypeVar("_T")

#: A float frame already scaled to ``[0, 1]`` never exceeds this; anything above
#: it is a float array still on the 0-255 scale and is left alone. Mirrors
#: ``app.models.base._UNIT_RANGE_TOLERANCE`` so the guard and the model agree on
#: what "already normalized" means.
_UNIT_RANGE_TOLERANCE = 1.0 + 1e-3


@dataclass(frozen=True)
class GuardResult:
    """The verdict on one frame.

    Attributes:
        passed: Whether the frame cleared every check.
        reason: ``None`` when it passed, otherwise a short machine-readable slug
            for the first failing check — one of ``"blurry"``,
            ``"underexposed"``, ``"overexposed"``, ``"too_small"`` or
            ``"invalid_aspect_ratio"``.
        metrics: The raw numeric value behind every check, always fully
            populated regardless of pass/fail. Callers are expected to log these
            per frame: watching the distribution of, say, ``laplacian_variance``
            drift downward over a shift is how you catch a lens slowly fouling
            *before* it crosses the threshold and starts rejecting frames.
    """

    passed: bool
    reason: str | None
    metrics: dict[str, float]


class GuardError(Exception):
    """Raised on the prediction path when a frame fails the quality gate.

    Carries the full :class:`GuardResult` so a handler can surface the reason to
    an operator and log the metrics without re-running the guard.

    Attributes:
        result: The failing :class:`GuardResult`.
        reason: Shortcut for ``result.reason``.
        metrics: Shortcut for ``result.metrics``.
    """

    def __init__(self, result: GuardResult) -> None:
        self.result = result
        self.reason = result.reason
        self.metrics = result.metrics
        super().__init__(f"frame rejected by input-quality guard: {result.reason}")


@dataclass(frozen=True)
class GuardConfig:
    """Thresholds for :class:`FrameGuard`, each overridable from the environment.

    Defaults match the project spec. :meth:`from_env` resolves each field from
    its environment variable, falling back to the default; an unset or blank
    variable uses the default rather than failing, matching
    :meth:`app.models.config.ModelConfig.from_env`.

    Attributes:
        blur_threshold: Reject when Laplacian variance is below this. ``50.0``.
        min_resolution: Reject when either side is below this, in pixels. ``64``.
        max_aspect_ratio: Reject when ``width/height`` is above this or below its
            reciprocal. ``10.0``.
        dark_level: A pixel at or below this counts as floor-clipped. ``10``.
        bright_level: A pixel at or above this counts as ceiling-clipped. ``245``.
        exposure_fraction: Reject when this fraction of pixels is floor- or
            ceiling-clipped. ``0.95``.
        blur_resize_edge: Longest edge, in pixels, the frame is downscaled to
            before the Laplacian variance is measured. See
            :meth:`FrameGuard._laplacian_variance` for why the blur measure is
            taken at a fixed resolution rather than on the raw frame. ``256``.
    """

    blur_threshold: float = 50.0
    min_resolution: int = 64
    max_aspect_ratio: float = 10.0
    dark_level: int = 10
    bright_level: int = 245
    exposure_fraction: float = 0.95
    blur_resize_edge: int = 256

    #: field -> environment variable it is read from.
    _ENV: dict[str, str] = field(
        default_factory=lambda: {
            "blur_threshold": "BLUR_THRESHOLD",
            "min_resolution": "MIN_RESOLUTION",
            "max_aspect_ratio": "MAX_ASPECT_RATIO",
            "dark_level": "EXPOSURE_DARK_LEVEL",
            "bright_level": "EXPOSURE_BRIGHT_LEVEL",
            "exposure_fraction": "EXPOSURE_FRACTION",
            "blur_resize_edge": "BLUR_RESIZE_EDGE",
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.blur_threshold < 0:
            msg = f"blur_threshold must be >= 0, got {self.blur_threshold}."
            raise ValueError(msg)
        if self.min_resolution < 1:
            msg = f"min_resolution must be >= 1, got {self.min_resolution}."
            raise ValueError(msg)
        if self.max_aspect_ratio < 1.0:
            msg = f"max_aspect_ratio must be >= 1.0 (it is compared against its own reciprocal), got {self.max_aspect_ratio}."
            raise ValueError(msg)
        if not 0 <= self.dark_level < self.bright_level <= 255:
            msg = f"require 0 <= dark_level < bright_level <= 255, got dark_level={self.dark_level}, bright_level={self.bright_level}."
            raise ValueError(msg)
        if not 0.0 < self.exposure_fraction <= 1.0:
            msg = f"exposure_fraction must be in (0, 1], got {self.exposure_fraction}."
            raise ValueError(msg)
        if self.blur_resize_edge < 1:
            msg = f"blur_resize_edge must be >= 1, got {self.blur_resize_edge}."
            raise ValueError(msg)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GuardConfig:
        """Build a config from the defaults, overridden by environment variables.

        Args:
            environ: Mapping to read. Defaults to ``os.environ``.

        Returns:
            A validated :class:`GuardConfig`. A blank or missing variable falls
            back to the field default.
        """
        source = os.environ if environ is None else environ
        defaults = cls()

        def _read(name: str, current: _T, cast: Callable[[str], _T]) -> _T:
            raw = source.get(name)
            if raw is None or raw.strip() == "":
                return current
            return cast(raw.strip())

        env = defaults._ENV
        return cls(
            blur_threshold=_read(env["blur_threshold"], defaults.blur_threshold, float),
            min_resolution=_read(env["min_resolution"], defaults.min_resolution, int),
            max_aspect_ratio=_read(env["max_aspect_ratio"], defaults.max_aspect_ratio, float),
            dark_level=_read(env["dark_level"], defaults.dark_level, int),
            bright_level=_read(env["bright_level"], defaults.bright_level, int),
            exposure_fraction=_read(env["exposure_fraction"], defaults.exposure_fraction, float),
            blur_resize_edge=_read(env["blur_resize_edge"], defaults.blur_resize_edge, int),
        )

    def with_overrides(self, **overrides: object) -> GuardConfig:
        """Return a copy with ``overrides`` applied; ``None`` values are ignored."""
        return replace(self, **{key: value for key, value in overrides.items() if value is not None})


class FrameGuard:
    """Gate raw frames on image quality before an anomaly model ever sees them.

    Stateless apart from its :class:`GuardConfig`, so a single instance is safe
    to share across threads and requests — which is exactly how
    :mod:`app.guardrails` uses the module-level :data:`~app.guardrails.guard`.

    Example:
        >>> guard = FrameGuard()
        >>> guard.validate(np.zeros((256, 256, 3), dtype=np.uint8)).reason
        'underexposed'
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config if config is not None else GuardConfig.from_env()

    def __repr__(self) -> str:
        return f"FrameGuard(config={self.config!r})"

    def validate(self, image: np.ndarray) -> GuardResult:
        """Check one frame and report a verdict plus every metric behind it.

        Timed into ``guard_check_latency_seconds`` and, on a rejection, counted
        into ``guard_rejections_total{reason}`` — see the module docstring for
        why the instrumentation lives here and the log line does not. The timing
        is in a ``finally``, so a frame that raises on its way through is still
        observed: a malformed input is the one case where the guard could
        plausibly become slow, and it would otherwise be the one case missing
        from the histogram.

        Args:
            image: ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``
                array; ``uint8`` in ``[0, 255]`` or float (``[0, 1]`` or the
                0-255 scale — both are handled, see :meth:`_to_gray`).

        Returns:
            A :class:`GuardResult`.

        Raises:
            TypeError: If ``image`` is not a NumPy array.
            ValueError: If it is not a 2-D or 3-D array, or has a zero-sized
                dimension.
        """
        with observe_guard_check():
            result = self._evaluate(image)

        if not result.passed and result.reason is not None:
            record_guard_rejection(result.reason)
        return result

    def _evaluate(self, image: np.ndarray) -> GuardResult:
        """Run the checks. The verdict itself, with no instrumentation around it.

        The checks are evaluated most-fundamental first — resolution, then
        aspect ratio, then exposure, then blur — and the first failure decides
        :attr:`GuardResult.reason`. That order is deliberate: a 10x10 all-black
        tile fails several checks at once, and "too_small" is the more useful
        thing to tell an operator than "underexposed". Every metric is computed
        regardless, so a rejection still logs a full row for drift monitoring.
        """
        array = self._as_frame(image)
        height, width = array.shape[:2]
        gray = self._to_gray(array)
        cfg = self.config

        aspect_ratio = width / height
        metrics: dict[str, float] = {
            "laplacian_variance": self._laplacian_variance(gray),
            "dark_fraction": float((gray < cfg.dark_level).mean()),
            "bright_fraction": float((gray > cfg.bright_level).mean()),
            "width": float(width),
            "height": float(height),
            "min_dimension": float(min(height, width)),
            "aspect_ratio": float(aspect_ratio),
        }

        # (reason, failed?) in priority order; the first True wins.
        checks: list[tuple[str, bool]] = [
            ("too_small", metrics["min_dimension"] < cfg.min_resolution),
            (
                "invalid_aspect_ratio",
                aspect_ratio > cfg.max_aspect_ratio or aspect_ratio < 1.0 / cfg.max_aspect_ratio,
            ),
            ("underexposed", metrics["dark_fraction"] > cfg.exposure_fraction),
            ("overexposed", metrics["bright_fraction"] > cfg.exposure_fraction),
            ("blurry", metrics["laplacian_variance"] < cfg.blur_threshold),
        ]
        for reason, failed in checks:
            if failed:
                return GuardResult(passed=False, reason=reason, metrics=metrics)
        return GuardResult(passed=True, reason=None, metrics=metrics)

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _as_frame(image: np.ndarray) -> np.ndarray:
        """Validate the shape of a raw frame and return it as an array.

        Deliberately permissive on channel count — the guard's job is quality,
        not the channel/dtype contract that ``AnomalyModel._to_rgb_array``
        already enforces on the model path. It only insists the input is an
        image-shaped array with real pixels in it.
        """
        if not isinstance(image, np.ndarray):
            msg = f"validate() expects a numpy.ndarray, got {type(image).__name__}."
            raise TypeError(msg)
        if image.ndim not in {2, 3}:
            msg = f"validate() expects a 2-D or 3-D image array, got shape {image.shape}."
            raise ValueError(msg)
        if image.size == 0 or 0 in image.shape[:2]:
            msg = f"validate() got an empty frame with shape {image.shape}."
            raise ValueError(msg)
        return image

    @staticmethod
    def _to_gray(array: np.ndarray) -> np.ndarray:
        """Reduce a frame to a single-channel ``float64`` image on the 0-255 scale.

        Two normalizations happen here so the fixed thresholds mean the same
        thing for every frame that can reach the guard:

        * **Channels collapse to luminance** by averaging (alpha, if present, is
          dropped first). Averaging rather than a weighted BGR/RGB luma keeps the
          measure independent of channel order — the guard is not told whether it
          was handed BGR or RGB, and for a focus/exposure check the difference is
          immaterial.
        * **Float ``[0, 1]`` frames are rescaled to 0-255.** A model accepts both
          a ``uint8`` frame and a float ``[0, 1]`` one; left as-is, a perfectly
          exposed float frame would read as 100% floor-clipped (every value below
          ``dark_level=10``) and be wrongly rejected. The 0-255-scale float case
          (values already above 1) is left untouched.
        """
        if array.ndim == 3:
            channels = array.shape[2]
            if channels == 1:
                array = array[:, :, 0]
            elif channels >= 3:
                array = array[:, :, :3].mean(axis=2)
            else:  # a 2-channel oddity; average what is there
                array = array.mean(axis=2)

        gray = array.astype(np.float64)
        if np.issubdtype(np.asarray(array).dtype, np.floating) and gray.size and float(gray.max()) <= _UNIT_RANGE_TOLERANCE:
            gray = gray * 255.0
        return np.clip(gray, 0.0, 255.0)

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Focus measure: variance of the Laplacian, taken at a fixed resolution.

        The subtlety worth stating, because it is the one non-obvious decision in
        this module: **Laplacian variance is not scale-free.** Downscaling a
        sharp image concentrates its edge energy into fewer pixels and raises the
        variance; upscaling spreads it and lowers it. On this project's data the
        effect is large — a sharp MVTec bottle scores ~30 at its native 900x900,
        ~90 at 512, ~280 at 256. A single absolute threshold like ``50`` is
        therefore meaningless unless it is tied to a resolution, and a threshold
        that worked for a 256px camera would reject every sharp frame from a 4K
        one.

        So the measure is taken at a fixed working resolution: the frame is
        downscaled so its longest edge is :attr:`GuardConfig.blur_resize_edge`
        (default 256), and never *upscaled* — a frame already smaller than that
        is judged at its own resolution, where a genuinely sharp small frame
        already scores high. This makes ``blur_threshold`` portable across camera
        resolutions, which is the only way it can have one default.

        The area interpolation used for the downscale is the right one for
        shrinking: it averages the source pixels rather than sampling them, so it
        does not manufacture aliasing edges that would inflate the variance of a
        frame that is actually soft.
        """
        edge = self.config.blur_resize_edge
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest > edge:
            scale = edge / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        # float64 in, CV_64F out: OpenCV 5 rejects a float32 source with a 64-bit
        # destination, and float64 keeps the variance exact.
        laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        return float(laplacian.var())
