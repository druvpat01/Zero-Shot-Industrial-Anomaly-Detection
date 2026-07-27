"""The interface every anomaly-detection backend in this project implements.

Why this exists
---------------
PatchCore, EfficientAD and WinCLIP are all reached through anomalib, but they
have meaningfully different APIs there — different constructor arguments,
different training semantics (memory bank vs. distillation vs. zero-shot), and
different output containers. If the serving layer talked to anomalib directly it
would have to know which of those it was holding.

:class:`AnomalyModel` is the seam instead. Everything downstream depends on four
methods and one output type, so swapping the backend behind
``MODEL_BACKEND=patchcore`` is a factory change, not a rewrite.

The mirror of this idea on the data side is :class:`app.data.DataModule`, which
keeps anomalib's *input* types from leaking; :class:`ModelOutput` keeps its
*output* types from leaking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias

from app.data.transforms import to_tensor
from app.guardrails import GuardError, GuardResult, guard
from app.models.config import ModelConfig
from app.observability.logging_config import get_logger

__all__ = ["AnomalyModel", "ModelOutput"]

log = get_logger(__name__)

#: A float image already scaled to [0, 1] never exceeds this; anything above it
#: is a float array still on the 0-255 scale, which we rescale rather than reject.
_UNIT_RANGE_TOLERANCE = 1.0 + 1e-3


@dataclass(frozen=True)
class ModelOutput:
    """The result of scoring one image, in plain NumPy/Python types.

    Attributes:
        anomaly_score: Image-level score. Calibrated models emit ``[0, 1]``,
            where ``0.5`` is the adaptive decision boundary fitted at training
            time (see :meth:`AnomalyModel.is_calibrated`); an uncalibrated model
            emits raw, unbounded distances instead.
        anomaly_map: ``(H, W)`` ``float32`` pixel-level heatmap at the *input*
            image's resolution, not the model's internal working resolution — a
            caller can overlay it on the frame they passed in without rescaling.
            Uses the same scale as ``anomaly_score``.
        is_defective: Whether ``anomaly_score`` cleared the configured
            :attr:`~app.models.config.ModelConfig.anomaly_threshold`.
        model_name: Backend that produced this result, e.g. ``"patchcore"``.
            Carried on the output so logs and API responses can attribute a
            score without the caller tracking which model served it.
    """

    anomaly_score: float
    anomaly_map: np.ndarray
    is_defective: bool
    model_name: str

    def __post_init__(self) -> None:
        if self.anomaly_map.ndim != 2:
            msg = f"anomaly_map must be a 2-D (H, W) array, got shape {self.anomaly_map.shape}."
            raise ValueError(msg)
        if not np.isfinite(self.anomaly_score):
            msg = f"anomaly_score must be finite, got {self.anomaly_score}."
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the anomaly map."""
        height, width = self.anomaly_map.shape
        return (int(height), int(width))

    def defective_area_ratio(self, threshold: float) -> float:
        """Fraction of pixels scoring at or above ``threshold``.

        The cheapest useful summary of *how much* of the part is affected, which
        image-level score alone cannot express: a hairline crack and a shattered
        bottle can both saturate ``anomaly_score``.
        """
        return float((self.anomaly_map >= threshold).mean())


class AnomalyModel(ABC):
    """Base class for anomaly-detection backends.

    Subclasses are expected to be constructible without touching disk or the
    network beyond backbone weights, and to raise a clear error from
    :meth:`predict` if they have not been trained or loaded yet.

    Attributes:
        model_name: Short identifier used in :class:`ModelOutput`, checkpoint
            filenames and log lines. Subclasses must override it.
    """

    model_name: str = "anomaly-model"

    #: Interpolation used to resize a raw frame to ``image_size``. Bilinear is
    #: right for a convolutional backbone; a backend whose weights were
    #: pretrained under a different resize (WinCLIP's ViT, which open_clip
    #: preprocesses with bicubic) overrides it rather than resizing twice.
    _resize_mode: str = "bilinear"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # -- lifecycle -------------------------------------------------------------

    @abstractmethod
    def train(self, datamodule: object) -> None:
        """Fit the model on the normal (defect-free) training split.

        Args:
            datamodule: An :class:`app.data.DataModule` supplying the category's
                train/val/test splits.
        """

    @abstractmethod
    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        """Score a single raw image.

        Args:
            image: ``(H, W, 3)``, ``(H, W, 4)`` or ``(H, W)`` array of any size,
                ``uint8`` in ``[0, 255]`` or float in ``[0, 1]``.
            color_order: ``"rgb"`` (default) or ``"bgr"``. OpenCV hands back
                BGR, so anything coming from ``cv2.imread``/``VideoCapture``
                must say so — the backbone is ImageNet-pretrained and its
                features degrade on channel-swapped input.

        Returns:
            A :class:`ModelOutput` whose ``anomaly_map`` matches ``image``'s
            height and width.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model to ``path``."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore a model previously written by :meth:`save`."""

    # -- introspection ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`predict` can run. Subclasses should override."""
        return False

    @property
    def is_calibrated(self) -> bool:
        """Whether scores are normalized to ``[0, 1]``.

        Calibration statistics are fitted on a validation pass, so a model that
        was trained without one still predicts — just on a raw distance scale.
        """
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_name={self.model_name!r}, trained={self.is_trained})"

    # -- shared image plumbing -------------------------------------------------
    #
    # :meth:`predict` promises the same thing for every backend: any raw frame
    # in, a heatmap at *that frame's* resolution out. Everything below is the
    # machinery that keeps that promise, and it lives here rather than in each
    # wrapper so the promise cannot quietly drift apart between them. Backends
    # differ only in :meth:`_scale_for_model`.

    def _preprocess(self, image: np.ndarray, *, color_order: str = "rgb") -> torch.Tensor:
        """Turn an arbitrary raw frame into a ``(1, 3, S, S)`` model input tensor.

        Args:
            image: Raw image array.
            color_order: ``"rgb"`` or ``"bgr"``.

        Returns:
            Batched tensor at ``image_size``, scaled the way this backend wants.
        """
        return self._to_model_input(self._to_rgb_array(image, color_order=color_order))

    @staticmethod
    def _guard_frame(frame: np.ndarray) -> GuardResult:
        """Run the input-quality guard and raise if the frame is not worth scoring.

        Called by every backend's :meth:`predict` before the resize-and-normalize
        that feeds the network, so a camera glitch, fouled lens or lighting
        failure is refused loudly here instead of being turned into a
        plausible-looking anomaly score downstream. The guard is model-agnostic,
        so it lives once on the base class and each :meth:`predict` invokes it —
        the check is identical no matter which backend serves the frame.

        It runs on the parsed 3-channel frame from :meth:`_to_rgb_array` rather
        than the caller's raw argument, so structural problems (wrong type, rank
        or channel count) still surface as the :class:`TypeError`/
        :class:`ValueError` that method raises; the guard only judges quality.
        The channel-order and channel-count normalization :meth:`_to_rgb_array`
        does leaves the pixels' luminance untouched, so the quality verdict is
        the raw frame's.

        Args:
            frame: The ``(H, W, 3)`` RGB frame returned by :meth:`_to_rgb_array`.

        Returns:
            The passing :class:`~app.guardrails.GuardResult`, so a caller can log
            its metrics for drift monitoring.

        Raises:
            GuardError: If the frame fails any quality check, with the reason and
                the metrics attached.
        """
        verdict = guard.validate(frame)
        if not verdict.passed:
            # The one log line covering *every* caller of the model layer — a
            # script, a notebook, a benchmark run — not just the API. The API
            # path rejects in its own handler before a model is ever fetched, so
            # these two never fire for the same frame.
            #
            # The guard's metrics are spread into the record rather than nested
            # under a `metrics` key: `laplacian_variance` as a top-level field is
            # a number a log backend will aggregate and alert on, whereas the
            # same value inside a JSON blob is a string somebody has to parse.
            log.warning(
                "guard_rejected",
                reason=verdict.reason,
                model_backend=self.model_name,
                source="model",
                **verdict.metrics,
            )
            raise GuardError(verdict)
        return verdict

    @staticmethod
    def _to_rgb_array(image: np.ndarray, *, color_order: str = "rgb") -> np.ndarray:
        """Validate a raw frame and return it as a contiguous ``(H, W, 3)`` RGB array.

        Args:
            image: Raw image array.
            color_order: ``"rgb"`` or ``"bgr"``.

        Returns:
            The image as 3-channel RGB, dtype unchanged.

        Raises:
            TypeError: If ``image`` is not a NumPy array.
            ValueError: On an unsupported rank, channel count or ``color_order``.
        """
        if not isinstance(image, np.ndarray):
            msg = f"predict() expects a numpy.ndarray, got {type(image).__name__}."
            raise TypeError(msg)

        order = color_order.lower()
        if order not in {"rgb", "bgr"}:
            msg = f"color_order must be 'rgb' or 'bgr', got {color_order!r}."
            raise ValueError(msg)

        array = np.asarray(image)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            msg = f"predict() expects (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got shape {image.shape}."
            raise ValueError(msg)

        channels = array.shape[2]
        if channels == 1:  # grayscale sensor feed
            array = np.repeat(array, 3, axis=2)
        elif channels == 4:  # RGBA/BGRA screenshot or PNG with alpha
            array = array[:, :, :3]
        elif channels != 3:
            msg = f"predict() expects 1, 3 or 4 channels, got {channels} (shape {image.shape})."
            raise ValueError(msg)

        if order == "bgr":
            array = array[:, :, ::-1]

        return np.ascontiguousarray(array)

    def _to_model_input(self, array: np.ndarray) -> torch.Tensor:
        """Scale an ``(H, W, 3)`` RGB array to a ``(1, 3, S, S)`` model input tensor."""
        tensor = to_tensor(array)
        if float(tensor.max()) > _UNIT_RANGE_TOLERANCE:
            # A float frame still on the 0-255 scale; to_tensor only rescales uint8.
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0).unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=self.config.image_hw,
            mode=self._resize_mode,
            align_corners=False,
            antialias=True,
        )
        return self._scale_for_model(tensor)

    def _scale_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Hook: put a ``[0, 1]`` batch on the scale this backend's network expects.

        The default is the identity, which is right for any network that
        normalizes internally. Backends fed a bare timm backbone override this
        to apply ImageNet statistics.
        """
        return tensor

    @staticmethod
    def _to_input_resolution(anomaly_map: torch.Tensor, height: int, width: int) -> np.ndarray:
        """Upsample a model anomaly map to ``height x width`` as ``(H, W)`` float32.

        Models score at their own working resolution — a coarse feature grid for
        PatchCore, the padded input grid for EfficientAD. Returning the map at
        the *caller's* resolution is part of the :class:`ModelOutput` contract:
        the caller should never have to know what ``image_size`` was configured.
        """
        tensor = anomaly_map.detach().float()
        while tensor.ndim > 3:  # (N, 1, H, W) -> (N, H, W)
            tensor = tensor[:, 0]
        if tensor.ndim == 2:  # already (H, W)
            tensor = tensor.unsqueeze(0)

        resized = F.interpolate(
            tensor[:1].unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized[0, 0].cpu().numpy().astype(np.float32)
