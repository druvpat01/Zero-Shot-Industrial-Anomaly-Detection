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

from app.models.config import ModelConfig

__all__ = ["AnomalyModel", "ModelOutput"]


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
