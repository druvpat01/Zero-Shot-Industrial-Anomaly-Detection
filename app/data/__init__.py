"""Data layer for the zero-shot industrial defect detector.

This package is the codebase's only dependency on anomalib's data classes.
Import :class:`DataModule` and :class:`DefectBatch` from here; do not import
``anomalib.data`` anywhere else.
"""

from app.data.batch import BATCH_KEYS, DefectBatch
from app.data.datamodule import (
    DEFAULT_CATEGORY,
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGE_SIZE,
    DataModule,
)
from app.data.transforms import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize_image,
    normalize_image,
    to_tensor,
    validate_image_shape,
)

__all__ = [
    "BATCH_KEYS",
    "CLIP_MEAN",
    "CLIP_STD",
    "DEFAULT_CATEGORY",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DataModule",
    "DefectBatch",
    "denormalize_image",
    "normalize_image",
    "to_tensor",
    "validate_image_shape",
]
