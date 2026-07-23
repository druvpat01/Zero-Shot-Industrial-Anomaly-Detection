"""Model layer for the zero-shot industrial defect detector.

This package is the codebase's only dependency on ``anomalib.models``. Import
:class:`AnomalyModel`, :class:`ModelOutput` and a concrete backend from here;
downstream code should type against :class:`AnomalyModel`, not against a
specific model.
"""

from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig, get_model_config
from app.models.patchcore import PatchCoreModel

__all__ = [
    "AnomalyModel",
    "ModelConfig",
    "ModelOutput",
    "PatchCoreModel",
    "get_model_config",
]
