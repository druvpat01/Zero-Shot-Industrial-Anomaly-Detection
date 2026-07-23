"""Evaluation layer: metrics and the model benchmark runner.

Import the four metrics and :class:`BenchmarkRunner` from here. Everything in
this package is written against :class:`~app.models.base.ModelOutput` and plain
NumPy — it never imports anomalib — so the numbers it produces can be explained
line by line and stay decoupled from how any single backend works.
"""

from app.evaluation.benchmark import BenchmarkResult, BenchmarkRunner
from app.evaluation.metrics import au_pro, f1_at_best_threshold, image_auroc, pixel_auroc

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "au_pro",
    "f1_at_best_threshold",
    "image_auroc",
    "pixel_auroc",
]
