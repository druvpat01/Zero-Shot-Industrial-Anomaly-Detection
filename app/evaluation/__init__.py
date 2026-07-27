"""Evaluation layer: metrics, the benchmark runner, and the two reliability pieces.

Import the four metrics and :class:`BenchmarkRunner` from here. Everything in
this package is written against :class:`~app.models.base.ModelOutput` and plain
NumPy — it never imports anomalib — so the numbers it produces can be explained
line by line and stay decoupled from how any single backend works.

Two of the modules answer offline questions and two answer live ones:

* :mod:`~app.evaluation.metrics` and :mod:`~app.evaluation.benchmark` measure a
  model against ground truth, threshold-free, on a test split.
* :mod:`~app.evaluation.calibration` picks the operating point that turns those
  scores into pass/scrap decisions, and
  :mod:`~app.evaluation.drift` watches whether the score distribution the
  operating point was fitted to is still the one arriving. They are a pair: drift
  is what makes a threshold stale, and recalibration is the cheapest fix for it.
"""

from app.evaluation.benchmark import BenchmarkResult, BenchmarkRunner
from app.evaluation.calibration import (
    CALIBRATION_METRICS,
    DEFAULT_METRIC,
    evaluate_threshold,
    find_optimal_threshold,
)
from app.evaluation.drift import (
    DEFAULT_KS_DRIFT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    REASON_KS_DRIFT,
    ScoreDistributionMonitor,
    ks_drift_threshold,
)
from app.evaluation.metrics import au_pro, f1_at_best_threshold, image_auroc, pixel_auroc

__all__ = [
    "CALIBRATION_METRICS",
    "DEFAULT_KS_DRIFT_THRESHOLD",
    "DEFAULT_METRIC",
    "DEFAULT_WINDOW_SIZE",
    "REASON_KS_DRIFT",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ScoreDistributionMonitor",
    "au_pro",
    "evaluate_threshold",
    "f1_at_best_threshold",
    "find_optimal_threshold",
    "image_auroc",
    "ks_drift_threshold",
    "pixel_auroc",
]
