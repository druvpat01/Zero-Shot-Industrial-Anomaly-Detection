"""Choosing the operating point: turning a score into a decision.

The gap this fills
==================
Every model in this project emits a continuous ``anomaly_score``, and every
metric in :mod:`app.evaluation.metrics` that judges it well —
:func:`~app.evaluation.metrics.image_auroc`, :func:`~app.evaluation.metrics.au_pro`
— is deliberately **threshold-free**. That is the right way to measure a
detector: AUROC reads the ranking the scores induce and is invariant to any
monotonic rescaling of them, so it compares PatchCore's raw distances against
WinCLIP's softmax probabilities without either being penalised for its scale.

A production line cannot ship a ranking. It has to answer *pass or scrap* for
one part, which means committing to a single number: the **operating point**,
the score at or above which a frame is called defective. AUROC says nothing
about where that number should sit — a model with 0.99 AUROC still fails
completely at a badly chosen threshold, flagging everything or nothing. Picking
it is a separate decision from training the model, it is made on labelled data
the model has never seen, and it is the only decision in the pipeline that
directly trades false alarms against missed defects.

This module makes that choice explicit and re-makeable. ``ANOMALY_THRESHOLD``
(default ``0.5``) is a sensible starting point for a calibrated backend and
nothing more; :func:`find_optimal_threshold` replaces the guess with a number
fitted to real labelled scores, and ``POST /calibrate`` applies it to a running
service without a redeploy.

Why the threshold goes stale, and what re-fitting it means
==========================================================
The operating point is fitted to a score *distribution*, so it is only as valid
as that distribution is current. When the line changes — new supplier, aged
camera, re-exported graph — the scores move and the threshold does not follow.
:mod:`app.evaluation.drift` is what notices this; recalibration is the cheapest
of the responses it recommends, because in the common case the model is entirely
fine and only the boundary is stale. That is why the two modules sit next to each
other and why ``POST /calibrate`` sets the drift reference at the same time it
sets the threshold: the calibration set *is* the new definition of normal.

Which metric to maximise
========================
:func:`find_optimal_threshold` sweeps every threshold the scores can produce and
returns the one maximising a metric of the caller's choosing. The choice encodes
what the line's mistakes cost, and it is a business decision rather than a
statistical one:

===================  ==========================================================
``metric``           Maximises, and when to pick it
===================  ==========================================================
``f1``               Harmonic mean of precision and recall. The default, and
                     right when false alarms and missed defects cost roughly the
                     same. Balanced by construction: it collapses if either term
                     does, so it cannot be gamed by flagging everything.
``balanced_accuracy``Mean of recall and specificity. Prefer it over ``f1`` when
                     the calibration set is heavily imbalanced and you want the
                     threshold judged on both classes equally rather than
                     weighted toward the positives.
``recall``           Catch every defect, at whatever false-alarm cost. Honest
                     only when a missed defect is genuinely catastrophic *and* a
                     human re-inspects the flagged parts — see the degeneracy
                     note below.
``precision``        Never cry wolf. For an automatic-scrap line where a false
                     positive destroys a good part.
===================  ==========================================================

**``precision`` and ``recall`` alone are degenerate and the docstring says so
rather than hiding it.** Recall is maximised by flagging everything (threshold
below the minimum score: recall 1.0, precision near the defect rate); precision
is maximised by flagging almost nothing. Both are therefore only meaningful with
a floor on the other, which this module does not implement — it would be a
constrained optimisation with a second parameter, and inventing that API before
anyone has asked for it would be fiction. What it does instead is break ties
toward the *lowest* qualifying threshold, which turns "maximise precision" into
"the most sensitive threshold that achieves the best precision" — the useful
reading of the request rather than the useless one.

The tie-break, and why it points that way
=========================================
Many thresholds usually achieve the same maximum, because the metric only
changes where the confusion matrix does. Among them this returns the **lowest**,
which is the most sensitive: at equal F1 it catches more defects and raises more
false alarms. That is the correct default for industrial QA, where a false alarm
costs an operator thirty seconds at a re-inspection station and a missed defect
ships. A deployment that disagrees should choose ``precision`` explicitly rather
than have the tie-break quietly reversed under it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.observability.logging_config import get_logger

__all__ = [
    "CALIBRATION_METRICS",
    "DEFAULT_METRIC",
    "evaluate_threshold",
    "find_optimal_threshold",
]

log = get_logger(__name__)

#: The metric maximised when the caller does not say.
DEFAULT_METRIC = "f1"

#: Metrics :func:`find_optimal_threshold` knows how to maximise. A closed set,
#: validated at the boundary, so a typo is an immediate error rather than a
#: threshold fitted to something nobody asked for.
CALIBRATION_METRICS: tuple[str, ...] = ("f1", "precision", "recall", "balanced_accuracy")


def find_optimal_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    metric: str = DEFAULT_METRIC,
) -> float:
    """Return the decision threshold maximising ``metric`` on a labelled set.

    A frame is called defective when ``score >= threshold``, matching every
    ``is_defective`` in the model layer, so the number returned here can be
    dropped straight into
    :attr:`~app.models.config.ModelConfig.anomaly_threshold`.

    The sweep is **exact, not a grid**. Candidate thresholds are the distinct
    score values themselves: the confusion matrix can only change where a score
    crosses the boundary, so every distinct score is one candidate and there is
    nothing between them to miss. Sorting once and taking cumulative sums makes
    the whole sweep O(n log n) rather than O(n²) — a 10,000-sample calibration
    set is milliseconds, which is what lets this run inside a request handler.

    Args:
        scores: Image-level anomaly scores; higher means "more anomalous".
        labels: Ground truth aligned with ``scores``: ``1`` defective, ``0``
            normal.
        metric: One of :data:`CALIBRATION_METRICS`. See the module docstring for
            which to pick, and for why ``precision`` and ``recall`` on their own
            are degenerate.

    Returns:
        The threshold. Ties are broken toward the lowest (most sensitive)
        qualifying value — see the module docstring.

    Raises:
        ValueError: If the inputs are empty or mismatched in length, if a label
            is outside ``{0, 1}``, if a score is not finite, if only one class is
            present, or if ``metric`` is not in :data:`CALIBRATION_METRICS`.
            Every one of these produces a meaningless threshold rather than a
            wrong-but-plausible one, and a calibration endpoint that silently
            accepts a single-class dataset would install a boundary that flags
            every part on the line.
    """
    score_array, label_array = _validate(scores, labels)
    metric_name = _validate_metric(metric)

    thresholds, values = _sweep(score_array, label_array, metric_name)

    # `thresholds` descends, so the last index achieving the maximum is the
    # lowest threshold that achieves it. Reversing before argmax picks that one.
    best = values.size - 1 - int(np.argmax(values[::-1]))

    log.info(
        "threshold_calibrated",
        metric=metric_name,
        threshold=round(float(thresholds[best]), 6),
        metric_value=round(float(values[best]), 6),
        samples=int(score_array.size),
        positives=int(label_array.sum()),
    )
    return float(thresholds[best])


def evaluate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    metric: str = DEFAULT_METRIC,
) -> float:
    """Score one threshold on a labelled set, without sweeping.

    The companion to :func:`find_optimal_threshold`: it reports what the fitted
    threshold actually achieves, which is the number that decides whether the
    calibration was worth applying. A "best" F1 of 0.42 is still the best
    available and still means the operating point cannot be rescued by moving it.

    Args:
        scores: Image-level anomaly scores.
        labels: Ground truth aligned with ``scores``; ``1`` defective.
        threshold: The boundary to evaluate; a frame counts as defective when
            ``score >= threshold``.
        metric: One of :data:`CALIBRATION_METRICS`.

    Returns:
        The metric's value at ``threshold``, in ``[0, 1]``.

    Raises:
        ValueError: Same conditions as :func:`find_optimal_threshold`.
    """
    score_array, label_array = _validate(scores, labels)
    metric_name = _validate_metric(metric)

    predicted = score_array >= float(threshold)
    positives = label_array == 1
    tp = float(np.count_nonzero(predicted & positives))
    fp = float(np.count_nonzero(predicted & ~positives))
    fn = float(np.count_nonzero(~predicted & positives))
    tn = float(np.count_nonzero(~predicted & ~positives))
    return float(_metric_value(metric_name, np.array([tp]), np.array([fp]), np.array([fn]), np.array([tn]))[0])


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check a calibration set, or explain exactly what is wrong with it.

    Every message names the offending value, because these errors surface to an
    operator through ``POST /calibrate`` as a 422 body rather than to a developer
    through a traceback.
    """
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels))

    if score_array.size == 0:
        msg = "a calibration set needs at least one sample, got none."
        raise ValueError(msg)
    if score_array.shape != label_array.shape:
        msg = f"scores and labels must be the same length, got {score_array.shape} and {label_array.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(score_array)):
        msg = "every score must be finite; a NaN or inf makes the threshold sweep meaningless."
        raise ValueError(msg)

    label_array = label_array.astype(int)
    unknown = np.unique(label_array[(label_array != 0) & (label_array != 1)])
    if unknown.size:
        msg = f"labels must be 0 (normal) or 1 (defective), got {unknown.tolist()}."
        raise ValueError(msg)
    if np.unique(label_array).size < 2:
        present = "defective" if label_array[0] == 1 else "normal"
        msg = (
            f"a calibration set needs both classes; every sample here is {present}. "
            "With one class the sweep is degenerate: the best threshold is whichever "
            "extreme flags them all, which is not an operating point."
        )
        raise ValueError(msg)

    return score_array, label_array


def _validate_metric(metric: str) -> str:
    """Normalise and check the requested metric name."""
    name = metric.strip().lower()
    if name not in CALIBRATION_METRICS:
        msg = f"unknown metric {metric!r}; choose from {list(CALIBRATION_METRICS)}."
        raise ValueError(msg)
    return name


def _sweep(scores: np.ndarray, labels: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Every candidate threshold and the metric it achieves, descending by threshold.

    Sorts by descending score and accumulates: at the ``i``-th sorted position,
    everything up to and including it is predicted defective, so ``TP`` and
    ``FP`` are running counts of the labels seen so far. Keeping only the last
    index of each run of equal scores is what makes ``score >= threshold`` exact
    — a threshold landing mid-run would split identical scores into different
    verdicts, which no real decision rule can do.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels).astype(float)
    fp = np.cumsum(1 - sorted_labels).astype(float)

    last_of_run = np.append(np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1)
    thresholds = sorted_scores[last_of_run]
    tp, fp = tp[last_of_run], fp[last_of_run]

    total_positive = float(labels.sum())
    total_negative = float(labels.size) - total_positive
    fn = total_positive - tp
    tn = total_negative - fp

    return thresholds, _metric_value(metric, tp, fp, fn, tn)


def _metric_value(metric: str, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Evaluate ``metric`` from confusion-matrix counts, vectorised over thresholds.

    Every division guards its own denominator and yields ``0.0`` rather than
    ``nan`` when it is empty. That is the operationally correct reading: a
    threshold that predicts no positives has undefined precision in the textbook
    but is worth exactly nothing as an operating point, and letting a ``nan``
    through would make ``argmax`` pick it.
    """
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)

    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    if metric == "balanced_accuracy":
        specificity = _safe_divide(tn, tn + fp)
        return (recall + specificity) / 2.0
    return _safe_divide(2.0 * precision * recall, precision + recall)
