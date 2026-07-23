"""Anomaly-detection metrics, implemented from scratch on sklearn and NumPy.

Why re-implement these rather than call anomalib's own metrics
-------------------------------------------------------------
anomalib ships ``AUROC``/``AUPRO`` torchmetrics, and they are fine. This module
deliberately does not use them, for two reasons that matter for a portfolio
project:

1. **Explainability.** Every number this project reports should be one I can
   derive on a whiteboard in an interview. So the ROC curves come from
   ``sklearn.metrics`` and the region logic is a dozen lines of NumPy —
   nothing here hides inside a metric's ``.update()/.compute()`` state machine.
2. **Decoupling.** The evaluation layer talks to :class:`ModelOutput`, not to
   anomalib tensors. A metric that imported anomalib would re-couple the two.

The four public functions map one-to-one onto the four numbers in the benchmark
table (:mod:`app.evaluation.benchmark`):

* :func:`image_auroc`  — is the *image* flagged? (detection)
* :func:`pixel_auroc`  — is each *pixel* flagged? (segmentation, area-weighted)
* :func:`au_pro`       — is each *defect region* flagged? (segmentation, region-weighted)
* :func:`f1_at_best_threshold` — the best single operating point for deployment.

The centrepiece is :func:`au_pro`; its docstring explains at length why it is the
metric to trust for industrial segmentation and pixel-AUROC is not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score

__all__ = [
    "au_pro",
    "f1_at_best_threshold",
    "image_auroc",
    "pixel_auroc",
]

logger = logging.getLogger(__name__)

#: MVTec AD's AU-PRO is, by convention, the area under the PRO/FPR curve
#: integrated only up to this false-positive rate and then normalized by it.
#: Beyond a 30% FPR the operating points are useless in practice, so scoring
#: them would reward behaviour no inspection line would ever run at. This is the
#: same limit anomalib's ``AUPRO`` and the original Bergmann et al. paper use, so
#: the numbers here are comparable to published ones. See :func:`au_pro`.
DEFAULT_FPR_LIMIT = 0.3

#: 8-connectivity (diagonals count) when labelling ground-truth defect regions,
#: matching ``skimage.measure.label``'s default for 2-D input, which is what the
#: reference AU-PRO implementations use. A hairline crack running diagonally is
#: one defect, not a dotted line of many, and region-averaging makes that choice
#: visible in the score.
_FULL_CONNECTIVITY = np.ones((3, 3), dtype=int)


# ---------------------------------------------------------------------------
# Image level
# ---------------------------------------------------------------------------


def image_auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Image-level ROC-AUC: how well the anomaly *score* ranks defective above normal.

    ROC-AUC is the probability that a randomly chosen defective image outscores a
    randomly chosen normal one. It reads the *ranking* the scores induce and is
    invariant to any monotonic rescaling of them — which is exactly what we want
    when different backends emit scores on different scales (PatchCore's raw
    distances, WinCLIP's softmax probabilities). It needs no threshold, so it
    measures the detector's ceiling rather than one chosen operating point.

    Args:
        y_true: Ground-truth labels, ``1`` anomalous and ``0`` normal.
        scores: Image-level anomaly scores; higher means "more anomalous".

    Returns:
        ROC-AUC in ``[0, 1]``. ``0.5`` is chance, ``1.0`` is a perfect ranking.
        Returns ``nan`` (with a warning) if only one class is present, since ROC
        is undefined without both a positive and a negative example.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        msg = f"y_true and scores must be the same length, got {y_true.shape} and {scores.shape}."
        raise ValueError(msg)
    if np.unique(y_true).size < 2:
        logger.warning("image_auroc: only one class present in y_true; ROC-AUC is undefined, returning nan.")
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def f1_at_best_threshold(y_true: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """Best achievable F1 over all thresholds, and the threshold that achieves it.

    ROC-AUC judges the ranking; deployment needs a *decision*, and that means
    committing to a threshold. This sweeps every threshold the scores can
    produce and returns the one maximising F1 — the harmonic mean of precision
    and recall, which balances "don't cry wolf" against "don't miss defects" in
    a single number. It is the natural companion to :func:`image_auroc`: the
    AUROC says how separable the classes are, this says how good the best cut is.

    The candidate thresholds come straight from
    :func:`sklearn.metrics.precision_recall_curve`, which returns precision and
    recall at exactly the score values where the confusion matrix changes — so
    the maximum found is exact, not a grid approximation.

    Args:
        y_true: Ground-truth labels, ``1`` anomalous and ``0`` normal.
        scores: Image-level anomaly scores; higher means "more anomalous".

    Returns:
        ``(best_f1, threshold)``. A frame scores as defective when
        ``score >= threshold``. Returns ``(nan, nan)`` if only one class is
        present.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(y_true).size < 2:
        logger.warning("f1_at_best_threshold: only one class present in y_true; F1 sweep is undefined.")
        return (float("nan"), float("nan"))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (the trailing (P=1, R=0) point that no finite threshold
    # produces); drop it so the arrays line up with `thresholds`.
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    best = int(np.argmax(f1))
    return (float(f1[best]), float(thresholds[best]))


# ---------------------------------------------------------------------------
# Pixel level
# ---------------------------------------------------------------------------


def _stack_pixels(gt_masks: object, pred_maps: object) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a batch of masks and heatmaps into two aligned 1-D pixel vectors.

    Accepts either a stacked ``(N, H, W)`` array or a sequence of 2-D arrays
    (the images need not share a resolution in the latter case), so callers can
    pass whatever is convenient. Ground truth is binarized at ``> 0``.
    """
    # Iterating a (N, H, W) array yields its N 2-D slices; iterating a sequence
    # yields its 2-D elements. Either way we get a per-image list to zip.
    gt_list = list(gt_masks)
    pred_list = list(pred_maps)
    if len(gt_list) != len(pred_list):
        msg = f"gt_masks and pred_maps must have the same length, got {len(gt_list)} and {len(pred_list)}."
        raise ValueError(msg)

    gt_flat, pred_flat = [], []
    for gt, pred in zip(gt_list, pred_list, strict=True):
        gt = np.asarray(gt)
        pred = np.asarray(pred, dtype=np.float32)
        if gt.shape != pred.shape:
            msg = f"each mask and heatmap must match in shape, got {gt.shape} vs {pred.shape}."
            raise ValueError(msg)
        gt_flat.append((gt > 0).ravel())
        pred_flat.append(pred.ravel())

    return np.concatenate(gt_flat), np.concatenate(pred_flat)


def pixel_auroc(gt_masks: object, pred_maps: object) -> float:
    """Pixel-level ROC-AUC over every pixel in the test set, pooled.

    Same idea as :func:`image_auroc` but the "examples" are pixels: pool every
    pixel of every test image (defective and normal alike), label each ``1`` if
    it lies inside a ground-truth defect and ``0`` otherwise, and take the ROC-AUC
    of the heatmap values against those labels. It answers "if I pick a random
    defect pixel and a random clean pixel, how often does the heatmap rank the
    defect pixel higher?".

    **Read this number with care — it is the metric this module also implements
    :func:`au_pro` to correct for.** Because every pixel votes equally, the score
    is dominated by *area*: MVTec's ``broken_large`` bottle covers tens of
    thousands of defect pixels while ``broken_small`` covers a few hundred, so a
    model that nails the big defects and entirely misses the small ones still
    posts a high pixel-AUROC. The overwhelming majority of pixels are also
    normal, which inflates the score further. It is reported because it is the
    field's most-quoted segmentation number and makes comparison to papers
    possible; it is not the number to optimise.

    Args:
        gt_masks: ``(N, H, W)`` array or a sequence of 2-D masks; nonzero means
            defect.
        pred_maps: Matching anomaly heatmaps, same shape(s).

    Returns:
        ROC-AUC in ``[0, 1]``, or ``nan`` if no defect pixels exist anywhere.
    """
    gt_flat, pred_flat = _stack_pixels(gt_masks, pred_maps)
    if np.unique(gt_flat).size < 2:
        logger.warning("pixel_auroc: no defect pixels (or no normal pixels) present; returning nan.")
        return float("nan")
    return float(roc_auc_score(gt_flat, pred_flat))


# ---------------------------------------------------------------------------
# Region level — the one to trust
# ---------------------------------------------------------------------------


def au_pro(
    gt_masks: object,
    pred_maps: object,
    num_thresholds: int = 100,
    *,
    fpr_limit: float = DEFAULT_FPR_LIMIT,
) -> float:
    """Area Under the Per-Region-Overlap curve — the gold-standard AD segmentation metric.

    Why AU-PRO exists, and why it beats pixel-AUROC
    ===============================================
    Pixel-AUROC (:func:`pixel_auroc`) weights every *pixel* equally, so its score
    is really a score for *area*. On MVTec bottle that is a problem you can see
    with the naked eye: a single ``broken_large`` smash contributes ~30x the
    defect pixels of a ``broken_small`` chip. A detector that segments the big
    defects perfectly and misses every small one is objectively poor at the job —
    small defects are the ones a human inspector misses and the ones that matter
    — yet it scores a near-perfect pixel-AUROC, because the pixels it got right
    vastly outnumber the ones it got wrong.

    AU-PRO removes area from the equation by weighting every *defect region*
    equally instead of every pixel:

    1. **Label the ground truth into connected components** — each contiguous
       defect blob is one "region", whether it is 300 pixels or 30,000
       (:data:`_FULL_CONNECTIVITY`, 8-connected, so a diagonal crack stays one
       region).
    2. **Per-Region Overlap (PRO) at a threshold ``t``.** Binarize the heatmap at
       ``t``. For each region, compute its *overlap* = the fraction of that
       region's pixels the threshold caught (its per-region recall). ``PRO(t)`` is
       the plain average of those overlaps **across regions** — so the
       300-pixel chip and the 30,000-pixel smash each contribute exactly one
       vote. This is the whole trick, and the single line where AU-PRO and
       pixel-AUROC part ways.
    3. **False-positive rate (FPR) at ``t``.** The fraction of *normal* pixels
       (outside every region) that the same threshold wrongly flagged. This is
       the cost axis: catching more of each region always means lighting up more
       of the clean background too.
    4. **Sweep ``t``** to trace ``PRO`` against ``FPR``, then integrate. Lowering
       the threshold moves you right (more false positives) and up (more region
       coverage); the area under that curve is the score.

    The integral is taken only up to ``fpr_limit`` (0.3 by convention) and
    normalized by it, because operating points past a 30% false-positive rate are
    useless on a real line and should not earn credit. The result is in ``[0, 1]``
    and directly comparable to the AU-PRO figures in the literature.

    The upshot to say out loud in an interview: **pixel-AUROC asks "what fraction
    of defective area did you find?"; AU-PRO asks "what fraction of defects did
    you find, counting each the same?"** The second is the question quality
    inspection actually cares about.

    Args:
        gt_masks: ``(N, H, W)`` array or a sequence of 2-D masks; nonzero means
            defect. Normal images (all-zero masks) contribute only to the FPR.
        pred_maps: Matching anomaly heatmaps, same shape(s).
        num_thresholds: Number of thresholds swept between the heatmaps' min and
            max. 100 is plenty for a smooth curve; more only refines the integral.
        fpr_limit: Upper bound on the FPR axis for the integral. Keep the default
            to stay comparable with published AU-PRO.

    Returns:
        Normalized AU-PRO in ``[0, 1]``, or ``nan`` if there are no defect
        regions to average over.
    """
    if not 0.0 < fpr_limit <= 1.0:
        msg = f"fpr_limit must be in (0, 1], got {fpr_limit}."
        raise ValueError(msg)

    gt_stack, pred_stack = _to_stack(gt_masks, pred_maps)
    gt_binary = gt_stack > 0

    # Label every image's defect blobs, offsetting ids so each region is unique
    # across the whole set. One flat vector of region ids lets a single
    # `ndimage.mean` compute all per-region overlaps at once, per threshold.
    region_labels = np.zeros_like(gt_stack, dtype=np.int64)
    total_regions = 0
    for i in range(gt_stack.shape[0]):
        labelled, count = ndimage.label(gt_binary[i], structure=_FULL_CONNECTIVITY)
        region_labels[i] = np.where(labelled > 0, labelled + total_regions, 0)
        total_regions += count

    if total_regions == 0:
        logger.warning("au_pro: no defect regions found in the ground truth; returning nan.")
        return float("nan")

    normal_mask = ~gt_binary
    normal_pixel_count = int(normal_mask.sum())
    region_ids = np.arange(1, total_regions + 1)

    lo, hi = float(pred_stack.min()), float(pred_stack.max())
    if lo == hi:  # a constant heatmap has no ranking to sweep
        logger.warning("au_pro: heatmaps are constant; returning nan.")
        return float("nan")
    thresholds = np.linspace(hi, lo, num_thresholds)  # high -> low: FPR sweeps 0 -> 1

    fprs = np.empty(num_thresholds)
    pros = np.empty(num_thresholds)
    for j, t in enumerate(thresholds):
        predicted = pred_stack >= t
        # Per-region recall: the mean of the (0/1) prediction over each region's
        # pixels *is* the fraction of that region caught. Averaging those means
        # equally over regions is the definition of PRO.
        overlaps = ndimage.mean(predicted, labels=region_labels, index=region_ids)
        pros[j] = float(np.mean(overlaps))
        fprs[j] = float((predicted & normal_mask).sum() / normal_pixel_count) if normal_pixel_count else 0.0

    return _integrate_to_limit(fprs, pros, fpr_limit)


def _to_stack(gt_masks: object, pred_maps: object) -> tuple[np.ndarray, np.ndarray]:
    """Coerce masks/heatmaps into aligned ``(N, H, W)`` arrays for region analysis.

    Unlike :func:`_stack_pixels`, region labelling needs the 2-D layout intact,
    so every image must share a resolution here. That is true by construction in
    the benchmark (all predictions are made at a single ``image_size``), which is
    where these come from.
    """
    gt_stack = np.asarray(gt_masks) if isinstance(gt_masks, np.ndarray) else np.stack([np.asarray(m) for m in gt_masks])
    pred_stack = (
        np.asarray(pred_maps, dtype=np.float32)
        if isinstance(pred_maps, np.ndarray)
        else np.stack([np.asarray(m, dtype=np.float32) for m in pred_maps])
    )
    if gt_stack.ndim != 3 or pred_stack.ndim != 3:
        msg = f"au_pro expects (N, H, W) stacks or equal-sized 2-D sequences, got {gt_stack.shape} and {pred_stack.shape}."
        raise ValueError(msg)
    if gt_stack.shape != pred_stack.shape:
        msg = f"masks and heatmaps must be the same shape, got {gt_stack.shape} vs {pred_stack.shape}."
        raise ValueError(msg)
    return gt_stack, pred_stack.astype(np.float32)


def _integrate_to_limit(fprs: np.ndarray, pros: np.ndarray, fpr_limit: float) -> float:
    """Normalized area under ``pros`` vs ``fprs`` for ``fpr <= fpr_limit``.

    Anchors the curve at the origin — a threshold above the maximum score
    predicts nothing, so ``PRO = 0`` at ``FPR = 0``, a point the finite sweep may
    not land exactly on. Then sorts by FPR, enforces the monotonicity a PRO/FPR
    curve has by construction (lowering the threshold cannot reduce either axis),
    collapses duplicate FPRs, resamples onto a dense grid over ``[0, fpr_limit]``,
    and applies the trapezoidal rule. Dividing by ``fpr_limit`` rescales the
    result to ``[0, 1]`` so a perfect detector scores exactly 1.0.
    """
    # The theoretical (0, 0) start of the curve. Without it, interpolation would
    # hold the lowest swept PRO flat down to FPR=0 and overstate the area.
    fprs = np.concatenate([[0.0], fprs])
    pros = np.concatenate([[0.0], pros])

    order = np.argsort(fprs)
    fprs, pros = fprs[order], np.maximum.accumulate(pros[order])

    # Keep the last point of each run of equal FPR (its accumulated max PRO), so
    # the FPR axis is strictly increasing for `np.interp`.
    keep = np.append(np.diff(fprs) > 0, True)
    fprs, pros = fprs[keep], pros[keep]

    grid = np.linspace(0.0, fpr_limit, 512)
    pro_on_grid = np.interp(grid, fprs, pros)
    return float(np.trapezoid(pro_on_grid, grid) / fpr_limit)
