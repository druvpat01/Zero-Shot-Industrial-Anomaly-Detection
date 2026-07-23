"""Unit tests for the evaluation metrics, on synthetic inputs with known answers.

Every test here uses hand-built arrays whose correct metric value can be reasoned
about on paper, so a failure points at the metric, not at a model or a dataset.
No data download and no trained checkpoint is needed — this file runs anywhere.

The three cases the spec calls out are:

* ``test_image_auroc_perfect_predictor``   — perfectly separated scores -> 1.0
* ``test_image_auroc_random_predictor``     — random scores -> ~0.5
* ``test_au_pro_single_region_perfect``     — one region, perfect map -> 1.0

The rest pin down the properties that make these metrics worth trusting — most
importantly ``test_au_pro_is_not_dominated_by_large_defects``, which is the whole
reason AU-PRO exists alongside pixel-AUROC.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import au_pro, f1_at_best_threshold, image_auroc, pixel_auroc

# ---------------------------------------------------------------------------
# image_auroc
# ---------------------------------------------------------------------------


def test_image_auroc_perfect_predictor() -> None:
    """All defective scores above all clean scores -> a flawless ranking, AUROC 1.0."""
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert image_auroc(y_true, scores) == 1.0


def test_image_auroc_inverted_predictor_is_zero() -> None:
    """A predictor that ranks every defect *below* every normal scores 0.0."""
    y_true = [0, 0, 1, 1]
    scores = [1.0, 1.0, 0.0, 0.0]
    assert image_auroc(y_true, scores) == 0.0


def test_image_auroc_random_predictor_is_about_half() -> None:
    """Random scores carry no information, so AUROC concentrates around 0.5."""
    rng = np.random.default_rng(0)
    n = 4000
    y_true = rng.integers(0, 2, size=n)
    # Guarantee both classes are present so the metric is defined.
    y_true[0], y_true[1] = 0, 1
    scores = rng.random(n)

    auroc = image_auroc(y_true.tolist(), scores.tolist())
    assert auroc == pytest.approx(0.5, abs=0.05)


def test_image_auroc_single_class_returns_nan() -> None:
    """ROC is undefined without both classes; the metric says so rather than crashing."""
    assert np.isnan(image_auroc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_image_auroc_is_invariant_to_monotonic_rescaling() -> None:
    """AUROC reads the ranking only, so any order-preserving rescale leaves it fixed.

    This is why models on different score scales (raw distances vs softmax
    probabilities) can be compared by AUROC at all.
    """
    y_true = [0, 1, 0, 1, 1]
    scores = [0.1, 0.6, 0.2, 0.9, 0.7]
    rescaled = [10 * s + 3 for s in scores]  # strictly increasing transform
    assert image_auroc(y_true, scores) == pytest.approx(image_auroc(y_true, rescaled))


# ---------------------------------------------------------------------------
# f1_at_best_threshold
# ---------------------------------------------------------------------------


def test_f1_perfectly_separable_reaches_one() -> None:
    """When a threshold separates the classes cleanly, best F1 is 1.0."""
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    best_f1, threshold = f1_at_best_threshold(y_true, scores)
    assert best_f1 == pytest.approx(1.0)
    # Any cut in (0.2, 0.8] separates them; the one returned must be such a cut.
    assert 0.2 < threshold <= 0.8


def test_f1_threshold_actually_maximises_f1() -> None:
    """The returned threshold reproduces the returned F1 when applied by hand."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_true[0], y_true[1] = 0, 1
    # Give the score a real (noisy) signal so the best cut is non-trivial.
    scores = y_true * 0.5 + rng.normal(0, 0.3, size=200)

    best_f1, threshold = f1_at_best_threshold(y_true.tolist(), scores.tolist())

    predicted = (scores >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    recomputed = 2 * tp / (2 * tp + fp + fn)
    assert recomputed == pytest.approx(best_f1)


def test_f1_single_class_returns_nan() -> None:
    best_f1, threshold = f1_at_best_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(best_f1)
    assert np.isnan(threshold)


# ---------------------------------------------------------------------------
# pixel_auroc
# ---------------------------------------------------------------------------


def test_pixel_auroc_perfect_segmentation_is_one() -> None:
    """A heatmap that is hot exactly on the defect and cold elsewhere scores 1.0."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    heatmap = mask.astype(np.float32)  # 1.0 inside, 0.0 outside
    assert pixel_auroc(np.stack([mask]), np.stack([heatmap])) == 1.0


def test_pixel_auroc_accepts_ragged_sequences() -> None:
    """Images of differing resolutions can be scored by pooling their pixels."""
    m1 = np.zeros((8, 8), dtype=np.uint8)
    m1[0:2, 0:2] = 1
    m2 = np.zeros((10, 12), dtype=np.uint8)
    m2[3:6, 3:6] = 1
    p1, p2 = m1.astype(np.float32), m2.astype(np.float32)
    assert pixel_auroc([m1, m2], [p1, p2]) == 1.0


def test_pixel_auroc_no_defect_pixels_returns_nan() -> None:
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    preds = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    assert np.isnan(pixel_auroc(masks, preds))


# ---------------------------------------------------------------------------
# au_pro
# ---------------------------------------------------------------------------


def test_au_pro_single_region_perfect_prediction_is_one() -> None:
    """The spec's case: one connected component, a perfect map -> AU-PRO 1.0.

    A perfect heatmap covers the whole region at every threshold below its value
    while never touching a normal pixel, so PRO is 1.0 all along the curve and
    the normalized area under it is exactly 1.0.
    """
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 8:20] = 1  # a single 12x12 connected component
    heatmap = mask.astype(np.float32)  # 1.0 inside, 0.0 outside

    assert au_pro(np.stack([mask]), np.stack([heatmap])) == pytest.approx(1.0)


def test_au_pro_perfect_prediction_multiple_images_is_one() -> None:
    """Perfect maps across several images, some normal, still integrate to 1.0."""
    masks, heatmaps = [], []
    rng = np.random.default_rng(2)
    for _ in range(5):
        mask = np.zeros((40, 40), dtype=np.uint8)
        y, x = rng.integers(2, 25), rng.integers(2, 25)
        mask[y : y + 6, x : x + 6] = 1
        masks.append(mask)
        heatmaps.append(mask.astype(np.float32))
    masks.append(np.zeros((40, 40), dtype=np.uint8))  # a normal image, all-zero mask
    heatmaps.append(np.zeros((40, 40), dtype=np.float32))

    assert au_pro(masks, heatmaps) == pytest.approx(1.0)


def test_au_pro_random_prediction_is_low() -> None:
    """Noise has no localisation, so AU-PRO sits far below 1.0."""
    rng = np.random.default_rng(3)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:16, 10:16] = 1
    heatmap = rng.random((32, 32)).astype(np.float32)

    score = au_pro(np.stack([mask]), np.stack([heatmap]))
    assert 0.0 <= score < 0.6


def test_au_pro_no_regions_returns_nan() -> None:
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    preds = np.random.default_rng(0).random((2, 16, 16)).astype(np.float32)
    assert np.isnan(au_pro(masks, preds))


def test_au_pro_counts_diagonal_touching_pixels_as_one_region() -> None:
    """8-connectivity: a diagonal defect is one region, not a string of many.

    Detecting only half of it should therefore give one region ~0.5 covered, not
    a pile of tiny regions each fully hit or fully missed.
    """
    mask = np.zeros((16, 16), dtype=np.uint8)
    for i in range(2, 10):
        mask[i, i] = 1  # a diagonal line, only 8-connected
    # Cover the first half of the diagonal only.
    heatmap = np.zeros((16, 16), dtype=np.float32)
    for i in range(2, 6):
        heatmap[i, i] = 1.0

    # One region, ~half covered -> PRO tops out near 0.5 before any false
    # positives appear, so the (normalized) area stays well under 1.0.
    score = au_pro(np.stack([mask]), np.stack([heatmap]))
    assert score < 0.75


def test_au_pro_is_not_dominated_by_large_defects() -> None:
    """AU-PRO's reason to exist: a big defect must not drown out a missed small one.

    Two images: a huge defect the model segments perfectly, and a tiny defect the
    model misses entirely. Pixel-AUROC — which weights pixels — is dragged up by
    the thousands of correct big-defect pixels. AU-PRO weights the two *regions*
    equally, so missing one of two defects pulls it toward 0.5. The test asserts
    that gap, which is the whole argument for preferring AU-PRO on industrial data.
    """
    big_mask = np.zeros((64, 64), dtype=np.uint8)
    big_mask[8:56, 8:56] = 1  # ~2300-pixel defect
    big_pred = big_mask.astype(np.float32)  # segmented perfectly

    small_mask = np.zeros((64, 64), dtype=np.uint8)
    small_mask[0:3, 0:3] = 1  # 9-pixel defect
    small_pred = np.zeros((64, 64), dtype=np.float32)  # missed entirely

    masks = [big_mask, small_mask]
    preds = [big_pred, small_pred]

    px = pixel_auroc(masks, preds)
    region = au_pro(masks, preds)

    # Pixel-AUROC is inflated by the large, perfectly-caught defect...
    assert px > 0.9
    # ...while AU-PRO, weighting each region equally, is dragged toward 0.5 by
    # the missed small one. That divergence is the point.
    assert region < 0.65
    assert px - region > 0.25
