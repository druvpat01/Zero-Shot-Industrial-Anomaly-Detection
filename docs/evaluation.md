# Evaluation

How this project measures a defect detector, why it measures it that way, and
what the numbers came out to. Everything here is produced by
[`app/evaluation/`](../app/evaluation) — the metrics are implemented from scratch
on scikit-learn and NumPy (no anomalib metric internals), so every figure below
is one I can derive on a whiteboard.

Reproduce it with:

```bash
python scripts/run_benchmark.py --category bottle
```

which trains or loads all three models, scores them on the `bottle` test set, and
writes a timestamped JSON report to `results/`.

---

## The four metrics

Anomaly detection is really two tasks — *is this part defective?* (detection) and
*where is the defect?* (segmentation) — and a single number cannot speak to both.
The benchmark reports four, deliberately chosen so that each answers a question an
interviewer (or a plant manager) would actually ask.

### 1. Image-AUROC — *"Did you catch the bad part?"*

The ROC-AUC of the image-level anomaly score against the ground-truth label
(defective = 1, normal = 0). It equals the probability that a randomly chosen
defective image is scored higher than a randomly chosen normal one.

**Why it exists.** It is the headline detection number, and it has one property
that makes it the fair way to compare *these three* models: it reads only the
*ranking* the scores induce, so it is invariant to any monotonic rescaling. That
matters because the three backends emit scores on incompatible scales — PatchCore
produces unbounded nearest-neighbour distances, WinCLIP produces a softmax
probability in `[0, 1]`. AUROC compares them without anyone having to agree on a
threshold first, so it measures each detector's *ceiling* rather than one chosen
operating point.

**What it hides.** It says nothing about *where* the defect is, and it can look
excellent while the localisation is poor — a model that scores the whole image
correctly but paints the heatmap in the wrong place still posts a high
Image-AUROC. That is what the pixel metrics are for.

### 2. Pixel-AUROC — *"How much of the defective area did you paint?"*

The same ROC-AUC, but over pixels: pool every pixel of every test image, label
each `1` if it falls inside a ground-truth defect and `0` otherwise, and take the
ROC-AUC of the heatmap values against those labels.

**Why it exists.** It is the field's most-quoted segmentation number, so
reporting it makes comparison to published results possible. It answers "if I
pick a random defect pixel and a random clean pixel, how often does the heatmap
rank the defect pixel higher?"

**Why not to trust it on its own.** Because every pixel votes equally, the score
is really a score for *area*, and it is dominated by two things that have nothing
to do with detection quality: (a) the overwhelming majority of pixels are normal,
which inflates the baseline, and (b) large defects contribute far more pixels
than small ones. On MVTec `bottle`, a `broken_large` smash covers tens of
thousands of defect pixels and a `broken_small` chip a few hundred — so a model
that segments the big defects perfectly and misses every small one still scores a
near-perfect Pixel-AUROC. That is exactly backwards from what inspection cares
about, and it is why the next metric exists.

### 3. AU-PRO — *"What fraction of defects did you find, counting each the same?"*

Area Under the Per-Region-Overlap curve. This is the gold-standard segmentation
metric for industrial anomaly detection, and the one to trust. It is built to be
immune to the area bias that distorts Pixel-AUROC:

1. **Label the ground truth into connected components.** Each contiguous defect
   blob is one *region*, whether it is 300 pixels or 30,000 (8-connectivity, so a
   diagonal crack stays a single region).
2. **Per-Region Overlap (PRO) at a threshold `t`.** Binarize the heatmap at `t`.
   For each region, compute the fraction of *that region's* pixels the threshold
   caught — its per-region recall. `PRO(t)` is the plain average of those
   overlaps **across regions**. This is the whole trick: the 300-pixel chip and
   the 30,000-pixel smash each contribute exactly one vote.
3. **False-positive rate (FPR) at `t`.** The fraction of *normal* pixels the same
   threshold wrongly flagged — the cost axis.
4. **Sweep `t`** to trace PRO against FPR, and integrate the area under that
   curve. By convention the integral runs only up to **FPR ≤ 0.30** and is
   normalized by it, because operating points past a 30% false-positive rate are
   useless on a real line and should earn no credit. The result is in `[0, 1]` and
   comparable to the AU-PRO figures in the literature.

### 4. Best-F1 — *"What is the best single operating point?"*

The maximum F1 (harmonic mean of precision and recall) over all thresholds, plus
the threshold that achieves it. AUROC judges separability across every threshold;
deployment has to *commit* to one. Best-F1 reports how good the best available cut
is — the natural companion to Image-AUROC. The candidate thresholds come straight
from `precision_recall_curve`, which changes the confusion matrix at exactly the
score values where it can change, so the maximum is exact rather than a grid
approximation.

---

## Why AU-PRO is preferred over Pixel-AUROC for industrial defect segmentation

This is the single most likely follow-up question, so it gets its own section.

**Pixel-AUROC weights pixels; AU-PRO weights defects.** That one-line difference
is the whole argument. In quality inspection the unit that matters is the
*defect* — a missed hairline crack fails the part just as surely as a missed
shattered rim — but Pixel-AUROC counts a missed hairline crack as a few hundred
wrong pixels against a background of hundreds of thousands of correct ones, and
barely moves. A metric that can be maximized while systematically ignoring small
defects is measuring the wrong thing for this domain.

The project's own test suite pins this down with a synthetic case
([`test_au_pro_is_not_dominated_by_large_defects`](../tests/test_metrics.py)): two
images, one with a large defect segmented perfectly and one with a tiny defect
missed entirely. Pixel-AUROC comes out above 0.90 — dragged up by the thousands of
correctly-classified large-defect pixels — while AU-PRO is pulled toward 0.5,
because one of the two equally-weighted regions was missed. The gap between those
two numbers *is* the reason the field standardized on AU-PRO (Bergmann et al., *The
MVTec Anomaly Detection Dataset*, IJCV 2021) for segmentation quality.

The honest caveat: AU-PRO is a segmentation metric, not a detection one. It says
nothing about whether the *image* was flagged, only about how well the flagged
regions line up with ground truth. That is why the table reports it alongside
Image-AUROC rather than instead of it — the two answer different questions, and a
production system needs both.

---

## Results

Category **bottle**, 83 test images (63 defective, 20 normal), evaluated at
256×256 on CPU (4 threads). Source:
[`results/benchmark_bottle_20260723T185736Z.json`](../results/benchmark_bottle_20260723T185736Z.json).

| Model        | Img-AUROC | Px-AUROC | AU-PRO  | Best-F1 | Latency (s/img) | Training data |
|--------------|-----------|----------|---------|---------|-----------------|---------------|
| PatchCore    |  100.0%   |  98.6%   |  94.4%  |  1.000  |     0.49        | ~200 good imgs |
| EfficientAD  |   96.4%   |  76.5%   |  50.5%  |  0.960  |     0.49        | ~200 good imgs (1 epoch) |
| WinCLIP      |   98.2%   |  84.8%   |  70.2%  |  0.969  |     3.68        | **none (zero-shot)** |

Two caveats belong next to the table, not buried under it:

* **The image-level scores are optimistic for the two trained models.** PatchCore
  and EfficientAD fit their score-normalization and 0.5 decision threshold on a
  validation split anomalib derives from the *test* set, so Best-F1 = 1.000 and
  0.960 flatter the operating point slightly. AUROC and AU-PRO are ranking
  metrics and are far less sensitive to this. WinCLIP is scored uncalibrated —
  its Best-F1 threshold (0.406) is below 0.5 precisely because nothing was fitted.
* **EfficientAD here is deliberately under-trained.** It was fit for a single
  epoch (~200 steps) against the paper's ~70k. Its numbers are a smoke-trained
  student, not a converged one — see below.

---

## What the numbers mean, relative to published SOTA

**PatchCore essentially reproduces the paper.** Roth et al. (CVPR 2022) report
~100% image-AUROC and ~98% pixel-AUROC on MVTec `bottle`; this run lands at 100.0%
and 98.6%, with 94.4% AU-PRO (the paper-family range for bottle is ~94–96%). There
is no meaningful gap to explain — a frozen WideResNet-50 memory bank on a clean
~200-image training set is a solved problem on this category, and the harness
confirms it. This is the "accuracy ceiling when you have good data" reference the
other two are measured against.

**WinCLIP is the headline: it beats an under-trained EfficientAD on every
segmentation metric with zero training images.** 98.2% image-AUROC and 84.8%
pixel-AUROC, with no `train/good` set at all — matching the WinCLIP paper's ~85%
zero-shot pixel-AUROC (Jeong et al., CVPR 2023). The 70.2% AU-PRO reflects the
metric's own structural ceiling: every WinCLIP score originates on a 15×15 patch
grid, so it localises defects as warm blobs rather than sharp outlines, and
region-overlap punishes that coarseness harder than pixel-AUROC does. The
significance is operational, not just a number — this column needed no labelled
data, no curation, and no training cycle, which is the entire cold-start argument
for the platform serving all three backends.

**EfficientAD's gap is a training-budget gap, not a defect.** Its 96.4%
image-AUROC shows the student already ranks defective above normal after one epoch
— and the wrapper's own docstring predicted "~0.97 image AUROC here" for exactly
this setting, which the run confirms. But its 76.5% pixel-AUROC and 50.5% AU-PRO
are far below the paper's converged ~98%/~96%, because segmentation *sharpness* is
the last thing student-teacher distillation learns: the highest-error features are
hard-mined last, so at 200 steps the anomaly map is diffuse and the per-region
overlap collapses toward chance. Converging it needs the full ~70k-step schedule
(and a GPU to make that practical); the one-epoch fit is what keeps the benchmark
runnable on CPU. That is an honest limitation of the *run*, not the wrapper — a
converged EfficientAD would sit second, between PatchCore and WinCLIP.

**Reading the table as a whole.** The three columns tell the deployment story the
project is built around: PatchCore is the model to fit when you have good data
(top accuracy, cheap inference), WinCLIP is the model to reach for when you have
*no* data (near-PatchCore detection, zero-shot, at ~7.5× the latency), and
EfficientAD is the model to *serve* once converged (constant inference cost in
training-set size — the property the latency work in the next step measures).
