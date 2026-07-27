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

---

## Keeping the numbers true in production: drift and the operating point

Everything above is measured **offline**, on a fixed test split, with ground
truth. A deployed line has neither: no labels, no fixed split, and a data
distribution that moves. Two modules cover the gap, and they are a pair —
[`app/evaluation/drift.py`](../app/evaluation/drift.py) notices the problem and
[`app/evaluation/calibration.py`](../app/evaluation/calibration.py) fixes the
common case.

### The failure they exist to catch

[`app/guardrails/`](../app/guardrails) catches a bad *frame*. Nothing above
catches a bad *score* — because there is nothing wrong with it structurally. The
model returns a well-formed float in a plausible range; it simply no longer means
what it meant when the threshold was chosen. Three ordinary causes:

- **The product line changes.** A supplier switches material, a mould is retuned,
  a new SKU joins the line. Every frame is now slightly out-of-distribution
  relative to the nominal set PatchCore's memory bank was built from.
- **The camera ages.** Sensor gain drifts, a lens coating hazes, the LED ring
  dims over thousands of hours — all of it slow enough to stay well clear of the
  blur and exposure guards.
- **Something was deployed.** A new checkpoint, a re-export to ONNX, an
  `IMAGE_SIZE` change. The service is healthy; the operating point is stale.

In all three `/health` stays green, latency is unchanged, and the defect rate
quietly moves. That is the silent failure, and it is the expensive kind: by the
time anyone notices, a shift's worth of parts has been graded against a boundary
that no longer holds.

### Why the Kolmogorov-Smirnov test

The obvious monitor — *alert if the mean score moves* — assumes the thing that
changes is the centre of a distribution whose shape you already know. Anomaly
scores break both halves. They are bounded below, heavily right-skewed and
usually bimodal (a dense nominal cluster plus a thin defect tail), and the shifts
that matter are often changes in *shape*: the tail thickening while the nominal
mode sits exactly where it was, which barely moves the mean at all.

The two-sample KS test needs none of those assumptions. It builds the empirical
CDF of each window — the reference and the current one — and takes the largest
vertical gap between the two curves. That statistic's distribution under "both
samples came from the same underlying distribution" depends only on the two
sample sizes, not on the shape of the data. It is therefore valid on skewed,
bimodal, bounded anomaly scores, which is exactly why it is the right test here
and a t-test is not.

**The p-value does not mean "the probability the model drifted".** It answers a
narrower question: *if nothing had changed, how often would random sampling alone
produce a gap at least this large?* `p = 0.30` is unremarkable; `p = 0.001` says
either something rare happened or the assumption that nothing changed is wrong.
Below `KS_DRIFT_THRESHOLD` (default `0.05`) the service calls it drift.

Three consequences worth stating plainly:

1. **`0.05` buys a 5% false-alarm rate by construction.** One comparison in
   twenty crosses the line on a perfectly stable process. That is the definition
   of the threshold, not a defect in it — and the reason the response to a single
   alert is *look*, not *halt*.
2. **A large p-value is not proof of stability.** Failing to detect drift is not
   detecting its absence; a short window may simply have no power. The monitor
   reports "no verdict" (`p_value: null`) rather than "no drift" until both
   windows hold at least 30 scores.
3. **Statistical significance is not operational significance.** With a large
   enough window, KS will flag a shift far too small to change a single verdict.
   `p` says *whether* it moved; the percentiles in the same response say *by how
   much*. Only the second answers "does this matter", which is why `GET /drift`
   returns both and the docstrings insist they be read together.

### The response ladder

Deliberately not starting at "stop the line" — a monitor whose only action is
drastic gets muted within a week:

| Step | Action | When |
| ---- | ------ | ---- |
| 1 | **Alert with the numbers.** Cross-check `guard_rejections_total` first | Always — a drift arriving with a rising `blurry` rate is a fouling lens, fixed with a cloth |
| 2 | **Flag the window for human review** | The frames scored since the drift began were graded against a boundary that may no longer hold |
| 3 | **Recalibrate the operating point** (`POST /calibrate`) | The scores shifted but stayed separable — minutes of work, and the usual answer |
| 4 | **Retrain / re-fit the reference** | Recalibration cannot recover the F1: the shape changed, not just the location |
| 5 | **Gate the line to 100% human inspection** | Only when the alternative is shipping scrap |

### Calibration: choosing the operating point

Every metric in the table above is deliberately **threshold-free** — that is what
makes AUROC a fair comparison across backends emitting incompatible score scales.
A production line cannot ship a ranking. It has to answer *pass or scrap*, which
means committing to one number, and AUROC says nothing about where that number
should sit: a model at 0.99 AUROC still fails completely at a badly chosen
threshold.

`find_optimal_threshold(scores, labels, metric="f1")` sweeps every threshold the
scores can produce and returns the one maximising the metric. The sweep is
**exact, not a grid** — the confusion matrix can only change where a score crosses
the boundary, so the distinct score values are the complete candidate set, and
sorting once with cumulative sums makes the whole thing `O(n log n)`. That is what
lets it run inside a request handler.

Two design choices worth defending:

- **The metric is the caller's.** `f1` by default (false alarms and missed
  defects cost about the same), `balanced_accuracy` for a heavily imbalanced
  calibration set, `precision` for an automatic-scrap line, `recall` where a
  missed defect is catastrophic *and* a human re-inspects the flagged parts.
  `precision` and `recall` alone are degenerate at the extremes and the module
  docstring says so rather than hiding it.
- **Ties break toward the lowest threshold.** Many thresholds usually achieve the
  same maximum; the lowest is the most sensitive, so at equal F1 it catches more
  defects and raises more false alarms. That is the right default for industrial
  QA, where a false alarm costs an operator thirty seconds at a re-inspection
  station and a missed defect ships.

`POST /calibrate` applies the result to the running service, and by default also
installs the submitted scores as the drift monitor's reference — because the
calibration set *is* the new definition of normal, and letting the threshold and
the baseline diverge is how a service ends up alerting against a distribution
nobody chose.

### Two limitations, stated rather than discovered

- **The change is in memory only.** It is lost on restart, and under
  `uvicorn --workers N` it applies to the one worker that served the request.
  A threshold worth keeping belongs in `ANOMALY_THRESHOLD`; the endpoint is for
  finding it and trying it, not for storing it.
- **`metric_value` is fitted on the submitted data**, so it is an upper bound on
  production performance rather than an estimate of it. A calibration set drawn
  from images the model was fitted on produces a threshold that looks excellent
  and generalises poorly, and nothing in the endpoint can detect that. Hold it
  out.

```bash
# Fit an operating point and install it, plus the drift baseline
curl -s -X POST localhost:8000/calibrate -H "X-API-Key: $OPERATOR_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"category":"bottle","model_backend":"patchcore","metric":"f1",
       "samples":[{"score":0.11,"label":0},{"score":0.94,"label":1}]}'

# Then watch the distribution the line is actually producing
curl -s localhost:8000/drift -H "X-API-Key: $OPERATOR_KEY" | python -m json.tool
```
