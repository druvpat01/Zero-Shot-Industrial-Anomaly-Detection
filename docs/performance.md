# Performance: ONNX export, quantization, and CPU latency

How this project turns a trained PatchCore or EfficientAD checkpoint into a
faster CPU inference path, what that actually bought (measured, not quoted), and
where it did *not* pay off. Every number below comes from the two scripts named
under each table, run on the `bottle` category; nothing here is aspirational.

Reproduce the whole thing:

```bash
python scripts/export_onnx.py                 # -> results/exported/weights/onnx/*.onnx  (FP32 + INT8)
python scripts/benchmark_latency.py           # -> results/latency_benchmark.json
python scripts/compare_int8_accuracy.py       # -> results/int8_accuracy_comparison.json
```

---

## The path: export → quantize → serve

anomalib exports the **whole** Lightning `forward`, not the bare network — resize,
normalization and score calibration are all baked into the graph:

```
input frame ─▶ pre-process (resize; ImageNet-normalize for PatchCore)
            ─▶ model
            ─▶ post-process (min-max score/​map calibration, thresholds)
            ─▶ (pred_score, anomaly_map, …)
```

[`app/models/onnx_runner.py`](../app/models/onnx_runner.py) serves that graph as
an ordinary [`AnomalyModel`](../app/models/base.py): same `predict(image) ->
ModelOutput` contract, same input-quality guard, same "heatmap comes back at the
caller's resolution" promise as the PyTorch wrappers. Because it *is* an
`AnomalyModel`, the serving layer swaps `PatchCoreModel` for `ONNXRunner` behind
one config flag with no route changes — which is the whole point of the wrapper
seam. The one thing `ONNXRunner` does differently is preprocessing: since the
graph normalizes internally, the runner hands it a plain `[0, 1]` batch and skips
the ImageNet step the PyTorch PatchCore wrapper applies itself (doing it twice is
silent and only makes the model worse).

That faithfulness is verifiable, and it is the control for everything below: the
FP32 ONNX image score reproduces the PyTorch wrapper's to ~1e-6, and the FP32
pixel metrics are **identical** to PyTorch's (see the accuracy table). So any gap
in the tables is quantization's or the runtime's, never the export's.

---

## Latency

`bottle` test split, 83 frames, 5 warm-up iterations, batch size 1, end-to-end
`predict()` (guard + preprocess + model + upsample to input resolution — i.e.
what the serving layer actually pays per request). FPS is sustained
single-image throughput, `1000 / mean_ms`.

| Backend               | p50 (ms) | p95 (ms) | p99 (ms) |  FPS |
|-----------------------|---------:|---------:|---------:|-----:|
| PatchCore PyTorch     |    499.2 |    578.9 |    603.4 |  1.9 |
| **PatchCore ONNX FP32** | **437.5** | **455.0** | **461.3** | **2.3** |
| PatchCore ONNX INT8   |    904.9 |    943.4 |    976.3 |  1.2 |
| EfficientAD PyTorch   |    577.0 |    747.8 |    770.1 |  1.6 |
| EfficientAD ONNX FP32 |    706.9 |    717.6 |    728.5 |  1.4 |

*Source: `python scripts/benchmark_latency.py` → `results/latency_benchmark.json`.*

**Measurement conditions, stated up front because they matter.** Intel Core
i7-1165G7 (4 physical / 8 logical cores), CPU-only. Both runtimes are pinned to
**4 threads** (`torch.set_num_threads(4)` and onnxruntime `intra_op_num_threads=4`)
so neither oversubscribes the other on the hyperthreaded CPU — without this,
onnxruntime defaulted to one thread per *logical* core and contended with torch's
pool, inflating the ONNX numbers by ~35%. This machine was also thermally/​power
throttled to roughly a quarter of its rated clock during the run, so the
**absolute** milliseconds are much higher than the same code on an unthrottled
server; the throttle hits every backend proportionally, so the **relative**
comparisons — which is what this table is for — hold.

### Reading the table

Three findings, one of them the intended win and two honest negatives:

1. **ONNX FP32 is the faster PatchCore path — this is the headline.** −12% at p50
   (437.5 vs 499.2 ms) and, more usefully for an inspection line with a cycle-time
   budget, **−24% at p99** (461.3 vs 603.4 ms): the ONNX tail is far tighter than
   PyTorch's. Throughput rises 1.9 → 2.3 FPS. The runtime does the memory-bank
   nearest-neighbour search and the WideResNet forward as one optimized graph
   instead of eager PyTorch ops, and it shows most in the tail.

2. **INT8 dynamic quantization is counterproductive here — it is *slower*, not
   faster** (904.9 vs 437.5 ms, ~2× the FP32 ONNX graph). This is the result
   worth understanding rather than hiding. `quantize_dynamic` inserts
   quantize/​dequantize nodes and computes activation scales at runtime, and its
   INT8 kernels chiefly accelerate `MatMul`/attention. PatchCore is
   **convolution-dominated** (a WideResNet-50 backbone), so it pays the
   quantization overhead on almost every op while getting little kernel speedup in
   return — the classic case where dynamic quantization backfires on a conv-heavy
   graph on a CPU without INT8-VNNI acceleration. It still *shrinks* the file (see
   sizes below); it just does not make this model faster.

3. **EfficientAD's ONNX is slower than its PyTorch (706.9 vs 577.0 ms).**
   EfficientAD is two Patch-Description Networks plus an autoencoder — wide,
   shallow convolutions at near-input resolution, which is exactly the shape
   PyTorch's oneDNN backend already fuses and vectorizes well. The ONNX graph does
   not beat it on CPU. (EfficientAD's real advantage is elsewhere: its inference
   cost is constant in the training-set size where PatchCore's grows — see its
   [wrapper docstring](../app/models/efficientad.py).)

**Takeaway to serve:** PatchCore via ONNX FP32 — faster and tighter-tailed than
PyTorch at identical accuracy. INT8 and ONNX-EfficientAD are measured, documented,
and not recommended for this workload on this hardware.

---

## What INT8 costs in accuracy

Latency is only half the quantization story; the other half is what you give up.
Scored on the same defective `bottle` frames with the same Step 5 metrics
([`pixel_auroc`, `au_pro`](../app/evaluation/metrics.py)), PyTorch as the baseline
and ONNX FP32 as the control:

| Backend    | Pixel-AUROC | AU-PRO | Δ Pixel-AUROC | Δ AU-PRO |
|------------|------------:|-------:|--------------:|---------:|
| PyTorch    |      97.99% | 93.20% |      baseline | baseline |
| ONNX FP32  |      97.99% | 93.20% |     +0.00 pp  | +0.00 pp |
| ONNX INT8  |      96.75% | 89.65% |     −1.24 pp  | −3.56 pp |

*Source: `python scripts/compare_int8_accuracy.py` (10 defective frames) →
`results/int8_accuracy_comparison.json`.*

Dynamic INT8 quantization of PatchCore costs **−1.24 pp pixel-AUROC and −3.56 pp
AU-PRO** against the PyTorch baseline. The drop is larger on AU-PRO than on
pixel-AUROC, and that is the more honest of the two numbers to weigh: AU-PRO
weights every defect *region* equally rather than every pixel, so it is sensitive
to small defects being caught, and quantization noise blurs exactly the
fine-grained heatmap contrast that separates a small chip from clean background.
(The quantized model's *image* scores actually drift **upward** — mean 0.944 →
0.964 — because the noise lifts the whole map; the ranking that the metrics
measure is what degrades, not the raw magnitude.) The ONNX FP32 control lands at
0.00 pp on both, confirming the loss is quantization's and not the export's.

Combined with the latency result, INT8 dynamic quantization here is a **lose–lose**
for PatchCore — slower *and* less accurate — and earns its keep only as a
disk/​memory reduction. If INT8 were needed for a genuinely memory-constrained
deployment, static (calibrated) quantization or OpenVINO INT8, which quantize
convolutions properly, would be the paths to try next; dynamic quantization is not
the right tool for this graph.

---

## Artifact sizes

| Model        | Checkpoint (`.ckpt`) | ONNX FP32 | ONNX INT8 |
|--------------|---------------------:|----------:|----------:|
| PatchCore    |              ~220 MB |  345.6 MB |  212.0 MB |
| EfficientAD  |               ~74 MB |   30.8 MB |    7.8 MB |

PatchCore's FP32 ONNX is large because the coreset memory bank — here
`(21401, 1536)` float32 patch descriptors — is baked into the graph as a
constant alongside the backbone weights. INT8 shrinks it to 61% of FP32.
EfficientAD's graph is a fixed ~31 MB of PDN + autoencoder weights regardless of
the training set, and quantizes down to a quarter of that (7.8 MB) — the size win
INT8 *does* deliver, even where the latency win does not.

---

## Why WinCLIP is not exported

WinCLIP is deliberately skipped, and `scripts/export_onnx.py` refuses it with the
reason rather than producing a broken artifact. Its CLIP ViT backbone is a poor
ONNX target: WinCLIP scores by sliding a *variable* number of windows over the
patch grid and pooling their CLIP embeddings, so the number of attention
operations depends on the input. The (traced) ONNX exporter freezes that dynamic,
input-dependent control flow into a graph specialised to one shape — if it exports
cleanly at all. The honest engineering call is that PatchCore and EfficientAD are
the two models worth serving through ONNX; WinCLIP stays on the PyTorch path,
where its zero-shot flexibility is the entire point.

---

## Bottom line

- **Ship PatchCore as ONNX FP32.** Same accuracy as PyTorch to the fourth decimal,
  ~12% lower median latency, and a materially tighter p99 — the metric a
  fixed-cycle inspection line actually lives or dies on.
- **Don't ship INT8 for latency.** Dynamic quantization made a conv-heavy model
  slower *and* dropped AU-PRO 3.6 pp. It is a size lever, not a speed one.
- **EfficientAD stays on PyTorch for latency;** oneDNN already serves its PDN
  convolutions better than the ONNX graph does on CPU.
- **Trust the FP32 control.** It reproduces PyTorch exactly, which is what makes
  every delta in these tables attributable to a single cause.
