# Architecture

One frame's journey through the service, in the order it actually happens, with
the file that owns each step. The companion documents go deeper on three of these
stages: [`evaluation.md`](evaluation.md) on the metrics and the drift monitor,
[`performance.md`](performance.md) on the ONNX path, and
[`security.md`](security.md) on the gate in front of it all.

---

## The data flow

```
                     [Camera / Image Upload]
                                |
                                |  POST /predict {image_b64, model_backend, category}
                                v
              [FrameGuard: blur / exposure / size]
                                |
                                |  passes  (a rejection stops here -> 422)
                                v
            [ModelRegistry: lazy-load + Redis cache]
                                |
              ┌─────────────────┼─────────────────┐
              v                 v                 v
      [PatchCore ONNX]    [EfficientAD]       [WinCLIP]
              └─────────────────┼─────────────────┘
                                v
              [ModelOutput: score + pixel heatmap]
                                |
                                v
           [ScoreDistributionMonitor: KS drift check]
                                |
                                v
            [FastAPI Response: JSON + base64 heatmap]
                                |
              ┌─────────────────┼─────────────────┐
              v                 v                 v
    [Prometheus /metrics]  [Audit Log]       [Structlog]
```

Three things the diagram compresses, said plainly so it is read as a map rather
than as a contract:

* **The observability lane is not a final stage.** It is drawn last because it is
  where the data lands, but the writes happen *during* the pass:
  `FrameGuard.validate` times itself and counts its own rejections before the
  registry is ever consulted, and a rejected frame therefore produces metrics and
  a log line without ever reaching a model.
* **The audit log is not on this path at all.** `/predict` is high-volume and gets
  a structured log record; the audit trail is reserved for `/benchmark` and
  `/calibrate`. See [`security.md`](security.md) for why those two and nothing
  else.
* **There are five servable backends, not three.** The middle row names the three
  *models*; `patchcore` and `efficientad` each have an ONNX twin, so the wire
  accepts `patchcore`, `efficientad`, `winclip`, `onnx_patchcore` and
  `onnx_efficientad`. WinCLIP has no twin, for the reason in
  [`performance.md`](performance.md#why-winclip-is-not-exported).

---

## Stage by stage

### Ingress — `POST /predict`

[`app/serving/main.py`](../app/serving/main.py). The frame arrives base64-encoded
in a JSON body alongside the backend that should score it and the product
category. Two things happen before any image processing: the `X-API-Key` header
is resolved to a `Principal` by [`app/serving/auth.py`](../app/serving/auth.py)
(`/predict` needs `viewer`), and `bind_log_context(model_backend, category)`
pins those two words onto every log record the request will emit — the same two
words that label the Prometheus series, so a spike and the lines explaining it
are selected with one query.

[`app/serving/imaging.py`](../app/serving/imaging.py) decodes the payload to a
BGR NumPy array. A payload that is not a decodable image raises
`InvalidImageError` → **422**, before a model is loaded.

### FrameGuard — blur / exposure / size

[`app/guardrails/quality.py`](../app/guardrails/quality.py). Five checks, run in
most-fundamental-first order, with the **first** failure deciding the reason:

| Order | Reason | Fails when | Default |
|---|---|---|---|
| 1 | `too_small` | either side is under `min_resolution` px | 64 |
| 2 | `invalid_aspect_ratio` | `w/h` above `max_aspect_ratio` or below its reciprocal | 10.0 |
| 3 | `underexposed` | more than `exposure_fraction` of pixels strictly below `dark_level` | 0.95 / 10 |
| 4 | `overexposed` | more than `exposure_fraction` of pixels strictly above `bright_level` | 0.95 / 245 |
| 5 | `blurry` | Laplacian variance below `blur_threshold` | 50.0 |

The ordering is the whole design of this stage: a 10×10 all-black tile fails
several checks at once, and `too_small` is the more useful thing to tell an
operator than `underexposed`. Every metric is computed regardless of which check
fires, so a rejection still logs a full row — watching `laplacian_variance` sag
across a shift is how a fouling lens is caught *before* it starts rejecting
frames. Blur is measured after downscaling to a 256 px longest edge, so the
threshold means the same thing on a 4 MP frame and a thumbnail.

A failure raises `GuardError` → **422**, and the count, the log line and the
`images_processed_total{result="rejected_by_guard"}` increment all happen in the
handler rather than the exception handler, because the handler runs in
Starlette's thread pool and the bound log context does not survive the unwind
back to the event loop.

Every threshold is overridable from the environment (`BLUR_THRESHOLD`,
`MIN_RESOLUTION`, `MAX_ASPECT_RATIO`, `EXPOSURE_DARK_LEVEL`,
`EXPOSURE_BRIGHT_LEVEL`, `EXPOSURE_FRACTION`, `BLUR_RESIZE_EDGE`).

**The guard runs twice, on purpose.** Every `AnomalyModel.predict` validates the
frame internally as well, so the API's call is a deliberate duplicate. It is not
free — 1.3 ms on a 256×256 frame, 23 ms on a 900×900 one, because the cost scales
with pixel count — but against a forward pass of several hundred milliseconds it
buys a structured 422 with the failing reason *and* a short circuit before the
registry is touched, while the model's own call keeps the guarantee true for
every caller of the model layer rather than only this one. Those measured values
are also where `GUARD_LATENCY_BUCKETS` comes from.

### ModelRegistry — lazy-load + Redis cache

[`app/serving/model_registry.py`](../app/serving/model_registry.py) holds one
model per `(backend, category)` pair and loads nothing until somebody asks. That
is what keeps `GET /health` answering in milliseconds and stops an orchestrator
killing a container for taking half a minute to become ready; the cost is that
the first request after every restart pays a load.

[`app/serving/model_cache.py`](../app/serving/model_cache.py) is the answer to
that cost, and it is worth being precise about what it caches: **four fields per
model** — `backend`, `category`, `checkpoint_path`, `loaded_at` — and **not the
weights**. A PatchCore checkpoint is 221 MB and its ONNX export 346 MB; pushing
those through Redis would be slower than reading the file already on the volume.
What is stored is a note saying "this pair was in use", written once at load time
with a one-hour TTL, so a restarted API can rebuild its working set in the
background while it is already answering.

Redis is optional in every direction. Nothing in the module raises: a failure
demotes the cache to an in-process dict that implements the same TTL, is logged,
and is not retried for 30 seconds, so an unreachable Redis costs one 0.5-second
timeout rather than one per load. The two modes differ only in surviving a
restart.

The load is timed separately and **subtracted from the reported latency** — a
cold load is a one-off startup cost, and folding a 30-second outlier into the
same series as the sub-second steady state would ruin every percentile computed
from it. A load over `_SLOW_LOAD_SECONDS` (1.0 s) emits `model_cold_loaded`
saying so.

If no artifact can serve the requested pair, `ModelNotReadyError` → **503**.

### The three backends

All of them implement [`AnomalyModel`](../app/models/base.py) — four methods and
one output type — which is the seam that lets the serving layer swap a PyTorch
wrapper for an ONNX graph without a route change.

| Backend | Module | Needs | Notes |
|---|---|---|---|
| PatchCore | [`patchcore.py`](../app/models/patchcore.py) | ~200 good images | Frozen WideResNet-50 memory bank. Top accuracy on `bottle`. |
| EfficientAD | [`efficientad.py`](../app/models/efficientad.py) | ~200 good images | Student-teacher; inference cost is constant in training-set size. |
| WinCLIP | [`winclip.py`](../app/models/winclip.py) | **nothing** | Zero-shot via CLIP text prompts. 240 px, CLIP normalization. |
| `onnx_*` | [`onnx_runner.py`](../app/models/onnx_runner.py) | an export | Serves the exported graph as an ordinary `AnomalyModel`. |

Two resolution rules are worth knowing because they show up in responses:

* **ONNX fallback.** Asking for `patchcore` when only `patchcore.onnx` exists
  serves the graph rather than 503-ing — but the response says `onnx_patchcore`,
  never `patchcore`, so a caller can explain a shift in numbers.
* **The frame is passed as BGR.** `model.predict(frame, color_order="bgr")`,
  because the decoder hands back OpenCV's channel order. Defaulting to `"rgb"`
  would swap channels under an ImageNet- or CLIP-pretrained backbone: no error,
  no shape change, quietly worse scores.

The export path itself — what `scripts/export_onnx.py` produces, what INT8 cost
in latency and accuracy, and why it is not recommended here — is
[`performance.md`](performance.md).

### ModelOutput — score + pixel heatmap

[`app/models/base.py`](../app/models/base.py). A frozen dataclass in plain
NumPy/Python types, which is what keeps anomalib's output containers from
leaking into the serving layer:

| Field | Meaning |
|---|---|
| `anomaly_score` | Image-level score. Calibrated models emit `[0, 1]` with 0.5 the fitted boundary; uncalibrated ones emit raw distances. |
| `anomaly_map` | `(H, W)` float32 heatmap **at the input frame's resolution**, so a caller can overlay it without rescaling. |
| `is_defective` | Whether the score cleared `ANOMALY_THRESHOLD`. |
| `model_name` | What actually scored the frame — the resolved name, not the requested one. |

`__post_init__` refuses a non-2-D map or a non-finite score, so a malformed
result fails at construction rather than three stages downstream.

### ScoreDistributionMonitor — KS drift check

[`app/evaluation/drift.py`](../app/evaluation/drift.py). Every score is appended
to a rolling window of 500, keyed by `(resolved model name, category)` — keyed by
the *resolved* name for the same reason the counter is: an ONNX graph scores a
little differently from the checkpoint it was exported from, and a window pooling
the two would report that difference as drift.

`GET /drift` compares that window against a reference distribution with a
two-sample Kolmogorov-Smirnov test, reporting `ks_drift` below
`KS_DRIFT_THRESHOLD` (default 0.05) alongside the window's mean/std/p10/p50/p90.
Below 30 samples in either window it reports `p_value: null` — *no verdict*,
rather than *no drift*. The reasoning for the test, the 5% false-alarm rate the
threshold buys by construction, and the response ladder are in
[`evaluation.md`](evaluation.md#keeping-the-numbers-true-in-production-drift-and-the-operating-point).

The append is deliberately placed **after** the verdict and **outside** the timed
section. It is an O(1) deque append behind an uncontended lock, but it is
observability, and nothing observational belongs inside the number a latency SLO
is read from. It is also unconditional: a monitor that only ran once someone had
configured a reference would have no history the first time anyone asked.

### FastAPI response — JSON + base64 heatmap

`InferenceResponse` carries `anomaly_score`, `is_defective`, `model_name`,
`anomaly_map_b64` (a PNG at the submitted frame's resolution), `latency_ms`
(model load excluded), and `guard_passed` / `guard_reason`.

The error contract is one code per cause, so a caller can branch on it:

| Condition | Status | Detail |
|---|---|---|
| Undecodable `image_b64` | 422 | `invalid_image` |
| Frame failed a quality check | 422 | the guard reason |
| Body fails schema validation | 422 | field errors, payload not echoed |
| No artifact can serve the pair | 503 | `model_not_ready` |
| Missing / unknown key, insufficient role | 401 / 403 | see [`security.md`](security.md) |

### The observability lane

[`app/observability/`](../app/observability). Three correlated views plus the
trail, joined on the same field names:

* **Metrics** — five series on `GET /metrics`: `images_processed_total`,
  `inference_latency_seconds`, `guard_check_latency_seconds`,
  `guard_rejections_total`, `models_loaded_count`. `inference_latency_seconds`
  covers the forward pass alone, excluding decode and heatmap encoding — the two
  ends that scale with the payload rather than with the model, and whose presence
  would stop the histogram answering "did the model get slower".
* **Traces** — one span per stage: `preprocess`, `guard`, `model_load`,
  `model_inference`, `postprocess`, each carrying the attributes that explain an
  outlier (`cold_load`, `guard.reason`, `anomaly_score`).
* **Structured logs** — structlog to stderr, human-readable by default and JSON
  under `LOG_FORMAT=json`, every record carrying `request_id` and `trace_id`.
* **Audit trail** — `results/audit.jsonl`, append-only, fed by `/benchmark` and
  `/calibrate` only.

---

## The write path: `/calibrate`

Everything above is a read. `POST /calibrate` is the one call that changes how
later requests are graded: it sweeps every threshold a labelled score set can
produce ([`app/evaluation/calibration.py`](../app/evaluation/calibration.py)),
installs the best on the resident model, and — by default — installs the same
scores as the drift monitor's reference, because the calibration set *is* the new
definition of normal.

Two properties of that write belong next to the diagram rather than in a
footnote: it is **in memory** (lost on restart) and **per-worker** (under
`uvicorn --workers N` it reaches the one worker that served the request). A
threshold worth keeping belongs in `ANOMALY_THRESHOLD`; the endpoint is for
finding it, not for storing it.

---

## Deployment shape

Five compose services, described in the [README](../README.md#quickstart):
`api` (this pipeline), `dashboard` (a Streamlit client that imports nothing from
`app/`), `redis` (the metadata cache above, optional), `prometheus` (scrapes
`GET /metrics`, waits for the api container to be *healthy*), and `grafana` (a
provisioned dashboard against a provisioned datasource).
