# Zero-Shot Industrial Defect Detection & Pixel-Level Anomaly Segmentation

## Problem Statement

Industrial quality assurance traditionally relies on large volumes of labeled
defect examples — expensive to collect and often unavailable for rare or novel
faults. This platform tackles defect detection in an **unsupervised / zero-shot**
setting: models are trained only on nominal (defect-free) samples, or use
vision-language priors, and at inference time flag anomalous frames *and* produce
**pixel-level segmentation masks** localizing where the defect is. The goal is a
production-shaped service that generalizes across product categories with little
or no defect supervision, suitable for real-time QA on a manufacturing line.

## Architecture

_Placeholder — to be filled in as components land._

- **Data** — dataset download, preprocessing, and datamodule wrappers.
- **Models** — PatchCore, EfficientAD, and WinCLIP wrappers behind a common interface.
- **Evaluation** — AUROC, AU-PRO, and F1 metrics with a benchmark runner, plus
  score-distribution drift detection and threshold calibration.
- **Guardrails** — input frame quality validation before inference.
- **Serving** — FastAPI routes, schemas, and session management.
- **Observability** — structured logging, Prometheus metrics, and tracing.

## Quickstart

```bash
# Create the virtual environment and install dependencies
make setup

# Verify the environment
python -c "import anomalib, fastapi, torch, cv2; print('env ok')"

# Run the test suite
make test

# Start the API server
make serve
```

Copy `.env.example` to `.env` and adjust configuration as needed.

`.env` is also where API keys live. Every endpoint except `/health` is gated by
an `X-API-Key` header, and a server with no keys configured refuses them all with
`503 auth_not_configured` — set `VIEWER_API_KEYS` and `OPERATOR_API_KEYS` before
`make serve`. `/health` stays open for liveness probes.
[`docs/security.md`](docs/security.md) explains what is gated, what is audited,
and what production would still require.

## Reliability: drift and the operating point

The guardrails catch a bad *frame*. Nothing about a bad *score* is structurally
wrong — the model returns a well-formed float that simply no longer means what it
meant when the threshold was chosen, because the product line changed, the camera
aged, or something was deployed. `/health` stays green throughout.

[`app/evaluation/drift.py`](app/evaluation/drift.py) watches for it. Every
`/predict` feeds its score into a rolling window per `(model, category)`;
`GET /drift` compares that window against a reference distribution with a
two-sample Kolmogorov-Smirnov test — the right test here precisely because it
assumes nothing about the shape of the data, and anomaly scores are skewed,
bounded and usually bimodal. Below `KS_DRIFT_THRESHOLD` (default `0.05`) it
reports `ks_drift`, alongside the window's mean/std/p10/p50/p90 so an operator
can tell a shift that matters from one that is merely significant.

The usual fix is not a retrain but a new operating point.
[`app/evaluation/calibration.py`](app/evaluation/calibration.py) sweeps every
threshold a labelled score set can produce and returns the one maximising F1 (or
precision, recall, balanced accuracy); `POST /calibrate` installs it in memory and
— by default — installs the same scores as the drift reference, because the
calibration set *is* the new definition of normal.

```bash
# Fit an operating point from held-out labelled scores, and set the baseline
curl -s -X POST localhost:8000/calibrate -H "X-API-Key: $OPERATOR_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"category":"bottle","model_backend":"patchcore","metric":"f1",
       "samples":[{"score":0.11,"label":0},{"score":0.94,"label":1}]}'

# Then watch what the line is actually producing
curl -s localhost:8000/drift -H "X-API-Key: $OPERATOR_KEY" | python -m json.tool
```

Both endpoints are operator-only, and `/calibrate` is audited: it is the only
call that changes how the service grades parts. The change is in-memory and
per-worker, so a threshold worth keeping belongs in `ANOMALY_THRESHOLD` —
[`docs/evaluation.md`](docs/evaluation.md) has the full argument, including why
`p < 0.05` buys a 5% false-alarm rate by construction and what a production
system should do at each rung of the response ladder.

## Observability

Every request produces three correlated views of itself, all set up in
[`app/observability/`](app/observability/). Each module's docstring carries the
reasoning; the short version:

| | What it answers | Where |
|---|---|---|
| **Structured logs** | What happened in *this* request | stderr, via structlog |
| **Metrics** | What is happening across *all* requests | `GET /metrics` |
| **Traces** | Where the milliseconds went | console, or an OTLP collector |
| **Audit trail** | Who ran the expensive, privacy-relevant call | `results/audit.jsonl` |

They are joined on the same field names. A spike on a dashboard filters to a
`model` and a `category`; those are the same two keys on the log records; those
records carry the `trace_id` that opens the trace.

```bash
# Human-readable logs (the default) — or LOG_FORMAT=json for a log shipper
make serve

# Scrape the metrics
curl -s localhost:8000/metrics | grep images_processed_total

# Turn tracing off for a load test; point it at a collector otherwise
OTEL_TRACES_EXPORTER=none make serve
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces make serve
```

Five application metrics are exported: `images_processed_total`,
`inference_latency_seconds`, `guard_check_latency_seconds`,
`guard_rejections_total` and `models_loaded_count`. `GET /metrics` is
unauthenticated so a Prometheus server can scrape it without holding a
credential — protect it with a network control, not a key.

Ready-made monitoring config lives in [`docker/`](docker/):

```bash
prometheus --config.file=docker/prometheus.yml
# then import docker/grafana/dashboards/defect_detection.json into Grafana
```

The dashboard has four panels — requests/sec, p50/p95 inference latency, guard
rejection rate by reason, and model backend distribution — and prompts for a
Prometheus datasource on import.
