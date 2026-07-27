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
- **Evaluation** — AUROC, AU-PRO, and F1 metrics with a benchmark runner.
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

`.env` is also where API keys live. `/predict`, `/models` and `/benchmark` are
gated by an `X-API-Key` header, and a server with no keys configured refuses all
three with `503 auth_not_configured` — set `VIEWER_API_KEYS` and
`OPERATOR_API_KEYS` before `make serve`. `/health` stays open for liveness
probes. [`docs/security.md`](docs/security.md) explains what is gated, what is
audited, and what production would still require.

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
