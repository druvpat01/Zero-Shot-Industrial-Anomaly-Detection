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
- **Deployment** — a four-service compose stack (api, redis, prometheus,
  grafana) built from [`docker/`](docker/); see the Quickstart.

## Quickstart

Three commands, no Python on the host:

```bash
git clone <repo> && cd defect-detection
cp .env.example .env
docker compose up --build
```

Then, from a second terminal:

```bash
docker compose exec api python scripts/run_api_demo.py --base-url http://localhost:8000
```

That scores five real MVTec bottle frames — three defective, two clean — through
`POST /predict` over HTTP and asserts every defect outscored every clean part.
The demo runs *inside* the api container because it imports `app/` to find the
sample images and read the API keys; from the host it is the same line, after a
`make setup` to build the venv it needs. Nothing else is required either way:
the image carries every dependency, and `.env.example` ships working development
keys.

| | | |
|---|---|---|
| API | <http://localhost:8000/docs> | OpenAPI, and `GET /health` for liveness |
| Prometheus | <http://localhost:9090/targets> | the `defect-detection` target, UP |
| Grafana | <http://localhost:3000> | `admin` / `$GRAFANA_ADMIN_PASSWORD` |

The Grafana dashboard is provisioned, not imported by hand — it is under
**Dashboards → Defect Detection → Defect Detection — Inference** on first load,
already pointed at the Prometheus that is already scraping the API.

Four services, and what each is for:

| Service | Why it is in the stack |
|---|---|
| `api` | The FastAPI app, built from [`docker/Dockerfile`](docker/Dockerfile). `./data` is mounted read-only (models read images, and nothing should write to a dataset); `./results` read-write, so checkpoints, benchmark JSON and the audit log outlive the container. |
| `redis` | Remembers which models were loaded — metadata only, one-hour TTL — so a restarted API rebuilds its working set instead of making the next caller pay a cold load. Optional: with it down, the API logs a warning and behaves exactly as before. |
| `prometheus` | Scrapes `GET /metrics`. Waits for the api container to be *healthy*, not merely started, so the first scrape lands on a listening socket. |
| `grafana` | Renders [the dashboard](docker/grafana/dashboards/defect_detection.json) against a provisioned datasource. |

`docker compose down` stops it; add `-v` to discard the Prometheus history, the
Grafana database and the cached backbone weights along with it.

If your Docker predates Compose V2, the command is the hyphenated
`docker-compose`; everything else is identical.

### Working on it locally

```bash
make setup     # venv + requirements-dev.txt (runtime deps, plus pytest/ruff/onnx)
make test
make serve     # uvicorn on :8000, without the container
```

`requirements.txt` is the runtime set and the only thing the image installs;
`requirements-dev.txt` includes it and adds the tooling. Keeping them apart is
most of how the image stays under 2 GB — that, and installing CPU-only torch,
which is the difference between a 1.7 GB image and a 6 GB one.

`.env` is where API keys live. Every endpoint except `/health` is gated by
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

`docker compose up` brings the scraper and the dashboard up with the service, so
this needs no setup — but the config is ordinary and works standalone too:

```bash
# Already running as part of the stack; this is the same config, by hand
prometheus --config.file=docker/prometheus.yml   # retarget localhost:8000 first
# or import docker/grafana/dashboards/defect_detection.json into any Grafana
```

The dashboard has four panels — requests/sec, p50/p95 inference latency, guard
rejection rate by reason, and model backend distribution. It is kept in Grafana's
portable export form, with a `${DS_PROMETHEUS}` datasource input, so importing it
by hand prompts for a datasource; under compose,
[`docker/grafana/entrypoint.sh`](docker/grafana/entrypoint.sh) binds that input
to the provisioned Prometheus instead.
