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

One frame's path through the service, in the order it happens:

```
frame ─▶ FrameGuard ─▶ ModelRegistry ─▶ backend ─▶ ModelOutput ─▶ drift monitor ─▶ JSON + heatmap
         blur/size/     lazy-load +     one of 5    score +        rolling KS       base64 PNG
         exposure       Redis metadata              heatmap        window
```

[`docs/architecture.md`](docs/architecture.md) is the full diagram with the
thresholds, the error codes and the reasoning for each stage.

- **Data** — dataset download, preprocessing, and datamodule wrappers, keeping
  anomalib's input types out of everything downstream.
- **Models** — PatchCore, EfficientAD, and WinCLIP behind one
  [`AnomalyModel`](app/models/base.py) interface: four methods and one output
  type, which is what lets a PyTorch wrapper be swapped for an ONNX graph without
  a route change. PatchCore needs ~200 good images, EfficientAD the same,
  **WinCLIP needs none** — it scores zero-shot from CLIP text prompts, which is
  the cold-start argument for the whole platform.
- **ONNX export** — [`scripts/export_onnx.py`](scripts/export_onnx.py) exports
  PatchCore and EfficientAD to FP32 and INT8 graphs, served by
  [`ONNXRunner`](app/models/onnx_runner.py) as ordinary `AnomalyModel`s. So the
  wire accepts five backends: `patchcore`, `efficientad`, `winclip`,
  `onnx_patchcore`, `onnx_efficientad`. Asking for `patchcore` when only the
  export exists serves the graph and says `onnx_patchcore` in the response rather
  than 503-ing. **PatchCore ONNX FP32 is the path to ship** — same accuracy to the
  fourth decimal, −12% median and −24% p99 latency; INT8 measured *slower* and
  less accurate, and WinCLIP is not exportable at all.
  [`docs/performance.md`](docs/performance.md) has every number and why.
- **Evaluation** — AUROC, AU-PRO, and F1 metrics implemented from scratch with a
  benchmark runner, plus score-distribution drift detection and threshold
  calibration. [`docs/evaluation.md`](docs/evaluation.md).
- **Guardrails** — five input-quality checks (size, aspect ratio, under/over
  exposure, blur) in most-fundamental-first order, so a bad frame is rejected
  without loading a model. Measured at 1.3 ms on a 256×256 frame and 23 ms on a
  900×900 one — the cost scales with pixel count, not with the model.
- **Serving** — FastAPI routes and schemas, a lazy model registry, and an
  optional Redis cache of *which* models were resident — metadata, never weights.
- **Auth** — two roles on one `X-API-Key` header. `viewer` opens `/predict`;
  `operator` adds `/models`, `/drift`, `/calibrate` and `/benchmark`, and nests
  viewer. `/health` and `/metrics` are open — a liveness probe and a scraper have
  nowhere good to hold a credential — and a server with no keys configured
  refuses everything else with `503`. [`docs/security.md`](docs/security.md).
- **Observability** — structlog to stderr (JSON on demand), five Prometheus
  metrics, OpenTelemetry spans per pipeline stage, and an append-only audit trail
  for the two calls that cost real money or change how parts are graded.
- **Deployment** — a five-service compose stack (api, redis, prometheus,
  grafana, dashboard) built from [`docker/`](docker/); see the Quickstart.

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
| Dashboard | <http://localhost:8501> | upload a frame, see the verdict and the heatmap |
| API | <http://localhost:8000/docs> | OpenAPI, and `GET /health` for liveness |
| Prometheus | <http://localhost:9090/targets> | the `defect-detection` target, UP |
| Grafana | <http://localhost:3000> | `admin` / `$GRAFANA_ADMIN_PASSWORD` |

The dashboard is the one to open first if you want to see what this does rather
than read about it: drop in an image, get a verdict, a localisation heatmap and a
latency. [`docs/demo_script.md`](docs/demo_script.md) is a seven-step script for
walking somebody else through it in three minutes.

The Grafana dashboard is provisioned, not imported by hand — it is under
**Dashboards → Defect Detection → Defect Detection — Inference** on first load,
already pointed at the Prometheus that is already scraping the API.

Five services, and what each is for:

| Service | Why it is in the stack |
|---|---|
| `dashboard` | [`dashboard.py`](dashboard.py) — a Streamlit client on a stock `python:3.11-slim`, because it imports nothing from `app/` and needs none of the API's ~1.8 GB of dependencies. Three tabs: score a frame, compare backends, watch the live metrics. |
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
