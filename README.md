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
