# Three-minute live demo

A literal script for showing this project on a call: what to click, in what
order, and the one sentence to say while it happens. Seven steps, roughly three
minutes if nobody interrupts — and they will interrupt, which is the point, so
the steps are ordered to survive being cut short. Stopping after step 4 still
tells a complete story.

Everything below happens in a browser at <http://localhost:8501>, against the
containerized API. The dashboard imports nothing from `app/`; it is an HTTP
client in its own `python:3.11-slim` container. Worth saying out loud once,
because it means every picture on the screen was produced by the service, not by
the thing drawing the screen.

---

## Before the call

Ten minutes ahead, not two. Two of these steps take minutes and one needs the
network.

```bash
cp .env.example .env                      # if you have not already
docker compose up -d --build              # ~2 min warm, ~10 min from cold
python scripts/make_demo_frames.py        # writes results/demo_frames/
docker compose ps                         # all five services (healthy)
```

`make_demo_frames.py` produces the three files the demo uploads:

| File | What it is |
| --- | --- |
| `results/demo_frames/1_clean_bottle.png` | An unmodified good bottle from the test split |
| `results/demo_frames/2_defective_bottle_broken_large.png` | An unmodified defective one |
| `results/demo_frames/3_blurred_bottle.png` | The clean frame, Gaussian-blurred until the guard refuses it |

**Have those three files open in a file picker before you share your screen.**
Hunting through `data/MVTecAD/bottle/test/broken_large/` on a call is thirty
seconds of dead air.

### Warm WinCLIP if you plan to do step 5

The first WinCLIP request downloads and loads ~830 MB of CLIP weights. Measured
on this machine: **2 min 50 s cold, 3.5 s warm.** Do it before the call:

```bash
python - <<'EOF'
import base64, requests
frame = open("results/demo_frames/2_defective_bottle_broken_large.png", "rb").read()
requests.post("http://localhost:8000/predict",
    headers={"X-API-Key": "dev-viewer-key"},
    json={"category": "bottle", "model_backend": "winclip",
          "image_b64": base64.b64encode(frame).decode()}, timeout=600)
print("winclip warm")
EOF
```

If you skip this, skip step 5 too. A three-minute demo cannot afford a
three-minute model load, and "it's just loading" is the least persuasive
sentence available to you.

---

## Step 1 — Open the dashboard

**Do:** Open <http://localhost:8501>. In the sidebar, confirm **Category =
bottle** and **Model backend = patchcore**. The sidebar should show a green
`API ok · N model(s) resident`.

**Say:** *"This is a Streamlit client talking to the containerized inference API
over HTTP — same endpoints, same API keys as any other caller. The green line is
its `/health` probe; the keys are prefilled from `.env` so you can watch the app
instead of watching me type."*

---

## Step 2 — Score a clean part

**Do:** Tab **Live Inspection** → upload `1_clean_bottle.png`.

**Expect:** A green **NORMAL** badge (score ≈ 0.23), latency ≈ 600–700 ms, and
an overlay that is uniformly cool blue across the whole frame.

**Say:** *"PatchCore scored it 0.23 against a 0.5 threshold, and the heatmap is
cold everywhere — no region of this part looks unlike the training set of good
parts."*

> The heatmap is rendered server-side on a **fixed [0, 1] ramp** for a calibrated
> backend, which is what makes this frame and the next one directly comparable.
> If it were min-max normalized per frame, this clean part would come back
> covered in false colour — sensor noise stretched across the full spectrum.
> That is worth a sentence if the interviewer is technical.

---

## Step 3 — Score a defective part

**Do:** Upload `2_defective_bottle_broken_large.png` (same tab, same settings).

**Expect:** A red **DEFECTIVE** badge (score ≈ 0.95), similar latency, and a hot
red/orange region sitting exactly over the broken rim.

**Say:** *"0.95, and the localisation is the part that matters — this is not just
'something is wrong with this image', it is 'the defect is here', which is what
lets an operator act on it."*

**Say next, before they ask:** *"The score is per-image, the heatmap is
per-pixel, and they are on the same scale — one model produced both in a single
forward pass, in about 600 milliseconds on CPU."*

---

## Step 4 — Show the input-quality guard

**Do:** Upload `3_blurred_bottle.png`.

**Expect:** No heatmap at all. An amber **GUARD REJECTED** panel with
`reason: blurry`, and the frame shown beside it marked *rejected, not scored*.

**Say:** *"The API returned 422 and no model ran. On a real line the camera goes
out of focus, a light dies, a part moves through the exposure — and a model will
happily give you a confident score on any of those. This is the check that says
'I will not answer that question', which is a different and more useful thing
than answering it badly."*

**If asked how it decides:** Laplacian variance measured at a fixed scale, plus
exposure and resolution checks, all thresholds configurable
(`BLUR_THRESHOLD`, `DARK_THRESHOLD`, ...). It costs 1–23 ms per frame depending
on frame size. The rejection is counted in `guard_rejections_total{reason}`,
which you will point at in step 7.

**Worth conceding, unprompted:** the blurred frame is synthetic. MVTec contains
defective *objects*, not defective *photographs*, so the failure mode this guard
exists for is not in the dataset — which is exactly why it is not in most
projects built on it either.

---

## Step 5 — Same defect, zero-shot

*Only if you warmed WinCLIP. Otherwise skip to step 6.*

**Do:** Sidebar → **Model backend = winclip**. Re-upload
`2_defective_bottle_broken_large.png`.

**Expect:** **DEFECTIVE** again, score ≈ 0.51, latency ≈ 3.5 s, and a coarser
heatmap that still lands on the defect.

**Say:** *"That is the same defect found by a model that has never seen a single
bottle — WinCLIP scores against text prompts, so it needs no training data and no
checkpoint. It is slower and the score sits much closer to the threshold, but it
is what you deploy on day one for a new part, while you are still collecting
images to train PatchCore on."*

That trade — no training data, weaker margin, 5× the latency — is the whole
point of the backend, and step 6 is where it gets a number.

---

## Step 6 — Benchmark comparison

**Do:** Tab **Benchmark Comparison**. Category `bottle`, models
`patchcore` + `onnx_efficientad`, click **Run Benchmark**.

**Expect:** A spinner for **1–2 minutes**, then a table (Img-AUROC, Px-AUROC,
AU-PRO, Best-F1, s/img) and a bar chart of AU-PRO.

**Say while it runs:** *"This is scoring every model over the whole test split —
83 images each — which is why it is operator-gated and audited to
`results/audit.jsonl`. It is the endpoint that is also a denial-of-service
primitive if you leave it open."*

**Say about the table:** *"Image-AUROC saturates near 1.0 on MVTec, so it stops
discriminating between models — AU-PRO is the honest column, because it weights
every defect region equally and does not let a model coast on finding only the
big ones."*

**Do not run WinCLIP here on a live call.** It is minutes per backend on its own.
If they ask for its numbers, they are in `docs/evaluation.md` and in the dated
JSON under `results/`.

> **Timing note:** if three minutes is a hard limit, kick this off at the *start*
> of step 6 and talk through steps 2–5's screenshots while it runs, or run it
> before the call — results persist for the session, so the table will already be
> on the tab when you switch to it.

---

## Step 7 — Live system health

**Do:** Tab **System Health**. It refreshes itself every 10 seconds.

**Expect:** Three live counters (images processed, guard rejections, p50
inference latency), the drift table with the window's percentiles, and the API /
Redis / resident-models row.

**Say:** *"These are Prometheus counters read from the API's own `/metrics` —
that guard rejection is the blurred frame from two minutes ago. The table under
it is the drift monitor: a rolling window of anomaly scores per model, KS-tested
against a reference distribution, so a lens that fouls gradually shows up as a
distribution shift rather than as a Tuesday where the reject rate was quietly
wrong."*

**If `KS p` reads `—`:** say so plainly — *"no reference distribution has been
set in this process, so it is not claiming anything yet; `POST /calibrate`
establishes one."* A dash is a third state, not a pass, and knowing the
difference is a better answer than a green tick.

**If asked what else is behind it:** Prometheus at <http://localhost:9090> and a
provisioned Grafana dashboard at <http://localhost:3000> are already running in
the same compose stack.

---

## Recovery

| Symptom | Fix |
| --- | --- |
| Sidebar shows *API unreachable* | `docker compose ps` — the API takes ~15 s past container start to import torch. Wait, then rerun from the browser. |
| First upload hangs for a minute | Cold model load, excluded from the reported `latency_ms`. Say so; it is a real property of the system, not a stall. |
| `401 invalid_api_key` | The key boxes are prefilled from `.env`; if you edited it, restart the dashboard: `docker compose restart dashboard`. |
| Benchmark 503 `dataset_not_available` | `data/MVTecAD/<category>/test` is not mounted. Fall back to a saved run in `results/`. |
| Something is broken beyond rescue | Switch to `docs/evaluation.md` and the checked-in benchmark JSON. A candidate who can talk through their own metrics without the demo is not in trouble. |

---

## What each step is actually evidence of

Useful if the conversation turns from "show me" to "why should I care":

| Step | The claim it supports |
| --- | --- |
| 2, 3 | The model works, localises, and is fast enough to be real |
| 4 | Input validation before inference — the difference between a demo and a system |
| 5 | A cold-start story for a new part with no training data |
| 6 | Honest evaluation, with the metric argued rather than assumed |
| 7 | It is observable in production, and the failure it watches for is gradual |
