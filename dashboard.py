"""A Streamlit front end for the inspection API: upload a frame, see a verdict.

Everything this project does is already reachable over HTTP. What was missing is
a way to *watch* it happen — a heatmap next to the frame that produced it, a
guard rejection as a red panel rather than a 422, a benchmark as a table rather
than a JSON file. That is this module's entire job.

It is a client, not a shortcut
==============================
Nothing here imports :mod:`app`. Every number on the screen arrives over HTTP
from the containerized service, through the same endpoints, the same API keys and
the same auth as any other caller — so a dashboard that renders is also a proof
that the deployment works, and a broken model cannot look healthy here because
the dashboard is holding its own copy of it. The import cost is the point too: a
`python:3.11-slim` with `streamlit` and `requests` starts in a couple of seconds,
because it has no torch in it.

The consequence to keep in mind while reading: every failure mode below is a
*network* failure mode. The API being down, a key being wrong, a benchmark taking
four minutes — these are the normal states of a client, and each one is rendered
as a sentence somebody can act on rather than a traceback.

Three tabs, three different latencies
=====================================
Worth naming, because it drives most of the design:

* **Live Inspection** — one ``POST /predict``, 150 ms to a few seconds. Results
  are cached on the frame's bytes so that fiddling with an unrelated widget does
  not silently re-run a model.
* **Benchmark Comparison** — one ``POST /benchmark``, *minutes*. Long-timeout
  request, a spinner, and a warning about what is being asked before it is asked.
* **System Health** — three cheap GETs on a 10 s loop, in a fragment so the
  refresh redraws that panel alone and does not throw you back to tab one every
  ten seconds while you are talking.

Run it::

    streamlit run dashboard.py                    # against localhost:8000
    DASHBOARD_API_URL=http://api:8000 streamlit run dashboard.py

or ``docker compose up`` and open http://localhost:8501. See
``docs/demo_script.md`` for what to do with it once it is open.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Any

import matplotlib

# Before pyplot, and not negotiable: the default backend probes for a display,
# and there is no display in a container. Agg renders to a buffer, which is all
# st.pyplot ever wanted.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow matplotlib.use
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Where the API lives. ``http://api:8000`` inside compose (the service name),
#: ``http://localhost:8000`` for a dashboard run on the host against a published
#: port. Overridable in the sidebar, because pointing this at a staging host is
#: the one thing somebody will want to do that no environment variable was set for.
DEFAULT_API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:8000").rstrip("/")

#: MVTec AD's fifteen categories. A static list rather than something discovered
#: from the API: there is no endpoint that enumerates categories, and inventing
#: one to populate a dropdown would be a schema change driven by a widget. The
#: configured default is floated to the top so the common case is zero clicks.
MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

#: The five names ``model_backend`` accepts, mirroring the ``ModelBackend``
#: literal in :mod:`app.serving.schemas`. Duplicated rather than imported for the
#: reason in the module docstring — this process does not import the app — and a
#: name that drifts fails loudly as a 422 from pydantic, not silently.
MODEL_BACKENDS = (
    "patchcore",
    "efficientad",
    "winclip",
    "onnx_patchcore",
    "onnx_efficientad",
)

#: Header the API authenticates on (``app.serving.auth.API_KEY_HEADER``).
API_KEY_HEADER = "X-API-Key"

#: Connect and read timeouts for the cheap endpoints. The read budget is generous
#: because the *first* request for a cold backend pays that backend's load —
#: WinCLIP's CLIP weights are ~830 MB and can take a minute on a cold cache — and
#: a demo that times out on its first upload is worse than one that waits.
PREDICT_TIMEOUT = (5, 180)
CHEAP_TIMEOUT = (3, 10)

#: ``POST /benchmark`` scores every model over an entire test split: tens of
#: seconds for an ONNX graph, minutes for WinCLIP. There is no server-side
#: timeout that would save us, so the client's is set to an hour and the UI warns
#: instead of pretending this is a request anyone should wait on.
BENCHMARK_TIMEOUT = (5, 3600)

#: Tab 3's refresh period, in seconds.
REFRESH_SECONDS = 10

#: Blend weight of the heatmap over the frame. Half and half: enough colour to
#: locate a defect, enough frame underneath to see *what* it is on.
HEATMAP_ALPHA = 0.5


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class ApiError(RuntimeError):
    """Any call that did not produce a usable answer.

    One class for connection failures, timeouts, 401s and 503s alike, because
    every one of them ends the same way here: a red box with a sentence in it.
    The distinction that *does* matter — a frame the guard refused — is not an
    error and is not raised; see :class:`PredictResult`.
    """


def _headers(api_key: str) -> dict[str, str]:
    """Auth header, omitted entirely when no key is set.

    Sending ``X-API-Key: `` would be authenticated as an empty key and rejected
    as invalid, which reports a wrong credential where the truth is a missing
    one. Omitting it gets the honest 401 (``missing_api_key``).
    """
    return {API_KEY_HEADER: api_key} if api_key else {}


def _describe_http_error(response: requests.Response) -> str:
    """Turn an error response into one sentence for a human.

    The API's error bodies are ``{"detail": <slug>, "reason": <detail>}``, and
    both halves matter: the slug says which failure this is, the reason says
    which instance of it. Anything that is not that shape (a proxy's HTML error
    page, say) falls back to the status line plus a bounded slice of the body,
    which is the difference between "502" and "502 from something that is not
    the API".
    """
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"

    if isinstance(body, dict):
        detail = body.get("detail")
        reason = body.get("reason")
        if isinstance(detail, list):  # pydantic's validation-error shape
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        parts = [str(part) for part in (detail, reason) if part]
        if parts:
            return f"HTTP {response.status_code}: {' — '.join(parts)}"
    return f"HTTP {response.status_code}: {str(body)[:200]}"


def _get(base_url: str, path: str, api_key: str = "", timeout: tuple[int, int] = CHEAP_TIMEOUT) -> Any:
    """GET ``path`` and return parsed JSON, or raise :class:`ApiError`."""
    try:
        response = requests.get(f"{base_url}{path}", headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach {base_url}{path}: {type(exc).__name__}") from exc
    if not response.ok:
        raise ApiError(_describe_http_error(response))
    return response.json()


def _get_text(base_url: str, path: str, timeout: tuple[int, int] = CHEAP_TIMEOUT) -> str:
    """GET ``path`` as text. ``/metrics`` is Prometheus exposition, not JSON."""
    try:
        response = requests.get(f"{base_url}{path}", timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach {base_url}{path}: {type(exc).__name__}") from exc
    if not response.ok:
        raise ApiError(_describe_http_error(response))
    return response.text


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictResult:
    """What came back from ``POST /predict``, including the ways it said no.

    A guard rejection is a *result*, not an exception: the API answers 422 with
    ``{"detail": "guard_failed", "reason": "blurry"}``, and that is a real
    verdict about a real frame — the camera is out of focus — which the operator
    is meant to see and act on. Modelling it as an error would file "your lens is
    dirty" alongside "the server is down".

    Attributes:
        scored: True when a model ran and there is a heatmap to draw.
        guard_reason: The failing check (``blurry``, ``too_dark``, ...) when the
            frame was refused, else None.
        body: The decoded ``InferenceResponse`` when scored, else ``{}``.
    """

    scored: bool
    guard_reason: str | None = None
    body: dict[str, Any] | None = None

    @property
    def guard_rejected(self) -> bool:
        return self.guard_reason is not None


@st.cache_data(show_spinner=False, max_entries=32)
def predict(base_url: str, api_key: str, category: str, backend: str, image_bytes: bytes) -> PredictResult:
    """Score one frame. Cached on every argument, including the frame's bytes.

    The cache is not an optimisation, it is a correctness fix for the way
    Streamlit works: the whole script re-runs on *any* widget interaction, and an
    uncached call here would re-score the uploaded frame every time somebody
    touched an unrelated control on another tab. On WinCLIP that is seconds of
    wall clock and a needless load on the service being demonstrated.

    Args:
        base_url: API root, no trailing slash.
        api_key: A viewer (or operator) key.
        category: MVTec-style category, selecting the checkpoint or prompt noun.
        backend: One of :data:`MODEL_BACKENDS`.
        image_bytes: The uploaded file's raw bytes — PNG, JPEG, whatever OpenCV
            can decode. Base64 happens here, not in the caller.

    Returns:
        A :class:`PredictResult`; a refused frame comes back with
        ``scored=False`` rather than raising.

    Raises:
        ApiError: Unreachable service, bad credential, undecodable image, or a
            backend with no artifact to serve the category.
    """
    payload = {
        "category": category,
        "model_backend": backend,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers=_headers(api_key),
            timeout=PREDICT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach {base_url}/predict: {type(exc).__name__}") from exc

    if response.status_code == 422:
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if isinstance(body, dict) and body.get("detail") == "guard_failed":
            return PredictResult(scored=False, guard_reason=str(body.get("reason") or "unknown"))
        # The other 422 is `invalid_image`, which is a client-side mistake
        # (a corrupt upload, an unsupported container) and belongs in the
        # error path with everything else the caller has to fix.
        raise ApiError(_describe_http_error(response))

    if not response.ok:
        raise ApiError(_describe_http_error(response))
    return PredictResult(scored=True, body=response.json())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Uploaded file bytes -> ``(H, W, 3)`` RGB ``uint8``.

    ``convert("RGB")`` rather than a channel check: a PNG with an alpha channel
    or a greyscale JPEG would otherwise reach the blend below with a shape the
    arithmetic cannot broadcast, and the failure would arrive as a numpy error
    about dimensions rather than as anything to do with the picture.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def decode_heatmap(anomaly_map_b64: str) -> np.ndarray:
    """The API's base64 PNG heatmap -> ``(H, W, 3)`` RGB ``uint8``."""
    return decode_image(base64.b64decode(anomaly_map_b64))


def overlay_heatmap(frame_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    """Alpha-blend the anomaly heatmap over the frame that produced it.

    The colormap is applied **server-side** — ``app.serving.imaging`` renders the
    anomaly map through JET at the submitted frame's resolution — so this blends
    the PNG as it arrives rather than re-deriving a scalar field from coloured
    pixels and re-applying a colormap to it. Two reasons that ordering is worth
    keeping. OpenCV's ``COLORMAP_JET`` *is* matplotlib's ``jet`` ramp, so nothing
    is lost by not doing it here; and, more importantly, the server is the only
    party that knows whether the producing model is calibrated, which decides
    whether the ramp spans a fixed ``[0, 1]`` or is normalized per frame. Inverting
    the colormap client-side would throw that away and quietly stretch the sensor
    noise on a clean part across the full spectrum — a picture of a defect that
    is not there. See :func:`app.serving.imaging.encode_heatmap_png_b64`.

    Args:
        frame_rgb: The original frame, ``(H, W, 3)`` RGB uint8.
        heatmap_rgb: The decoded heatmap, same shape (resized here if not).
        alpha: Heatmap weight; ``0.5`` splits it evenly with the frame.

    Returns:
        The blend, ``(H, W, 3)`` RGB uint8.
    """
    if heatmap_rgb.shape[:2] != frame_rgb.shape[:2]:
        # Documented not to happen — the API renders at the submitted
        # resolution — but a client that assumes its input is well-formed is a
        # client that crashes instead of drawing something slightly wrong.
        height, width = frame_rgb.shape[:2]
        heatmap_rgb = np.asarray(
            Image.fromarray(heatmap_rgb).resize((width, height), Image.BILINEAR),
            dtype=np.uint8,
        )
    blended = (1.0 - alpha) * frame_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def colorbar_figure() -> plt.Figure:
    """A JET strip labelled low -> high, so the overlay is readable without a caption.

    A heatmap without a legend is a picture with an implied claim in it. This is
    the claim: blue is a low anomaly score, red is a high one.
    """
    figure, axes = plt.subplots(figsize=(6, 0.42))
    ramp = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    axes.imshow(ramp, aspect="auto", cmap="jet", vmin=0.0, vmax=1.0)
    axes.set_yticks([])
    axes.set_xticks([0, 255])
    axes.set_xticklabels(["low anomaly score", "high anomaly score"], fontsize=8)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.2)
    return figure


def verdict_badge(is_defective: bool, score: float) -> str:
    """The headline verdict, as an HTML block.

    Streamlit has no "large coloured banner" primitive, and the alternatives —
    ``st.error`` / ``st.success`` — are sized for a sentence of body text. The
    verdict is the one thing on this page that has to be legible from across a
    room during a screen share, so it gets markup.
    """
    label = "DEFECTIVE" if is_defective else "NORMAL"
    background = "#b3261e" if is_defective else "#1b7f3b"
    return (
        f"<div style='background:{background};color:#ffffff;border-radius:10px;"
        f"padding:18px 24px;text-align:center;margin:6px 0 14px 0;'>"
        f"<div style='font-size:44px;font-weight:800;letter-spacing:2px;line-height:1.1;'>{label}</div>"
        f"<div style='font-size:17px;opacity:0.92;margin-top:4px;'>anomaly score {score:.3f}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Prometheus exposition parsing
# ---------------------------------------------------------------------------


def parse_samples(exposition: str, metric_name: str) -> list[tuple[dict[str, str], float]]:
    """Pull every ``(labels, value)`` sample of one metric out of ``/metrics`` text.

    A hand-rolled parser rather than ``prometheus_client.parser`` for one reason:
    this container installs four packages, and adding a fifth to read three
    numbers off a text endpoint is a dependency for a line of code. The format
    being parsed is the stable exposition format — ``name{label="value",...}
    number`` — and the only subtlety is that ``images_processed_total`` and
    ``images_processed_created`` share a prefix, so names are matched exactly and
    never by ``startswith``.

    Args:
        exposition: The body of ``GET /metrics``.
        metric_name: Exact sample name, e.g. ``images_processed_total``.

    Returns:
        One entry per label combination. Empty when the metric has no samples
        yet — a counter that has never been incremented is absent from the
        exposition, which is a zero and not a missing metric.
    """
    samples: list[tuple[dict[str, str], float]] = []
    for line in exposition.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, raw_value = line.rpartition(" ")
        if not head:
            continue
        name, brace, raw_labels = head.partition("{")
        if name.strip() != metric_name:
            continue
        labels: dict[str, str] = {}
        if brace:
            for pair in raw_labels.rstrip("}").split(","):
                key, _, value = pair.partition("=")
                if key:
                    labels[key.strip()] = value.strip().strip('"')
        try:
            samples.append((labels, float(raw_value)))
        except ValueError:  # NaN, or a line this parser has no business reading
            continue
    return samples


def sum_samples(exposition: str, metric_name: str) -> float:
    """Total one counter across every label combination."""
    return sum(value for _, value in parse_samples(exposition, metric_name))


def histogram_p50(exposition: str, metric_name: str = "inference_latency_seconds") -> float | None:
    """Median latency, interpolated from the histogram's cumulative buckets.

    This is ``histogram_quantile(0.5, ...)`` done client-side, aggregated over
    every ``(model, backend)`` series — the same arithmetic Prometheus would do,
    reproduced here so tab 3 needs the API and nothing else. Being explicit about
    what that number is worth: bucket boundaries are coarse (10 ms, 50 ms, 100 ms,
    250 ms, 500 ms, 1 s, ...), so a p50 reported as 375 ms means "somewhere
    between 250 ms and 500 ms, linearly guessed". It is an order-of-magnitude
    reading, and the per-request ``latency_ms`` on tab 1 is the exact one.

    Returns:
        Seconds, or None when nothing has been observed yet.
    """
    buckets: dict[float, float] = {}
    for labels, value in parse_samples(exposition, f"{metric_name}_bucket"):
        edge = labels.get("le")
        if edge is None:
            continue
        bound = float("inf") if edge in {"+Inf", "Inf"} else float(edge)
        buckets[bound] = buckets.get(bound, 0.0) + value

    if not buckets:
        return None
    ordered = sorted(buckets.items())
    total = ordered[-1][1]  # the +Inf bucket is the observation count
    if total <= 0:
        return None

    target = total * 0.5
    previous_bound, previous_count = 0.0, 0.0
    for bound, cumulative in ordered:
        if cumulative >= target:
            if bound == float("inf"):
                # Everything above the last finite edge. No upper bound exists to
                # interpolate towards, so report that edge and let it read as the
                # ">= 5 s" it is, rather than inventing a number.
                return previous_bound or None
            span = cumulative - previous_count
            if span <= 0:
                return bound
            return previous_bound + (bound - previous_bound) * (target - previous_count) / span
        previous_bound, previous_count = bound, cumulative
    return ordered[-1][0]


# ---------------------------------------------------------------------------
# Tab 1 — Live Inspection
# ---------------------------------------------------------------------------


def render_live_inspection(base_url: str, viewer_key: str, category: str, backend: str) -> None:
    """Upload a frame, show the verdict, the heatmap and what it cost."""
    st.subheader("Live Inspection")
    st.caption(
        f"`POST {base_url}/predict` — one frame, scored by **{backend}** on category "
        f"**{category}**. Viewer key or better."
    )

    upload = st.file_uploader(
        "Frame to inspect",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        help="An MVTec test image, or anything from a camera pointed at the same kind of part.",
    )
    if upload is None:
        st.info("Upload a JPG or PNG to score it. `data/MVTecAD/bottle/test/` is a good place to start.")
        return

    image_bytes = upload.getvalue()
    try:
        frame = decode_image(image_bytes)
    except Exception as exc:  # noqa: BLE001 - Pillow raises a family of these
        st.error(f"Could not read **{upload.name}** as an image: {type(exc).__name__}.")
        return

    with st.spinner(f"Scoring with {backend}… (a cold backend loads on its first frame)"):
        try:
            result = predict(base_url, viewer_key, category, backend, image_bytes)
        except ApiError as exc:
            st.error(str(exc))
            st.caption(
                "Check the API URL and key in the sidebar, and that the backend has an "
                "artifact for this category (`GET /models`)."
            )
            return

    # The guard path. Deliberately loud and deliberately *instead of* the
    # heatmap: no model ran, so there is no heatmap to draw, and showing a stale
    # or blank one next to a rejection would imply the frame was scored.
    if result.guard_rejected:
        left, right = st.columns([1, 1])
        with left:
            st.image(frame, caption=f"{upload.name} — rejected, not scored")
        with right:
            st.markdown(
                "<div style='background:#8a5a00;color:#ffffff;border-radius:10px;padding:18px 24px;"
                "text-align:center;margin:6px 0 14px 0;'>"
                "<div style='font-size:38px;font-weight:800;letter-spacing:2px;'>GUARD REJECTED</div>"
                f"<div style='font-size:18px;margin-top:6px;'>reason: <code style='color:#fff;'>"
                f"{result.guard_reason}</code></div></div>",
                unsafe_allow_html=True,
            )
            st.warning(
                f"The input-quality guard refused this frame (`{result.guard_reason}`) and **no model ran**. "
                "An anomaly score from an unusable frame is a guess wearing a number, so the API "
                "returns 422 rather than one."
            )
            st.caption(
                "Thresholds are configurable — `BLUR_THRESHOLD`, `DARK_THRESHOLD`, "
                "`BRIGHT_THRESHOLD` — see `app/guardrails/quality.py`."
            )
        return

    body = result.body or {}
    score = float(body.get("anomaly_score", 0.0))
    is_defective = bool(body.get("is_defective", False))

    st.markdown(verdict_badge(is_defective, score), unsafe_allow_html=True)

    served_by = str(body.get("model_name", backend))
    columns = st.columns(3)
    columns[0].metric("Latency", f"{float(body.get('latency_ms', 0.0)):.0f} ms", help="Server-side, excluding cold model load.")
    columns[1].metric("Served by", served_by, help="The model that actually ran; a missing checkpoint falls back to ONNX.")
    columns[2].metric("Guard", "passed" if body.get("guard_passed") else "failed")

    if served_by != backend:
        st.info(
            f"Requested **{backend}**, served by **{served_by}** — the requested backend had no "
            "checkpoint for this category, so the API fell back to its exported graph."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Original frame**")
        st.image(frame, caption=upload.name)
    with right:
        st.markdown(f"**Anomaly heatmap** (JET, {int(HEATMAP_ALPHA * 100)}% over the frame)")
        try:
            overlay = overlay_heatmap(frame, decode_heatmap(str(body.get("anomaly_map_b64", ""))))
            st.image(overlay, caption=f"{served_by} · score {score:.3f}")
            figure = colorbar_figure()
            st.pyplot(figure)
            plt.close(figure)
        except Exception as exc:  # noqa: BLE001 - a bad heatmap must not lose the verdict above
            st.warning(f"The verdict is above; the heatmap could not be rendered ({type(exc).__name__}).")

    st.caption(
        "Calibrated backends render on a fixed [0, 1] ramp, so two frames are directly "
        "comparable; an uncalibrated one emits raw distances and is normalized per frame."
    )


# ---------------------------------------------------------------------------
# Tab 2 — Benchmark Comparison
# ---------------------------------------------------------------------------

#: Column order for the results table. Keys are what ``BenchmarkResult.as_dict``
#: emits; labels are what the interviewer reads.
BENCHMARK_COLUMNS = (
    ("image_auroc", "Img-AUROC"),
    ("pixel_auroc", "Px-AUROC"),
    ("au_pro", "AU-PRO"),
    ("best_f1", "Best-F1"),
)


def benchmark_table(results: dict[str, dict]) -> pd.DataFrame:
    """``{model: metrics}`` -> the four headline metrics, best AU-PRO first.

    Sorted by AU-PRO because that is the metric this project argues is the honest
    one for segmentation quality (``docs/evaluation.md``): image-AUROC saturates
    near 1.0 on MVTec and stops discriminating between models, while AU-PRO keeps
    small defects from being drowned out by large ones.

    Missing metrics render as ``n/a`` rather than 0.000 — a model with no pixel
    ground truth has no pixel-AUROC, and a zero there is a claim that it scored
    badly rather than that it was not measured.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for key, label in BENCHMARK_COLUMNS:
            value = metrics.get(key)
            row[label] = f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"
        seconds = metrics.get("seconds_per_image")
        row["s/img"] = f"{float(seconds):.3f}" if isinstance(seconds, (int, float)) else "n/a"
        row["_sort"] = float(metrics.get("au_pro") or 0.0)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    return frame.reset_index(drop=True)


def au_pro_figure(results: dict[str, dict]) -> plt.Figure:
    """Bar chart of AU-PRO per model, worst to best left to right."""
    pairs = sorted(
        ((name, float(metrics.get("au_pro") or 0.0)) for name, metrics in results.items()),
        key=lambda item: item[1],
    )
    names = [name for name, _ in pairs]
    values = [value for _, value in pairs]

    figure, axes = plt.subplots(figsize=(7, 0.9 + 0.55 * len(names)))
    bars = axes.barh(names, values, color="#2f6f9f")
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("AU-PRO (higher is better)")
    axes.set_title("Per-region overlap, by model")
    for bar, value in zip(bars, values):
        axes.text(
            min(value + 0.015, 0.93),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def render_benchmark(base_url: str, operator_key_default: str, category_default: str) -> None:
    """Run ``POST /benchmark`` and render the comparison it returns."""
    st.subheader("Benchmark Comparison")
    st.caption(f"`POST {base_url}/benchmark` — every selected model over the full test split. **Operator key.**")

    left, right = st.columns([1, 1])
    with left:
        category = st.selectbox(
            "Category",
            MVTEC_CATEGORIES,
            index=MVTEC_CATEGORIES.index(category_default) if category_default in MVTEC_CATEGORIES else 0,
            key="benchmark_category",
        )
        operator_key = st.text_input(
            "Operator API key",
            value=operator_key_default,
            type="password",
            key="benchmark_operator_key",
            help="/benchmark is operator-only and every call is audited to results/audit.jsonl.",
        )
    with right:
        backends = st.multiselect(
            "Models to compare",
            MODEL_BACKENDS,
            default=["patchcore", "onnx_efficientad"],
            key="benchmark_backends",
            help="Each runs over the whole split. WinCLIP is the slow one — minutes, not seconds.",
        )

    st.warning(
        "This scores every selected model over every test image — roughly a minute per trained "
        "backend on `bottle`'s 83 images, and several for WinCLIP. It holds a worker for its whole "
        "duration, which is why it is operator-gated and audited.",
        icon="⏱️",
    )

    if st.button("Run Benchmark (operator key required)", type="primary", disabled=not backends):
        if not operator_key:
            st.error("An operator key is required. `/benchmark` fails closed without one.")
        else:
            started = time.perf_counter()
            with st.spinner(f"Benchmarking {', '.join(backends)} on {category}… minutes, not seconds."):
                try:
                    response = requests.post(
                        f"{base_url}/benchmark",
                        json={"category": category, "model_backends": list(backends)},
                        headers=_headers(operator_key),
                        timeout=BENCHMARK_TIMEOUT,
                    )
                    if not response.ok:
                        raise ApiError(_describe_http_error(response))
                    # Kept in session state rather than returned into the local
                    # flow: Streamlit re-runs this function on every interaction,
                    # and a result that lived in a local would vanish the moment
                    # anybody touched a widget on another tab.
                    st.session_state["benchmark_results"] = response.json().get("results", {})
                    st.session_state["benchmark_meta"] = {
                        "category": category,
                        "elapsed": time.perf_counter() - started,
                    }
                except requests.RequestException as exc:
                    st.error(f"Could not reach {base_url}/benchmark: {type(exc).__name__}")
                except ApiError as exc:
                    st.error(str(exc))

    results = st.session_state.get("benchmark_results")
    if not results:
        st.info("No benchmark run yet in this session. Results also land in `results/benchmark_<category>_<ts>.json`.")
        return

    meta = st.session_state.get("benchmark_meta", {})
    st.success(
        f"{len(results)} model(s) on **{meta.get('category', category)}** in "
        f"{meta.get('elapsed', 0.0):.1f}s wall clock."
    )
    st.table(benchmark_table(results))

    figure = au_pro_figure(results)
    st.pyplot(figure)
    plt.close(figure)

    st.caption(
        "AU-PRO weights every defect region equally, so a model is not rewarded for finding only "
        "the large ones. See `docs/evaluation.md` for why it is the metric to argue over on MVTec."
    )


# ---------------------------------------------------------------------------
# Tab 3 — System Health
# ---------------------------------------------------------------------------


def render_health_panel(base_url: str, operator_key: str) -> None:
    """One pass of the live panel: metrics, drift, and the cache's connection.

    Every call in here is one of the cheap endpoints — no model is loaded, no
    frame is scored — which is what makes a 10 s refresh reasonable. Each of the
    three sections degrades on its own: a missing operator key costs the drift
    table and nothing else.
    """
    st.caption(f"Refreshed {time.strftime('%H:%M:%S')} · every {REFRESH_SECONDS}s from `{base_url}`")

    # -- Prometheus counters ------------------------------------------------
    try:
        exposition = _get_text(base_url, "/metrics")
    except ApiError as exc:
        st.error(f"`GET /metrics` — {exc}")
        exposition = None

    if exposition is not None:
        processed = sum_samples(exposition, "images_processed_total")
        rejected = sum_samples(exposition, "guard_rejections_total")
        p50 = histogram_p50(exposition)

        columns = st.columns(3)
        columns[0].metric("Images processed", f"{processed:,.0f}", help="images_processed_total, summed over model/category/result.")
        columns[1].metric("Guard rejections", f"{rejected:,.0f}", help="guard_rejections_total, summed over reason.")
        columns[2].metric(
            "p50 inference",
            f"{p50 * 1000:.0f} ms" if p50 else "—",
            help="Interpolated from inference_latency_seconds buckets; coarse by construction.",
        )

        by_result = {
            labels.get("result", "?"): value
            for labels, value in parse_samples(exposition, "images_processed_total")
        }
        if by_result:
            st.caption(" · ".join(f"**{name}**: {value:,.0f}" for name, value in sorted(by_result.items())))

    st.divider()

    # -- Score distribution / drift ----------------------------------------
    st.markdown("**Score distribution** — `GET /drift`")
    if not operator_key:
        st.info("Set an operator key in the sidebar to read the drift monitors (`/drift` is operator-only).")
    else:
        try:
            monitors = _get(base_url, "/drift", operator_key)
        except ApiError as exc:
            st.error(f"`GET /drift` — {exc}")
            monitors = None

        if monitors is not None:
            if not monitors:
                st.info("No monitors yet — a monitor appears once a model has scored its first frame.")
            else:
                rows = []
                for monitor in monitors:
                    summary = monitor.get("summary", {}) or {}
                    p_value = monitor.get("p_value")
                    rows.append(
                        {
                            "Model": monitor.get("model_name", "?"),
                            "Category": monitor.get("category", "?"),
                            "n": summary.get("count", 0),
                            "mean": _fmt(summary.get("mean")),
                            "p10": _fmt(summary.get("p10")),
                            "p50": _fmt(summary.get("p50")),
                            "p90": _fmt(summary.get("p90")),
                            "KS p": "—" if p_value is None else f"{float(p_value):.3f}",
                            "Drifted": "⚠️ yes" if monitor.get("drifted") else "no",
                            "Ref": monitor.get("reference_size", 0),
                        }
                    )
                st.table(pd.DataFrame(rows))
                if any(monitor.get("p_value") is None for monitor in monitors):
                    st.caption(
                        "`KS p = —` is a third state, not a pass: no reference distribution has been "
                        "set (`POST /calibrate`), or the window is still filling."
                    )

    st.divider()

    # -- Liveness and the checkpoint cache ---------------------------------
    st.markdown("**Service** — `GET /health`")
    try:
        health = _get(base_url, "/health")
    except ApiError as exc:
        st.error(f"`GET /health` — {exc}")
        return

    cache = health.get("cache", {}) or {}
    connected = bool(cache.get("connected"))
    columns = st.columns(3)
    columns[0].metric("API", health.get("status", "?").upper())
    columns[1].metric(
        "Redis (checkpoint cache)",
        "connected" if connected else "fallback",
        help="Optional by design: without it, model loads still work but the warm set does not survive a restart.",
    )
    columns[2].metric("Models resident", len(health.get("models_loaded", [])))

    if connected:
        st.success(f"Checkpoint cache on Redis, {cache.get('ttl_seconds', '?')}s TTL per record.")
    else:
        st.warning(f"Checkpoint cache on its in-process dict — {cache.get('detail', 'not connected')}.")

    loaded = health.get("models_loaded", [])
    st.caption("Resident: " + (", ".join(f"`{key}`" for key in loaded) if loaded else "_nothing yet — loading is lazy_"))


def _fmt(value: Any, digits: int = 3) -> str:
    """Numbers to fixed decimals, None to an em dash. Used all over tab 3's tables."""
    return "—" if value is None else f"{float(value):.{digits}f}"


def _fragment(run_every: int):
    """``st.fragment`` when the installed Streamlit has it, a no-op decorator otherwise.

    Why a fragment and not the obvious ``time.sleep(10); st.rerun()``: a full
    rerun re-executes the whole script, which resets the tab selection back to
    the first tab. In an interview that means the health panel yanks you away
    from itself every ten seconds. A fragment reruns *only its own container*, so
    the page stays where it is. The sleep-and-rerun loop is kept as a fallback
    for a Streamlit too old to have fragments (< 1.37), where the choice is
    between a jumping tab and no refresh at all.
    """
    fragment = getattr(st, "fragment", None)
    if fragment is None:  # pragma: no cover - only on an unpinned, older Streamlit
        return lambda function: function
    return fragment(run_every=run_every)


@_fragment(run_every=REFRESH_SECONDS)
def render_system_health() -> None:
    """The auto-refreshing panel. Reads its configuration from session state.

    Arguments would have to be bound at decoration time, which happens once at
    import; session state is read on every fragment rerun, so changing the API
    URL in the sidebar is picked up by the next tick.
    """
    st.subheader("System Health")
    base_url = st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")
    render_health_panel(base_url, st.session_state.get("operator_key", ""))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _default_key(single_var: str, list_var: str) -> str:
    """Prefill a key box from the environment, so a demo needs no typing.

    ``DASHBOARD_VIEWER_KEY`` wins; otherwise the first entry of the API's own
    ``VIEWER_API_KEYS``, which compose hands this container through the same
    ``.env`` the API reads. That is a development convenience and nothing more —
    it works because both processes are configured from one file on one machine,
    and a real deployment gives the dashboard its own key rather than the
    service's list.
    """
    direct = os.getenv(single_var, "").strip()
    if direct:
        return direct
    return next((part.strip() for part in os.getenv(list_var, "").split(",") if part.strip()), "")


def main() -> None:
    """Draw the page: sidebar, then the three tabs."""
    st.set_page_config(page_title="Defect Detection — Live Inspection", page_icon="🔍", layout="wide")
    st.title("🔍 Zero-Shot Industrial Defect Detection")
    st.caption("A client for the inspection API. Every number on this page came over HTTP from the running service.")

    default_category = os.getenv("DEFAULT_CATEGORY", "bottle").strip() or "bottle"

    with st.sidebar:
        st.header("Configuration")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, key="api_url").rstrip("/")
        category = st.selectbox(
            "Category",
            MVTEC_CATEGORIES,
            index=MVTEC_CATEGORIES.index(default_category) if default_category in MVTEC_CATEGORIES else 0,
            key="category",
        )
        backend = st.selectbox("Model backend", MODEL_BACKENDS, index=0, key="backend")

        st.divider()
        st.subheader("Authentication")
        viewer_key = st.text_input(
            "Viewer API key",
            value=_default_key("DASHBOARD_VIEWER_KEY", "VIEWER_API_KEYS"),
            type="password",
            key="viewer_key",
            help="Opens /predict. The role an inspection line runs as.",
        )
        st.text_input(
            "Operator API key",
            value=_default_key("DASHBOARD_OPERATOR_KEY", "OPERATOR_API_KEYS"),
            type="password",
            key="operator_key",
            help="Opens /drift and /benchmark. Prefilled from .env for local demos.",
        )

        st.divider()
        try:
            health = _get(api_url, "/health")
            st.success(f"API {health.get('status', '?')} · {len(health.get('models_loaded', []))} model(s) resident")
        except ApiError as exc:
            st.error(f"API unreachable\n\n{exc}")
        st.caption(f"[OpenAPI docs]({api_url}/docs) · [Prometheus](http://localhost:9090) · [Grafana](http://localhost:3000)")

    inspection_tab, benchmark_tab, health_tab = st.tabs(["Live Inspection", "Benchmark Comparison", "System Health"])

    with inspection_tab:
        render_live_inspection(api_url, viewer_key, category, backend)

    with benchmark_tab:
        render_benchmark(api_url, st.session_state.get("operator_key", ""), category)

    with health_tab:
        render_system_health()
        if not hasattr(st, "fragment"):  # pragma: no cover - see _fragment
            time.sleep(REFRESH_SECONDS)
            st.rerun()


if __name__ == "__main__":
    main()
