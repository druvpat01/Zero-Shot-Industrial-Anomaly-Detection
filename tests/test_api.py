"""Tests for the FastAPI serving layer.

What is actually being tested here
----------------------------------
Not "does PatchCore work" — ``tests/test_patchcore.py`` owns that. This file
tests the *seam*: that a base64 payload becomes a frame, that a bad frame is
refused with a structured error rather than a traceback, that the response
carries the schema it advertises, and that the model registry hands back the
model it says it does.

So the error cases are the bulk of it, and they are the cases a production API
gets hit with: an image that will not decode, a frame that decodes but is not
worth scoring, a backend with no artifact behind it. Each has one right status
code and one right body shape, and each is asserted on both.

The synthetic corruptions mirror ``tests/test_guardrails.py``'s — a Gaussian
blur is a fouled lens, a 10x10 array is a truncated read — because the guard is
what the API is delegating to, and the API test should fail for the same reasons
the guard test does.

Dataset- and checkpoint-backed tests skip rather than fail when their artifacts
are absent, matching the model suites. Populate them with::

    python scripts/download_dataset.py --category bottle
    python scripts/train_patchcore.py --category bottle
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data.datamodule import DEFAULT_DATA_ROOT
from app.models.config import get_model_config
from app.models.onnx_runner import DEFAULT_EXPORTED_DIR, onnx_artifact_path
from app.serving.main import app
from app.serving.model_registry import ModelNotReadyError, ModelRegistry, get_registry
from app.serving.schemas import MODEL_BACKENDS

CATEGORY = "bottle"
BACKEND = "patchcore"

CATEGORY_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY
GOOD_TEST_DIR: Path = CATEGORY_DIR / "test" / "good"
DEFECT_TEST_DIR: Path = CATEGORY_DIR / "test" / "broken_large"
CHECKPOINT: Path = get_model_config().checkpoint_path(BACKEND, CATEGORY)
ONNX_PATCHCORE: Path = onnx_artifact_path(BACKEND, "fp32")

requires_dataset = pytest.mark.skipif(
    not GOOD_TEST_DIR.is_dir(),
    reason=f"{CATEGORY_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(),
    reason=f"{CHECKPOINT} not found; run `python scripts/train_patchcore.py --category {CATEGORY}`",
)
requires_onnx = pytest.mark.skipif(
    not ONNX_PATCHCORE.is_file(),
    reason=f"{ONNX_PATCHCORE} not found; run `python scripts/export_onnx.py`",
)

#: Every field ``InferenceResponse`` promises. Asserted as an exact set, not a
#: subset: an accidentally added field is a wire-contract change too.
EXPECTED_RESPONSE_KEYS = {
    "anomaly_score",
    "is_defective",
    "model_name",
    "anomaly_map_b64",
    "latency_ms",
    "guard_passed",
    "guard_reason",
}


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _png_b64(image: np.ndarray) -> str:
    """PNG-encode a BGR array and base64 it, exactly as an API client would."""
    ok, buffer = cv2.imencode(".png", image)
    assert ok, "failed to encode the test image"
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _predict_body(image_b64: str, *, backend: str = BACKEND, category: str = CATEGORY) -> dict[str, str]:
    return {"category": category, "model_backend": backend, "image_b64": image_b64}


def _blurred_frame() -> np.ndarray:
    """Sharp noise put through a heavy Gaussian blur: a fouled or defocused lens.

    Noise is used rather than a real photo so the case is self-contained, and
    because it makes the failure unambiguous — random pixels have the highest
    Laplacian variance available, so a blur that drops them below the threshold
    leaves no doubt about which check fired.
    """
    noise = np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 20)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client over the real app, with the process-wide registry in place."""
    return TestClient(app)


@pytest.fixture
def override_registry():
    """Swap the registry FastAPI injects, and put the real one back afterwards.

    The endpoints take their registry through ``Depends(get_registry)`` precisely
    so a test can do this: point the model lookup at a directory it controls
    without monkeypatching a module global or evicting the real registry's cache
    (which other tests in this module are relying on staying warm).
    """
    def _install(registry: ModelRegistry) -> ModelRegistry:
        app.dependency_overrides[get_registry] = lambda: registry
        return registry

    yield _install
    app.dependency_overrides.pop(get_registry, None)


@pytest.fixture(scope="module")
def clean_bottle_b64() -> str:
    """A real defect-free bottle from the test split, as an API payload."""
    if not GOOD_TEST_DIR.is_dir():
        pytest.skip(f"{GOOD_TEST_DIR} not found")
    path = sorted(GOOD_TEST_DIR.glob("*.png"))[0]
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture(scope="module")
def defective_bottle_b64() -> str:
    """A real defective bottle (``broken_large``) from the test split."""
    if not DEFECT_TEST_DIR.is_dir():
        pytest.skip(f"{DEFECT_TEST_DIR} not found")
    path = sorted(DEFECT_TEST_DIR.glob("*.png"))[0]
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    """The liveness probe answers, with the schema it advertises."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["models_loaded"], list)


def test_health_loads_no_models(client: TestClient, override_registry, tmp_path: Path) -> None:
    """Health must not trigger a model load — the whole reason loading is lazy.

    A fresh registry is injected so the assertion is about what ``/health`` did,
    not about whatever other tests have already warmed in the shared one.
    """
    registry = override_registry(ModelRegistry(warmup=False))

    body = client.get("/health").json()

    assert body["models_loaded"] == []
    assert registry.loaded_keys() == []


@pytest.mark.asyncio
async def test_health_over_asgi_transport() -> None:
    """The app also answers over a plain async httpx client, not just TestClient.

    TestClient drives the app through a synchronous portal; this exercises the
    ASGI interface the way uvicorn will, which is what ``scripts/run_api_demo.py``
    talks to.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


def test_models_returns_a_list(client: TestClient) -> None:
    """``GET /models`` returns one entry per backend, and loads nothing to do it."""
    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert [row["backend"] for row in body] == list(MODEL_BACKENDS)
    for row in body:
        assert isinstance(row["available"], bool)
        assert isinstance(row["loaded"], bool)


def test_models_reports_winclip_available_without_an_artifact(client: TestClient) -> None:
    """WinCLIP is available with nothing on disk — the zero-shot claim, via HTTP."""
    rows = {row["backend"]: row for row in client.get("/models").json()}

    assert rows["winclip"]["available"] is True


def test_models_reports_unavailable_backends(client: TestClient, override_registry, tmp_path: Path) -> None:
    """With no artifacts anywhere, *only* the zero-shot backend is available.

    The ONNX rows are the ones worth asserting. They need no checkpoint, which
    makes it tempting to treat them like WinCLIP; they need their exported graph,
    which makes that wrong. Advertising ``onnx_patchcore`` as available with no
    ``.onnx`` file on disk would send every caller that trusts ``/models``
    straight into a 503.
    """
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    override_registry(ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False))

    rows = {row["backend"]: row for row in client.get("/models").json()}

    assert rows["patchcore"]["available"] is False
    assert "no checkpoint" in rows["patchcore"]["detail"]
    assert rows["onnx_patchcore"]["available"] is False
    assert "no exported graph" in rows["onnx_patchcore"]["detail"]
    assert rows["onnx_efficientad"]["available"] is False
    assert rows["winclip"]["available"] is True


def test_models_agrees_with_what_get_model_does(tmp_path: Path) -> None:
    """Whatever ``/models`` claims is available must actually load.

    Pins the two halves together: ``describe`` answers from the filesystem and
    ``get_model`` opens files, and it is entirely possible to make them disagree
    (this test exists because an earlier version did). An availability report
    that lies is worse than no report.
    """
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False)

    for row in registry.describe(CATEGORY):
        if not row["available"]:
            with pytest.raises(ModelNotReadyError):
                registry.get_model(str(row["backend"]), CATEGORY)


# ---------------------------------------------------------------------------
# /predict — the success path
# ---------------------------------------------------------------------------


@requires_dataset
@requires_checkpoint
def test_predict_returns_the_documented_schema(client: TestClient, clean_bottle_b64: str) -> None:
    """A real bottle scored by PatchCore comes back as a well-formed InferenceResponse."""
    response = client.post("/predict", json=_predict_body(clean_bottle_b64))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == EXPECTED_RESPONSE_KEYS

    assert isinstance(body["is_defective"], bool)
    assert isinstance(body["anomaly_score"], float)
    assert np.isfinite(body["anomaly_score"])
    assert body["model_name"] == BACKEND
    assert body["guard_passed"] is True
    assert body["guard_reason"] is None
    assert body["latency_ms"] > 0.0


@requires_dataset
@requires_checkpoint
def test_predict_heatmap_matches_the_submitted_resolution(client: TestClient, clean_bottle_b64: str) -> None:
    """The returned heatmap is a PNG the size of the frame that was sent.

    This is the ``ModelOutput`` contract surviving the trip through the API: a
    caller overlays the heatmap on the image they submitted, so a map at the
    model's internal working resolution would be useless to them.
    """
    submitted = cv2.imdecode(np.frombuffer(base64.b64decode(clean_bottle_b64), np.uint8), cv2.IMREAD_COLOR)

    body = client.post("/predict", json=_predict_body(clean_bottle_b64)).json()
    heatmap = cv2.imdecode(np.frombuffer(base64.b64decode(body["anomaly_map_b64"]), np.uint8), cv2.IMREAD_COLOR)

    assert heatmap is not None, "anomaly_map_b64 did not decode as an image"
    assert heatmap.shape == submitted.shape


@requires_dataset
@requires_checkpoint
def test_predict_scores_a_defect_above_a_clean_part(
    client: TestClient,
    clean_bottle_b64: str,
    defective_bottle_b64: str,
) -> None:
    """End to end, the API preserves the thing the whole system exists to do.

    Not a model-quality test — one image each proves nothing about AUROC, which
    ``scripts/run_benchmark.py`` measures properly. It is a wiring test: a
    channel-order slip or a stale checkpoint in the serving path would show up
    here as scores that no longer separate.
    """
    clean = client.post("/predict", json=_predict_body(clean_bottle_b64)).json()
    defective = client.post("/predict", json=_predict_body(defective_bottle_b64)).json()

    assert defective["anomaly_score"] > clean["anomaly_score"]


# ---------------------------------------------------------------------------
# /predict — rejections
# ---------------------------------------------------------------------------


def test_predict_rejects_a_blurred_frame(client: TestClient) -> None:
    """A defocused frame is refused with the reason, not scored."""
    response = client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "guard_failed"
    assert "blurry" in body["reason"]


def test_predict_rejects_a_tiny_frame(client: TestClient) -> None:
    """A 10x10 read is refused as too_small — the guard's priority order, over HTTP."""
    tiny = np.random.default_rng(0).integers(0, 256, size=(10, 10, 3), dtype=np.uint8)

    body = client.post("/predict", json=_predict_body(_png_b64(tiny))).json()

    assert body["detail"] == "guard_failed"
    assert body["reason"] == "too_small"


def test_predict_rejects_garbage_base64(client: TestClient) -> None:
    """A payload outside the base64 alphabet is ``invalid_image``, not a traceback."""
    response = client.post("/predict", json=_predict_body("not-a-valid-base64-image!!!"))

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_image"


def test_predict_rejects_valid_base64_that_is_not_an_image(client: TestClient) -> None:
    """Decodable base64 carrying non-image bytes fails the same way, with a distinct reason."""
    response = client.post("/predict", json=_predict_body(base64.b64encode(b"hello world").decode("ascii")))

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "invalid_image"
    assert body["reason"] == "undecodable"


def test_predict_rejects_an_unknown_backend(client: TestClient) -> None:
    """An unknown backend is caught by the schema, before any model is touched."""
    response = client.post("/predict", json=_predict_body("aGk=", backend="not_a_model"))

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_request"


def test_validation_errors_do_not_echo_the_payload(client: TestClient) -> None:
    """A 422 must not reflect ``image_b64`` back — it can be megabytes.

    pydantic's default error payload includes the offending input. The custom
    handler strips it; this pins that down, because the leak would be invisible
    on the small payloads a test would otherwise use.
    """
    payload = _predict_body("A" * 5000, backend="not_a_model")

    text = client.post("/predict", json=payload).text

    assert "A" * 100 not in text


def test_guard_runs_before_the_model_is_loaded(client: TestClient, override_registry, tmp_path: Path) -> None:
    """A bad frame is rejected as such even when the backend could not have served it.

    The ordering that makes this pass is the point: guard first, registry second.
    Reversed, this request would return 503 ``model_not_ready`` and the caller
    would never learn their camera is out of focus — and the server would have
    paid for a model load to tell them nothing.
    """
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    override_registry(ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False))

    response = client.post("/predict", json=_predict_body(_png_b64(_blurred_frame())))

    assert response.status_code == 422
    assert response.json()["reason"] == "blurry"


@requires_dataset
def test_predict_returns_503_when_no_artifact_can_serve_the_backend(
    client: TestClient,
    override_registry,
    tmp_path: Path,
    clean_bottle_b64: str,
) -> None:
    """A good frame against a backend with nothing on disk is 503, not 500."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    override_registry(ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False))

    response = client.post("/predict", json=_predict_body(clean_bottle_b64))

    assert response.status_code == 503
    body = response.json()
    assert body["detail"] == "model_not_ready"
    assert body["backend"] == BACKEND


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


@requires_checkpoint
def test_registry_caches_by_backend_and_category() -> None:
    """The second request for a pair gets the identical object, not a second load."""
    registry = ModelRegistry(warmup=False)

    first = registry.get_model(BACKEND, CATEGORY)
    second = registry.get_model(BACKEND, CATEGORY)

    assert first is second
    assert registry.loaded_keys() == [f"{BACKEND}:{CATEGORY}"]


def test_registry_rejects_an_unknown_backend() -> None:
    """An unknown name is a programming error, and says so."""
    with pytest.raises(ValueError, match="Unknown backend"):
        ModelRegistry(warmup=False).get_model("nope", CATEGORY)


def test_registry_raises_model_not_ready_without_artifacts(tmp_path: Path) -> None:
    """No checkpoint and no export means not-ready, with the missing paths named."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False)

    with pytest.raises(ModelNotReadyError) as excinfo:
        registry.get_model(BACKEND, CATEGORY)

    assert excinfo.value.backend == BACKEND
    assert "no checkpoint" in excinfo.value.detail


def test_registry_serves_winclip_with_no_artifact_at_all(tmp_path: Path) -> None:
    """WinCLIP is constructible from an empty directory — it needs a noun, not a file."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False)

    model = registry.get_model("winclip", CATEGORY)

    assert model.model_name == "winclip"
    assert model.is_trained  # zero-shot: ready before it has been given anything


@requires_onnx
def test_registry_falls_back_to_onnx_and_says_so(tmp_path: Path) -> None:
    """With the checkpoint missing but the export present, ONNX serves — under its own name.

    The renaming is the part worth pinning: a response claiming ``patchcore``
    while an ONNX graph did the work would make a latency or accuracy shift
    impossible to attribute.
    """
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, warmup=False)  # real exported dir, empty checkpoint dir

    model = registry.get_model(BACKEND, CATEGORY)

    assert model.model_name == f"onnx_{BACKEND}"


@requires_dataset
@requires_onnx
def test_predict_reports_the_onnx_fallback_in_model_name(
    client: TestClient,
    override_registry,
    tmp_path: Path,
    clean_bottle_b64: str,
) -> None:
    """The fallback is visible to the caller, through the API, in ``model_name``."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    override_registry(ModelRegistry(config, warmup=False))

    body = client.post("/predict", json=_predict_body(clean_bottle_b64)).json()

    assert body["model_name"] == f"onnx_{BACKEND}"


def test_registry_describe_touches_no_models() -> None:
    """``describe`` answers from the filesystem, so ``/models`` stays cheap to poll."""
    registry = ModelRegistry(exported_dir=DEFAULT_EXPORTED_DIR, warmup=False)

    rows = registry.describe(CATEGORY)

    assert [row["backend"] for row in rows] == list(MODEL_BACKENDS)
    assert registry.loaded_keys() == []


# ---------------------------------------------------------------------------
# /benchmark
# ---------------------------------------------------------------------------


def test_benchmark_rejects_an_unknown_backend(client: TestClient) -> None:
    """The backend list is validated at the schema, before a test split is read."""
    response = client.post("/benchmark", json={"category": CATEGORY, "model_backends": ["patchcore", "nope"]})

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_request"


def test_benchmark_rejects_an_empty_backend_list(client: TestClient) -> None:
    """An empty comparison is a caller error, not an empty result set."""
    response = client.post("/benchmark", json={"category": CATEGORY, "model_backends": []})

    assert response.status_code == 422
