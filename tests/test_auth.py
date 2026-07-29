"""Tests for API-key access control and the benchmark audit trail.

What is being tested here
-------------------------
Two things, and neither of them is anomaly detection. First, that each endpoint
admits exactly the roles it is supposed to and refuses the rest with the *right*
status — 401 for "I do not know you", 403 for "I know you and the answer is no",
and 200 for the four combinations that should work. Second, that a successful
``/benchmark`` leaves one honest line in ``results/audit.jsonl``.

Which is why there are no real models in this file. The subject is the gate, so
the thing behind the gate is a stub that returns a fixed score instantly, and the
test set is four synthetic images written to ``tmp_path``. That keeps the whole
module hermetic and sub-second — it needs no checkpoint, no MVTec download and no
CLIP weights — and, more usefully, it means a failure here is unambiguously an
*auth* failure. ``tests/test_api.py`` covers the real request path.

Two isolation properties are load-bearing and set up by fixtures below:

* The audit trail is redirected into ``tmp_path`` for **every** test in this
  module, so running the suite never appends to the repository's real
  ``results/audit.jsonl``.
* Keys are injected through :func:`app.serving.auth.get_auth_config` rather than
  exported into the environment, so the tests neither read a developer's ``.env``
  nor leave anything behind in ``os.environ``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.evaluation.drift import ScoreDistributionMonitor
from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig
from app.observability import audit_log
from app.observability.audit_log import get_audit_log, record_benchmark
from app.serving.auth import (
    API_KEY_HEADER,
    AuthConfig,
    Principal,
    hash_api_key,
    require_role,
)
from app.serving.main import app
from app.serving.model_cache import CheckpointCache
from app.serving.model_registry import get_registry

CATEGORY = "widget"
BACKEND = "patchcore"

#: The two keys every test in this file authenticates with. Deliberately
#: unguessable-looking rather than "abc": several assertions below check that a
#: raw key never reaches a file, and a short key would match by accident.
VIEWER_KEY = "viewer-key-1c4f9a2e"
OPERATOR_KEY = "operator-key-7b3d05fa"

#: What the server is configured with for these tests.
AUTH_CONFIG = AuthConfig(viewer_keys=(VIEWER_KEY,), operator_keys=(OPERATOR_KEY,))

#: Small enough to keep the synthetic benchmark instant, large enough to clear
#: the input guard's 64 px ``min_resolution``.
IMAGE_EDGE = 64


# ---------------------------------------------------------------------------
# Stubs: a model that scores nothing, and the registry that hands it out
# ---------------------------------------------------------------------------


class StubModel(AnomalyModel):
    """An :class:`AnomalyModel` that answers instantly and means nothing.

    Its score is the frame's mean intensity, which is enough for the benchmark's
    metrics to compute over a mixed test set without importing a single weight.
    """

    model_name = "stub"

    def train(self, datamodule: object) -> None:  # pragma: no cover - never called
        raise NotImplementedError

    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        frame = self._to_rgb_array(image, color_order=color_order)
        height, width = frame.shape[:2]
        score = float(np.asarray(frame, dtype=np.float64).mean())
        # Normalised into [0, 1] whatever the input dtype, so ModelOutput's
        # finite-score contract holds for uint8 frames and float ones alike.
        score = score / 255.0 if score > 1.0 else score
        return ModelOutput(
            anomaly_score=score,
            anomaly_map=np.full((height, width), score, dtype=np.float32),
            is_defective=score >= self.config.anomaly_threshold,
            model_name=self.model_name,
        )

    def save(self, path: str | Path) -> None:  # pragma: no cover - never called
        raise NotImplementedError

    def load(self, path: str | Path) -> None:  # pragma: no cover - never called
        raise NotImplementedError

    @property
    def is_trained(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True


class StubRegistry:
    """The subset of :class:`~app.serving.model_registry.ModelRegistry` the routes use.

    Injected through ``Depends(get_registry)``, which is what that dependency
    exists for. Nothing here touches the filesystem, so a request reaches the
    handler at the speed of the auth check — the only thing this module is
    trying to measure.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = StubModel(config=config)
        self._monitors: dict[tuple[str, str], ScoreDistributionMonitor] = {}
        # ``/health`` reports the checkpoint cache's connection state. The empty
        # URL is the seam documented in app.serving.model_cache: fallback mode,
        # chosen without reading the environment, so this stub stays as offline
        # as the rest of it and cannot reach a developer's real Redis.
        self.cache = CheckpointCache(url="")

    def get_model(self, backend: str, category: str) -> AnomalyModel:
        return self._model

    def loaded_keys(self) -> list[str]:
        return [f"{self._model.model_name}:{self.config.category}"]

    def describe(self, category: str) -> list[dict[str, object]]:
        return [{"backend": BACKEND, "available": True, "loaded": True, "artifact": None, "detail": None}]

    # ``/predict`` records every score into a drift monitor and ``/drift`` reads
    # them back, so the stub carries real monitors rather than no-ops: a stub
    # that silently swallowed scores would let a broken wiring pass this module.
    # ``tests/test_drift.py`` owns the behaviour; these two keep the routes
    # callable at the speed the auth tests need.

    def monitor_for(self, model_name: str, category: str) -> ScoreDistributionMonitor:
        return self._monitors.setdefault((model_name, category), ScoreDistributionMonitor())

    def monitors(self) -> dict[tuple[str, str], ScoreDistributionMonitor]:
        return dict(self._monitors)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array), f"failed to write {path}"


def _synthetic_dataset(root: Path, category: str = CATEGORY, *, per_split: int = 4) -> Path:
    """Write a minimal MVTec-shaped category: train/good, test/good, test/crack + masks.

    Sharp uniform noise, so every frame clears the input-quality guard, and a
    solid mask block so the pixel metrics have something to score against. Four
    images per split is the smallest set that leaves the test split with both
    labels after anomalib's val split.
    """
    rng = np.random.default_rng(0)

    def noise() -> np.ndarray:
        return rng.integers(0, 256, size=(IMAGE_EDGE, IMAGE_EDGE, 3), dtype=np.uint8)

    for index in range(per_split):
        _write_image(root / category / "train" / "good" / f"{index:03d}.png", noise())
        _write_image(root / category / "test" / "good" / f"{index:03d}.png", noise())
        _write_image(root / category / "test" / "crack" / f"{index:03d}.png", noise())

        mask = np.zeros((IMAGE_EDGE, IMAGE_EDGE), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        _write_image(root / category / "ground_truth" / "crack" / f"{index:03d}_mask.png", mask)

    return root


@pytest.fixture(autouse=True)
def audit_trail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the audit trail at ``tmp_path`` for every test in this module.

    Autouse and unconditional: a test that appended to the repository's real
    ``results/audit.jsonl`` would both pollute it and make "exactly one entry"
    depend on how many times the suite had been run before.
    """
    destination = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit_log, "AUDIT_LOG_PATH", destination)
    return destination


@pytest.fixture
def stub_registry(tmp_path: Path) -> StubRegistry:
    """A registry over the stub model, pointed at a synthetic dataset in ``tmp_path``."""
    data_root = _synthetic_dataset(tmp_path / "data")
    config = ModelConfig(
        category=CATEGORY,
        data_root=data_root,
        image_size=IMAGE_EDGE,
        batch_size=4,
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    return StubRegistry(config)


@pytest.fixture
def client(stub_registry: StubRegistry):
    """A ``TestClient`` with the stub registry and the test key set installed.

    No default ``X-API-Key`` header: this module's whole subject is which key was
    sent, so every request states its own.
    """
    from app.serving.auth import get_auth_config

    app.dependency_overrides[get_auth_config] = lambda: AUTH_CONFIG
    app.dependency_overrides[get_registry] = lambda: stub_registry
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def frame_b64() -> str:
    """A sharp, well-exposed synthetic frame, base64-encoded as a client would send it."""
    noise = np.random.default_rng(1).integers(0, 256, size=(IMAGE_EDGE, IMAGE_EDGE, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", noise)
    assert ok, "failed to encode the test frame"
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _predict_body(image_b64: str) -> dict[str, str]:
    return {"category": CATEGORY, "model_backend": BACKEND, "image_b64": image_b64}


def _benchmark_body() -> dict[str, object]:
    return {"category": CATEGORY, "model_backends": [BACKEND]}


def _key(api_key: str) -> dict[str, str]:
    return {API_KEY_HEADER: api_key}


# ---------------------------------------------------------------------------
# /health — the one open endpoint
# ---------------------------------------------------------------------------


def test_health_needs_no_key(client: TestClient) -> None:
    """The liveness probe answers unauthenticated, because a kubelet cannot hold a key.

    This is the load-bearing exception in the whole scheme. A health check that
    can fail on a credential problem will eventually restart a healthy pod during
    a key rotation, which is a worse outage than the one authentication was
    protecting against.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict — viewer and above
# ---------------------------------------------------------------------------


def test_predict_without_a_key_is_401(client: TestClient, frame_b64: str) -> None:
    """No credential is 401, not 403: the server does not know who this is."""
    response = client.post("/predict", json=_predict_body(frame_b64))

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_api_key"


def test_predict_with_a_viewer_key_is_200(client: TestClient, frame_b64: str) -> None:
    """The inspection line's own key opens the endpoint the line actually uses."""
    response = client.post("/predict", json=_predict_body(frame_b64), headers=_key(VIEWER_KEY))

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == StubModel.model_name
    assert body["guard_passed"] is True


def test_predict_with_an_operator_key_is_200(client: TestClient, frame_b64: str) -> None:
    """Roles nest: an operator can do everything a viewer can, without being listed twice.

    The inverse of ``test_benchmark_with_a_viewer_key_is_403``. Together they pin
    the hierarchy down in both directions, which is the part of a role scheme
    that silently breaks when a route is added.
    """
    response = client.post("/predict", json=_predict_body(frame_b64), headers=_key(OPERATOR_KEY))

    assert response.status_code == 200


def test_predict_with_an_unknown_key_is_401(client: TestClient, frame_b64: str) -> None:
    """A key the server has never seen is unauthenticated, not merely unauthorised."""
    response = client.post("/predict", json=_predict_body(frame_b64), headers=_key("not-a-real-key"))

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_api_key"


# ---------------------------------------------------------------------------
# /benchmark and /models — operator only
# ---------------------------------------------------------------------------


def test_benchmark_with_a_viewer_key_is_403(client: TestClient) -> None:
    """A valid key with the wrong role is 403 — and the distinction matters.

    Answering 401 here would send a viewer off hunting for a typo in a key that
    is perfectly correct. 403 tells them the truth: the credential is fine, the
    role is not, and they need a different one.
    """
    response = client.post("/benchmark", json=_benchmark_body(), headers=_key(VIEWER_KEY))

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient_role"


def test_benchmark_with_an_operator_key_is_200(client: TestClient) -> None:
    """The expensive endpoint runs end to end for an operator."""
    response = client.post("/benchmark", json=_benchmark_body(), headers=_key(OPERATOR_KEY))

    assert response.status_code == 200
    results = response.json()["results"]
    assert set(results) == {StubModel.model_name}
    assert results[StubModel.model_name]["num_images"] > 0


def test_benchmark_denied_to_a_viewer_runs_nothing(client: TestClient, audit_trail: Path) -> None:
    """A 403 costs the server nothing — the guard runs before the handler does.

    The point of gating ``/benchmark`` is resource exhaustion, so "denied" has to
    mean "no test split was read", not "the work happened and the answer was
    withheld". An empty audit trail is the observable proof of that: the handler
    writes an entry on every path it reaches, including failure, so no entry
    means it was never entered.
    """
    client.post("/benchmark", json=_benchmark_body(), headers=_key(VIEWER_KEY))

    assert not audit_trail.exists()
    assert get_audit_log() == []


def test_models_requires_an_operator_key(client: TestClient) -> None:
    """``/models`` is cheap but discloses artifact paths, so it is operator-gated."""
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers=_key(VIEWER_KEY)).status_code == 403
    assert client.get("/models", headers=_key(OPERATOR_KEY)).status_code == 200


# ---------------------------------------------------------------------------
# The audit trail
# ---------------------------------------------------------------------------


def test_successful_benchmark_writes_exactly_one_audit_entry(
    client: TestClient,
    audit_trail: Path,
) -> None:
    """One call, one line, with every field an incident review would need.

    The assertions are on *content*, not just on the count. An audit entry that
    exists but says nothing useful — no caller, no duration, no metrics — would
    pass a count check and fail the only purpose the file has.
    """
    response = client.post("/benchmark", json=_benchmark_body(), headers=_key(OPERATOR_KEY))
    assert response.status_code == 200

    entries = get_audit_log()
    assert len(entries) == 1

    entry = entries[0]
    assert entry["event"] == "benchmark"
    assert entry["caller"] == hash_api_key(OPERATOR_KEY)
    assert entry["role"] == "operator"
    assert entry["category"] == CATEGORY
    assert entry["models"] == [BACKEND]
    assert entry["outcome"] == "ok"
    assert entry["duration_seconds"] > 0.0
    # What the caller actually received, which is the part that makes the trail
    # worth keeping: "someone ran a benchmark" answers nothing on its own.
    assert set(entry["metrics"]) == set(response.json()["results"])
    assert entry["metrics"][StubModel.model_name]["num_images"] > 0

    # UTC with an explicit offset. A naive local timestamp is how an audit file
    # becomes unreadable the first time it is opened in another timezone.
    assert entry["timestamp"].endswith("+00:00")

    # And it is one JSON object per line, not a JSON document.
    lines = audit_trail.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry


def test_audit_entry_never_contains_the_raw_key(client: TestClient, audit_trail: Path) -> None:
    """The trail identifies the caller without handing out their credential.

    An audit file gets copied, shipped to a log store and read by people who
    should not thereby be able to call the API. This is the assertion that keeps
    that true.
    """
    client.post("/benchmark", json=_benchmark_body(), headers=_key(OPERATOR_KEY))

    text = audit_trail.read_text(encoding="utf-8")
    assert OPERATOR_KEY not in text
    assert "hmac-sha256:" in text


def test_failed_benchmark_is_audited_too(client: TestClient, stub_registry: StubRegistry) -> None:
    """A run that 503s still cost CPU, so it still leaves a record.

    Pointing the config at an empty data root makes ``DataModule.setup`` raise,
    which the app turns into a 503. The audit entry is what makes a caller who
    triggers minute-long failures over and over visible at all.
    """
    stub_registry.config = stub_registry.config.with_overrides(data_root=stub_registry.config.data_root / "missing")

    response = client.post("/benchmark", json=_benchmark_body(), headers=_key(OPERATOR_KEY))

    assert response.status_code == 503
    entries = get_audit_log()
    assert len(entries) == 1
    assert entries[0]["outcome"] == "failed:DatasetNotAvailableError"
    assert entries[0]["metrics"] == {}


def test_get_audit_log_returns_the_last_n_entries_oldest_first(audit_trail: Path) -> None:
    """The reader tails the file and preserves write order."""
    for index in range(5):
        record_benchmark(
            caller="hmac-sha256:0000000000000000",
            role="operator",
            category=f"cat{index}",
            models=[BACKEND],
            duration_seconds=float(index),
        )

    entries = get_audit_log(limit=3)

    assert [entry["category"] for entry in entries] == ["cat2", "cat3", "cat4"]


def test_get_audit_log_tolerates_a_truncated_line(audit_trail: Path) -> None:
    """A half-written line is skipped, not fatal.

    A crash mid-append leaves exactly this. Refusing to show the other entries
    because the last one is damaged is the opposite of what a forensic tool
    should do.
    """
    record_benchmark(caller="c", role="operator", category="ok", models=[], duration_seconds=1.0)
    with audit_trail.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2026-07-2')

    entries = get_audit_log()

    assert [entry["category"] for entry in entries] == ["ok"]


def test_get_audit_log_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    """No trail yet is an empty trail, not an error — the state of every fresh deployment."""
    assert get_audit_log(path=tmp_path / "never-written.jsonl") == []


def test_audit_trail_is_not_world_readable(audit_trail: Path) -> None:
    """The file is created 0600: it names callers and carries the metrics they got."""
    record_benchmark(caller="c", role="operator", category=CATEGORY, models=[], duration_seconds=0.1)

    assert audit_trail.stat().st_mode & 0o077 == 0


# ---------------------------------------------------------------------------
# AuthConfig and the role grants, without going through HTTP
# ---------------------------------------------------------------------------


def test_an_unconfigured_server_refuses_gated_endpoints(client: TestClient, frame_b64: str) -> None:
    """No keys configured means 503, not "everything is open".

    A missing environment variable is the most likely way this protection ever
    gets switched off by accident, and a deployment that answers 200 to everyone
    because ``OPERATOR_API_KEYS`` was misspelled is the failure this prevents.
    503 also names the right owner: the server's configuration, not the request.
    """
    from app.serving.auth import get_auth_config

    app.dependency_overrides[get_auth_config] = lambda: AuthConfig()

    response = client.post("/predict", json=_predict_body(frame_b64), headers=_key(VIEWER_KEY))

    assert response.status_code == 503
    assert response.json()["detail"] == "auth_not_configured"
    # /health stays up, so an orchestrator sees a live pod with a broken config
    # rather than a dead one it will restart forever.
    assert client.get("/health").status_code == 200


def test_from_env_parses_comma_separated_keys() -> None:
    """Blank entries and stray whitespace are dropped rather than becoming keys.

    An empty-string "key" would be matched by a request sending an empty header,
    which is the kind of bug that turns a trailing comma in a ``.env`` file into
    an authentication bypass.
    """
    config = AuthConfig.from_env(
        {"VIEWER_API_KEYS": " v1 , v2 ,", "OPERATOR_API_KEYS": "o1"},
        use_dotenv=False,
    )

    assert config.viewer_keys == ("v1", "v2")
    assert config.operator_keys == ("o1",)
    assert config.is_configured


def test_from_env_with_nothing_set_is_unconfigured() -> None:
    assert not AuthConfig.from_env({}, use_dotenv=False).is_configured


def test_a_key_in_both_lists_gets_the_higher_privilege() -> None:
    """Ambiguous configuration resolves deterministically, not by iteration order."""
    config = AuthConfig(viewer_keys=("shared",), operator_keys=("shared",))

    assert config.principal_for("shared") == Principal(key_id=hash_api_key("shared"), role="operator")


def test_principal_for_an_unknown_key_is_none() -> None:
    assert AUTH_CONFIG.principal_for("nope") is None


def test_hash_api_key_is_stable_and_reveals_nothing() -> None:
    """Same key in, same id out — and the key itself is not recoverable from it."""
    first = hash_api_key(OPERATOR_KEY)

    assert first == hash_api_key(OPERATOR_KEY)
    assert first != hash_api_key(VIEWER_KEY)
    assert OPERATOR_KEY not in first
    assert first.startswith("hmac-sha256:")


def test_require_role_rejects_an_unknown_role() -> None:
    """A typo'd role is a programming error caught at import, not a route that admits nobody."""
    with pytest.raises(ValueError, match="Unknown role"):
        require_role("superuser")  # type: ignore[arg-type]
