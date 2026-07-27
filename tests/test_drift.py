"""Tests for the reliability layer: drift detection, threshold calibration, and their endpoints.

What is being tested, and how the randomness is handled
-------------------------------------------------------
The drift tests are statistical, which makes them the one place in this suite
where a naive test would be *flaky by construction*. A KS test at ``p < 0.05``
declares drift on 5% of comparisons between samples from the same distribution —
that is the definition of the threshold, not a defect — so a
``test_no_drift_when_distributions_match`` drawing fresh randomness would fail
roughly one run in twenty and teach everyone to re-run the suite until it passed.

Every draw here therefore comes from a seeded
:func:`numpy.random.default_rng`. The distributions are the spec's — N(0.2, 0.05)
against N(0.8, 0.05) for the drifted case, one distribution against itself for
the stable one — and the separated pair is twelve standard deviations apart, so
the drift assertion holds for any seed. The *stable* case is the fragile one, and
:func:`test_no_drift_across_many_seeds` measures its false-alarm rate across
fifty seeds rather than trusting the single lucky one, which is the honest way to
pin a probabilistic assertion.

The API tests use a registry the test owns, with drift monitors sized down so a
handful of scores fills a window. They assert the *wiring* — that a score reaches
a monitor, that a threshold reaches a model's config — not the statistics, which
the unit tests above already cover.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data.datamodule import DEFAULT_DATA_ROOT
from app.evaluation.calibration import (
    CALIBRATION_METRICS,
    evaluate_threshold,
    find_optimal_threshold,
)
from app.evaluation.drift import (
    DEFAULT_KS_DRIFT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    KS_DRIFT_THRESHOLD_VAR,
    REASON_KS_DRIFT,
    ScoreDistributionMonitor,
    ks_drift_threshold,
)
from app.models.config import get_model_config
from app.serving.auth import API_KEY_HEADER, AuthConfig, get_auth_config
from app.serving.main import app
from app.serving.model_registry import ModelRegistry, ThresholdOutOfRangeError, get_registry
from app.serving.schemas import CALIBRATION_METRICS as SCHEMA_CALIBRATION_METRICS

CATEGORY = "bottle"

#: An operator key. Both new endpoints are operator-gated; ``tests/test_auth.py``
#: owns the access-control behaviour, so this module authenticates once and then
#: ignores the subject.
OPERATOR_KEY = "test-operator-key"

#: The spec's two distributions: a nominal line, and the same line after a shift
#: large enough that no reasonable test could miss it.
REFERENCE_MEAN, REFERENCE_STD = 0.2, 0.05
DRIFTED_MEAN, DRIFTED_STD = 0.8, 0.05
SAMPLE_SIZE = 100

#: The one test here that needs real artifacts skips rather than fails without
#: them, matching every other suite in this project.
GOOD_TEST_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY / "test" / "good"
CHECKPOINT: Path = get_model_config().checkpoint_path("patchcore", CATEGORY)

requires_dataset = pytest.mark.skipif(
    not GOOD_TEST_DIR.is_dir(),
    reason=f"{GOOD_TEST_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(),
    reason=f"{CHECKPOINT} not found; run `python scripts/train_patchcore.py --category {CATEGORY}`",
)


def _normal(rng: np.random.Generator, mean: float, std: float, size: int = SAMPLE_SIZE) -> list[float]:
    """A list of ``size`` draws from N(mean, std), as plain floats."""
    return [float(value) for value in rng.normal(mean, std, size)]


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every distributional assertion in this file is reproducible."""
    return np.random.default_rng(20260727)


@pytest.fixture(scope="module")
def clean_bottle_b64() -> str:
    """A real defect-free bottle from the test split, as an API payload."""
    if not GOOD_TEST_DIR.is_dir():
        pytest.skip(f"{GOOD_TEST_DIR} not found")
    return base64.b64encode(sorted(GOOD_TEST_DIR.glob("*.png"))[0].read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# ScoreDistributionMonitor — the drift verdict
# ---------------------------------------------------------------------------


def test_drift_detected_when_the_distribution_shifts(rng: np.random.Generator) -> None:
    """N(0.2, 0.05) reference against an N(0.8, 0.05) window is drift.

    The spec's headline case, and the one a production line lives or dies by: the
    scores are all still well-formed floats in a plausible range, the model has
    not raised, and everything downstream of the score has silently changed
    meaning.
    """
    monitor = ScoreDistributionMonitor(window_size=DEFAULT_WINDOW_SIZE)
    monitor.set_reference(_normal(rng, REFERENCE_MEAN, REFERENCE_STD))

    for score in _normal(rng, DRIFTED_MEAN, DRIFTED_STD):
        monitor.record_score(score)

    drifted, reason = monitor.is_drifted()

    assert drifted is True
    assert reason == REASON_KS_DRIFT
    p_value = monitor.ks_p_value()
    assert p_value is not None and p_value < DEFAULT_KS_DRIFT_THRESHOLD


def test_no_drift_when_the_distribution_holds(rng: np.random.Generator) -> None:
    """The same distribution on both sides is not drift, and reports no reason."""
    monitor = ScoreDistributionMonitor()
    monitor.set_reference(_normal(rng, REFERENCE_MEAN, REFERENCE_STD))

    for score in _normal(rng, REFERENCE_MEAN, REFERENCE_STD):
        monitor.record_score(score)

    drifted, reason = monitor.is_drifted()

    assert drifted is False
    assert reason is None


def test_no_drift_across_many_seeds() -> None:
    """The false-alarm rate on an unchanged process stays near the 5% the threshold buys.

    The test above passes on one seed; this is the assertion that it is not luck.
    Fifty independent comparisons of a distribution against itself should trip the
    ``p < 0.05`` line about 5% of the time — the threshold's *definition*, not a
    bug — so the bound is generous (20%) and is really watching for a monitor that
    has become systematically trigger-happy: a reversed comparison or a mishandled
    window would push this to 100%, not to 8%.
    """
    false_alarms = 0
    for seed in range(50):
        generator = np.random.default_rng(seed)
        monitor = ScoreDistributionMonitor()
        monitor.set_reference(_normal(generator, REFERENCE_MEAN, REFERENCE_STD))
        for score in _normal(generator, REFERENCE_MEAN, REFERENCE_STD):
            monitor.record_score(score)
        false_alarms += monitor.is_drifted()[0]

    assert false_alarms <= 10, f"{false_alarms}/50 false alarms; expected roughly 2-3 at p<0.05"


def test_drift_survives_a_shift_in_shape_alone(rng: np.random.Generator) -> None:
    """A distribution that keeps its mean and changes its spread is still drift.

    The case that justifies choosing KS over a t-test, spelled out as a test: both
    windows are centred on 0.5, so a monitor watching the mean sees nothing at
    all, while the ECDFs are visibly different curves.
    """
    monitor = ScoreDistributionMonitor()
    monitor.set_reference(_normal(rng, 0.5, 0.02, size=200))

    for score in _normal(rng, 0.5, 0.20, size=200):
        monitor.record_score(score)

    assert monitor.is_drifted() == (True, REASON_KS_DRIFT)


# ---------------------------------------------------------------------------
# ScoreDistributionMonitor — window, summary and the "no verdict" state
# ---------------------------------------------------------------------------


def test_window_is_bounded_and_keeps_the_newest_scores() -> None:
    """The deque evicts the oldest score, so memory is bounded regardless of uptime."""
    monitor = ScoreDistributionMonitor(window_size=10)

    for score in range(100):
        monitor.record_score(float(score))

    summary = monitor.get_summary()
    assert monitor.count == 10
    # The last ten values are 90..99, whose median is 94.5.
    assert summary["count"] == 10
    assert summary["p50"] == pytest.approx(94.5)


def test_summary_reports_the_documented_statistics() -> None:
    """mean/std/p10/p50/p90 over a window whose answers can be computed by hand."""
    monitor = ScoreDistributionMonitor()
    values = [float(value) / 100.0 for value in range(101)]  # 0.00 .. 1.00
    for value in values:
        monitor.record_score(value)

    summary = monitor.get_summary()

    assert summary["count"] == 101
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["p10"] == pytest.approx(0.1)
    assert summary["p50"] == pytest.approx(0.5)
    assert summary["p90"] == pytest.approx(0.9)
    assert summary["std"] == pytest.approx(float(np.std(values)))


def test_summary_of_an_empty_window_is_null_not_nan() -> None:
    """An empty window reports nulls, so ``GET /drift`` stays valid JSON.

    ``nan`` is not representable in JSON, and a serializer that emitted it would
    produce a response some clients parse and others reject.
    """
    summary = ScoreDistributionMonitor().get_summary()

    assert summary["count"] == 0
    assert summary["mean"] is None
    assert summary["p50"] is None


def test_no_verdict_without_a_reference(rng: np.random.Generator) -> None:
    """A monitor with scores but no reference offers no verdict, and does not claim health.

    The distinction the ``(False, None)`` return deliberately blurs and
    ``ks_p_value`` keeps: nothing has said what normal looks like, so there is
    nothing to be different from.
    """
    monitor = ScoreDistributionMonitor()
    for score in _normal(rng, DRIFTED_MEAN, DRIFTED_STD):
        monitor.record_score(score)

    assert monitor.has_reference is False
    assert monitor.ks_p_value() is None
    assert monitor.is_drifted() == (False, None)


def test_no_verdict_until_the_window_has_enough_samples(rng: np.random.Generator) -> None:
    """Below ``min_samples`` the KS test is powerless, so the monitor withholds a verdict.

    Four wildly out-of-distribution scores are not evidence of drift, and a
    monitor that alerted on them would be muted before it ever caught a real one.
    """
    monitor = ScoreDistributionMonitor(min_samples=30)
    monitor.set_reference(_normal(rng, REFERENCE_MEAN, REFERENCE_STD))

    for score in [0.9, 0.95, 0.92, 0.98]:
        monitor.record_score(score)

    assert monitor.ks_p_value() is None
    assert monitor.is_drifted() == (False, None)


def test_reset_window_clears_scores_but_keeps_the_reference(rng: np.random.Generator) -> None:
    """What a system calls after acting on an alert: forget the scores, keep the baseline."""
    monitor = ScoreDistributionMonitor()
    reference = _normal(rng, REFERENCE_MEAN, REFERENCE_STD)
    monitor.set_reference(reference)
    for score in _normal(rng, DRIFTED_MEAN, DRIFTED_STD):
        monitor.record_score(score)

    monitor.reset_window()

    assert monitor.count == 0
    assert monitor.reference_size == len(reference)
    assert monitor.is_drifted() == (False, None)


def test_reference_is_copied_not_aliased(rng: np.random.Generator) -> None:
    """Mutating the caller's list must not move the baseline under a live monitor."""
    monitor = ScoreDistributionMonitor()
    reference = _normal(rng, REFERENCE_MEAN, REFERENCE_STD)
    monitor.set_reference(reference)

    reference.clear()

    assert monitor.reference_size == SAMPLE_SIZE


def test_non_finite_input_is_refused() -> None:
    """A NaN would poison every later percentile and p-value silently, so it is refused."""
    monitor = ScoreDistributionMonitor()

    with pytest.raises(ValueError, match="finite"):
        monitor.record_score(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        monitor.set_reference([0.1, float("inf")])


def test_monitor_rejects_a_nonsense_configuration() -> None:
    """Construction validates, so a bad window or threshold fails at startup, not at alert time."""
    with pytest.raises(ValueError, match="window_size"):
        ScoreDistributionMonitor(window_size=0)
    with pytest.raises(ValueError, match="threshold"):
        ScoreDistributionMonitor(threshold=1.0)


# ---------------------------------------------------------------------------
# The KS_DRIFT_THRESHOLD environment variable
# ---------------------------------------------------------------------------


def test_ks_threshold_defaults_and_reads_the_environment() -> None:
    """A blank or missing variable falls back to 0.05; a set one wins."""
    assert ks_drift_threshold({}) == DEFAULT_KS_DRIFT_THRESHOLD
    assert ks_drift_threshold({KS_DRIFT_THRESHOLD_VAR: "  "}) == DEFAULT_KS_DRIFT_THRESHOLD
    assert ks_drift_threshold({KS_DRIFT_THRESHOLD_VAR: "0.01"}) == pytest.approx(0.01)


def test_ks_threshold_rejects_a_value_that_would_disable_the_monitor() -> None:
    """0 can never fire and 1 always fires; both are refused rather than silently accepted."""
    for bad in ("0", "1", "1.5", "not-a-number"):
        with pytest.raises(ValueError, match=KS_DRIFT_THRESHOLD_VAR):
            ks_drift_threshold({KS_DRIFT_THRESHOLD_VAR: bad})


def test_monitor_honours_the_configured_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stricter threshold is picked up at construction from the environment."""
    monkeypatch.setenv(KS_DRIFT_THRESHOLD_VAR, "0.001")

    assert ScoreDistributionMonitor().threshold == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------


def _separable_calibration_set(rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """A synthetic set a good threshold separates almost perfectly.

    Normal parts around 0.2, defects around 0.8, both with a 0.05 spread — the
    same two distributions the drift tests use, which is not a coincidence: this
    is what the line looks like before and after the shift.
    """
    normal_scores = _normal(rng, REFERENCE_MEAN, REFERENCE_STD)
    defect_scores = _normal(rng, DRIFTED_MEAN, DRIFTED_STD)
    return normal_scores + defect_scores, [0] * len(normal_scores) + [1] * len(defect_scores)


def test_find_optimal_threshold_reaches_high_f1(rng: np.random.Generator) -> None:
    """On a separable synthetic set the fitted threshold achieves F1 > 0.9.

    The spec's assertion. Separating N(0.2, 0.05) from N(0.8, 0.05) is easy by
    design — this is a test of the *sweep*, not of a model, and a sweep that
    cannot find the boundary between two distributions twelve standard deviations
    apart is broken in a way no subtler dataset would show more clearly.
    """
    scores, labels = _separable_calibration_set(rng)

    threshold = find_optimal_threshold(scores, labels, metric="f1")

    assert REFERENCE_MEAN < threshold < DRIFTED_MEAN
    assert evaluate_threshold(scores, labels, threshold, metric="f1") > 0.9


def test_fitted_threshold_beats_the_shipped_default(rng: np.random.Generator) -> None:
    """The point of calibrating: the fitted point is at least as good as ANOMALY_THRESHOLD.

    Not a tautology — the sweep is exact over every candidate, so 0.5 is one of
    the values it considered. What this pins is that the returned threshold is the
    *maximum* rather than merely a good one, on data where the default happens to
    be defensible too.
    """
    scores, labels = _separable_calibration_set(rng)
    threshold = find_optimal_threshold(scores, labels)

    fitted = evaluate_threshold(scores, labels, threshold)
    default = evaluate_threshold(scores, labels, 0.5)

    assert fitted >= default


def test_threshold_is_exact_on_a_hand_checkable_set() -> None:
    """A tiny set whose optimum can be read off by eye, with the tie-break pinned.

    Scores 0.1/0.2 are normal and 0.8/0.9 are defective, so every threshold in
    (0.2, 0.8] achieves F1 = 1.0. The sweep only proposes observed scores, and the
    tie-break takes the lowest qualifying one — 0.8, the most sensitive threshold
    that is still perfect. A tie-break pointing the other way would return 0.9,
    which is equally perfect here and would miss a defect scoring 0.85 in
    production.
    """
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    assert find_optimal_threshold(scores, labels) == pytest.approx(0.8)
    assert evaluate_threshold(scores, labels, 0.8) == pytest.approx(1.0)


def test_every_supported_metric_is_maximised(rng: np.random.Generator) -> None:
    """Each metric returns a threshold no other candidate beats on that same metric."""
    scores, labels = _separable_calibration_set(rng)

    for metric in CALIBRATION_METRICS:
        threshold = find_optimal_threshold(scores, labels, metric=metric)
        best = evaluate_threshold(scores, labels, threshold, metric=metric)
        assert all(
            evaluate_threshold(scores, labels, candidate, metric=metric) <= best + 1e-12
            for candidate in scores
        ), f"{metric}: a candidate threshold beat the one that was returned"


def test_recall_is_maximised_by_flagging_everything() -> None:
    """The documented degeneracy, asserted rather than left as a claim in a docstring."""
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    threshold = find_optimal_threshold(scores, labels, metric="recall")

    assert threshold == pytest.approx(0.1)  # the lowest score: everything is flagged
    assert evaluate_threshold(scores, labels, threshold, metric="recall") == pytest.approx(1.0)


def test_calibration_rejects_an_unusable_set() -> None:
    """Every way a calibration set can be meaningless is a ValueError naming the problem."""
    with pytest.raises(ValueError, match="both classes"):
        find_optimal_threshold([0.1, 0.2, 0.3], [0, 0, 0])
    with pytest.raises(ValueError, match="same length"):
        find_optimal_threshold([0.1, 0.2], [0, 1, 1])
    with pytest.raises(ValueError, match="at least one sample"):
        find_optimal_threshold([], [])
    with pytest.raises(ValueError, match="finite"):
        find_optimal_threshold([0.1, float("nan")], [0, 1])
    with pytest.raises(ValueError, match="0 .normal. or 1"):
        find_optimal_threshold([0.1, 0.9], [0, 2])
    with pytest.raises(ValueError, match="unknown metric"):
        find_optimal_threshold([0.1, 0.9], [0, 1], metric="accuracy")


def test_schema_metric_literal_matches_the_implementation() -> None:
    """The API's metric enum and the calibrator's are spelled out separately; pin them together.

    ``app.serving.schemas`` writes the ``Literal`` by hand so the wire contract
    reads without an import. This is the assertion that keeps that from drifting
    into a metric the API advertises and the calibrator rejects with a 500.
    """
    assert SCHEMA_CALIBRATION_METRICS == CALIBRATION_METRICS


# ---------------------------------------------------------------------------
# The serving layer: /drift and /calibrate
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """A client with an operator key and a registry this test owns.

    The registry is fresh per test — a shared one would carry another test's
    monitors into this one's ``/drift`` response — with ``warmup=False`` to keep
    the suite quick and a small drift window so a handful of scores is a full one.

    Overriding the auth *config* rather than the dependency means these requests
    still run the whole check, matching ``tests/test_api.py``.
    """
    app.dependency_overrides[get_auth_config] = lambda: AuthConfig(operator_keys=(OPERATOR_KEY,))
    registry = ModelRegistry(warmup=False, drift_window_size=64)
    app.dependency_overrides[get_registry] = lambda: registry

    test_client = TestClient(app, headers={API_KEY_HEADER: OPERATOR_KEY})
    test_client.registry = registry  # type: ignore[attr-defined]
    yield test_client

    app.dependency_overrides.pop(get_auth_config, None)
    app.dependency_overrides.pop(get_registry, None)


def test_drift_is_empty_before_any_prediction(client: TestClient) -> None:
    """No frame scored means no monitor, and an empty list rather than an error."""
    response = client.get("/drift")

    assert response.status_code == 200
    assert response.json() == []


def test_drift_reports_a_monitor_fed_directly(client: TestClient, rng: np.random.Generator) -> None:
    """A monitor's window and verdict reach the endpoint in the documented shape.

    Fed through the registry rather than through ``/predict`` so the test needs no
    checkpoint and no dataset: what is under test here is the endpoint's
    serialisation, and ``test_predict_feeds_the_drift_monitor`` covers the wiring
    from a real scored frame.
    """
    monitor = client.registry.monitor_for("patchcore", CATEGORY)  # type: ignore[attr-defined]
    monitor.set_reference(_normal(rng, REFERENCE_MEAN, REFERENCE_STD))
    for score in _normal(rng, DRIFTED_MEAN, DRIFTED_STD, size=64):
        monitor.record_score(score)

    body = client.get("/drift").json()

    assert len(body) == 1
    row = body[0]
    assert row["model_name"] == "patchcore"
    assert row["category"] == CATEGORY
    assert row["drifted"] is True
    assert row["reason"] == REASON_KS_DRIFT
    assert row["p_value"] < DEFAULT_KS_DRIFT_THRESHOLD
    assert row["threshold"] == pytest.approx(DEFAULT_KS_DRIFT_THRESHOLD)
    assert row["reference_size"] == SAMPLE_SIZE
    assert row["window_size"] == 64
    assert set(row["summary"]) == {"count", "mean", "std", "p10", "p50", "p90"}
    assert row["summary"]["count"] == 64
    assert row["summary"]["p50"] == pytest.approx(DRIFTED_MEAN, abs=0.05)


def test_drift_reports_no_verdict_without_a_reference(client: TestClient) -> None:
    """A monitor with traffic and no baseline reports a summary and a null p-value.

    The state a fresh deployment is in, and the reason ``drifted: false`` must not
    be read as "healthy" — the endpoint has to be able to say "I do not know".
    """
    monitor = client.registry.monitor_for("patchcore", CATEGORY)  # type: ignore[attr-defined]
    for score in [0.4] * 40:
        monitor.record_score(score)

    row = client.get("/drift").json()[0]

    assert row["drifted"] is False
    assert row["reason"] is None
    assert row["p_value"] is None
    assert row["reference_size"] == 0
    assert row["summary"]["count"] == 40
    # Reported so the null p-value above is explicable from the response alone.
    assert row["min_samples"] == 30


def test_drift_separates_monitors_by_model_and_category(client: TestClient) -> None:
    """One row per (model, category), sorted, so two polls are diffable."""
    registry = client.registry  # type: ignore[attr-defined]
    registry.monitor_for("patchcore", "bottle").record_score(0.1)
    registry.monitor_for("onnx_patchcore", "bottle").record_score(0.2)
    registry.monitor_for("patchcore", "cable").record_score(0.3)

    body = client.get("/drift").json()

    assert [(row["model_name"], row["category"]) for row in body] == [
        ("onnx_patchcore", "bottle"),
        ("patchcore", "bottle"),
        ("patchcore", "cable"),
    ]


@requires_dataset
@requires_checkpoint
def test_predict_feeds_the_drift_monitor(client: TestClient, clean_bottle_b64: str) -> None:
    """A real scored frame lands in the monitor for the model that scored it.

    The wiring assertion the rest of this section stands on: everything above
    feeds monitors directly, and this is the one test that proves ``/predict``
    does too. It also pins the *key* — the monitor is created under the resolved
    ``model_name`` from the response, so an ONNX fallback accumulates its own
    distribution rather than mixing into the PyTorch backend's.
    """
    payload = {"category": CATEGORY, "model_backend": "patchcore", "image_b64": clean_bottle_b64}

    predicted = client.post("/predict", json=payload)
    assert predicted.status_code == 200
    scored_by = predicted.json()["model_name"]

    row = client.get("/drift").json()[0]

    assert row["model_name"] == scored_by
    assert row["category"] == CATEGORY
    assert row["summary"]["count"] == 1
    assert row["summary"]["p50"] == pytest.approx(predicted.json()["anomaly_score"], rel=1e-6)


def test_drift_requires_an_operator_key() -> None:
    """A viewer key is refused: the summary describes production traffic."""
    app.dependency_overrides[get_auth_config] = lambda: AuthConfig(viewer_keys=("viewer-key",))
    try:
        with TestClient(app, headers={API_KEY_HEADER: "viewer-key"}) as viewer_client:
            response = viewer_client.get("/drift")
    finally:
        app.dependency_overrides.pop(get_auth_config, None)

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient_role"


# ---------------------------------------------------------------------------
# POST /calibrate
# ---------------------------------------------------------------------------


def _calibration_payload(rng: np.random.Generator, **overrides) -> dict:
    scores, labels = _separable_calibration_set(rng)
    payload = {
        "category": CATEGORY,
        "model_backend": "winclip",
        "samples": [{"score": score, "label": label} for score, label in zip(scores, labels, strict=True)],
    }
    payload.update(overrides)
    return payload


def test_calibrate_updates_the_models_decision_threshold(client: TestClient, rng: np.random.Generator) -> None:
    """The fitted threshold reaches the loaded model's config, which is where it takes effect.

    ``winclip`` is the backend under test because it needs no checkpoint — it is
    the zero-shot one — so this runs in a fresh clone. What it asserts is the full
    path: sweep, install, and the model's own ``anomaly_threshold`` afterwards.
    """
    response = client.post("/calibrate", json=_calibration_payload(rng))

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "winclip"
    assert body["category"] == CATEGORY
    assert body["metric"] == "f1"
    assert body["previous_threshold"] == pytest.approx(get_model_config().anomaly_threshold)
    assert REFERENCE_MEAN < body["threshold"] < DRIFTED_MEAN
    assert body["metric_value"] > 0.9
    assert body["samples"] == 2 * SAMPLE_SIZE
    assert body["positives"] == SAMPLE_SIZE

    model = client.registry.get_model("winclip", CATEGORY)  # type: ignore[attr-defined]
    assert model.config.anomaly_threshold == pytest.approx(body["threshold"])


def test_calibrate_installs_the_drift_reference(client: TestClient, rng: np.random.Generator) -> None:
    """The calibration set becomes the baseline the next drift check runs against."""
    body = client.post("/calibrate", json=_calibration_payload(rng)).json()

    assert body["drift_reference_size"] == 2 * SAMPLE_SIZE
    monitor = client.registry.monitor_for("winclip", CATEGORY)  # type: ignore[attr-defined]
    assert monitor.reference_size == 2 * SAMPLE_SIZE


def test_calibrate_can_skip_the_drift_reference(client: TestClient, rng: np.random.Generator) -> None:
    """``set_drift_reference: false`` fits the threshold and leaves the baseline alone."""
    body = client.post("/calibrate", json=_calibration_payload(rng, set_drift_reference=False)).json()

    assert body["drift_reference_size"] == 0
    assert client.registry.monitor_for("winclip", CATEGORY).has_reference is False  # type: ignore[attr-defined]


def test_calibrated_threshold_survives_a_model_reload(client: TestClient, rng: np.random.Generator) -> None:
    """An evicted and rebuilt model comes back calibrated, not at ANOMALY_THRESHOLD.

    Without the override being recorded in the registry, the threshold would live
    only on the model object and quietly revert the next time one was rebuilt —
    a regression that is invisible until a shift's worth of parts has been graded
    against the wrong boundary.
    """
    registry = client.registry  # type: ignore[attr-defined]
    threshold = client.post("/calibrate", json=_calibration_payload(rng)).json()["threshold"]

    # Evict the model without touching the recorded override, as a reload would.
    registry._models.clear()  # noqa: SLF001 - simulating an eviction is the point

    assert registry.get_model("winclip", CATEGORY).config.anomaly_threshold == pytest.approx(threshold)


def test_calibrate_rejects_a_single_class_set(client: TestClient) -> None:
    """All-normal samples are a 422 explaining why, not a 500 and not a fitted threshold."""
    payload = {
        "category": CATEGORY,
        "model_backend": "winclip",
        "samples": [{"score": 0.1, "label": 0}, {"score": 0.2, "label": 0}],
    }

    response = client.post("/calibrate", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "invalid_calibration_set"
    assert "both classes" in body["reason"]


def test_calibrate_rejects_a_bad_label_at_the_schema(client: TestClient) -> None:
    """A label outside {0, 1} is caught by pydantic, before any model is touched."""
    payload = {
        "category": CATEGORY,
        "model_backend": "winclip",
        "samples": [{"score": 0.1, "label": 0}, {"score": 0.9, "label": 7}],
    }

    response = client.post("/calibrate", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_request"


def test_calibrate_rejects_an_unknown_metric(client: TestClient, rng: np.random.Generator) -> None:
    """The metric enum is enforced at the boundary by the schema."""
    response = client.post("/calibrate", json=_calibration_payload(rng, metric="accuracy"))

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_request"


def test_calibrate_requires_an_operator_key(rng: np.random.Generator) -> None:
    """A viewer cannot change how the service grades parts."""
    app.dependency_overrides[get_auth_config] = lambda: AuthConfig(viewer_keys=("viewer-key",))
    try:
        with TestClient(app, headers={API_KEY_HEADER: "viewer-key"}) as viewer_client:
            response = viewer_client.post("/calibrate", json=_calibration_payload(rng))
    finally:
        app.dependency_overrides.pop(get_auth_config, None)

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient_role"


def test_threshold_outside_the_normalized_range_is_refused(rng: np.random.Generator) -> None:
    """A threshold fitted to raw distances cannot be installed, and says why.

    ``anomaly_threshold`` is bounded to [0, 1] because it is documented as a
    normalized score. An uncalibrated backend emits raw distances, so this is the
    honest failure rather than a silently clamped threshold that would grade every
    part on the line against a boundary nobody chose.
    """
    registry = ModelRegistry(warmup=False)

    with pytest.raises(ThresholdOutOfRangeError) as excinfo:
        registry.set_threshold("winclip", CATEGORY, 4.2)

    assert excinfo.value.threshold == pytest.approx(4.2)
    assert "outside the [0, 1] range" in excinfo.value.detail
