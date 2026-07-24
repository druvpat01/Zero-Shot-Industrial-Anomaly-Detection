"""Tests for the input-quality guard.

Every synthetic case is generated in-process with OpenCV/NumPy — no disk, no
fixtures — so the file runs anywhere. The one dataset-backed test (a real bottle
must pass every check) skips rather than fails when ``data/MVTecAD/bottle`` is
absent, exactly like the model suites.

The corruptions here mirror the spec's, and the reason they are the right ones
is that each isolates a single failure mode a real line produces: a Gaussian
blur is a fouled lens, an all-black frame is a dead camera, an all-white frame
is a blown-out light, and a 10x10 array is a truncated or thumbnail read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.data.datamodule import DEFAULT_DATA_ROOT
from app.guardrails import FrameGuard, GuardConfig, GuardError, GuardResult, guard

CATEGORY = "bottle"
TRAIN_GOOD_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY / "train" / "good"

requires_dataset = pytest.mark.skipif(
    not TRAIN_GOOD_DIR.is_dir(),
    reason=f"{TRAIN_GOOD_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)

#: Every metric the guard promises, one per check (blur; exposure ×2; resolution
#: ×3; aspect). Callers log these for drift monitoring, so their presence is part
#: of the contract, not an implementation detail.
EXPECTED_METRIC_KEYS = {
    "laplacian_variance",
    "dark_fraction",
    "bright_fraction",
    "width",
    "height",
    "min_dimension",
    "aspect_ratio",
}


@pytest.fixture(scope="module")
def frame_guard() -> FrameGuard:
    """A guard on the documented defaults, independent of the process environment."""
    return FrameGuard(GuardConfig())


def _sharp_well_exposed_frame() -> np.ndarray:
    """A synthetic frame that passes every check: mid-grey with high-frequency detail.

    Random noise has crisp pixel-to-pixel transitions (high Laplacian variance)
    and a spread of intensities (nothing pinned to the floor or ceiling), so it
    clears blur and exposure without needing a real image — useful for the tests
    that are about the guard's plumbing rather than about real photographs.
    """
    return np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------


def test_gaussian_blur_is_rejected(frame_guard: FrameGuard) -> None:
    """A heavily blurred frame (sigma=20) fails the blur check.

    Built from a sharp source so only the blur changes: the same frame passes
    before blurring and fails after, which is what pins the failure on focus and
    not on some incidental property of the image.
    """
    sharp = _sharp_well_exposed_frame()
    assert frame_guard.validate(sharp).passed, "the pre-blur frame should pass"

    blurred = cv2.GaussianBlur(sharp, ksize=(0, 0), sigmaX=20)
    result = frame_guard.validate(blurred)

    assert not result.passed
    assert result.reason == "blurry"
    assert result.metrics["laplacian_variance"] < frame_guard.config.blur_threshold


def test_blur_threshold_is_configurable(frame_guard: FrameGuard) -> None:
    """Lowering BLUR_THRESHOLD far enough lets a blurred frame back through.

    Confirms the check reads its threshold from config rather than a hardcoded
    constant — a line with a softer optical path can retune it.
    """
    blurred = cv2.GaussianBlur(_sharp_well_exposed_frame(), ksize=(0, 0), sigmaX=20)
    measured = frame_guard.validate(blurred).metrics["laplacian_variance"]

    lenient = FrameGuard(GuardConfig(blur_threshold=measured / 2))
    assert lenient.validate(blurred).passed, "a threshold below the measured variance must pass the frame"


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def test_all_black_frame_is_underexposed(frame_guard: FrameGuard) -> None:
    """A dead camera: every pixel on the floor."""
    result = frame_guard.validate(np.zeros((256, 256, 3), dtype=np.uint8))

    assert not result.passed
    assert result.reason == "underexposed"
    assert result.metrics["dark_fraction"] == pytest.approx(1.0)


def test_all_white_frame_is_overexposed(frame_guard: FrameGuard) -> None:
    """A blown-out light: every pixel on the ceiling."""
    result = frame_guard.validate(np.full((256, 256, 3), 255, dtype=np.uint8))

    assert not result.passed
    assert result.reason == "overexposed"
    assert result.metrics["bright_fraction"] == pytest.approx(1.0)


def test_float_unit_range_frame_is_not_falsely_underexposed(frame_guard: FrameGuard) -> None:
    """A well-exposed float ``[0, 1]`` frame must not read as all-black.

    The trap: a model accepts both uint8 and float ``[0, 1]`` frames, and every
    value in a ``[0, 1]`` frame is numerically below the ``dark_level`` of 10. If
    the guard did not rescale, it would reject every float frame as underexposed.
    """
    sharp_uint8 = _sharp_well_exposed_frame()
    assert frame_guard.validate(sharp_uint8).passed

    as_float = sharp_uint8.astype(np.float32) / 255.0
    assert frame_guard.validate(as_float).passed, "a [0, 1] float frame must be treated on the 0-255 scale"


# ---------------------------------------------------------------------------
# Resolution and aspect ratio
# ---------------------------------------------------------------------------


def test_tiny_frame_is_too_small(frame_guard: FrameGuard) -> None:
    """A 10x10 array is below the minimum on both sides."""
    result = frame_guard.validate(np.zeros((10, 10, 3), dtype=np.uint8))

    assert not result.passed
    assert result.reason == "too_small"
    assert result.metrics["min_dimension"] == 10.0


def test_resolution_is_checked_before_exposure(frame_guard: FrameGuard) -> None:
    """The 10x10 frame is also all-black, but 'too_small' is the more useful reason.

    Documents the deliberate check ordering: the most fundamental problem wins,
    so an operator is told the frame is unusably small rather than merely dark.
    """
    result = frame_guard.validate(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.reason == "too_small"
    # The exposure metric is still computed and available for logging.
    assert result.metrics["dark_fraction"] == pytest.approx(1.0)


def test_extreme_aspect_ratio_is_rejected(frame_guard: FrameGuard) -> None:
    """A wide sliver with enough pixels on each side still fails on shape.

    Sized so the short side clears ``min_resolution`` — otherwise 'too_small'
    would fire first — isolating the aspect-ratio check. This is the transposed
    buffer / single-row-read case.
    """
    sliver = np.random.default_rng(1).integers(0, 256, size=(70, 900, 3), dtype=np.uint8)
    result = frame_guard.validate(sliver)

    assert not result.passed
    assert result.reason == "invalid_aspect_ratio"
    assert result.metrics["aspect_ratio"] > frame_guard.config.max_aspect_ratio


def test_tall_sliver_is_rejected(frame_guard: FrameGuard) -> None:
    """The reciprocal bound: a tall sliver fails just as a wide one does."""
    sliver = np.random.default_rng(2).integers(0, 256, size=(900, 70, 3), dtype=np.uint8)
    result = frame_guard.validate(sliver)

    assert not result.passed
    assert result.reason == "invalid_aspect_ratio"
    assert result.metrics["aspect_ratio"] < 1.0 / frame_guard.config.max_aspect_ratio


# ---------------------------------------------------------------------------
# A real frame passes, and the metrics contract
# ---------------------------------------------------------------------------


@requires_dataset
def test_a_real_bottle_passes_every_check(frame_guard: FrameGuard) -> None:
    """The frames the guard exists to let through must actually get through.

    Reads with OpenCV, the way the serving layer will, hence BGR — the guard is
    channel-order agnostic, so it makes no difference here, which is itself worth
    not breaking.
    """
    path = sorted(TRAIN_GOOD_DIR.glob("*.png"))[0]
    frame = cv2.imread(str(path))
    assert frame is not None, f"OpenCV could not read {path}"

    result = frame_guard.validate(frame)

    assert result.passed, f"a real good bottle was rejected as {result.reason!r} (metrics {result.metrics})"
    assert result.reason is None


@requires_dataset
def test_metrics_are_populated_and_numeric_on_a_real_frame(frame_guard: FrameGuard) -> None:
    """Every check contributes a numeric metric, so callers can log them for drift.

    Asserted on a passing frame precisely because a caller will want these even
    when nothing is wrong — the point is to watch the distribution move over a
    shift, not only to explain a rejection.
    """
    path = sorted(TRAIN_GOOD_DIR.glob("*.png"))[0]
    result = frame_guard.validate(cv2.imread(str(path)))

    assert set(result.metrics) == EXPECTED_METRIC_KEYS
    for key, value in result.metrics.items():
        assert isinstance(value, float), f"metric {key!r} is {type(value).__name__}, not a float"
        assert np.isfinite(value), f"metric {key!r} is not finite: {value}"


def test_metrics_are_populated_even_on_a_synthetic_reject(frame_guard: FrameGuard) -> None:
    """A rejection still logs a full, numeric metric row (no dataset needed)."""
    result = frame_guard.validate(np.zeros((256, 256, 3), dtype=np.uint8))

    assert not result.passed
    assert set(result.metrics) == EXPECTED_METRIC_KEYS
    assert all(isinstance(v, float) and np.isfinite(v) for v in result.metrics.values())


# ---------------------------------------------------------------------------
# Malformed input and the GuardError type
# ---------------------------------------------------------------------------


def test_validate_rejects_non_arrays_and_bad_ranks(frame_guard: FrameGuard) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        frame_guard.validate("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2-D or 3-D"):
        frame_guard.validate(np.zeros((2, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        frame_guard.validate(np.zeros((0, 32, 3), dtype=np.uint8))


def test_guard_error_carries_the_result() -> None:
    """The exception the model path raises exposes the reason and metrics.

    A serving handler catches one :class:`GuardError` and has everything it needs
    to tell the operator why and to log the row — without re-running the guard.
    """
    result = GuardResult(passed=False, reason="blurry", metrics={"laplacian_variance": 1.0})
    error = GuardError(result)

    assert error.reason == "blurry"
    assert error.metrics == {"laplacian_variance": 1.0}
    assert error.result is result
    assert "blurry" in str(error)


def test_module_level_guard_is_a_frame_guard() -> None:
    """The shared instance the model wrappers import is ready to use."""
    assert isinstance(guard, FrameGuard)
    assert guard.validate(_sharp_well_exposed_frame()).passed


# ---------------------------------------------------------------------------
# Config resolution from the environment
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_spec() -> None:
    config = GuardConfig()
    assert config.blur_threshold == 50.0
    assert config.min_resolution == 64
    assert config.max_aspect_ratio == 10.0
    assert config.dark_level == 10
    assert config.bright_level == 245
    assert config.exposure_fraction == 0.95


def test_config_reads_the_environment() -> None:
    env = {
        "BLUR_THRESHOLD": "12.5",
        "MIN_RESOLUTION": "128",
        "MAX_ASPECT_RATIO": "4",
        "EXPOSURE_DARK_LEVEL": "5",
        "EXPOSURE_BRIGHT_LEVEL": "250",
        "EXPOSURE_FRACTION": "0.9",
    }
    config = GuardConfig.from_env(env)

    assert config.blur_threshold == 12.5
    assert config.min_resolution == 128
    assert config.max_aspect_ratio == 4.0
    assert config.dark_level == 5
    assert config.bright_level == 250
    assert config.exposure_fraction == 0.9


def test_blank_environment_values_fall_back_to_defaults() -> None:
    """An empty ``BLUR_THRESHOLD=`` in a .env file must not blow up parsing."""
    assert GuardConfig.from_env({"BLUR_THRESHOLD": "", "MIN_RESOLUTION": "  "}).blur_threshold == 50.0


def test_config_rejects_incoherent_thresholds() -> None:
    with pytest.raises(ValueError, match="max_aspect_ratio"):
        GuardConfig(max_aspect_ratio=0.5)
    with pytest.raises(ValueError, match="dark_level"):
        GuardConfig(dark_level=250, bright_level=10)
    with pytest.raises(ValueError, match="exposure_fraction"):
        GuardConfig(exposure_fraction=1.5)
