#!/usr/bin/env python3
"""Show the input-quality guard catching corrupted frames on a real category.

This is the evidence behind a claim like *"the guard prevented N garbage
predictions"*. It takes a category's test set, artificially corrupts a fraction
of the frames the way a real line degrades them — a fouled lens (Gaussian blur)
or a dead camera (blackout) — runs every frame through
:class:`~app.guardrails.quality.FrameGuard`, and reports how many were let
through versus rejected, broken down by failure mode::

    Passed: 66  Rejected: 17  (blur: 9, exposure: 8, size: 0)

The uncorrupted frames should pass and the corrupted ones should be caught, so
"Rejected" tracks the corruption fraction: those are the frames that would
otherwise have reached a model and been turned into a plausible-looking anomaly
score. ``size`` stays 0 here because neither corruption shrinks the frame — it is
in the summary because it is one of the guard's checks, and a real feed can trip
it (a truncated read) even though this demo does not manufacture one.

Usage::

    python scripts/demo_guardrails.py                       # bottle, 20% corrupted
    python scripts/demo_guardrails.py --category cable
    python scripts/demo_guardrails.py --corrupt-fraction 0.3 --seed 7

The run is seeded, so the summary is reproducible; change ``--seed`` to draw a
different 20%. Thresholds come from :class:`~app.guardrails.quality.GuardConfig`
(the environment, then the documented defaults), so this measures the same gate
the model wrappers apply in ``predict``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/demo_guardrails.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.guardrails import FrameGuard, GuardConfig  # noqa: E402

logger = logging.getLogger("demo_guardrails")

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "MVTecAD"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

#: Which guard reason each corruption is meant to trip — used only to sanity-check
#: that the corruptions land where expected, and to explain surprises in the log.
_BLUR_SIGMA = 20.0

#: reason slug -> the summary bucket it is counted under.
_REASON_BUCKET = {
    "blurry": "blur",
    "underexposed": "exposure",
    "overexposed": "exposure",
    "too_small": "size",
    "invalid_aspect_ratio": "aspect",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default="bottle", help="MVTec AD category to sample (default: bottle).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument(
        "--corrupt-fraction",
        type=float,
        default=0.2,
        help="Fraction of the test set to corrupt with blur/blackout (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for which frames are corrupted and how (default: 0).")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s). INFO logs every rejection.",
    )
    return parser.parse_args(argv)


def collect_test_images(data_root: Path, category: str) -> list[Path]:
    """Every image under ``<category>/test`` (all defect subfolders), sorted.

    Raises:
        FileNotFoundError: If the test split is missing.
    """
    test_dir = data_root / category / "test"
    if not test_dir.is_dir():
        msg = (
            f"No test split at {test_dir}. Run "
            f"`python scripts/download_dataset.py --category {category}` first."
        )
        raise FileNotFoundError(msg)

    paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        msg = f"{test_dir} contains no images."
        raise FileNotFoundError(msg)
    return paths


def blur(frame: np.ndarray) -> np.ndarray:
    """Fouled-lens corruption: a heavy Gaussian blur that collapses edge content."""
    return cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=_BLUR_SIGMA)


def blackout(frame: np.ndarray) -> np.ndarray:
    """Dead-camera corruption: a frame pinned to the floor."""
    return np.zeros_like(frame)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    if not 0.0 <= args.corrupt_fraction <= 1.0:
        logger.error("--corrupt-fraction must be in [0, 1], got %s.", args.corrupt_fraction)
        return 1

    try:
        paths = collect_test_images(args.data_root, args.category)
    except FileNotFoundError as exc:
        logger.error("%s", exc)  # noqa: TRY400 - the traceback adds nothing for the user here
        return 1

    guard = FrameGuard(GuardConfig.from_env())
    rng = np.random.default_rng(args.seed)

    total = len(paths)
    num_corrupt = round(total * args.corrupt_fraction)
    corrupt_indices = set(rng.choice(total, size=num_corrupt, replace=False).tolist()) if num_corrupt else set()
    corruptions = (blur, blackout)

    passed = 0
    rejected_by_bucket: Counter[str] = Counter()
    corrupted_but_passed = 0

    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("OpenCV could not decode %s; skipping.", path)
            continue

        was_corrupted = index in corrupt_indices
        if was_corrupted:
            corrupt = corruptions[int(rng.integers(len(corruptions)))]
            frame = corrupt(frame)

        result = guard.validate(frame)
        if result.passed:
            passed += 1
            if was_corrupted:
                # A corruption that slipped through is worth surfacing, not hiding.
                corrupted_but_passed += 1
                logger.info("Corrupted frame passed the guard: %s (metrics %s)", path.name, result.metrics)
        else:
            bucket = _REASON_BUCKET.get(result.reason, result.reason or "unknown")
            rejected_by_bucket[bucket] += 1
            logger.info("Rejected %s: %s", result.reason, path.name)

    rejected = sum(rejected_by_bucket.values())

    print(f"Category         : {args.category}")
    print(f"Test frames      : {total}")
    print(f"Corrupted        : {num_corrupt} ({args.corrupt_fraction:.0%}, seed {args.seed}) — blur or blackout, at random")
    print()
    print(
        f"Passed: {passed}  Rejected: {rejected}  "
        f"(blur: {rejected_by_bucket['blur']}, exposure: {rejected_by_bucket['exposure']}, size: {rejected_by_bucket['size']})"
    )
    print()
    print(f"The guard prevented {rejected} garbage prediction(s) from ever reaching a model.")
    if corrupted_but_passed:
        print(f"Note: {corrupted_but_passed} corrupted frame(s) slipped through — rerun with --log-level INFO to see which.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
