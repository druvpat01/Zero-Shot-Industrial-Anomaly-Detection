#!/usr/bin/env python3
"""Score a raw image with WinCLIP and save a heatmap overlay. **No training.**

This is the demo for the project's headline claim, so it is worth being precise
about what it does *not* do. There is no ``model.train(...)`` below, no
checkpoint is read, no datamodule is constructed, and nothing has ever looked at
``data/MVTecAD/bottle/train/``. The script picks an image, hands it to a freshly
constructed :class:`~app.models.winclip.WinCLIPModel`, and writes the result.
The only thing the model knows about bottles is the string ``"bottle"``, which
it turns into ~150 text prompts and encodes with CLIP.

Compare with ``scripts/train_patchcore.py`` and ``scripts/train_efficientad.py``,
which cannot produce a single number without fitting on a curated defect-free
training set first. That is the whole argument for shipping this backend.

Usage::

    python scripts/run_winclip_zeroshot.py                       # a broken bottle
    python scripts/run_winclip_zeroshot.py --defect good         # an intact one
    python scripts/run_winclip_zeroshot.py --image path/to.png --class-name cable
    python scripts/run_winclip_zeroshot.py --category metal_nut --class-name "metal nut"

Every flag is optional and falls through to
:class:`~app.models.config.ModelConfig`. The first run downloads ~830 MB of CLIP
weights (open_clip caches them under ``~/.cache/clip``); later runs load from
that cache. Expect ~4 s per image on CPU — see the latency discussion in
``app/models/winclip.py``.

Output is a three-panel figure at ``results/winclip_demo.png``: the input, the
raw anomaly map, and the map overlaid on the input.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/run_winclip_zeroshot.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display on a build machine or in a container
import matplotlib.pyplot as plt  # noqa: E402

from app.models import ModelConfig, ModelOutput, WinCLIPModel  # noqa: E402
from app.models.winclip import PATCH_GRID_SIZE  # noqa: E402

logger = logging.getLogger("run_winclip_zeroshot")

DEFAULT_DEFECT = "broken_large"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "winclip_demo.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category to pull a test image from.")
    parser.add_argument(
        "--defect",
        default=DEFAULT_DEFECT,
        help=f"Subdirectory of <category>/test to sample from, e.g. 'good' (default: {DEFAULT_DEFECT}).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Score this exact file instead of sampling from the dataset. Any image, any size.",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Noun for the text prompts (default: the category, underscores turned into spaces).",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=None,
        help="Reference images to consult. Leave at 0 — the point of this script is that it needs none.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Figure path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_image_path(args: argparse.Namespace, config: ModelConfig) -> Path:
    """Pick the image to score: an explicit ``--image``, else the first test frame.

    Raises:
        FileNotFoundError: If the file or the category's test split is missing.
    """
    if args.image is not None:
        path = args.image.expanduser()
        if not path.is_file():
            msg = f"No image at {path}."
            raise FileNotFoundError(msg)
        return path

    defect_dir = config.data_root / config.category / "test" / args.defect
    if not defect_dir.is_dir():
        msg = (
            f"No test images at {defect_dir}. Run "
            f"`python scripts/download_dataset.py --category {config.category}` first, "
            f"or point --image at any file."
        )
        raise FileNotFoundError(msg)

    candidates = sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not candidates:
        msg = f"{defect_dir} contains no images."
        raise FileNotFoundError(msg)
    return candidates[0]


def save_overlay(image_bgr: np.ndarray, result: ModelOutput, destination: Path, *, title: str) -> Path:
    """Write a three-panel figure: input, anomaly map, and the two combined.

    The heatmap is rescaled to its own min/max before colouring, which is a
    presentation choice worth flagging rather than hiding. An uncalibrated
    zero-shot map occupies a narrow slice of ``[0, 1]`` — around 0.30-0.41 on a
    broken bottle — so colouring it on an absolute scale would render a
    uniformly lukewarm image regardless of what the model found. Stretching
    shows *where* the model is looking. It does not change how confident the
    model is, and the printed score is the number to quote.

    Args:
        image_bgr: The frame as read by OpenCV.
        result: What :meth:`WinCLIPModel.predict` returned for it.
        destination: Figure path. Parent directories are created.
        title: Suptitle, used to record the prompt and the source image.

    Returns:
        ``destination``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    heatmap = result.anomaly_map
    span = float(heatmap.max() - heatmap.min())
    # A genuinely flat map would divide by zero; it also means something is
    # wrong (see test_anomaly_map_is_not_a_uniform_field), so say so.
    if span < 1e-6:
        logger.warning("Anomaly map is uniform at %.4f; check the open_clip version pin.", float(heatmap.min()))
        stretched = np.zeros_like(heatmap)
    else:
        stretched = (heatmap - heatmap.min()) / span

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for axis in axes:
        axis.axis("off")

    axes[0].imshow(image_rgb)
    axes[0].set_title("Input (never seen in training — there was none)")

    mappable = axes[1].imshow(heatmap, cmap="inferno")
    axes[1].set_title(f"Anomaly map ({heatmap.min():.3f}–{heatmap.max():.3f})")
    figure.colorbar(mappable, ax=axes[1], fraction=0.046)

    axes[2].imshow(image_rgb)
    axes[2].imshow(stretched, cmap="inferno", alpha=0.55)
    verdict = "DEFECTIVE" if result.is_defective else "OK"
    axes[2].set_title(f"Overlay — score {result.anomaly_score:.4f} [{verdict}]")

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    config = ModelConfig.from_env(
        category=args.category,
        class_name=args.class_name,
        k_shot=args.k_shot,
        data_root=args.data_root,
    )

    try:
        image_path = resolve_image_path(args, config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)  # noqa: TRY400 - the traceback adds nothing for the user here
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("OpenCV could not decode %s.", image_path)
        return 1

    # The whole demo, in three lines. Note the absence of a train() call: there
    # is no checkpoint to load and no training split to load it from.
    model = WinCLIPModel(class_name=config.prompt_class_name, category=config.category, config=config)

    shot = "zero-shot — no reference images" if model.is_zero_shot else "few-shot"
    print("Model      : WinCLIP (ViT-B-16-plus-240, CLIP zero-shot)")
    print(
        f'Prompts    : "a photo of a {model.class_name} without defect." vs '
        f'"a photo of a damaged {model.class_name}." (+ ~150 more)',
    )
    print(f"k_shot     : {model.k_shot}  ({shot})")
    print(f"Scales     : {tuple(model.scales)} patches over a {PATCH_GRID_SIZE}x{PATCH_GRID_SIZE} grid")
    print(f"Trained    : {model.is_trained}  (nothing was fitted; train() was never called)")
    print(f"Image      : {image_path}  {image.shape[1]}x{image.shape[0]}")
    print()

    started = time.perf_counter()
    result = model.predict(image, color_order="bgr")
    elapsed = time.perf_counter() - started

    title = (
        f'WinCLIP zero-shot — prompt class "{model.class_name}", no training data\n'
        f"{image_path.parent.name}/{image_path.name}"
    )
    output = save_overlay(image, result, args.output.expanduser(), title=title)

    low, high = float(result.anomaly_map.min()), float(result.anomaly_map.max())
    hot_half = low + 0.5 * (high - low)
    verdict = "DEFECTIVE" if result.is_defective else "OK"

    print(f"Anomaly score  : {result.anomaly_score:.4f}  ({verdict} at threshold {config.anomaly_threshold})")
    print(f"Anomaly map    : {result.anomaly_map.shape[0]}x{result.anomaly_map.shape[1]} (range {low:.4f}-{high:.4f})")
    print(f"Hot region     : {result.defective_area_ratio(hot_half):.1%} of pixels in the upper half of that range")
    print(f"Inference time : {elapsed:.2f}s (CPU)")
    print(f"Heatmap saved  : {output}")
    print()
    print(
        "Note: scores are uncalibrated softmax probabilities over the prompt ensemble, so they sit in a "
        "narrow band around 0.5 rather than spanning [0, 1]. That is expected without a calibration pass — "
        "run model.train(datamodule) to fit score normalization against a category, or read the ordering "
        "(defective > clean) rather than the absolute value.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
