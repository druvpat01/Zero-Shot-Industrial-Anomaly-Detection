#!/usr/bin/env python3
"""Quantify the accuracy cost of INT8 quantization for PatchCore.

Usage::

    python scripts/compare_int8_accuracy.py                    # bottle, 10 defective frames
    python scripts/compare_int8_accuracy.py --num-images 20

Latency is only half the quantization story; the other half is what you give up
for it. This scores three PatchCore backends — PyTorch, ONNX FP32, ONNX INT8 —
on the *same* frames with the *same* Step 5 metrics (:func:`app.evaluation.metrics.pixel_auroc`
and :func:`~app.evaluation.metrics.au_pro`) and reports the drop from the PyTorch
baseline. FP32 is included as a control: it should match PyTorch almost exactly,
so any real gap in the table is INT8's, not the export's.

Frames are drawn the way Step 5's :class:`~app.evaluation.benchmark.BenchmarkRunner`
draws them — from the datamodule at ``image_size``, defective ones only (they are
the frames with a ground-truth mask, hence the only ones the pixel metrics can
score) — so these numbers sit on the same footing as the accuracy benchmark.
Results are written to ``results/int8_accuracy_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow `python scripts/compare_int8_accuracy.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import DataModule  # noqa: E402
from app.evaluation.metrics import au_pro, image_auroc, pixel_auroc  # noqa: E402
from app.models import AnomalyModel, ModelConfig, ONNXRunner, PatchCoreModel, onnx_artifact_path  # noqa: E402
from app.models.onnx_runner import DEFAULT_EXPORTED_DIR  # noqa: E402

logger = logging.getLogger("compare_int8_accuracy")

_MODEL = "patchcore"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category (default: bottle).")
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="How many defective frames to score (default: 10; a quick, representative subset).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Evaluation resolution (default: 256).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument("--exported-dir", type=Path, default=None, help="ONNX artifact root (default: results/exported).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint root (default: results/checkpoints).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Where to write the JSON (default: ./results).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _defective_frames(
    datamodule: DataModule,
    limit: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pull up to ``limit`` defective frames and their masks, as Step 5 does.

    Returns ``(images, masks)`` where each image is an ``(H, W, 3)`` RGB array and
    each mask a binary ``(H, W)`` array at the same resolution — exactly the pair
    the pixel metrics compare.
    """
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in datamodule.test_dataloader():
        if batch.mask is None:
            continue
        batch_images = batch.image.detach().cpu().numpy()
        batch_labels = batch.label.detach().cpu().numpy()
        batch_masks = batch.mask.detach().cpu().numpy()
        for i in range(len(batch_labels)):
            if int(batch_labels[i]) != 1:  # defective only: normal frames have no defect pixels to score
                continue
            images.append(np.ascontiguousarray(batch_images[i].transpose(1, 2, 0)))
            masks.append((np.asarray(batch_masks[i]) > 0).astype(np.uint8))
            if len(images) >= limit:
                return images, masks
    return images, masks


def _score_backend(model: AnomalyModel, images: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    """Run one backend over the frames and compute the Step 5 metrics."""
    scores: list[float] = []
    pred_maps: list[np.ndarray] = []
    for image in images:
        out = model.predict(image, color_order="rgb")
        scores.append(out.anomaly_score)
        pred_maps.append(np.asarray(out.anomaly_map, dtype=np.float32))
    # Every frame here is defective, so image_auroc has one class only and is nan;
    # it is reported for completeness but the pixel metrics are the story.
    return {
        "image_auroc": image_auroc([1] * len(scores), scores),
        "pixel_auroc": pixel_auroc(masks, pred_maps),
        "au_pro": au_pro(masks, pred_maps),
        "mean_score": float(np.mean(scores)),
    }


def _build_backends(config: ModelConfig, exported_dir: Path) -> dict[str, AnomalyModel]:
    """PyTorch baseline plus the two ONNX backends, all sharing one config."""
    checkpoint = config.checkpoint_path(_MODEL, config.category)
    if not checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {checkpoint}. Train it with `python scripts/train_{_MODEL}.py`.")
    pytorch = PatchCoreModel(config=config)
    pytorch.load(checkpoint)

    backends: dict[str, AnomalyModel] = {"PyTorch": pytorch}
    for precision, label in (("fp32", "ONNX FP32"), ("int8", "ONNX INT8")):
        path = onnx_artifact_path(_MODEL, precision, exported_dir)
        if not path.is_file():
            raise SystemExit(f"No ONNX artifact at {path}. Export it with `python scripts/export_onnx.py`.")
        backends[label] = ONNXRunner(path, model_name=f"{_MODEL}_{precision}", config=config)
    return backends


def _print_table(rows: dict[str, dict[str, float]], baseline: str) -> None:
    """Print pixel-AUROC and AU-PRO per backend, with the delta from the baseline."""
    base = rows[baseline]
    header = f"{'Backend':<14} | {'Px-AUROC':>9} | {'AU-PRO':>8} | {'ΔPx-AUROC':>10} | {'ΔAU-PRO':>9}"
    print()
    print(header)
    print(f"{'-' * 14}-|-{'-' * 9}-|-{'-' * 8}-|-{'-' * 10}-|-{'-' * 9}")
    for label, metrics in rows.items():
        d_px = metrics["pixel_auroc"] - base["pixel_auroc"]
        d_pro = metrics["au_pro"] - base["au_pro"]
        delta_px = "  baseline" if label == baseline else f"{d_px * 100:+9.2f}pp"
        delta_pro = " baseline" if label == baseline else f"{d_pro * 100:+8.2f}pp"
        print(
            f"{label:<14} | {metrics['pixel_auroc'] * 100:>8.2f}% | {metrics['au_pro'] * 100:>7.2f}% | "
            f"{delta_px:>10} | {delta_pro:>9}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    config = ModelConfig.from_env(
        category=args.category,
        image_size=args.image_size,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        accelerator="cpu",
    )
    exported_dir = args.exported_dir or DEFAULT_EXPORTED_DIR

    datamodule = DataModule(
        category=config.category,
        image_size=config.image_size,
        batch_size=config.batch_size,
        root=config.data_root,
        num_workers=config.num_workers,
    )
    try:
        datamodule.setup()
    except FileNotFoundError as exc:
        logger.error("%s", exc)  # noqa: TRY400 - the traceback adds nothing here
        return 1

    images, masks = _defective_frames(datamodule, args.num_images)
    if not images:
        raise SystemExit(f"No defective frames with masks found for {config.category!r}.")

    print(f"Category   : {config.category}")
    print(f"Frames     : {len(images)} defective (with masks), at {config.image_size}x{config.image_size}")
    print(f"Metrics    : pixel-AUROC and AU-PRO (same as Step 5)")

    backends = _build_backends(config, exported_dir)
    rows: dict[str, dict[str, float]] = {}
    for label, model in backends.items():
        logger.info("Scoring %s ...", label)
        rows[label] = _score_backend(model, images, masks)

    _print_table(rows, baseline="PyTorch")

    payload = {
        "category": config.category,
        "image_size": config.image_size,
        "num_images": len(images),
        "model": _MODEL,
        "baseline": "PyTorch",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": rows,
    }
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "int8_accuracy_comparison.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
