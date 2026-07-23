#!/usr/bin/env python3
"""Train EfficientAD on one MVTec AD category and save the checkpoint.

Usage::

    python scripts/train_efficientad.py                                # bottle, defaults
    python scripts/train_efficientad.py --category bottle --epochs 20
    python scripts/train_efficientad.py --category cable --model-size medium

Every flag is optional and defaults to :class:`~app.models.config.ModelConfig`,
which resolves each value from the environment (or ``.env``) and finally from a
hardcoded default. Flags therefore override the environment, not the other way
round, and nothing is duplicated between here and the config.

Note on ``--epochs``: unlike ``scripts/train_patchcore.py``, this one means it.
EfficientAD is gradient-trained and the paper's schedule is ~70k steps; at one
image per step that is hundreds of epochs on a typical MVTec category. The
default of 1 exists so a run finishes in minutes and proves the pipeline works
— it does not produce a converged model, and the script says so when it ends.

First run only, this downloads two things anomalib caches for good: the
pretrained teacher weights (~40 MB) and ImageNette (~1.5 GB, to
``--imagenet-dir``), which EfficientAD needs to stop the student generalising
beyond the training distribution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/train_efficientad.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import DataModule  # noqa: E402
from app.models import EfficientADModel, ModelConfig  # noqa: E402

logger = logging.getLogger("train_efficientad")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category to train on (default: bottle).")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs; EfficientAD honours this.")
    parser.add_argument(
        "--model-size",
        default=None,
        choices=("small", "medium"),
        help="Patch Description Network size (default: small).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square input resolution (default: 256, which is also the minimum the autoencoder accepts).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Eval batch size. EfficientAD pins the *train* loader to 1 regardless (default: 8).",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (default: 0).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=None,
        help="ImageNette root for the distillation penalty (default: ./data/imagenette).",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        choices=("auto", "cpu", "gpu", "mps"),
        help="Lightning accelerator (default: auto).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path (default: results/checkpoints/efficientad_<category>.ckpt).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    config = ModelConfig.from_env(
        category=args.category,
        model_size=args.model_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        data_root=args.data_root,
        imagenet_dir=args.imagenet_dir,
    )

    datamodule = DataModule(
        category=config.category,
        image_size=config.image_size,
        batch_size=config.batch_size,
        root=config.data_root,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    try:
        datamodule.setup()
    except FileNotFoundError as exc:
        logger.error("%s", exc)  # noqa: TRY400 - the traceback adds nothing for the user here
        return 1

    counts = datamodule.num_samples()
    print(f"Category   : {config.category}")
    print(f"Model      : EfficientAD, {config.model_size} PDN (33x33 receptive field, stride 4)")
    print(f"Image size : {config.image_size}x{config.image_size}")
    print(f"Epochs     : {config.max_epochs}  (train batch size forced to 1)")
    print(f"ImageNette : {config.imagenet_dir}")
    print(f"Samples    : train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print()

    model = EfficientADModel(config=config)

    started = time.perf_counter()
    model.train(datamodule)
    elapsed = time.perf_counter() - started

    checkpoint = model.checkpoint_path
    if args.checkpoint is not None:
        model.save(args.checkpoint)
        checkpoint = model.checkpoint_path

    steps = config.max_epochs * counts["train"]
    minutes, seconds = divmod(elapsed, 60)

    print()
    print(f"Training time  : {elapsed:.1f}s ({int(minutes)}m {seconds:04.1f}s)")
    print(f"Optimizer steps: {steps} ({config.max_epochs} epoch(s) x {counts['train']} images)")
    print(f"Map quantiles  : {model.has_map_quantiles}")
    print(f"Calibrated     : {model.is_calibrated}")
    print(f"Checkpoint     : {checkpoint}")
    print(f"Checkpoint size: {checkpoint.stat().st_size / 1024**2:.1f} MB")

    # The paper trains for 70k steps. Anything far short of that predicts, but
    # its numbers should not be quoted against PatchCore's without this caveat.
    if steps < 10_000:
        print()
        print(
            f"NOTE: {steps} steps is well short of the paper's ~70k schedule. This checkpoint is "
            f"usable and will separate obvious defects, but it is not converged — raise --epochs "
            f"before comparing AUROC against PatchCore.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
