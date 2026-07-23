#!/usr/bin/env python3
"""Train PatchCore on one MVTec AD category and save the checkpoint.

Usage::

    python scripts/train_patchcore.py                                  # bottle, defaults
    python scripts/train_patchcore.py --category bottle --epochs 1
    python scripts/train_patchcore.py --category cable --coreset-sampling-ratio 0.01

Every flag is optional and defaults to :class:`~app.models.config.ModelConfig`,
which resolves each value from the environment (or ``.env``) and finally from a
hardcoded default. Flags therefore override the environment, not the other way
round, and nothing is duplicated between here and the config.

Note on ``--epochs``: PatchCore fits a memory bank in a single pass over the
training set and anomalib pins training to one epoch regardless. The flag is
accepted because the interface is shared with the models added later; anything
other than 1 logs a warning and is ignored.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/train_patchcore.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import DataModule  # noqa: E402
from app.models import ModelConfig, PatchCoreModel  # noqa: E402

logger = logging.getLogger("train_patchcore")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category to train on (default: bottle).")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs; PatchCore always uses 1.")
    parser.add_argument("--backbone", default=None, help="timm backbone (default: wide_resnet50_2).")
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=None,
        help="Fraction of patch embeddings kept in the memory bank (default: 0.1).",
    )
    parser.add_argument("--num-neighbors", type=int, default=None, help="Neighbours for scoring (default: 9).")
    parser.add_argument("--image-size", type=int, default=None, help="Square input resolution (default: 256).")
    parser.add_argument("--batch-size", type=int, default=None, help="Train batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (default: 0).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
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
        help="Explicit checkpoint path (default: results/checkpoints/patchcore_<category>.ckpt).",
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
        backbone=args.backbone,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        data_root=args.data_root,
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
    print(f"Backbone   : {config.backbone}  layers={list(config.layers)}")
    print(f"Image size : {config.image_size}x{config.image_size}")
    print(f"Coreset    : {config.coreset_sampling_ratio}  num_neighbors={config.num_neighbors}")
    print(f"Samples    : train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print()

    model = PatchCoreModel(config=config)

    started = time.perf_counter()
    model.train(datamodule)
    elapsed = time.perf_counter() - started

    checkpoint = model.checkpoint_path
    if args.checkpoint is not None:
        model.save(args.checkpoint)
        checkpoint = model.checkpoint_path

    bank = model._module.model.memory_bank  # noqa: SLF001 - reporting only
    minutes, seconds = divmod(elapsed, 60)

    print()
    print(f"Training time  : {elapsed:.1f}s ({int(minutes)}m {seconds:04.1f}s)")
    print(f"Memory bank    : {tuple(bank.shape)} ({bank.numel() * bank.element_size() / 1024**2:.1f} MB)")
    print(f"Calibrated     : {model.is_calibrated}")
    print(f"Checkpoint     : {checkpoint}")
    print(f"Checkpoint size: {checkpoint.stat().st_size / 1024**2:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
