#!/usr/bin/env python3
"""Benchmark PatchCore, EfficientAD and WinCLIP on one MVTec AD category.

Usage::

    python scripts/run_benchmark.py                                 # bottle, all three models
    python scripts/run_benchmark.py --category bottle --accelerator cpu
    python scripts/run_benchmark.py --models patchcore,winclip      # a subset
    python scripts/run_benchmark.py --force-retrain                 # ignore existing checkpoints

For each requested model the script gets an inference-ready instance — loading an
existing checkpoint when one is present, training otherwise — then runs
:class:`~app.evaluation.benchmark.BenchmarkRunner` over the category's test split
and prints a comparison table:

    Model        | Img-AUROC | Px-AUROC | AU-PRO  | Best-F1
    -------------|-----------|----------|---------|--------
    PatchCore    |   99.4%   |  97.8%   |  93.1%  |  0.984
    ...

The full results (plus run metadata) are written to
``results/benchmark_<category>_<timestamp>.json``; runs accumulate, never
overwrite.

Notes on the three backends:

* **PatchCore / EfficientAD** need a fitted checkpoint. If one exists at
  ``results/checkpoints/<model>_<category>.ckpt`` it is loaded; otherwise the
  model is trained first (EfficientAD's one-epoch fit is a smoke-trained model,
  not a converged one — see its wrapper docstring).
* **WinCLIP** is zero-shot: it is benchmarked straight from the prompt ensemble
  with no training and no calibration, which is the honest headline number. (A
  calibrated WinCLIP would be fitted on the very test split it is then scored on;
  see ``WinCLIPModel.train``.) Pass ``--calibrate-winclip`` to opt into that.

Every flag defaults to :class:`~app.models.config.ModelConfig`, which resolves
values from the environment and finally from a hardcoded default, so nothing is
duplicated between here and the config.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

# Allow `python scripts/run_benchmark.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import DataModule  # noqa: E402
from app.evaluation import BenchmarkRunner  # noqa: E402
from app.models import (  # noqa: E402
    AnomalyModel,
    EfficientADModel,
    ModelConfig,
    PatchCoreModel,
    WinCLIPModel,
)

logger = logging.getLogger("run_benchmark")

#: Registry order is the order the table prints in and the order models train in.
_ALL_MODELS = ("patchcore", "efficientad", "winclip")

#: model_name -> the display label used in the printed table.
_DISPLAY = {"patchcore": "PatchCore", "efficientad": "EfficientAD", "winclip": "WinCLIP"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category to benchmark (default: bottle).")
    parser.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated subset of {','.join(_ALL_MODELS)} (default: all three).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Evaluation resolution (default: 256).")
    parser.add_argument("--batch-size", type=int, default=None, help="Test dataloader batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (default: 0).")
    parser.add_argument("--backbone", default=None, help="PatchCore timm backbone (default: wide_resnet50_2).")
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=None,
        help="PatchCore memory-bank fraction, used only when training (default: 0.1).",
    )
    parser.add_argument("--num-neighbors", type=int, default=None, help="PatchCore scoring neighbours (default: 9).")
    parser.add_argument("--max-epochs", type=int, default=None, help="EfficientAD training epochs (default: 1).")
    parser.add_argument(
        "--accelerator",
        default=None,
        choices=("auto", "cpu", "gpu", "mps"),
        help="Lightning accelerator for any training/inference (default: auto).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Where to write the report (default: ./results).")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain PatchCore/EfficientAD even if a checkpoint exists.",
    )
    parser.add_argument(
        "--calibrate-winclip",
        action="store_true",
        help="Calibrate WinCLIP on the test split before scoring (optimistic; off by default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _resolve_models(raw: str | None) -> list[str]:
    """Parse the ``--models`` list, validating names against the registry."""
    if raw is None:
        return list(_ALL_MODELS)
    requested = [name.strip().lower() for name in raw.split(",") if name.strip()]
    unknown = [name for name in requested if name not in _ALL_MODELS]
    if unknown:
        msg = f"Unknown model(s) {unknown}; choose from {list(_ALL_MODELS)}."
        raise SystemExit(msg)
    # Preserve registry order regardless of how the user listed them.
    return [name for name in _ALL_MODELS if name in requested]


def _prepare_model(
    name: str,
    config: ModelConfig,
    datamodule: DataModule,
    *,
    force_retrain: bool,
    calibrate_winclip: bool,
) -> AnomalyModel:
    """Return an inference-ready model, training or loading as needed.

    PatchCore and EfficientAD load an existing checkpoint unless ``force_retrain``
    is set or none exists, in which case they train first. WinCLIP is zero-shot
    and needs neither, unless ``--calibrate-winclip`` opts into a calibration
    pass over the labelled split.
    """
    if name == "winclip":
        model: AnomalyModel = WinCLIPModel(config=config)
        if calibrate_winclip:
            logger.info("Calibrating WinCLIP on the %r test split (optimistic — see the script docstring).", config.category)
            model.train(datamodule)
        else:
            logger.info("WinCLIP: pure zero-shot, no training or calibration.")
        return model

    if name == "patchcore":
        model = PatchCoreModel(config=config)
    else:
        model = EfficientADModel(config=config)

    checkpoint = config.checkpoint_path(name, config.category)
    if checkpoint.is_file() and not force_retrain:
        logger.info("%s: loading existing checkpoint %s", name, checkpoint)
        model.load(checkpoint)
    else:
        reason = "forced retrain" if force_retrain else f"no checkpoint at {checkpoint}"
        logger.info("%s: training (%s)", name, reason)
        started = time.perf_counter()
        model.train(datamodule)
        logger.info("%s: trained in %.1fs", name, time.perf_counter() - started)
    return model


def _fmt_pct(value: float) -> str:
    """A ratio in [0, 1] as ``xx.x%``, or ``n/a`` when the metric was undefined."""
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:.1f}%"


def _fmt_f1(value: float) -> str:
    """An F1 as ``x.xxx``, or ``n/a`` when undefined."""
    return "n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def _print_table(results: dict[str, dict[str, object]]) -> None:
    """Print the fixed-width comparison table the spec asks for."""
    header = f"{'Model':<12} | {'Img-AUROC':^9} | {'Px-AUROC':^8} | {'AU-PRO':^7} | {'Best-F1':^7}"
    rule = f"{'-' * 12}-|-{'-' * 9}-|-{'-' * 8}-|-{'-' * 7}-|-{'-' * 7}"
    print()
    print(header)
    print(rule)
    for name, metrics in results.items():
        label = _DISPLAY.get(name, name)
        row = (
            f"{label:<12} | "
            f"{_fmt_pct(metrics['image_auroc']):^9} | "
            f"{_fmt_pct(metrics['pixel_auroc']):^8} | "
            f"{_fmt_pct(metrics['au_pro']):^7} | "
            f"{_fmt_f1(metrics['best_f1']):^7}"
        )
        print(row)
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    model_names = _resolve_models(args.models)

    config = ModelConfig.from_env(
        category=args.category,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone=args.backbone,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        data_root=args.data_root,
        results_dir=args.results_dir,
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
    print(f"Models     : {', '.join(_DISPLAY.get(n, n) for n in model_names)}")
    print(f"Image size : {config.image_size}x{config.image_size}")
    print(f"Test set   : {counts['test']} images")
    print()

    models: list[AnomalyModel] = []
    for name in model_names:
        model = _prepare_model(
            name,
            config,
            datamodule,
            force_retrain=args.force_retrain,
            calibrate_winclip=args.calibrate_winclip,
        )
        models.append(model)

    runner = BenchmarkRunner(models, datamodule, results_dir=config.results_dir)
    started = time.perf_counter()
    results = runner.run()
    elapsed = time.perf_counter() - started

    _print_table(results)
    print(f"Benchmarked {len(models)} model(s) in {elapsed:.1f}s.")
    print(f"Full report: {config.results_dir}/benchmark_{config.category}_<timestamp>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
