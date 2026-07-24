#!/usr/bin/env python3
"""Export trained PatchCore / EfficientAD checkpoints to ONNX, then INT8-quantize them.

Usage::

    python scripts/export_onnx.py                              # both models, bottle
    python scripts/export_onnx.py --models patchcore
    python scripts/export_onnx.py --category cable --no-quantize
    python scripts/export_onnx.py --exported-dir results/exported

For each model this:

1. Loads the trained checkpoint through the PyTorch wrapper (so the resolution,
   backbone and calibration are exactly the ones it was fitted with).
2. Exports it with anomalib's built-in ``Engine.export(export_type=ONNX)``. The
   exported graph is the *whole* ``forward`` — resize, normalization and score
   calibration are baked in — which is why :class:`~app.models.onnx_runner.ONNXRunner`
   feeds it a plain ``[0, 1]`` frame and does no normalization of its own.
3. Dynamically quantizes the FP32 graph to INT8 with
   ``onnxruntime.quantization.quantize_dynamic`` (skip with ``--no-quantize``).

Artifacts land under ``results/exported/weights/onnx/`` (anomalib's own layout)::

    <model>.onnx        # FP32
    <model>_int8.onnx   # INT8, weights dynamically quantized

and :func:`~app.models.onnx_runner.onnx_artifact_path` is the single place that
names them, so the benchmark, the tests and the serving layer all locate an
artifact from the model name and precision alone.

WinCLIP is deliberately not exported
------------------------------------
WinCLIP is skipped, and the reason is worth stating plainly rather than hiding:
its CLIP ViT backbone is a poor ONNX target. WinCLIP scores by sliding windows
of *variable* count over the patch grid and pooling their CLIP embeddings, so the
number of attention operations depends on the input — dynamic control flow and
dynamic-shape attention that the (traced) ONNX exporter turns into a graph
specialised to one shape, if it exports at all. The honest engineering call is
that PatchCore and EfficientAD are the two models worth serving through ONNX;
WinCLIP stays on the PyTorch path where its zero-shot flexibility is the point.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python scripts/export_onnx.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomalib.deploy import ExportType  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

from app.models import (  # noqa: E402
    EfficientADModel,
    ModelConfig,
    PatchCoreModel,
    onnx_artifact_path,
)
from app.models.onnx_runner import DEFAULT_EXPORTED_DIR, ONNXRunner  # noqa: E402

logger = logging.getLogger("export_onnx")

#: model_name -> the wrapper class that loads its checkpoint. WinCLIP is absent
#: on purpose (see the module docstring).
_EXPORTABLE: dict[str, type] = {"patchcore": PatchCoreModel, "efficientad": EfficientADModel}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated subset of {','.join(_EXPORTABLE)} (default: both).",
    )
    parser.add_argument("--category", default=None, help="MVTec AD category whose checkpoint to export (default: bottle).")
    parser.add_argument("--image-size", type=int, default=None, help="Export resolution (default: the config's 256).")
    parser.add_argument(
        "--exported-dir",
        type=Path,
        default=None,
        help=f"Export root (default: {DEFAULT_EXPORTED_DIR}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Where to read checkpoints from (default: results/checkpoints).",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Export FP32 only; skip INT8 quantization.")
    parser.add_argument(
        "--accelerator",
        default="cpu",
        choices=("auto", "cpu", "gpu", "mps"),
        help="Accelerator anomalib exports from (default: cpu, the ONNX serving target).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _resolve_models(raw: str | None) -> list[str]:
    """Parse ``--models``, validating names and rejecting WinCLIP with the why."""
    if raw is None:
        return list(_EXPORTABLE)
    requested = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if "winclip" in requested:
        raise SystemExit(
            "WinCLIP is not exportable to ONNX: its CLIP ViT uses dynamic, input-dependent "
            "sliding-window attention that the ONNX exporter cannot capture. See the docstring "
            "of scripts/export_onnx.py. Export patchcore and/or efficientad instead."
        )
    unknown = [name for name in requested if name not in _EXPORTABLE]
    if unknown:
        raise SystemExit(f"Unknown or non-exportable model(s) {unknown}; choose from {list(_EXPORTABLE)}.")
    return [name for name in _EXPORTABLE if name in requested]


def _load_wrapper(name: str, config: ModelConfig) -> Any:
    """Load a trained wrapper, or fail with the exact command that trains one."""
    wrapper = _EXPORTABLE[name](config=config)
    checkpoint = config.checkpoint_path(name, config.category)
    if not checkpoint.is_file():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train one first with "
            f"`python scripts/train_{name}.py --category {config.category}`."
        )
    wrapper.load(checkpoint)
    return wrapper


def _export_one(name: str, config: ModelConfig, exported_dir: Path, *, quantize: bool) -> dict[str, Any]:
    """Export and (optionally) quantize one model; return a row for the summary."""
    wrapper = _load_wrapper(name, config)

    engine = Engine(accelerator=config.accelerator, devices=config.devices, logger=False)
    started = time.perf_counter()
    # `_module` is the anomalib LightningModule the wrapper caches; the training
    # scripts read it the same way. anomalib exports its whole forward().
    produced = engine.export(
        model=wrapper._module,  # noqa: SLF001 - the wrapper's cached anomalib module, as in scripts/train_*.py
        export_type=ExportType.ONNX,
        export_root=str(exported_dir),
        model_file_name=name,
        input_size=config.image_hw,
    )
    export_seconds = time.perf_counter() - started

    fp32_path = onnx_artifact_path(name, "fp32", exported_dir)
    if Path(produced) != fp32_path:
        # anomalib's own layout should already match onnx_artifact_path; guard
        # against a future change silently splitting the two conventions apart.
        logger.warning("Exporter wrote %s but the convention expects %s.", produced, fp32_path)
        fp32_path = Path(produced)
    fp32_mb = fp32_path.stat().st_size / 1024**2
    logger.info("Exported %s FP32 to %s (%.1f MB) in %.1fs", name, fp32_path, fp32_mb, export_seconds)

    row: dict[str, Any] = {"model": name, "fp32_path": fp32_path, "fp32_mb": fp32_mb, "int8_path": None, "int8_mb": None}

    if quantize:
        int8_path = onnx_artifact_path(name, "int8", exported_dir)
        started = time.perf_counter()
        # Dynamic quantization: weights -> INT8, activations quantized on the fly
        # at inference. No calibration dataset needed, unlike static PTQ. A benign
        # warning about one un-quantizable Slice tensor is expected and harmless.
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        int8_mb = int8_path.stat().st_size / 1024**2
        logger.info(
            "Quantized %s to INT8 at %s (%.1f MB, %.0f%% of FP32) in %.1fs",
            name,
            int8_path,
            int8_mb,
            100 * int8_mb / fp32_mb,
            time.perf_counter() - started,
        )
        row["int8_path"], row["int8_mb"] = int8_path, int8_mb

    # Verify each artifact loads and exposes the expected outputs before we claim success.
    for precision in ("fp32", "int8") if quantize else ("fp32",):
        ONNXRunner(row[f"{precision}_path"], model_name=f"{name}_{precision}", config=config)
    logger.info("Verified %s artifact(s) load through ONNXRunner.", name)
    return row


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact table of what was written and how big it is."""
    print()
    print(f"{'Model':<12} | {'FP32 (MB)':>10} | {'INT8 (MB)':>10} | Artifacts")
    print(f"{'-' * 12}-|-{'-' * 10}-|-{'-' * 10}-|-{'-' * 9}")
    for row in rows:
        int8 = f"{row['int8_mb']:>10.1f}" if row["int8_mb"] is not None else f"{'-':>10}"
        print(f"{row['model']:<12} | {row['fp32_mb']:>10.1f} | {int8} | {row['fp32_path'].parent}/")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    model_names = _resolve_models(args.models)
    exported_dir = args.exported_dir or DEFAULT_EXPORTED_DIR

    config = ModelConfig.from_env(
        category=args.category,
        image_size=args.image_size,
        accelerator=args.accelerator,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"Category    : {config.category}")
    print(f"Models      : {', '.join(model_names)}")
    print(f"Resolution  : {config.image_size}x{config.image_size}")
    print(f"Exported to : {exported_dir}")
    print(f"Quantize    : {not args.no_quantize}")
    print("WinCLIP     : skipped (CLIP ViT has dynamic-shape attention; see the script docstring)")
    print()

    rows = [
        _export_one(name, config, exported_dir, quantize=not args.no_quantize)
        for name in model_names
    ]

    _print_summary(rows)
    print(f"Exported {len(rows)} model(s). Run `python scripts/benchmark_latency.py` to time them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
