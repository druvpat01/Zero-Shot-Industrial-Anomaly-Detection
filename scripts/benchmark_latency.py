#!/usr/bin/env python3
"""Measure per-image latency and FPS for the PyTorch and ONNX inference paths.

Usage::

    python scripts/benchmark_latency.py                        # all backends, bottle
    python scripts/benchmark_latency.py --num-images 100
    python scripts/benchmark_latency.py --backends patchcore_pytorch,patchcore_onnx_int8

Runs the same set of raw test frames through five backends and reports the
per-image latency distribution of each ``predict`` call end to end — the input
guard, preprocessing, the model, and the upsample back to input resolution, i.e.
exactly what the serving layer pays per request:

    PatchCore   PyTorch  / ONNX FP32 / ONNX INT8
    EfficientAD PyTorch  / ONNX FP32

Every backend sees the byte-identical list of frames, so differences in the
table are differences between *backends*, not between inputs. Results are written
to ``results/latency_benchmark.json`` with a timestamp and host details; the file
is overwritten each run (it is a current-hardware snapshot, not a historical
series like the accuracy benchmark).

FPS here is sustained single-image throughput, ``1000 / mean_ms`` — the rate the
backend holds scoring frames one at a time, which is how an inspection endpoint
serves them.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow `python scripts/benchmark_latency.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.models import (  # noqa: E402
    AnomalyModel,
    EfficientADModel,
    ModelConfig,
    ONNXRunner,
    PatchCoreModel,
    onnx_artifact_path,
)
from app.models.onnx_runner import DEFAULT_EXPORTED_DIR  # noqa: E402

logger = logging.getLogger("benchmark_latency")

#: Default cap on how many test frames to time. The table is stable well below
#: this; more only tightens the tail percentiles.
_DEFAULT_NUM_IMAGES = 100

#: Backend registry. Order is the order the table prints in. Each entry knows how
#: to build its model from a config; ONNX ones also carry the artifact precision
#: so a missing export is skipped with a clear message rather than crashing.
_BACKENDS: dict[str, dict[str, Any]] = {
    "patchcore_pytorch": {"label": "PatchCore PyTorch", "kind": "torch", "model": "patchcore"},
    "patchcore_onnx_fp32": {"label": "PatchCore ONNX FP32", "kind": "onnx", "model": "patchcore", "precision": "fp32"},
    "patchcore_onnx_int8": {"label": "PatchCore ONNX INT8", "kind": "onnx", "model": "patchcore", "precision": "int8"},
    "efficientad_pytorch": {"label": "EfficientAD PyTorch", "kind": "torch", "model": "efficientad"},
    "efficientad_onnx_fp32": {"label": "EfficientAD ONNX FP32", "kind": "onnx", "model": "efficientad", "precision": "fp32"},
}

_TORCH_WRAPPERS: dict[str, type[AnomalyModel]] = {"patchcore": PatchCoreModel, "efficientad": EfficientADModel}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None, help="MVTec AD category to benchmark (default: bottle).")
    parser.add_argument(
        "--backends",
        default=None,
        help=f"Comma-separated subset of {','.join(_BACKENDS)} (default: all five).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=_DEFAULT_NUM_IMAGES,
        help=f"How many test frames to time (default: {_DEFAULT_NUM_IMAGES}; capped at what the split has).",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warm-up iterations per backend (default: 5).")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Intra-op threads for BOTH torch and onnxruntime, so neither oversubscribes the "
        "other (default: torch's own choice, typically the physical core count).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: ./data/MVTecAD).")
    parser.add_argument("--exported-dir", type=Path, default=None, help="Where ONNX artifacts live (default: results/exported).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Where checkpoints live (default: results/checkpoints).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Where to write the JSON (default: ./results).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _resolve_backends(raw: str | None) -> list[str]:
    if raw is None:
        return list(_BACKENDS)
    requested = [name.strip().lower() for name in raw.split(",") if name.strip()]
    unknown = [name for name in requested if name not in _BACKENDS]
    if unknown:
        raise SystemExit(f"Unknown backend(s) {unknown}; choose from {list(_BACKENDS)}.")
    return [name for name in _BACKENDS if name in requested]


def _load_test_frames(category_dir: Path, limit: int) -> list[np.ndarray]:
    """Load up to ``limit`` raw BGR test frames, defect classes and good alike.

    Frames are the original ~900x900 PNGs, not pre-resized: every backend does
    its own resize inside ``predict``, so timing them at full resolution keeps
    the preprocessing cost in the measurement where it belongs.
    """
    paths = sorted(category_dir.glob("test/*/*.png"))
    if not paths:
        raise SystemExit(
            f"No test images under {category_dir}/test/. Run "
            f"`python scripts/download_dataset.py --category {category_dir.name}` first."
        )
    frames = []
    for path in paths[:limit]:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("OpenCV could not read %s; skipping.", path)
            continue
        frames.append(frame)
    return frames


def _build_backend(spec: dict[str, Any], config: ModelConfig, exported_dir: Path, num_threads: int) -> AnomalyModel | None:
    """Instantiate one backend, or return ``None`` if its artifact is absent."""
    if spec["kind"] == "torch":
        checkpoint = config.checkpoint_path(spec["model"], config.category)
        if not checkpoint.is_file():
            logger.warning("Skipping %s: no checkpoint at %s.", spec["label"], checkpoint)
            return None
        model = _TORCH_WRAPPERS[spec["model"]](config=config)
        model.load(checkpoint)
        return model

    onnx_path = onnx_artifact_path(spec["model"], spec["precision"], exported_dir)
    if not onnx_path.is_file():
        logger.warning("Skipping %s: no ONNX artifact at %s (run scripts/export_onnx.py).", spec["label"], onnx_path)
        return None
    return ONNXRunner(
        onnx_path,
        model_name=f"{spec['model']}_{spec['precision']}",
        config=config,
        num_threads=num_threads,
    )


def _time_backend(model: AnomalyModel, frames: list[np.ndarray], warmup: int) -> dict[str, float]:
    """Time ``predict`` over every frame and summarise the latency distribution.

    A handful of warm-up calls are run and discarded first: onnxruntime lazily
    optimises and allocates on its first ``run``, and torch has its own one-off
    costs, so an un-warmed first call would land in the tail and skew p99.
    """
    for i in range(warmup):
        model.predict(frames[i % len(frames)], color_order="bgr")

    latencies_ms: list[float] = []
    for frame in frames:
        started = time.perf_counter()
        model.predict(frame, color_order="bgr")
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    latencies = np.asarray(latencies_ms)
    p50, p95, p99 = (float(v) for v in np.percentile(latencies, [50, 95, 99]))
    mean = float(latencies.mean())
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "fps": 1000.0 / mean if mean > 0 else float("nan"),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the latency table in the spec's layout, plus a p99 column."""
    header = f"{'Backend':<21} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | {'p99 (ms)':>8} | {'FPS':>5}"
    rule = f"{'-' * 21}-|-{'-' * 8}-|-{'-' * 8}-|-{'-' * 8}-|-{'-' * 5}"
    print()
    print(header)
    print(rule)
    for row in rows:
        s = row["stats"]
        print(
            f"{row['label']:<21} | {s['p50_ms']:>8.1f} | {s['p95_ms']:>8.1f} | "
            f"{s['p99_ms']:>8.1f} | {s['fps']:>5.1f}"
        )
    print()


def _host_info() -> dict[str, Any]:
    """Hardware/runtime context, so a latency number is reproducible from the JSON."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")

    backend_keys = _resolve_backends(args.backends)
    config = ModelConfig.from_env(
        category=args.category,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )
    exported_dir = args.exported_dir or DEFAULT_EXPORTED_DIR

    # Pin both runtimes to the same thread budget so neither oversubscribes the
    # other on a hyperthreaded CPU — the difference is large and would otherwise
    # penalise whichever backend runs second. Default to torch's own choice
    # (the physical core count), then hand the same number to onnxruntime.
    num_threads = args.threads or torch.get_num_threads()
    torch.set_num_threads(num_threads)

    frames = _load_test_frames(config.data_root / config.category, args.num_images)
    print(f"Category   : {config.category}")
    print(f"Frames     : {len(frames)} (requested {args.num_images}), warm-up {args.warmup}")
    print(f"Backends   : {', '.join(_BACKENDS[k]['label'] for k in backend_keys)}")
    print(f"Threads    : torch=onnxruntime={num_threads}")
    print()

    rows: list[dict[str, Any]] = []
    for key in backend_keys:
        spec = _BACKENDS[key]
        model = _build_backend(spec, config, exported_dir, num_threads)
        if model is None:
            continue
        logger.info("Timing %s over %d frames ...", spec["label"], len(frames))
        stats = _time_backend(model, frames, args.warmup)
        rows.append({"key": key, "label": spec["label"], "stats": stats})
        logger.info(
            "%s: p50=%.1fms p95=%.1fms p99=%.1fms mean=%.1fms -> %.1f FPS",
            spec["label"],
            stats["p50_ms"],
            stats["p95_ms"],
            stats["p99_ms"],
            stats["mean_ms"],
            stats["fps"],
        )

    if not rows:
        raise SystemExit("No backends could be benchmarked; export artifacts and train checkpoints first.")

    _print_table(rows)

    payload = {
        "category": config.category,
        "image_size": config.image_size,
        "num_images": len(frames),
        "warmup": args.warmup,
        "num_threads": num_threads,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": _host_info(),
        "results": {row["key"]: {"label": row["label"], **row["stats"]} for row in rows},
    }
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "latency_benchmark.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
