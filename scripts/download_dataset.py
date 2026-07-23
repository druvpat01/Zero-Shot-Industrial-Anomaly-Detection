#!/usr/bin/env python3
"""Download ONE MVTec AD category and lay it out the way anomalib expects.

Target layout (this is what ``app.data.DataModule`` reads)::

    data/MVTecAD/<category>/train/good/*.png
    data/MVTecAD/<category>/test/<defect_type>/*.png
    data/MVTecAD/<category>/ground_truth/<defect_type>/*_mask.png

Acquisition strategy
====================
Three sources are tried in order. Each one is a real fallback for a different
failure mode, not a retry of the same call:

1. ``hf-parquet`` (**primary**) — resolve the HuggingFace mirror's parquet
   shards for *this category only* with ``hf_hub_download`` and stream rows out
   of them.

   This is a deliberate optimisation over the obvious
   ``load_dataset("TheoM55/mvtec_all_objects_split", split="bottle.train")``.
   That repo publishes all 15 MVTec categories as splits of a single default
   config, and ``load_dataset`` materialises the whole config before slicing a
   split out of it — roughly 5.3 GB downloaded to obtain the ~157 MB of bottle
   images. Addressing ``data/bottle.{train,test}-*.parquet`` directly downloads
   only what we asked for.

2. ``hf-datasets`` — the literal ``load_dataset(..., split=f"{category}.train")``
   call. Kept because it only depends on the repo's declared split config, so it
   still works if the shard *filenames* are reorganised in a way that breaks the
   direct parquet addressing above. It is slow and bandwidth-hungry, hence
   second.

3. ``anomalib`` (**fallback for a dead mirror**) — hand off to anomalib's own
   ``MVTecAD`` datamodule, which downloads the official MVTec AD archive from
   mydrive.ch and extracts it into the output directory.

   This exists because sources 1 and 2 share a single point of failure: the
   HuggingFace Hub. If the Hub is unreachable, rate-limits us, or the mirror
   repo is renamed/deleted/gated, *neither* HF strategy can work, and without
   this path the project would have no way to obtain data at all. The trade-off
   is that the official archive is not per-category: it pulls ~4.9 GB covering
   all 15 categories and extracts every one of them, which is exactly why it is
   last. Its output for ``<output-dir>/<category>/`` is byte-for-byte the layout
   above, so downstream code cannot tell which source was used.

Usage::

    python scripts/download_dataset.py                        # bottle
    python scripts/download_dataset.py --category cable
    python scripts/download_dataset.py --source anomalib      # skip HF entirely
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

# Make `app` importable when this script is run directly from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CATEGORY = "bottle"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "MVTecAD"

HF_REPO_ID = "TheoM55/mvtec_all_objects_split"
SPLITS = ("train", "test")
SOURCES = ("auto", "hf-parquet", "hf-datasets", "anomalib")

logger = logging.getLogger("download_dataset")


class Record(NamedTuple):
    """One dataset row, normalised across all three acquisition sources."""

    split: str  # "train" | "test"
    defect: str  # "good" for normal samples, else the defect type
    image_name: str  # e.g. "000.png"
    image_bytes: bytes
    mask_name: str | None  # e.g. "000_mask.png"; None when there is no mask
    mask_bytes: bytes | None


# ---------------------------------------------------------------------------
# Writing records to disk
# ---------------------------------------------------------------------------


def _write_records(records: Iterator[Record], category: str, output_dir: Path) -> dict[str, int]:
    """Write records into the anomalib folder layout. Returns per-bucket counts."""
    counts = {"train": 0, "test": 0, "ground_truth": 0}
    root = output_dir / category

    for record in records:
        image_dir = root / record.split / record.defect
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / record.image_name).write_bytes(record.image_bytes)
        counts[record.split] += 1

        if record.mask_bytes is None:
            continue
        # Only anomalous test samples carry ground truth; anomalib matches a
        # mask to an image by requiring the image stem to be a substring of the
        # mask stem (000.png <-> 000_mask.png), so the upstream name is kept.
        mask_dir = root / "ground_truth" / record.defect
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_name = record.mask_name or f"{Path(record.image_name).stem}_mask.png"
        (mask_dir / mask_name).write_bytes(record.mask_bytes)
        counts["ground_truth"] += 1

    return counts


def _record_from_row(row: dict[str, Any], fallback_split: str) -> Record:
    """Normalise one HF row (``{bytes, path}`` structs) into a :class:`Record`."""
    image = row["image_path"] or {}
    mask = row.get("mask_path") or {}

    image_bytes = image.get("bytes")
    if not image_bytes:
        msg = f"Row for defect {row.get('defect')!r} has no image bytes; the mirror may have changed format."
        raise ValueError(msg)

    return Record(
        # The `split` column is authoritative, but fall back to the shard we
        # read the row from if the mirror ever drops the column.
        split=str(row.get("split") or fallback_split),
        defect=str(row.get("defect") or "good"),
        image_name=Path(image.get("path") or "image.png").name,
        image_bytes=image_bytes,
        mask_name=Path(mask["path"]).name if mask.get("path") else None,
        mask_bytes=mask.get("bytes"),
    )


# ---------------------------------------------------------------------------
# Source 1 (primary): direct parquet shards from the HuggingFace mirror
# ---------------------------------------------------------------------------


def _resolve_category_shards(category: str) -> dict[str, list[str]]:
    """Map each split to the mirror's parquet shard paths for ``category``."""
    from huggingface_hub import HfApi

    repo_files = HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset")
    shards: dict[str, list[str]] = {}
    for split in SPLITS:
        prefix = f"data/{category}.{split}-"
        matches = sorted(f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet"))
        if not matches:
            msg = f"No parquet shards named '{prefix}*.parquet' in {HF_REPO_ID}; is {category!r} a valid category?"
            raise FileNotFoundError(msg)
        shards[split] = matches
    return shards


def _iter_parquet_records(shard_path: Path, split: str) -> Iterator[Record]:
    """Stream a parquet shard row by row so a category never sits in RAM whole."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(shard_path)
    for record_batch in parquet_file.iter_batches(batch_size=16):
        for row in record_batch.to_pylist():
            yield _record_from_row(row, fallback_split=split)


def download_via_hf_parquet(category: str, output_dir: Path) -> dict[str, int]:
    """PRIMARY path: fetch only ``<category>``'s parquet shards from the Hub."""
    from huggingface_hub import hf_hub_download

    shards = _resolve_category_shards(category)
    counts = {"train": 0, "test": 0, "ground_truth": 0}

    for split, filenames in shards.items():
        for filename in filenames:
            logger.info("Downloading %s from %s", filename, HF_REPO_ID)
            local_path = Path(hf_hub_download(HF_REPO_ID, filename, repo_type="dataset"))
            written = _write_records(_iter_parquet_records(local_path, split), category, output_dir)
            for key, value in written.items():
                counts[key] += value

    return counts


# ---------------------------------------------------------------------------
# Source 2: the mirror via `datasets.load_dataset`
# ---------------------------------------------------------------------------


def download_via_hf_datasets(category: str, output_dir: Path) -> dict[str, int]:
    """Secondary HF path: the split-config API rather than raw shard paths.

    Warning:
        ``load_dataset`` resolves the repo's whole default config before taking
        a split, so this downloads every category (~5.3 GB) to keep one. It is
        only reached when shard-path resolution fails.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset

    counts = {"train": 0, "test": 0, "ground_truth": 0}
    for split in SPLITS:
        logger.info("load_dataset(%r, split=%r)", HF_REPO_ID, f"{category}.{split}")
        dataset = load_dataset(HF_REPO_ID, split=f"{category}.{split}")
        # Keep the original PNG bytes and filenames instead of letting the
        # Image feature decode to PIL and lose the upstream name.
        dataset = dataset.cast_column("image_path", HFImage(decode=False))
        dataset = dataset.cast_column("mask_path", HFImage(decode=False))

        records = (_record_from_row(row, fallback_split=split) for row in dataset)
        written = _write_records(records, category, output_dir)
        for key, value in written.items():
            counts[key] += value

    return counts


# ---------------------------------------------------------------------------
# Source 3 (fallback): anomalib's own auto-download
# ---------------------------------------------------------------------------


def download_via_anomalib(category: str, output_dir: Path) -> dict[str, int]:
    """FALLBACK for an unusable HuggingFace mirror.

    Delegates to ``anomalib.data.MVTecAD.prepare_data()``, which fetches the
    official MVTec AD archive and extracts it into ``output_dir``. Because that
    archive is monolithic, this downloads ~4.9 GB and writes all 15 categories,
    of which we only need one — acceptable as a last resort, unacceptable as the
    default. The resulting ``output_dir/<category>/`` tree is identical to what
    the HF paths produce.

    This is the *only* place in the project other than ``app/data/datamodule.py``
    that touches anomalib, and it is a deliberate dependency inversion: the data
    layer normally feeds anomalib, but when our own source is down we let
    anomalib feed us.
    """
    from anomalib.data import MVTecAD

    logger.warning(
        "Falling back to anomalib's MVTec AD auto-download: ~4.9 GB covering all 15 categories.",
    )
    datamodule = MVTecAD(root=output_dir, category=category)
    datamodule.prepare_data()

    category_dir = output_dir / category
    if not category_dir.is_dir():
        msg = f"anomalib reported success but {category_dir} is missing."
        raise FileNotFoundError(msg)

    return summarize_layout(category, output_dir)


# ---------------------------------------------------------------------------
# Verification / CLI
# ---------------------------------------------------------------------------


def summarize_layout(category: str, output_dir: Path) -> dict[str, int]:
    """Count the PNGs under each top-level bucket of the category directory."""
    root = output_dir / category
    return {
        bucket: sum(1 for _ in (root / bucket).rglob("*.png")) if (root / bucket).is_dir() else 0
        for bucket in ("train", "test", "ground_truth")
    }


def verify_layout(category: str, output_dir: Path) -> dict[str, int]:
    """Check the downloaded tree is usable, raising with a specific reason if not."""
    root = output_dir / category
    counts = summarize_layout(category, output_dir)

    if not (root / "train" / "good").is_dir():
        msg = f"Missing {root / 'train' / 'good'}; the layout anomalib expects was not produced."
        raise FileNotFoundError(msg)
    for bucket in ("train", "test"):
        if counts[bucket] == 0:
            msg = f"No images found under {root / bucket}."
            raise FileNotFoundError(msg)

    return counts


def is_already_downloaded(category: str, output_dir: Path) -> bool:
    """Whether a usable copy of ``category`` is already on disk."""
    try:
        verify_layout(category, output_dir)
    except FileNotFoundError:
        return False
    return True


def _run_sources(category: str, output_dir: Path, source: str) -> tuple[str, dict[str, int]]:
    """Try the configured source(s) in order; return the one that worked."""
    strategies = {
        "hf-parquet": download_via_hf_parquet,
        "hf-datasets": download_via_hf_datasets,
        "anomalib": download_via_anomalib,
    }
    chain = list(strategies) if source == "auto" else [source]

    failures: list[str] = []
    for name in chain:
        logger.info("Trying source %r for category %r", name, category)
        try:
            strategies[name](category, output_dir)
            counts = verify_layout(category, output_dir)
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next source"
            logger.warning("Source %r failed: %s: %s", name, type(exc).__name__, exc)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        return name, counts

    msg = "All download sources failed:\n  - " + "\n  - ".join(failures)
    raise RuntimeError(msg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="MVTec AD object category to download (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination root; the category is written to <output-dir>/<category> (default: ./data/MVTecAD).",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="auto",
        help="Acquisition source. 'auto' tries hf-parquet, then hf-datasets, then anomalib (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing copy of the category and download it again.",
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

    output_dir: Path = args.output_dir.expanduser().resolve()
    category: str = args.category
    category_dir = output_dir / category

    if args.force and category_dir.exists():
        logger.info("--force given; removing %s", category_dir)
        shutil.rmtree(category_dir)

    if is_already_downloaded(category, output_dir):
        counts = summarize_layout(category, output_dir)
        logger.info("%s already present at %s; nothing to do (use --force to redownload).", category, category_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            source, counts = _run_sources(category, output_dir, args.source)
        except RuntimeError as exc:
            logger.error("%s", exc)  # noqa: TRY400 - the traceback adds nothing for the user here
            return 1
        logger.info("Downloaded %s via source %r", category, source)

    print(f"\nCategory : {category}")
    print(f"Location : {category_dir}")
    print(f"train images        : {counts['train']}")
    print(f"test images         : {counts['test']}")
    print(f"ground-truth masks  : {counts['ground_truth']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
