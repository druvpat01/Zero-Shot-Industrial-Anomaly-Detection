"""Tests for the data layer: downloaded layout, datamodule, transforms.

The dataset-backed tests skip (rather than fail) when ``data/MVTecAD/bottle``
is absent, so a fresh clone can still run the suite. Populate it with::

    python scripts/download_dataset.py --category bottle
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image

from app.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DataModule,
    DefectBatch,
    denormalize_image,
    normalize_image,
    to_tensor,
    validate_image_shape,
)
from app.data.datamodule import DEFAULT_DATA_ROOT

CATEGORY = "bottle"
IMAGE_SIZE = 128  # deliberately not the 256 default, so the wrapper's resize is exercised
BATCH_SIZE = 4

CATEGORY_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY

requires_dataset = pytest.mark.skipif(
    not (CATEGORY_DIR / "train" / "good").is_dir(),
    reason=f"{CATEGORY_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)


@pytest.fixture(scope="module")
def download_script() -> ModuleType:
    """Import ``scripts/download_dataset.py``, which is not an installed package."""
    script_path = DEFAULT_DATA_ROOT.parents[1] / "scripts" / "download_dataset.py"
    spec = importlib.util.spec_from_file_location("download_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_layout(category: str, output_dir: Path) -> None:
    """Create the smallest tree that :func:`verify_layout` accepts."""
    pixel = Image.new("RGB", (4, 4))
    for relative in (f"{category}/train/good/000.png", f"{category}/test/broken/000.png"):
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixel.save(path)


@pytest.fixture(scope="module")
def datamodule() -> DataModule:
    """A single set-up datamodule shared by the dataset-backed tests."""
    dm = DataModule(category=CATEGORY, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=0)
    dm.setup()
    return dm


# ---------------------------------------------------------------------------
# 1. Downloaded directory structure
# ---------------------------------------------------------------------------


@requires_dataset
def test_category_directory_structure_exists() -> None:
    """The download script produced the tree anomalib expects."""
    assert CATEGORY_DIR.is_dir(), f"{CATEGORY_DIR} is missing"
    for subdir in ("train", "test", "ground_truth"):
        assert (CATEGORY_DIR / subdir).is_dir(), f"{CATEGORY_DIR / subdir} is missing"
    assert (CATEGORY_DIR / "train" / "good").is_dir(), "train split must contain a 'good' folder"


@requires_dataset
def test_at_least_one_train_and_test_image() -> None:
    """Both splits contain images, and train holds only normal samples."""
    train_images = list((CATEGORY_DIR / "train").rglob("*.png"))
    test_images = list((CATEGORY_DIR / "test").rglob("*.png"))

    assert len(train_images) >= 1, f"no train images under {CATEGORY_DIR / 'train'}"
    assert len(test_images) >= 1, f"no test images under {CATEGORY_DIR / 'test'}"

    train_classes = {p.parent.name for p in train_images}
    assert train_classes == {"good"}, f"train split must be defect-free, found {sorted(train_classes)}"


@requires_dataset
def test_ground_truth_masks_pair_with_defective_test_images() -> None:
    """Every defective test image has a matching mask; 'good' has none."""
    defect_dirs = [d for d in (CATEGORY_DIR / "test").iterdir() if d.is_dir() and d.name != "good"]
    assert defect_dirs, "expected at least one defect type in the test split"

    for defect_dir in defect_dirs:
        mask_dir = CATEGORY_DIR / "ground_truth" / defect_dir.name
        assert mask_dir.is_dir(), f"missing ground truth folder for defect {defect_dir.name!r}"

        image_stems = {p.stem for p in defect_dir.glob("*.png")}
        mask_stems = {p.stem for p in mask_dir.glob("*.png")}
        assert len(image_stems) == len(mask_stems), f"image/mask count mismatch for {defect_dir.name!r}"
        # anomalib pairs them by requiring the image stem inside the mask stem.
        for stem in image_stems:
            assert any(stem in mask_stem for mask_stem in mask_stems), f"no mask for {defect_dir.name}/{stem}.png"

    assert not (CATEGORY_DIR / "ground_truth" / "good").exists(), "normal samples must not have ground truth masks"


# ---------------------------------------------------------------------------
# 2. Download source fallback chain
# ---------------------------------------------------------------------------


def test_primary_source_short_circuits_the_chain(
    download_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the HF parquet path works, no other source is attempted."""
    attempted: list[str] = []

    def primary(category: str, output_dir: Path) -> dict[str, int]:
        attempted.append("hf-parquet")
        _write_minimal_layout(category, output_dir)
        return {}

    def unexpected(category: str, output_dir: Path) -> dict[str, int]:
        attempted.append("other")
        raise AssertionError("later sources must not run when the primary succeeds")

    monkeypatch.setattr(download_script, "download_via_hf_parquet", primary)
    monkeypatch.setattr(download_script, "download_via_hf_datasets", unexpected)
    monkeypatch.setattr(download_script, "download_via_anomalib", unexpected)

    source, counts = download_script._run_sources(CATEGORY, tmp_path, "auto")

    assert source == "hf-parquet"
    assert attempted == ["hf-parquet"]
    assert counts == {"train": 1, "test": 1, "ground_truth": 0}


def test_falls_back_to_anomalib_when_huggingface_is_unavailable(
    download_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A dead HuggingFace Hub takes out both HF strategies; anomalib takes over.

    This is the fallback the spec asks for: not a retry between two HF calls,
    but a switch to a different provider entirely when the mirror is gone.
    """
    attempted: list[str] = []

    def hub_is_down(category: str, output_dir: Path) -> dict[str, int]:
        attempted.append("hf")
        msg = "HuggingFace Hub unreachable"
        raise ConnectionError(msg)

    def anomalib_download(category: str, output_dir: Path) -> dict[str, int]:
        attempted.append("anomalib")
        _write_minimal_layout(category, output_dir)
        return {}

    monkeypatch.setattr(download_script, "download_via_hf_parquet", hub_is_down)
    monkeypatch.setattr(download_script, "download_via_hf_datasets", hub_is_down)
    monkeypatch.setattr(download_script, "download_via_anomalib", anomalib_download)

    source, counts = download_script._run_sources(CATEGORY, tmp_path, "auto")

    assert source == "anomalib"
    assert attempted == ["hf", "hf", "anomalib"]
    assert counts["train"] >= 1
    assert (tmp_path / CATEGORY / "train" / "good").is_dir()


def test_every_source_failing_reports_all_of_them(
    download_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no source left, the error names each failure rather than the last."""

    def fail(name: str):
        def _fail(category: str, output_dir: Path) -> dict[str, int]:
            raise ConnectionError(f"{name} down")

        return _fail

    monkeypatch.setattr(download_script, "download_via_hf_parquet", fail("parquet"))
    monkeypatch.setattr(download_script, "download_via_hf_datasets", fail("datasets"))
    monkeypatch.setattr(download_script, "download_via_anomalib", fail("anomalib"))

    with pytest.raises(RuntimeError) as excinfo:
        download_script._run_sources(CATEGORY, tmp_path, "auto")

    message = str(excinfo.value)
    assert "parquet down" in message
    assert "datasets down" in message
    assert "anomalib down" in message


def test_incomplete_download_is_rejected(download_script: ModuleType, tmp_path: Path) -> None:
    """A tree missing train/good is treated as not downloaded."""
    assert download_script.is_already_downloaded(CATEGORY, tmp_path) is False

    (tmp_path / CATEGORY / "test" / "broken").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="train/good"):
        download_script.verify_layout(CATEGORY, tmp_path)


@requires_dataset
def test_real_download_is_recognised_as_complete(download_script: ModuleType) -> None:
    """The tree produced by the actual run passes the script's own verification."""
    counts = download_script.verify_layout(CATEGORY, DEFAULT_DATA_ROOT)
    assert counts["train"] >= 1
    assert counts["test"] >= 1
    assert counts["ground_truth"] >= 1


# ---------------------------------------------------------------------------
# 3. DataModule
# ---------------------------------------------------------------------------


@requires_dataset
def test_setup_runs_without_error() -> None:
    """setup() completes and populates every split."""
    dm = DataModule(category=CATEGORY, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=0)
    assert dm.is_setup is False

    assert dm.setup() is dm  # returns self so it can be chained
    assert dm.is_setup is True

    counts = dm.num_samples()
    assert counts["train"] > 0
    assert counts["test"] > 0


def test_setup_raises_a_useful_error_when_data_is_missing(tmp_path: Path) -> None:
    """A missing dataset points the caller at the download script."""
    dm = DataModule(category=CATEGORY, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="download_dataset.py"):
        dm.setup()


# ---------------------------------------------------------------------------
# 4. Batches
# ---------------------------------------------------------------------------


@requires_dataset
def test_test_batch_has_expected_keys(datamodule: DataModule) -> None:
    """A single test batch exposes image, label and mask."""
    batch = next(iter(datamodule.test_dataloader()))

    assert {"image", "label", "mask"} <= set(batch.keys())
    assert dict(batch).keys() == set(batch.keys())
    for key in ("image", "label", "mask"):
        assert batch[key] is not None, f"{key} is None"
        assert getattr(batch, key) is batch[key], f"attribute and key access disagree for {key}"


@requires_dataset
def test_test_batch_shapes_and_dtypes(datamodule: DataModule) -> None:
    """Tensors come back at the configured size with predictable dtypes."""
    batch = next(iter(datamodule.test_dataloader()))
    n = batch.batch_size

    assert n == BATCH_SIZE
    validate_image_shape(batch.image, IMAGE_SIZE, name="test batch image")
    assert batch.image.dtype == torch.float32
    assert 0.0 <= float(batch.image.min()) and float(batch.image.max()) <= 1.0

    assert batch.label.shape == (n,)
    assert batch.label.dtype == torch.int64
    assert set(batch.label.tolist()) <= {0, 1}

    assert batch.mask is not None
    assert batch.mask.shape == (n, IMAGE_SIZE, IMAGE_SIZE)
    assert batch.mask.dtype == torch.uint8

    assert len(batch.image_path) == n
    assert all(Path(p).exists() for p in batch.image_path)


@requires_dataset
def test_train_batch_is_all_normal(datamodule: DataModule) -> None:
    """The train loader yields defect-free samples at the configured size."""
    batch = next(iter(datamodule.train_dataloader()))

    validate_image_shape(batch.image, IMAGE_SIZE, name="train batch image")
    assert batch.label.tolist() == [0] * batch.batch_size
    assert batch.mask is not None
    assert int(batch.mask.sum()) == 0, "normal samples must have empty ground truth masks"


@requires_dataset
def test_dataloaders_do_not_leak_anomalib_types(datamodule: DataModule) -> None:
    """The whole point of the wrapper: nothing anomalib-shaped escapes it.

    Guards the stated goal that no other module depends on anomalib data
    classes — including implicitly, via the objects the dataloaders return.
    """
    for loader in (datamodule.train_dataloader(), datamodule.test_dataloader()):
        batch = next(iter(loader))

        assert isinstance(batch, DefectBatch)
        assert type(batch).__module__.startswith("app.data")

        # Plain torch tensors, not tv_tensors/anomalib subclasses.
        for name in ("image", "label", "mask"):
            tensor = getattr(batch, name)
            assert type(tensor) is torch.Tensor, f"{name} is a {type(tensor).__name__}, not a plain torch.Tensor"

        # And no anomalib attribute names survived the conversion.
        for leaked in ("gt_label", "gt_mask", "pred_score", "anomaly_map"):
            assert not hasattr(batch, leaked), f"anomalib field {leaked!r} leaked into DefectBatch"


@requires_dataset
def test_batch_helpers_round_trip(datamodule: DataModule) -> None:
    """DefectBatch behaves like both a dataclass and a read-only mapping."""
    batch = next(iter(datamodule.test_dataloader()))

    assert len(batch) == batch.batch_size
    assert batch.has_mask is True
    assert "image" in batch
    assert batch.as_dict()["label"] is batch.label
    assert torch.equal(batch.to("cpu").image, batch.image)

    with pytest.raises(KeyError):
        _ = batch["not_a_field"]


# ---------------------------------------------------------------------------
# 5. Transforms
# ---------------------------------------------------------------------------


def test_to_tensor_accepts_pil_numpy_and_tensor() -> None:
    """All three input representations converge on (C, H, W) float32 in [0, 1]."""
    array = np.full((8, 6, 3), 255, dtype=np.uint8)

    from_numpy = to_tensor(array)
    from_pil = to_tensor(Image.fromarray(array))
    from_tensor = to_tensor(torch.zeros(3, 8, 6, dtype=torch.float32))

    for tensor in (from_numpy, from_pil, from_tensor):
        assert tensor.shape == (3, 8, 6)
        assert tensor.dtype == torch.float32

    assert torch.allclose(from_numpy, torch.ones(3, 8, 6))
    assert torch.equal(from_numpy, from_pil)
    assert torch.allclose(from_tensor, torch.zeros(3, 8, 6))


def test_to_tensor_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError):
        to_tensor("not-an-image")  # type: ignore[arg-type]


def test_normalize_image_uses_imagenet_statistics() -> None:
    """A constant image maps to the expected per-channel z-scores."""
    image = torch.full((3, 4, 4), 0.5)
    normalized = normalize_image(image)

    assert normalized.shape == image.shape
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        expected = (0.5 - mean) / std
        assert normalized[channel].allclose(torch.full((4, 4), expected))

    # Batched input is handled too, and the input is never modified in place.
    assert normalize_image(image.unsqueeze(0)).shape == (1, 3, 4, 4)
    assert torch.equal(image, torch.full((3, 4, 4), 0.5))


def test_normalize_denormalize_round_trip() -> None:
    image = torch.rand(3, 16, 16)
    assert torch.allclose(denormalize_image(normalize_image(image)), image, atol=1e-6)


def test_normalize_image_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="channel"):
        normalize_image(torch.rand(1, 4, 4))
    with pytest.raises(ValueError, match="got shape"):
        normalize_image(torch.rand(4, 4))
    with pytest.raises(ValueError, match="zeros"):
        normalize_image(torch.rand(3, 4, 4), std=(0.0, 1.0, 1.0))


def test_validate_image_shape_accepts_valid_images() -> None:
    """Returns the tensor so it can be used inline, for both ranks."""
    chw = torch.rand(3, 32, 32)
    nchw = torch.rand(2, 3, 32, 32)

    assert validate_image_shape(chw, 32) is chw
    assert validate_image_shape(nchw, 32) is nchw
    assert validate_image_shape(torch.rand(3, 16, 32), (16, 32)) is not None
    # Masks are single-channel, hence the opt-out.
    assert validate_image_shape(torch.zeros(1, 8, 8), 8, expected_channels=None) is not None


def test_validate_image_shape_rejects_invalid_images() -> None:
    with pytest.raises(ValueError, match="must be 64x64"):
        validate_image_shape(torch.rand(3, 32, 32), 64)
    with pytest.raises(ValueError, match="3 channel"):
        validate_image_shape(torch.rand(1, 32, 32), 32)
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        validate_image_shape(torch.rand(32, 32), 32)
    with pytest.raises(TypeError):
        validate_image_shape(np.zeros((3, 32, 32)), 32)  # type: ignore[arg-type]
