"""Tests for the ONNX inference path (:class:`app.models.onnx_runner.ONNXRunner`).

The core promise this file checks is the one Step 7 exists to deliver: an ONNX
backend is a drop-in for the PyTorch wrapper. So the central test scores the same
image through both and asserts the :class:`ModelOutput` comes back with the *same
schema, the same value ranges, and — for the FP32 graph — the same numbers*.

Getting an ONNX artifact to test against is self-contained: a module-scoped
fixture trains a small PatchCore (``coreset_sampling_ratio=0.01``, one epoch,
like tests/test_patchcore.py), exports it with anomalib, and dynamically
quantizes it — all into a temp dir, so the suite needs only the ``bottle``
dataset, no pre-run export. PatchCore is the model under test here because it is
calibrated (scores and maps land cleanly in ``[0, 1]``) and its exported graph
reproduces the PyTorch score almost exactly, which makes the equivalence
assertion sharp.

Dataset-backed tests skip (rather than fail) when ``data/MVTecAD/bottle`` is
absent. Populate it with::

    python scripts/download_dataset.py --category bottle
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.data.datamodule import DEFAULT_DATA_ROOT
from app.models import ModelConfig, ModelOutput, ONNXRunner, PatchCoreModel, onnx_artifact_path

CATEGORY = "bottle"
TEST_IMAGE_SIZE = 256
#: As in tests/test_patchcore.py: a real memory bank without a long CPU run. A
#: 1% coreset also keeps the exported graph small enough to write to a temp dir.
TEST_CORESET_RATIO = 0.01

CATEGORY_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY

requires_dataset = pytest.mark.skipif(
    not (CATEGORY_DIR / "train" / "good").is_dir(),
    reason=f"{CATEGORY_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_config(tmp_path_factory: pytest.TempPathFactory) -> ModelConfig:
    """A config tuned for a quick CPU fit, writing checkpoints to a temp dir."""
    return ModelConfig.from_env(
        category=CATEGORY,
        image_size=TEST_IMAGE_SIZE,
        coreset_sampling_ratio=TEST_CORESET_RATIO,
        max_epochs=1,
        batch_size=8,
        num_workers=0,
        accelerator="cpu",
        checkpoint_dir=tmp_path_factory.mktemp("checkpoints"),
        results_dir=tmp_path_factory.mktemp("results"),
    )


@pytest.fixture(scope="module")
def trained_patchcore(fast_config: ModelConfig) -> PatchCoreModel:
    """Train PatchCore on bottle once; the PyTorch reference the ONNX path must match."""
    from app.data import DataModule

    datamodule = DataModule(
        category=CATEGORY,
        image_size=fast_config.image_size,
        batch_size=fast_config.batch_size,
        root=fast_config.data_root,
        num_workers=fast_config.num_workers,
    )
    model = PatchCoreModel(config=fast_config)
    model.train(datamodule)
    return model


@pytest.fixture(scope="module")
def exported_paths(trained_patchcore: PatchCoreModel, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Export the trained model to ONNX and INT8, into a temp export root.

    Exercises the same anomalib + onnxruntime calls scripts/export_onnx.py uses,
    so this fixture also covers the export path end to end.
    """
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from onnxruntime.quantization import QuantType, quantize_dynamic

    export_dir = tmp_path_factory.mktemp("exported")
    engine = Engine(accelerator="cpu", devices=1, logger=False)
    engine.export(
        model=trained_patchcore._module,  # noqa: SLF001 - the cached anomalib module, as in the export script
        export_type=ExportType.ONNX,
        export_root=str(export_dir),
        model_file_name="patchcore",
        input_size=trained_patchcore.config.image_hw,
    )
    fp32 = onnx_artifact_path("patchcore", "fp32", export_dir)
    int8 = onnx_artifact_path("patchcore", "int8", export_dir)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    assert fp32.is_file(), "FP32 export did not land where onnx_artifact_path expects it"
    assert int8.is_file(), "INT8 quantization did not land where onnx_artifact_path expects it"
    return {"fp32": fp32, "int8": int8}


@pytest.fixture(scope="module")
def fp32_runner(exported_paths: dict[str, Path], fast_config: ModelConfig) -> ONNXRunner:
    return ONNXRunner(exported_paths["fp32"], model_name="patchcore_fp32", config=fast_config)


@pytest.fixture(scope="module")
def defective_image() -> np.ndarray:
    """A visibly damaged bottle, read as BGR the way the serving layer would."""
    path = sorted((CATEGORY_DIR / "test" / "broken_large").glob("*.png"))[0]
    image = cv2.imread(str(path))
    assert image is not None, f"OpenCV could not read {path}"
    return image


# ---------------------------------------------------------------------------
# The headline test: ONNX output matches the PyTorch wrapper's schema and values
# ---------------------------------------------------------------------------


@requires_dataset
def test_onnx_output_has_the_same_schema_as_the_pytorch_wrapper(
    fp32_runner: ONNXRunner,
    trained_patchcore: PatchCoreModel,
    defective_image: np.ndarray,
) -> None:
    """Same fields and same value ranges as PatchCoreModel.predict() on one image."""
    onnx_out = fp32_runner.predict(defective_image, color_order="bgr")
    torch_out = trained_patchcore.predict(defective_image, color_order="bgr")
    height, width = defective_image.shape[:2]

    # Same type and exactly the same set of fields — "same keys".
    assert isinstance(onnx_out, ModelOutput)
    onnx_fields = {f.name for f in dataclasses.fields(onnx_out)}
    torch_fields = {f.name for f in dataclasses.fields(torch_out)}
    assert onnx_fields == torch_fields == {"anomaly_score", "anomaly_map", "is_defective", "model_name"}

    # Same value ranges as the PyTorch contract (see tests/test_patchcore.py).
    assert isinstance(onnx_out.anomaly_score, float)
    assert 0.0 <= onnx_out.anomaly_score <= 1.0

    assert onnx_out.anomaly_map.shape == (height, width), "heatmap must come back at the input resolution"
    assert onnx_out.anomaly_map.dtype == np.float32
    assert np.isfinite(onnx_out.anomaly_map).all()
    assert 0.0 <= float(onnx_out.anomaly_map.min())
    assert float(onnx_out.anomaly_map.max()) <= 1.0 + 1e-4

    assert isinstance(onnx_out.is_defective, bool)
    assert onnx_out.is_defective == (onnx_out.anomaly_score >= fp32_runner.config.anomaly_threshold)
    assert isinstance(onnx_out.model_name, str) and onnx_out.model_name


@requires_dataset
def test_onnx_fp32_reproduces_the_pytorch_score(
    fp32_runner: ONNXRunner,
    trained_patchcore: PatchCoreModel,
    defective_image: np.ndarray,
) -> None:
    """The FP32 graph is the same computation: score and map match to numerical noise.

    This is the strongest evidence the ONNX path is faithful rather than merely
    plausible — the whole ``forward`` (resize, ImageNet-normalize, backbone,
    nearest-neighbour search, calibration) is preserved through the export.
    """
    onnx_out = fp32_runner.predict(defective_image, color_order="bgr")
    torch_out = trained_patchcore.predict(defective_image, color_order="bgr")

    assert onnx_out.anomaly_score == pytest.approx(torch_out.anomaly_score, abs=1e-3)
    assert onnx_out.anomaly_map.shape == torch_out.anomaly_map.shape
    # Maps agree pixelwise, not just in aggregate.
    assert np.abs(onnx_out.anomaly_map - torch_out.anomaly_map).max() < 1e-2


@requires_dataset
def test_onnx_preserves_the_clean_vs_defective_ordering(
    fp32_runner: ONNXRunner,
    defective_image: np.ndarray,
) -> None:
    """The point of the model survives export: a broken bottle outscores a clean one."""
    clean_path = sorted((CATEGORY_DIR / "test" / "good").glob("*.png"))[0]
    clean_image = cv2.imread(str(clean_path))

    defective = fp32_runner.predict(defective_image, color_order="bgr")
    clean = fp32_runner.predict(clean_image, color_order="bgr")

    assert defective.anomaly_score > clean.anomaly_score
    assert defective.is_defective
    assert not clean.is_defective


# ---------------------------------------------------------------------------
# INT8: a valid ModelOutput, just not bit-identical
# ---------------------------------------------------------------------------


@requires_dataset
def test_int8_runner_returns_a_well_formed_output(
    exported_paths: dict[str, Path],
    fast_config: ModelConfig,
    defective_image: np.ndarray,
) -> None:
    """INT8 keeps the contract (schema + ranges); quantization only perturbs values."""
    runner = ONNXRunner(exported_paths["int8"], model_name="patchcore_int8", config=fast_config)
    height, width = defective_image.shape[:2]

    out = runner.predict(defective_image, color_order="bgr")

    assert isinstance(out, ModelOutput)
    assert isinstance(out.anomaly_score, float) and np.isfinite(out.anomaly_score)
    assert 0.0 <= out.anomaly_score <= 1.0
    assert out.anomaly_map.shape == (height, width)
    assert out.anomaly_map.dtype == np.float32
    assert np.isfinite(out.anomaly_map).all()
    assert out.model_name == "patchcore_int8"


# ---------------------------------------------------------------------------
# Contract, lifecycle and construction (no dataset needed for the last few)
# ---------------------------------------------------------------------------


@requires_dataset
def test_runner_reconciles_config_to_the_graph_resolution(fp32_runner: ONNXRunner) -> None:
    """The .onnx file is the source of truth for input size, and predict() is ready."""
    assert fp32_runner.is_trained
    assert fp32_runner.is_calibrated
    assert fp32_runner.config.image_size == TEST_IMAGE_SIZE


@requires_dataset
def test_runner_accepts_any_size_and_channel_count(
    fp32_runner: ONNXRunner,
    defective_image: np.ndarray,
) -> None:
    """Same input plumbing as the wrappers: arbitrary resolution and channel count in."""
    resized = cv2.resize(defective_image, (512, 400))  # (W, H) -> array is (400, 512)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    assert fp32_runner.predict(resized, color_order="bgr").anomaly_map.shape == (400, 512)
    assert fp32_runner.predict(gray, color_order="bgr").anomaly_map.shape == (400, 512)


@requires_dataset
def test_runner_is_inference_only(fp32_runner: ONNXRunner, tmp_path: Path) -> None:
    """train()/save()/load() are unsupported and say why."""
    with pytest.raises(NotImplementedError, match="export"):
        fp32_runner.train(object())
    with pytest.raises(NotImplementedError):
        fp32_runner.save(tmp_path / "x.onnx")
    with pytest.raises(NotImplementedError, match="constructor"):
        fp32_runner.load(tmp_path / "x.onnx")


def test_runner_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        ONNXRunner(tmp_path / "does_not_exist.onnx")


def test_onnx_artifact_path_naming_convention(tmp_path: Path) -> None:
    """FP32 keeps the bare model name; INT8 gets the suffix; both under weights/onnx."""
    fp32 = onnx_artifact_path("patchcore", "fp32", tmp_path)
    int8 = onnx_artifact_path("patchcore", "int8", tmp_path)

    assert fp32 == tmp_path / "weights" / "onnx" / "patchcore.onnx"
    assert int8 == tmp_path / "weights" / "onnx" / "patchcore_int8.onnx"
