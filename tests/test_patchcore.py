"""Tests for the PatchCore wrapper, its config and the model interface.

The three tests the spec calls for — train, predict, and defective-scores-higher
— all need a fitted memory bank, so training happens once in a module-scoped
fixture and every dataset-backed test shares it. Fitting uses
``coreset_sampling_ratio=0.01`` and a single epoch so the suite stays viable on
CPU; the numbers are not meant to be publication-grade, only to show the model
is doing real work.

Dataset-backed tests skip (rather than fail) when ``data/MVTecAD/bottle`` is
absent. Populate it with::

    python scripts/download_dataset.py --category bottle
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.data import DataModule, denormalize_image
from app.data.datamodule import DEFAULT_DATA_ROOT
from app.models import AnomalyModel, ModelConfig, ModelOutput, PatchCoreModel
from app.models.config import get_model_config

CATEGORY = "bottle"
#: Enough to build a meaningful memory bank without a 20-minute CPU run.
TEST_IMAGE_SIZE = 256
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
def trained_model(fast_config: ModelConfig) -> PatchCoreModel:
    """Train PatchCore on bottle exactly once, and share it across tests."""
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


def _read_bgr(path: Path) -> np.ndarray:
    """Read an image the way the serving layer will: OpenCV, hence BGR."""
    image = cv2.imread(str(path))
    assert image is not None, f"OpenCV could not read {path}"
    return image


@pytest.fixture(scope="module")
def clean_image_path() -> Path:
    return sorted((CATEGORY_DIR / "test" / "good").glob("*.png"))[0]


@pytest.fixture(scope="module")
def defective_image_path() -> Path:
    """A visibly damaged bottle: the largest, least ambiguous defect class."""
    return sorted((CATEGORY_DIR / "test" / "broken_large").glob("*.png"))[0]


# ---------------------------------------------------------------------------
# 1. train()
# ---------------------------------------------------------------------------


@requires_dataset
def test_train_runs_and_builds_a_memory_bank(trained_model: PatchCoreModel) -> None:
    """train() completes on 'bottle' and leaves a usable, calibrated model."""
    assert trained_model.is_trained
    assert trained_model.is_calibrated, "the validation pass should have fitted normalization stats"

    bank = trained_model._module.model.memory_bank  # noqa: SLF001 - asserting on internals is the point here
    assert bank.ndim == 2
    assert bank.shape[0] > 0, "memory bank is empty"
    # layer2 (512ch) + layer3 (1024ch) of wide_resnet50_2, projected to 1024 dims.
    assert bank.shape[1] > 0


@requires_dataset
def test_train_writes_the_canonical_checkpoint(trained_model: PatchCoreModel, fast_config: ModelConfig) -> None:
    """The checkpoint lands at results/checkpoints/patchcore_<category>.ckpt."""
    checkpoint = trained_model.checkpoint_path

    assert checkpoint.name == f"patchcore_{CATEGORY}.ckpt"
    assert checkpoint.parent == fast_config.checkpoint_dir
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0


def test_train_rejects_a_raw_anomalib_datamodule() -> None:
    """The data contract is enforced: only app.data.DataModule is accepted."""
    model = PatchCoreModel(category=CATEGORY)
    with pytest.raises(TypeError, match="app.data.DataModule"):
        model.train(object())


@requires_dataset
def test_checkpoint_round_trips(trained_model: PatchCoreModel, defective_image_path: Path, tmp_path: Path) -> None:
    """save() then load() into a fresh instance reproduces the same score."""
    image = _read_bgr(defective_image_path)
    expected = trained_model.predict(image, color_order="bgr").anomaly_score

    destination = tmp_path / "round_trip.ckpt"
    trained_model.save(destination)

    reloaded = PatchCoreModel(config=trained_model.config)
    reloaded.load(destination)

    assert reloaded.is_trained
    assert reloaded.is_calibrated
    assert reloaded.predict(image, color_order="bgr").anomaly_score == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. predict()
# ---------------------------------------------------------------------------


@requires_dataset
def test_predict_returns_a_well_formed_model_output(
    trained_model: PatchCoreModel,
    defective_image_path: Path,
) -> None:
    """A single test image yields a score in [0, 1] and a map matching its dims."""
    image = _read_bgr(defective_image_path)
    height, width = image.shape[:2]

    result = trained_model.predict(image, color_order="bgr")

    assert isinstance(result, ModelOutput)
    assert isinstance(result.anomaly_score, float)
    assert 0.0 <= result.anomaly_score <= 1.0, f"score {result.anomaly_score} outside [0, 1]"

    assert result.anomaly_map.shape == (height, width), "heatmap must come back at the input resolution"
    assert result.anomaly_map.dtype == np.float32
    assert np.isfinite(result.anomaly_map).all()
    assert 0.0 <= float(result.anomaly_map.min())
    assert float(result.anomaly_map.max()) <= 1.0

    assert isinstance(result.is_defective, bool)
    assert result.is_defective == (result.anomaly_score >= trained_model.config.anomaly_threshold)
    assert result.model_name == "patchcore"


@requires_dataset
@pytest.mark.parametrize("size", [(64, 96), (512, 512), (900, 900)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_predict_accepts_any_size_and_channel_count(
    trained_model: PatchCoreModel,
    defective_image_path: Path,
    size: tuple[int, int],
    channels: int,
) -> None:
    """Arbitrary resolutions, grayscale, RGB and RGBA all produce a matching map."""
    bgr = cv2.resize(_read_bgr(defective_image_path), (size[1], size[0]))
    if channels == 1:
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    else:
        image = bgr

    result = trained_model.predict(image, color_order="bgr")

    assert result.anomaly_map.shape == size


@requires_dataset
def test_predict_handles_float_and_uint8_inputs_identically(
    trained_model: PatchCoreModel,
    clean_image_path: Path,
) -> None:
    """A float [0, 1] frame scores the same as the uint8 one it came from."""
    uint8_image = _read_bgr(clean_image_path)
    float_image = uint8_image.astype(np.float32) / 255.0

    from_uint8 = trained_model.predict(uint8_image, color_order="bgr").anomaly_score
    from_float = trained_model.predict(float_image, color_order="bgr").anomaly_score

    assert from_float == pytest.approx(from_uint8, abs=1e-4)


@requires_dataset
def test_predict_distinguishes_channel_order(trained_model: PatchCoreModel, clean_image_path: Path) -> None:
    """Mislabelling BGR as RGB changes the score, i.e. color_order is not cosmetic.

    The backbone is ImageNet-pretrained, so channel-swapped input is genuinely
    out of distribution. If this ever stops holding, the flag is being ignored.
    """
    image = _read_bgr(clean_image_path)

    as_bgr = trained_model.predict(image, color_order="bgr").anomaly_score
    as_rgb = trained_model.predict(image, color_order="rgb").anomaly_score

    assert as_bgr != pytest.approx(as_rgb, abs=1e-4)


@requires_dataset
def test_predict_rejects_malformed_input(trained_model: PatchCoreModel) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        trained_model.predict("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="color_order"):
        trained_model.predict(np.zeros((8, 8, 3), dtype=np.uint8), color_order="rgba")
    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        trained_model.predict(np.zeros((8, 8, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        trained_model.predict(np.zeros((2, 8, 8, 3), dtype=np.uint8))


def test_predict_without_a_model_explains_how_to_get_one(tmp_path: Path) -> None:
    """An untrained wrapper fails loudly, pointing at the training script."""
    model = PatchCoreModel(config=ModelConfig.from_env(category=CATEGORY, checkpoint_dir=tmp_path))
    assert model.is_trained is False

    with pytest.raises(RuntimeError, match="train_patchcore.py"):
        model.predict(np.zeros((32, 32, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 3. The model actually discriminates
# ---------------------------------------------------------------------------


@requires_dataset
def test_defective_image_scores_higher_than_a_clean_one(
    trained_model: PatchCoreModel,
    clean_image_path: Path,
    defective_image_path: Path,
) -> None:
    """The whole point: a broken bottle must outscore an intact one.

    A model returning noise, or one whose memory bank never got populated, would
    put these two within sampling distance of each other.
    """
    clean = trained_model.predict(_read_bgr(clean_image_path), color_order="bgr")
    defective = trained_model.predict(_read_bgr(defective_image_path), color_order="bgr")

    assert defective.anomaly_score > clean.anomaly_score, (
        f"defective {defective_image_path.name} scored {defective.anomaly_score:.4f} "
        f"but clean {clean_image_path.name} scored {clean.anomaly_score:.4f}"
    )
    # The gap should be a decision, not a rounding error.
    assert defective.anomaly_score - clean.anomaly_score > 0.05
    assert defective.is_defective
    assert not clean.is_defective


@requires_dataset
def test_anomaly_map_localises_the_defect(
    trained_model: PatchCoreModel,
    clean_image_path: Path,
    defective_image_path: Path,
) -> None:
    """Pixel-level scores separate too, and the map agrees with ground truth.

    Checks the heatmap is spatially meaningful rather than a uniform field
    scaled by the image-level score: the defect's masked region should score
    above the rest of the same image.
    """
    defective = trained_model.predict(_read_bgr(defective_image_path), color_order="bgr")
    clean = trained_model.predict(_read_bgr(clean_image_path), color_order="bgr")

    assert defective.anomaly_map.max() > clean.anomaly_map.max()

    mask_dir = CATEGORY_DIR / "ground_truth" / defective_image_path.parent.name
    mask_path = next(mask_dir.glob(f"{defective_image_path.stem}*.png"))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    assert mask.any(), f"ground truth {mask_path} is empty"

    inside = float(defective.anomaly_map[mask].mean())
    outside = float(defective.anomaly_map[~mask].mean())
    assert inside > outside, f"defect region scored {inside:.4f} vs {outside:.4f} elsewhere"


# ---------------------------------------------------------------------------
# 4. Interface and configuration (no dataset needed)
# ---------------------------------------------------------------------------


def test_patchcore_implements_the_anomaly_model_interface() -> None:
    assert issubclass(PatchCoreModel, AnomalyModel)
    for method in ("train", "predict", "save", "load"):
        assert callable(getattr(PatchCoreModel, method))
    assert PatchCoreModel.model_name == "patchcore"


def test_anomaly_model_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AnomalyModel(ModelConfig())  # type: ignore[abstract]


def test_model_output_validates_its_contract() -> None:
    output = ModelOutput(
        anomaly_score=0.75,
        anomaly_map=np.full((4, 6), 0.8, dtype=np.float32),
        is_defective=True,
        model_name="patchcore",
    )
    assert output.shape == (4, 6)
    assert output.defective_area_ratio(0.5) == 1.0
    assert output.defective_area_ratio(0.9) == 0.0

    with pytest.raises(ValueError, match="2-D"):
        ModelOutput(0.5, np.zeros((1, 4, 4), dtype=np.float32), False, "patchcore")
    with pytest.raises(ValueError, match="finite"):
        ModelOutput(float("nan"), np.zeros((4, 4), dtype=np.float32), False, "patchcore")


def test_config_defaults_match_the_documented_hyperparameters() -> None:
    """The spec's defaults live here and nowhere else."""
    config = ModelConfig()

    assert config.backbone == "wide_resnet50_2"
    assert config.layers == ("layer2", "layer3")
    assert config.coreset_sampling_ratio == 0.1
    assert config.num_neighbors == 9
    assert config.image_size == 256
    assert config.anomaly_threshold == 0.5


def test_config_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars override defaults, and comma-separated layers are parsed."""
    env = {
        "MODEL_BACKBONE": "resnet18",
        "MODEL_LAYERS": "layer1, layer2 ,layer3",
        "CORESET_SAMPLING_RATIO": "0.05",
        "NUM_NEIGHBORS": "3",
        "IMAGE_SIZE": "128",
        "ANOMALY_THRESHOLD": "0.7",
        "DEFAULT_CATEGORY": "cable",
    }

    config = ModelConfig.from_env(env)

    assert config.backbone == "resnet18"
    assert config.layers == ("layer1", "layer2", "layer3")
    assert config.coreset_sampling_ratio == 0.05
    assert config.num_neighbors == 3
    assert config.image_size == 128
    assert config.anomaly_threshold == 0.7
    assert config.category == "cable"


def test_explicit_arguments_beat_the_environment() -> None:
    """Resolution order is default < env < kwarg, and None means 'not supplied'."""
    env = {"IMAGE_SIZE": "128", "DEFAULT_CATEGORY": "cable"}

    config = ModelConfig.from_env(env, image_size=320, category=None)

    assert config.image_size == 320
    assert config.category == "cable"


def test_blank_environment_values_fall_back_to_defaults() -> None:
    """`IMAGE_SIZE=` in a .env file must not blow up validation."""
    assert ModelConfig.from_env({"IMAGE_SIZE": "", "MODEL_BACKBONE": "  "}).image_size == 256


def test_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="coreset_sampling_ratio"):
        ModelConfig(coreset_sampling_ratio=1.5)
    with pytest.raises(ValueError, match="coreset_sampling_ratio"):
        ModelConfig(coreset_sampling_ratio=0.0)
    with pytest.raises(ValueError, match="num_neighbors"):
        ModelConfig(num_neighbors=0)
    with pytest.raises(ValueError, match="anomaly_threshold"):
        ModelConfig(anomaly_threshold=1.2)
    with pytest.raises(ValueError, match="unexpected_knob"):
        ModelConfig(unexpected_knob=1)


def test_config_is_frozen() -> None:
    """A shared config cannot be mutated under a running model."""
    config = ModelConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.image_size = 512  # type: ignore[misc]

    assert config.with_overrides(image_size=512).image_size == 512
    assert config.image_size == 256, "with_overrides must not mutate the original"


def test_checkpoint_path_follows_the_naming_convention(tmp_path: Path) -> None:
    config = ModelConfig(checkpoint_dir=tmp_path, category="bottle")

    assert config.checkpoint_path("patchcore") == tmp_path / "patchcore_bottle.ckpt"
    assert config.checkpoint_path("patchcore", "cable") == tmp_path / "patchcore_cable.ckpt"


def test_get_model_config_caches_until_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_SIZE", "192")
    first = get_model_config(refresh=True)
    assert first.image_size == 192
    assert get_model_config() is first, "repeat calls must not re-read the environment"

    monkeypatch.setenv("IMAGE_SIZE", "224")
    assert get_model_config().image_size == 192
    assert get_model_config(refresh=True).image_size == 224

    # An override derives a new instance and leaves the cache alone.
    derived = get_model_config(image_size=96)
    assert derived.image_size == 96
    assert get_model_config().image_size == 224

    # Restore the cache to the real environment; it outlives this test.
    monkeypatch.delenv("IMAGE_SIZE")
    assert get_model_config(refresh=True).image_size == 256


def test_constructor_arguments_override_config_defaults() -> None:
    model = PatchCoreModel(
        category="cable",
        backbone="resnet18",
        coreset_sampling_ratio=0.02,
        num_neighbors=5,
        image_size=128,
    )

    assert model.category == "cable"
    assert model.config.backbone == "resnet18"
    assert model.config.coreset_sampling_ratio == 0.02
    assert model.config.num_neighbors == 5
    assert model.config.image_size == 128
    assert model.checkpoint_path.name == "patchcore_cable.ckpt"


def test_preprocess_produces_a_normalized_square_batch() -> None:
    """The documented preprocessing contract, independent of any trained weights."""
    model = PatchCoreModel(image_size=128)
    image = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)

    tensor = model._preprocess(image)  # noqa: SLF001 - the contract under test

    assert tensor.shape == (1, 3, 128, 128)
    assert tensor.dtype == torch.float32
    # ImageNet normalization pushes values outside [0, 1] in both directions.
    assert float(tensor.min()) < 0.0
    assert float(tensor.max()) > 1.0


def test_bgr_and_rgb_preprocessing_are_channel_swaps_of_each_other() -> None:
    """Reading the same bytes as BGR flips channels, and only channels."""
    model = PatchCoreModel(image_size=64)
    image = np.random.default_rng(1).integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    as_rgb = denormalize_image(model._preprocess(image, color_order="rgb"))  # noqa: SLF001
    as_bgr = denormalize_image(model._preprocess(image, color_order="bgr"))  # noqa: SLF001

    assert not torch.allclose(as_bgr, as_rgb), "color_order had no effect"
    assert torch.allclose(as_bgr, as_rgb.flip(1), atol=1e-5)
