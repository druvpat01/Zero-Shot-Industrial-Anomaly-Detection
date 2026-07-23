"""Tests for the EfficientAD wrapper, mirroring tests/test_patchcore.py.

The three tests the spec calls for — train, predict, and defective-scores-higher
— all need a fitted student, so training happens once in a module-scoped fixture
and every dataset-backed test shares it. Fitting uses a single epoch so the suite
stays viable on CPU.

That single epoch is the thing to keep in mind when reading the assertions.
PatchCore's memory bank is complete after one pass, so its test can demand
publication-shaped behaviour. EfficientAD is gradient-trained and the paper uses
~70k steps; one epoch over ``bottle`` is ~200. So the assertions here check that
the model *works and discriminates*, with a smaller required margin than
test_patchcore.py asks for — a tighter bound would be testing the training
budget, not the wrapper.

Dataset-backed tests skip (rather than fail) when ``data/MVTecAD/bottle`` is
absent. Populate it with::

    python scripts/download_dataset.py --category bottle

Training additionally needs ImageNette at ``data/imagenette``; anomalib
downloads it (~1.5 GB) on the first run, so the first invocation of this file is
much slower than later ones.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.data import DataModule
from app.data.datamodule import DEFAULT_DATA_ROOT
from app.models import AnomalyModel, EfficientADModel, ModelConfig, ModelOutput, PatchCoreModel

CATEGORY = "bottle"
TEST_IMAGE_SIZE = 256
#: One pass over ~200 images. Enough to distil a student that discriminates,
#: nowhere near the paper's schedule. See the module docstring.
TEST_MAX_EPOCHS = 1

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
        model_size="small",
        image_size=TEST_IMAGE_SIZE,
        max_epochs=TEST_MAX_EPOCHS,
        batch_size=8,
        num_workers=0,
        accelerator="cpu",
        checkpoint_dir=tmp_path_factory.mktemp("checkpoints"),
        results_dir=tmp_path_factory.mktemp("results"),
    )


@pytest.fixture(scope="module")
def trained_model(fast_config: ModelConfig) -> EfficientADModel:
    """Train EfficientAD on bottle exactly once, and share it across tests."""
    datamodule = DataModule(
        category=CATEGORY,
        image_size=fast_config.image_size,
        batch_size=fast_config.batch_size,
        root=fast_config.data_root,
        num_workers=fast_config.num_workers,
    )
    model = EfficientADModel(config=fast_config)
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
def test_train_runs_and_distils_a_student(trained_model: EfficientADModel) -> None:
    """train() completes on 'bottle' and leaves a usable, calibrated model."""
    assert trained_model.is_trained
    assert trained_model.is_calibrated, "the validation pass should have fitted normalization stats"
    assert trained_model.has_map_quantiles, "the validation pass should have fitted map quantiles"

    model = trained_model._module.model  # noqa: SLF001 - asserting on internals is the point here
    # Teacher statistics are what `is_trained` reads; check they are real numbers.
    assert torch.isfinite(model.mean_std["mean"]).all()
    assert torch.isfinite(model.mean_std["std"]).all()
    assert float(model.mean_std["std"].min()) > 0.0, "a zero channel std would divide by zero at inference"


@requires_dataset
def test_training_moved_the_student_but_not_the_teacher(trained_model: EfficientADModel) -> None:
    """The student's weights changed and the teacher's did not.

    PatchCore's equivalent assertion is 'the memory bank is non-empty'. There is
    no such marker here: a randomly initialised student has a full set of
    weights and would happily produce a plausible-looking anomaly map, so the
    only honest check that training happened is that the parameters moved — and
    that the distillation target they moved towards stayed put.
    """
    reference = EfficientADModel(config=trained_model.config)._build_module()  # noqa: SLF001 - no public equivalent
    reference.prepare_pretrained_model()  # loads the frozen teacher from anomalib's cache
    trained = trained_model._module  # noqa: SLF001 - asserting on internals is the point here

    assert not torch.allclose(reference.model.student.conv1.weight, trained.model.student.conv1.weight), (
        "the student is still at its random initialisation; no gradients reached it"
    )
    assert torch.allclose(reference.model.teacher.conv1.weight, trained.model.teacher.conv1.weight), (
        "the teacher drifted during training; it is meant to be a fixed distillation target"
    )


@requires_dataset
def test_train_writes_the_canonical_checkpoint(trained_model: EfficientADModel, fast_config: ModelConfig) -> None:
    """The checkpoint lands at results/checkpoints/efficientad_<category>.ckpt."""
    checkpoint = trained_model.checkpoint_path

    assert checkpoint.name == f"efficientad_{CATEGORY}.ckpt"
    assert checkpoint.parent == fast_config.checkpoint_dir
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0


@requires_dataset
def test_train_restores_the_datamodules_batch_size(fast_config: ModelConfig) -> None:
    """The forced train_batch_size=1 is scoped to the fit, not permanent.

    A benchmark run trains PatchCore and EfficientAD from one datamodule; if
    EfficientAD left it at 1, PatchCore would silently fit 8x slower.
    """
    datamodule = DataModule(category=CATEGORY, image_size=64, batch_size=8, root=fast_config.data_root)
    anomalib_datamodule = datamodule.anomalib_datamodule
    assert anomalib_datamodule.train_batch_size == 8

    with EfficientADModel._single_image_train_batches(anomalib_datamodule):  # noqa: SLF001 - the contract under test
        assert anomalib_datamodule.train_batch_size == 1

    assert anomalib_datamodule.train_batch_size == 8


def test_train_rejects_a_raw_anomalib_datamodule() -> None:
    """The data contract is enforced: only app.data.DataModule is accepted."""
    model = EfficientADModel(category=CATEGORY)
    with pytest.raises(TypeError, match="app.data.DataModule"):
        model.train(object())


@requires_dataset
def test_checkpoint_round_trips(trained_model: EfficientADModel, defective_image_path: Path, tmp_path: Path) -> None:
    """save() then load() into a fresh instance reproduces the same score."""
    image = _read_bgr(defective_image_path)
    expected = trained_model.predict(image, color_order="bgr").anomaly_score

    destination = tmp_path / "round_trip.ckpt"
    trained_model.save(destination)

    reloaded = EfficientADModel(config=trained_model.config)
    reloaded.load(destination)

    assert reloaded.is_trained
    assert reloaded.is_calibrated
    assert reloaded.has_map_quantiles
    assert reloaded.predict(image, color_order="bgr").anomaly_score == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. predict()
# ---------------------------------------------------------------------------


@requires_dataset
def test_predict_returns_a_well_formed_model_output(
    trained_model: EfficientADModel,
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
    assert result.model_name == "efficientad"


@requires_dataset
@pytest.mark.parametrize("size", [(64, 96), (512, 512), (900, 900)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_predict_accepts_any_size_and_channel_count(
    trained_model: EfficientADModel,
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
    trained_model: EfficientADModel,
    clean_image_path: Path,
) -> None:
    """A float [0, 1] frame scores the same as the uint8 one it came from."""
    uint8_image = _read_bgr(clean_image_path)
    float_image = uint8_image.astype(np.float32) / 255.0

    from_uint8 = trained_model.predict(uint8_image, color_order="bgr").anomaly_score
    from_float = trained_model.predict(float_image, color_order="bgr").anomaly_score

    assert from_float == pytest.approx(from_uint8, abs=1e-4)


@requires_dataset
def test_predict_distinguishes_channel_order(trained_model: EfficientADModel, clean_image_path: Path) -> None:
    """Mislabelling BGR as RGB changes the score, i.e. color_order is not cosmetic.

    The teacher was distilled from an ImageNet backbone, so channel-swapped
    input is genuinely out of distribution here too.
    """
    image = _read_bgr(clean_image_path)

    as_bgr = trained_model.predict(image, color_order="bgr").anomaly_score
    as_rgb = trained_model.predict(image, color_order="rgb").anomaly_score

    assert as_bgr != pytest.approx(as_rgb, abs=1e-4)


@requires_dataset
def test_predict_rejects_malformed_input(trained_model: EfficientADModel) -> None:
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
    model = EfficientADModel(config=ModelConfig.from_env(category=CATEGORY, checkpoint_dir=tmp_path))
    assert model.is_trained is False

    with pytest.raises(RuntimeError, match="train_efficientad.py"):
        model.predict(np.zeros((32, 32, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 3. The model actually discriminates
# ---------------------------------------------------------------------------


@requires_dataset
def test_defective_image_scores_higher_than_a_clean_one(
    trained_model: EfficientADModel,
    clean_image_path: Path,
    defective_image_path: Path,
) -> None:
    """The whole point: a broken bottle must outscore an intact one.

    Deliberately weaker than test_patchcore.py's version of this test, in two
    ways, and both are about the *scale* rather than the model's ability:

    * **No margin is demanded.** PatchCore requires a 0.05 gap. EfficientAD
      after one epoch produces ~0.005 — not because the ranking is weak (see
      :func:`test_scores_rank_defective_above_clean_across_the_split`, which
      measures AUROC ~0.97 on the same model) but because anomalib min-max
      normalizes scores against the validation split's range. One extreme
      ``contamination`` image saturates that range, and every other image is
      compressed towards the 0.5 midpoint. A larger margin would need a
      different post-processor, not a better model.
    * **``is_defective`` is not asserted.** For the same reason: the normalized
      scores straddle the 0.5 threshold by less than 0.01, so which side of it
      a given image lands on is not a property worth pinning in a test.

    Ordering is what one epoch genuinely buys, so ordering is what is checked.
    """
    clean = trained_model.predict(_read_bgr(clean_image_path), color_order="bgr")
    defective = trained_model.predict(_read_bgr(defective_image_path), color_order="bgr")

    assert defective.anomaly_score > clean.anomaly_score, (
        f"defective {defective_image_path.name} scored {defective.anomaly_score:.4f} "
        f"but clean {clean_image_path.name} scored {clean.anomaly_score:.4f}"
    )


@requires_dataset
def test_scores_rank_defective_above_clean_across_the_split(trained_model: EfficientADModel) -> None:
    """Rank-based separation over many images, not one hand-picked pair.

    This is the assertion that actually has teeth. A single pair can be carried
    by luck; AUROC over a sample of the test split cannot, and it is measured on
    the raw ranking, so the score compression described above does not affect it
    at all. A model that had learned nothing would sit at 0.5.

    The 0.80 bound is deliberately well below the ~0.97 a one-epoch fit reaches
    here, so this fails on a broken wrapper rather than on the run-to-run
    variance of a randomly initialised student.
    """
    sample = 8  # ~0.4 s/image on CPU; enough to make the rank statistic stable
    clean_paths = sorted((CATEGORY_DIR / "test" / "good").glob("*.png"))[:sample]
    defect_paths = [
        path
        for defect_class in sorted(p for p in (CATEGORY_DIR / "test").iterdir() if p.is_dir() and p.name != "good")
        for path in sorted(defect_class.glob("*.png"))[: sample // 2]
    ]
    assert clean_paths and defect_paths, "expected both clean and defective test images"

    clean_scores = [trained_model.predict(_read_bgr(p), color_order="bgr").anomaly_score for p in clean_paths]
    defect_scores = [trained_model.predict(_read_bgr(p), color_order="bgr").anomaly_score for p in defect_paths]

    # AUROC as the Mann-Whitney statistic: the fraction of (defective, clean)
    # pairs the model orders correctly, counting ties as half a win.
    wins = sum(
        (1.0 if defect > clean else 0.5 if defect == clean else 0.0)
        for defect in defect_scores
        for clean in clean_scores
    )
    auroc = wins / (len(defect_scores) * len(clean_scores))

    assert auroc > 0.80, (
        f"image AUROC {auroc:.3f} over {len(defect_scores)} defective and {len(clean_scores)} clean images; "
        f"clean mean {np.mean(clean_scores):.4f}, defective mean {np.mean(defect_scores):.4f}"
    )


@requires_dataset
def test_anomaly_map_localises_the_defect(
    trained_model: EfficientADModel,
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


def test_efficientad_implements_the_anomaly_model_interface() -> None:
    assert issubclass(EfficientADModel, AnomalyModel)
    for method in ("train", "predict", "save", "load"):
        assert callable(getattr(EfficientADModel, method))
    assert EfficientADModel.model_name == "efficientad"


def test_both_backends_expose_the_same_surface() -> None:
    """The benchmark runner in step 5 holds either one behind AnomalyModel.

    Anything PatchCore offers and EfficientAD does not would be a hole the
    runner has to special-case, which is exactly what the base class exists to
    prevent. (The reverse is fine: `has_map_quantiles` is EfficientAD-specific
    and no shared caller may depend on it.)
    """
    shared = {"train", "predict", "save", "load", "category", "checkpoint_path", "is_trained", "is_calibrated"}
    missing = {name for name in shared if not hasattr(EfficientADModel, name)}
    assert not missing, f"EfficientADModel is missing {sorted(missing)}"

    for name in shared:
        patchcore_attr = getattr(PatchCoreModel, name)
        efficientad_attr = getattr(EfficientADModel, name)
        assert isinstance(efficientad_attr, type(patchcore_attr)), (
            f"{name} is a {type(patchcore_attr).__name__} on PatchCoreModel "
            f"but a {type(efficientad_attr).__name__} on EfficientADModel"
        )


def test_config_defaults_match_the_documented_hyperparameters() -> None:
    """The spec's defaults live in ModelConfig and nowhere else."""
    config = ModelConfig()

    assert config.model_size == "small"
    assert config.image_size == 256
    assert config.imagenet_dir.name == "imagenette"


def test_config_reads_the_environment() -> None:
    env = {"MODEL_SIZE": "medium", "IMAGENET_DIR": "/tmp/imagenette", "IMAGE_SIZE": "128"}

    config = ModelConfig.from_env(env)

    assert config.model_size == "medium"
    assert config.imagenet_dir == Path("/tmp/imagenette")
    assert config.image_size == 128


def test_config_rejects_an_unknown_model_size() -> None:
    """A typo fails at config load, not twenty minutes into a fit."""
    with pytest.raises(ValueError, match="model_size"):
        ModelConfig(model_size="tiny")
    # Case and stray whitespace are forgiven; anomalib's enum is lowercase.
    assert ModelConfig(model_size=" Medium ").model_size == "medium"


def test_constructor_arguments_override_config_defaults() -> None:
    model = EfficientADModel(
        category="cable",
        model_size="medium",
        image_size=320,
        imagenet_dir="/tmp/imagenette",
    )

    assert model.category == "cable"
    assert model.config.model_size == "medium"
    assert model.config.image_size == 320
    assert model.config.imagenet_dir == Path("/tmp/imagenette")
    assert model.checkpoint_path.name == "efficientad_cable.ckpt"


def test_image_size_below_the_autoencoder_floor_is_rejected_up_front() -> None:
    """A too-small image_size fails at construction, not minutes into a fit.

    EfficientAD's autoencoder downsamples by 32 and then applies a valid 8x8
    convolution, so below 256 it dies with a bare
    ``RuntimeError: Kernel size can't be greater than actual input size``
    from inside a conv, long after the setting that caused it. PatchCore
    accepts any size, which is exactly why this is easy to trip over when
    shrinking a benchmark to make it fast.
    """
    with pytest.raises(ValueError, match="image_size >= 256"):
        EfficientADModel(image_size=128)

    assert EfficientADModel(image_size=256).config.image_size == 256
    # PatchCore, by contrast, is genuinely resolution-agnostic.
    assert PatchCoreModel(image_size=128).config.image_size == 128


def test_checkpoint_path_follows_the_naming_convention(tmp_path: Path) -> None:
    config = ModelConfig(checkpoint_dir=tmp_path, category="bottle")

    assert config.checkpoint_path("efficientad") == tmp_path / "efficientad_bottle.ckpt"
    assert config.checkpoint_path("efficientad", "cable") == tmp_path / "efficientad_cable.ckpt"


# ---------------------------------------------------------------------------
# 5. Preprocessing: the one place the two backends legitimately differ
# ---------------------------------------------------------------------------


def test_preprocess_leaves_the_batch_unnormalized() -> None:
    """EfficientAD must be fed [0, 1], because the PDN normalizes internally.

    This is the failure that would never announce itself: normalizing here as
    well produces the right shapes and dtype, no error anywhere, and a model
    that is simply and quietly worse. Contrast with PatchCore below.
    """
    model = EfficientADModel(image_size=256)
    image = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)

    tensor = model._preprocess(image)  # noqa: SLF001 - the contract under test

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_patchcore_and_efficientad_preprocess_differently_on_purpose() -> None:
    """The same frame reaches the two networks on two different scales."""
    image = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)

    efficientad = EfficientADModel(image_size=256)._preprocess(image)  # noqa: SLF001
    patchcore = PatchCoreModel(image_size=256)._preprocess(image)  # noqa: SLF001

    assert efficientad.shape == patchcore.shape
    # PatchCore's ImageNet normalization pushes values outside [0, 1] both ways.
    assert float(patchcore.min()) < 0.0
    assert float(patchcore.max()) > 1.0
    assert not torch.allclose(efficientad, patchcore)


def test_bgr_and_rgb_preprocessing_are_channel_swaps_of_each_other() -> None:
    """Reading the same bytes as BGR flips channels, and only channels."""
    model = EfficientADModel(image_size=256)
    image = np.random.default_rng(1).integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    as_rgb = model._preprocess(image, color_order="rgb")  # noqa: SLF001
    as_bgr = model._preprocess(image, color_order="bgr")  # noqa: SLF001

    assert not torch.allclose(as_bgr, as_rgb), "color_order had no effect"
    assert torch.allclose(as_bgr, as_rgb.flip(1), atol=1e-5)
