"""Tests for the WinCLIP wrapper, and for the zero-shot claim in particular.

This file is shaped differently from ``tests/test_patchcore.py`` and
``tests/test_efficientad.py``, and the difference is the point.

Those two open with a module-scoped fixture that *trains a model*, because
nothing they assert is meaningful until one exists. **This file never calls
:meth:`~app.models.winclip.WinCLIPModel.train`.** Every prediction below comes
out of a wrapper that has been constructed and nothing else — no fit, no
checkpoint, no dataset, in some cases not even an image off disk. If any of it
starts requiring a training step, the project's headline architectural claim has
quietly stopped being true, and these tests are what says so.

Two consequences worth knowing before reading the assertions:

* **The bar for accuracy is lower, deliberately.** A zero-shot model's scores
  come from prompt similarity, not from this category's statistics, so the
  clean/defective margin is thinner than PatchCore's and the pixel map is
  coarse by construction (15x15 patches upsampled). The assertions check
  ordering and localisation, not published AUROC.
* **It is slow.** A ViT-B/16-plus forward plus ~365 masked window poolings is
  ~4 s per image on CPU, so predictions are computed once in module-scoped
  fixtures and shared. Building the model loads ~830 MB of CLIP weights, which
  open_clip downloads on the first ever run and caches under ``~/.cache/clip``.

Dataset-backed tests skip (rather than fail) when ``data/MVTecAD/bottle`` is
absent. Populate it with::

    python scripts/download_dataset.py --category bottle

The calibration test — the only one that needs :meth:`train` — is skipped unless
``WINCLIP_SLOW_TESTS=1``, because it scores the whole test split twice and takes
several minutes on CPU::

    WINCLIP_SLOW_TESTS=1 pytest tests/test_winclip.py -k calibrat
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.data import DataModule
from app.data.datamodule import DEFAULT_DATA_ROOT
from app.models import (
    AnomalyModel,
    EfficientADModel,
    ModelConfig,
    ModelOutput,
    PatchCoreModel,
    WinCLIPModel,
)
from anomalib.callbacks.checkpoint import ModelCheckpoint

from app.models.winclip import PATCH_GRID_SIZE, REQUIRED_IMAGE_SIZE

CATEGORY = "bottle"

CATEGORY_DIR: Path = DEFAULT_DATA_ROOT / CATEGORY

requires_dataset = pytest.mark.skipif(
    not (CATEGORY_DIR / "test").is_dir(),
    reason=f"{CATEGORY_DIR} not found; run `python scripts/download_dataset.py --category {CATEGORY}`",
)

slow = pytest.mark.skipif(
    os.environ.get("WINCLIP_SLOW_TESTS") != "1",
    reason="scores the full test split on CPU; set WINCLIP_SLOW_TESTS=1 to run",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def untrained_config(tmp_path_factory: pytest.TempPathFactory) -> ModelConfig:
    """A config whose checkpoint directory is empty, and stays that way.

    Pointing ``checkpoint_dir`` at a fresh temp directory is not tidiness — it is
    part of the test. ``WinCLIPModel._require_module`` loads calibration from
    ``checkpoint_path`` when a file happens to be there, so a stray
    ``results/checkpoints/winclip_bottle.ckpt`` left by the demo script would
    make every "zero-shot" test below silently exercise the *calibrated* path.
    """
    return ModelConfig.from_env(
        category=CATEGORY,
        accelerator="cpu",
        checkpoint_dir=tmp_path_factory.mktemp("checkpoints"),
        results_dir=tmp_path_factory.mktemp("results"),
    )


@pytest.fixture(scope="module")
def zero_shot_model(untrained_config: ModelConfig) -> WinCLIPModel:
    """A WinCLIP that has been constructed and nothing else.

    Note what is *not* here, compared with the sibling test files: no
    ``model.train(datamodule)``. This fixture is the subject of the whole file.
    """
    model = WinCLIPModel(class_name=CATEGORY, config=untrained_config)
    assert not model.checkpoint_path.exists(), "the fixture must start with no calibration on disk"
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


@pytest.fixture(scope="module")
def defective_result(zero_shot_model: WinCLIPModel, defective_image_path: Path) -> ModelOutput:
    """One scored defective bottle, shared by every test that needs one (~4 s)."""
    return zero_shot_model.predict(_read_bgr(defective_image_path), color_order="bgr")


@pytest.fixture(scope="module")
def clean_result(zero_shot_model: WinCLIPModel, clean_image_path: Path) -> ModelOutput:
    """One scored intact bottle, shared by every test that needs one (~4 s)."""
    return zero_shot_model.predict(_read_bgr(clean_image_path), color_order="bgr")


# ---------------------------------------------------------------------------
# 1. The zero-shot claim: predict() with no training whatsoever
# ---------------------------------------------------------------------------


def test_predict_runs_with_no_training_and_no_dataset(zero_shot_model: WinCLIPModel) -> None:
    """The headline test. A bare wrapper scores an image; nothing was fitted.

    Deliberately uses a synthetic array rather than a bottle off disk, so it
    cannot pass by accident on a machine that happens to have the dataset. The
    only thing this model has ever been told about bottles is the string
    ``"bottle"``.

    The two sibling backends raise :class:`RuntimeError` from exactly this call.
    """
    assert zero_shot_model.is_zero_shot
    assert zero_shot_model.k_shot == 0
    assert zero_shot_model.is_trained, "a zero-shot model is ready the moment it is constructed"
    assert not zero_shot_model.is_calibrated, "nothing has been fitted, so there are no score statistics"

    image = np.random.default_rng(0).integers(0, 256, size=(320, 480, 3), dtype=np.uint8)

    result = zero_shot_model.predict(image)

    assert isinstance(result, ModelOutput)
    assert np.isfinite(result.anomaly_score)
    assert result.anomaly_map.shape == (320, 480)


def test_the_sibling_backends_cannot_do_this(tmp_path: Path) -> None:
    """The contrast that makes the test above worth having.

    PatchCore and EfficientAD raise on an untrained ``predict``. If this ever
    stops being true the comparison in ``app/models/winclip.py``'s docstring is
    wrong, and so is the reason this project ships three backends.
    """
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    config = ModelConfig.from_env(category=CATEGORY, checkpoint_dir=tmp_path)

    with pytest.raises(RuntimeError, match="not trained"):
        PatchCoreModel(config=config).predict(image)
    with pytest.raises(RuntimeError, match="not trained"):
        EfficientADModel(config=config).predict(image)


@requires_dataset
def test_predict_scores_a_real_bottle_without_training(defective_result: ModelOutput) -> None:
    """The same claim against a real MVTec image, and a well-formed output.

    WinCLIP's score is a softmax over the normal and anomalous prompt
    embeddings, so unlike PatchCore's raw nearest-neighbour distance it is in
    ``[0, 1]`` *before* any calibration — worth asserting, because it is why
    ``anomaly_threshold`` means something on an uncalibrated model here.
    """
    assert isinstance(defective_result.anomaly_score, float)
    assert 0.0 <= defective_result.anomaly_score <= 1.0

    assert defective_result.anomaly_map.dtype == np.float32
    assert np.isfinite(defective_result.anomaly_map).all()
    assert 0.0 <= float(defective_result.anomaly_map.min())
    assert float(defective_result.anomaly_map.max()) <= 1.0

    assert isinstance(defective_result.is_defective, bool)


def test_few_shot_without_references_refuses_rather_than_degrading(untrained_config: ModelConfig) -> None:
    """``k_shot > 0`` is the one configuration that *does* need data.

    The failure to avoid is a few-shot model quietly falling back to zero-shot
    when its reference images are missing: it would predict, the numbers would
    look plausible, and the k_shot setting would be a no-op.
    """
    model = WinCLIPModel(class_name=CATEGORY, k_shot=4, config=untrained_config)

    assert not model.is_zero_shot
    assert not model.is_trained

    with pytest.raises(RuntimeError, match="k_shot=4"):
        model.predict(np.zeros((32, 32, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 2. The ModelOutput contract
# ---------------------------------------------------------------------------


@requires_dataset
def test_model_name_is_winclip(defective_result: ModelOutput) -> None:
    """Results are self-describing, so a log line or API response can attribute a score."""
    assert defective_result.model_name == "winclip"
    assert WinCLIPModel.model_name == "winclip"


@requires_dataset
def test_anomaly_map_shape_matches_the_input_image(
    defective_result: ModelOutput,
    defective_image_path: Path,
) -> None:
    """The heatmap comes back at the caller's resolution, not the model's.

    WinCLIP works at 240x240 and scores on a 15x15 patch grid; MVTec images are
    900x900. Absorbing both numbers is the wrapper's job — a caller overlays the
    map on the frame it passed in without rescaling anything.
    """
    height, width = _read_bgr(defective_image_path).shape[:2]

    assert defective_result.anomaly_map.ndim == 2
    assert defective_result.anomaly_map.shape == (height, width)
    assert defective_result.shape == (height, width)


@requires_dataset
@pytest.mark.parametrize(("size", "channels"), [((64, 96), 1), ((512, 400), 4)])
def test_predict_accepts_any_size_and_channel_count(
    zero_shot_model: WinCLIPModel,
    defective_image_path: Path,
    size: tuple[int, int],
    channels: int,
) -> None:
    """Non-square resolutions, grayscale and RGBA all produce a matching map."""
    bgr = cv2.resize(_read_bgr(defective_image_path), (size[1], size[0]))
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY if channels == 1 else cv2.COLOR_BGR2BGRA)

    result = zero_shot_model.predict(image, color_order="bgr")

    assert result.anomaly_map.shape == size


@requires_dataset
def test_repeated_predicts_do_not_leak_forward_hooks(
    zero_shot_model: WinCLIPModel,
    defective_result: ModelOutput,  # noqa: ARG001 - forces the module to be built once, cheaply shared
) -> None:
    """anomalib's encode_image leaks a forward hook per call; the wrapper strips them.

    Left alone this is an unbounded ~5.7 MB-per-frame leak on the serving path
    (each leaked hook's closure pins the feature map it captured) and, because
    the hook is a local closure, it also makes the module unpicklable — which is
    the failure that broke ``train()`` before ``_build_engine`` was taught to
    suppress Lightning checkpointing. This asserts the containment directly.
    """
    module = zero_shot_model._module  # noqa: SLF001 - the leak lives in a private hook dict
    assert module is not None

    def leaked_hooks() -> int:
        return sum(
            1
            for submodule in module.model.modules()
            for fn in submodule._forward_hooks.values()  # noqa: SLF001 - the dict under test
            if getattr(fn, "__qualname__", "").endswith("get_feature_map.<locals>.hook")
        )

    # A guard-passing synthetic frame: predict() now runs FrameGuard first, and a
    # tiny all-black tile would be rejected (too_small + underexposed) before any
    # forward pass, which is not what this test is about. Random noise at a valid
    # resolution clears the gate cheaply.
    frame = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    before = leaked_hooks()
    for _ in range(3):
        zero_shot_model.predict(frame)

    assert leaked_hooks() == before == 0, "encode_image hooks are accumulating; _clear_leaked_hooks is not running"


def test_predict_rejects_malformed_input(zero_shot_model: WinCLIPModel) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        zero_shot_model.predict("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="color_order"):
        zero_shot_model.predict(np.zeros((8, 8, 3), dtype=np.uint8), color_order="rgba")
    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        zero_shot_model.predict(np.zeros((8, 8, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        zero_shot_model.predict(np.zeros((2, 8, 8, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 3. The model is actually doing zero-shot work
# ---------------------------------------------------------------------------


@requires_dataset
def test_defective_image_scores_higher_than_a_clean_one(
    clean_result: ModelOutput,
    defective_result: ModelOutput,
) -> None:
    """Prompt similarity alone separates a broken bottle from an intact one.

    No margin is demanded. PatchCore requires a 0.05 gap because it has seen
    hundreds of bottles; this model has seen the word. Ordering is the property
    zero-shot actually buys, so ordering is what is checked.
    """
    assert defective_result.anomaly_score > clean_result.anomaly_score, (
        f"defective scored {defective_result.anomaly_score:.4f} but clean scored "
        f"{clean_result.anomaly_score:.4f}"
    )


@requires_dataset
def test_anomaly_map_is_not_a_uniform_field(defective_result: ModelOutput, clean_result: ModelOutput) -> None:
    """Regression test for a failure mode that produces no error at all.

    anomalib drives WinCLIP's window pooling by permuting tokens to length-first
    and calling ``clip.visual.transformer`` by hand. open_clip made that stack
    batch-first in 2.26, and under a newer open_clip every window pools to the
    same vector — so every window scores identically and the anomaly map comes
    back a perfectly flat field. Nothing raises; the model simply detects
    nothing, while still returning a plausible image-level score.

    ``requirements.txt`` pins ``open-clip-torch<2.26.1`` to prevent it. This is
    what notices if that pin is ever lost.
    """
    for name, result in (("defective", defective_result), ("clean", clean_result)):
        spread = float(result.anomaly_map.max() - result.anomaly_map.min())
        assert spread > 1e-3, (
            f"{name} anomaly map is uniform (spread {spread:.2e}); the sliding windows are collapsing. "
            f"Check the installed open_clip version against the pin in requirements.txt."
        )


@requires_dataset
def test_anomaly_map_localises_the_defect(
    defective_result: ModelOutput,
    defective_image_path: Path,
) -> None:
    """The map is spatially meaningful, not the image score painted flat.

    This is the assertion with real content: a coarse 15x15 grid upsampled to
    900x900 still has to put its warm region where the ground-truth mask is.
    """
    mask_dir = CATEGORY_DIR / "ground_truth" / defective_image_path.parent.name
    mask_path = next(mask_dir.glob(f"{defective_image_path.stem}*.png"))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    assert mask.any(), f"ground truth {mask_path} is empty"

    inside = float(defective_result.anomaly_map[mask].mean())
    outside = float(defective_result.anomaly_map[~mask].mean())

    assert inside > outside, f"defect region scored {inside:.4f} vs {outside:.4f} elsewhere"


@requires_dataset
def test_the_prompt_is_load_bearing(zero_shot_model: WinCLIPModel, defective_image_path: Path) -> None:
    """Changing the class name changes the score, which is the mechanism working.

    If a nonsense class name produced the same answer, the model would not be
    reading the text encoder at all and the "zero-shot" story would be a
    coincidence of the image encoder.
    """
    image = _read_bgr(defective_image_path)
    as_bottle = zero_shot_model.predict(image, color_order="bgr").anomaly_score

    other = WinCLIPModel(class_name="hazelnut", category=CATEGORY, config=zero_shot_model.config)
    as_hazelnut = other.predict(image, color_order="bgr").anomaly_score

    assert as_bottle != pytest.approx(as_hazelnut, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. train() — calibration, not optimization
# ---------------------------------------------------------------------------


def test_train_rejects_a_raw_anomalib_datamodule() -> None:
    """The data contract is enforced here too, cheap as this model's train() is."""
    model = WinCLIPModel(class_name=CATEGORY)
    with pytest.raises(TypeError, match="app.data.DataModule"):
        model.train(object())


def test_train_engine_does_not_write_a_lightning_checkpoint(untrained_config: ModelConfig) -> None:
    """The calibration Engine claims the checkpoint slot only to suppress saving.

    Two things ride on this, both anomalib quirks specific to the zero-shot
    path. anomalib injects its own ``ModelCheckpoint`` when the list is empty and
    (uniquely for zero-/few-shot) saves it at *validation* end, which would dump
    ~830 MB of CLIP weights into a versioned results tree that ``save()`` exists
    to avoid — and that dump would raise ``AttributeError`` besides, because
    encode_image's leaked hook makes the module unpicklable. A ``save_top_k=0``
    callback claims the slot and writes nothing. No dataset or CLIP weights are
    needed to check the wiring, so this is a fast test.
    """
    engine = WinCLIPModel(class_name=CATEGORY, config=untrained_config)._build_engine()  # noqa: SLF001

    checkpoints = [c for c in engine._cache.args["callbacks"] if isinstance(c, ModelCheckpoint)]  # noqa: SLF001

    assert checkpoints, "no ModelCheckpoint installed, so anomalib will inject its own saving one"
    assert all(c.save_top_k == 0 for c in checkpoints), "the checkpoint callback would write an 830 MB Lightning dump"
    assert all(not c.save_last for c in checkpoints)


@slow
@requires_dataset
def test_train_calibrates_without_changing_the_backbone(untrained_config: ModelConfig, tmp_path: Path) -> None:
    """train() fits score statistics and leaves every CLIP weight untouched.

    The sibling models' equivalent test asserts that parameters *moved*
    (EfficientAD's student) or that a memory bank filled up (PatchCore). The
    assertion here is the opposite one, because "training" a zero-shot model
    must not train anything: what may change is the post-processor's
    normalization statistics, and nothing else.
    """
    model = WinCLIPModel(class_name=CATEGORY, config=untrained_config.with_overrides(checkpoint_dir=tmp_path))
    reference = model._build_module()  # noqa: SLF001 - a pristine backbone to compare against

    datamodule = DataModule(
        category=CATEGORY,
        image_size=REQUIRED_IMAGE_SIZE,
        batch_size=untrained_config.batch_size,
        root=untrained_config.data_root,
        num_workers=untrained_config.num_workers,
    )
    model.train(datamodule)

    assert model.is_calibrated, "the validation pass should have fitted normalization statistics"
    assert model.checkpoint_path.is_file()
    # A few hundred KB, not ~830 MB: the CLIP weights are deliberately not in it.
    assert model.checkpoint_path.stat().st_size < 10 * 1024**2

    trained = model._module  # noqa: SLF001 - asserting on internals is the point here
    assert torch.allclose(
        reference.model.clip.visual.conv1.weight,
        trained.model.clip.visual.conv1.weight,
    ), "a CLIP weight moved during train(); this model has no optimizer and must not fit anything"


@slow
@requires_dataset
def test_calibration_round_trips(untrained_config: ModelConfig, defective_image_path: Path, tmp_path: Path) -> None:
    """save() then load() into a fresh instance reproduces the same score."""
    model = WinCLIPModel(class_name=CATEGORY, config=untrained_config.with_overrides(checkpoint_dir=tmp_path))
    datamodule = DataModule(
        category=CATEGORY,
        image_size=REQUIRED_IMAGE_SIZE,
        batch_size=untrained_config.batch_size,
        root=untrained_config.data_root,
        num_workers=untrained_config.num_workers,
    )
    model.train(datamodule)

    image = _read_bgr(defective_image_path)
    expected = model.predict(image, color_order="bgr").anomaly_score

    destination = tmp_path / "round_trip.ckpt"
    model.save(destination)

    reloaded = WinCLIPModel(class_name=CATEGORY, config=untrained_config)
    reloaded.load(destination)

    assert reloaded.is_calibrated
    assert reloaded.predict(image, color_order="bgr").anomaly_score == pytest.approx(expected, abs=1e-5)


def test_load_refuses_a_sibling_backends_checkpoint(tmp_path: Path) -> None:
    """A Lightning checkpoint is the plausible mistake, so it fails clearly.

    WinCLIP's artifact is a small calibration payload rather than a full
    checkpoint (see ``WinCLIPModel.save``), so handing it a PatchCore ``.ckpt``
    would otherwise die on an opaque key error deep inside ``load_state_dict``.
    """
    foreign = tmp_path / "patchcore_bottle.ckpt"
    torch.save({"state_dict": {}, "pytorch-lightning_version": "2.0.0"}, foreign)

    with pytest.raises(ValueError, match="not a WinCLIP calibration artifact"):
        WinCLIPModel(class_name=CATEGORY).load(foreign)


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WinCLIPModel(class_name=CATEGORY).load(tmp_path / "nope.ckpt")


# ---------------------------------------------------------------------------
# 5. Interface and configuration (no dataset, no CLIP weights needed)
# ---------------------------------------------------------------------------


def test_winclip_implements_the_anomaly_model_interface() -> None:
    assert issubclass(WinCLIPModel, AnomalyModel)
    for method in ("train", "predict", "save", "load"):
        assert callable(getattr(WinCLIPModel, method))


def test_all_three_backends_expose_the_same_surface() -> None:
    """The benchmark runner in step 5 holds any of the three behind AnomalyModel.

    Anything two backends offer and the third does not is a hole the runner has
    to special-case, which is what the base class exists to prevent. (The
    reverse is fine: ``is_zero_shot`` is WinCLIP-specific and no shared caller
    may depend on it.)
    """
    shared = {"train", "predict", "save", "load", "category", "checkpoint_path", "is_trained", "is_calibrated"}

    missing = {name for name in shared if not hasattr(WinCLIPModel, name)}
    assert not missing, f"WinCLIPModel is missing {sorted(missing)}"

    for name in shared:
        patchcore_attr = getattr(PatchCoreModel, name)
        winclip_attr = getattr(WinCLIPModel, name)
        assert isinstance(winclip_attr, type(patchcore_attr)), (
            f"{name} is a {type(patchcore_attr).__name__} on PatchCoreModel "
            f"but a {type(winclip_attr).__name__} on WinCLIPModel"
        )


def test_config_defaults_match_the_documented_hyperparameters() -> None:
    """The spec's defaults live in ModelConfig and nowhere else."""
    config = ModelConfig()

    assert config.k_shot == 0, "zero-shot is the default, not an opt-in"
    assert config.scales == (2, 3)
    assert config.class_name is None
    assert config.prompt_class_name == "bottle"


def test_config_reads_the_environment() -> None:
    env = {"K_SHOT": "4", "WINCLIP_SCALES": "2, 3, 4", "WINCLIP_CLASS_NAME": "printed circuit board"}

    config = ModelConfig.from_env(env)

    assert config.k_shot == 4
    assert config.scales == (2, 3, 4)
    assert config.prompt_class_name == "printed circuit board"


def test_prompt_class_name_turns_a_folder_name_into_a_phrase() -> None:
    """MVTec categories are identifiers; CLIP's text encoder was trained on prose."""
    assert ModelConfig(category="metal_nut").prompt_class_name == "metal nut"
    assert ModelConfig(category="bottle").prompt_class_name == "bottle"
    # An explicit class name always wins over the derivation.
    assert ModelConfig(category="metal_nut", class_name="hex nut").prompt_class_name == "hex nut"


def test_config_rejects_a_degenerate_window_scale() -> None:
    """A 1-patch window carries no context and adds nothing the image scale lacks."""
    with pytest.raises(ValueError, match="scales"):
        ModelConfig(scales=(1, 2))
    with pytest.raises(ValueError, match="k_shot"):
        ModelConfig(k_shot=-1)


def test_constructor_arguments_override_config_defaults() -> None:
    model = WinCLIPModel(class_name="cable", k_shot=2, scales=(2, 3, 4))

    assert model.class_name == "cable"
    # class_name seeds the category when the latter is not given explicitly.
    assert model.category == "cable"
    assert model.k_shot == 2
    assert model.scales == (2, 3, 4)
    assert model.checkpoint_path.name == "winclip_cable.ckpt"


def test_category_and_class_name_can_differ() -> None:
    """The dataset folder and the prompt noun are not the same string."""
    model = WinCLIPModel(class_name="metal nut", category="metal_nut")

    assert model.category == "metal_nut"
    assert model.class_name == "metal nut"
    assert model.checkpoint_path.name == "winclip_metal_nut.ckpt"


# ---------------------------------------------------------------------------
# 6. Preprocessing: WinCLIP disagrees with both siblings, on purpose
# ---------------------------------------------------------------------------


def test_image_size_is_pinned_to_the_backbones_fixed_resolution() -> None:
    """A configured image_size is absorbed, not honoured and not rejected.

    ViT-B-16-plus-240's positional embeddings are a fixed 15x15 grid, so 240 is
    not a preference. Rejecting a 256 config would be worse than overriding it:
    256 is the process-wide default that the other two backends need.
    """
    assert WinCLIPModel().config.image_size == REQUIRED_IMAGE_SIZE
    assert WinCLIPModel(config=ModelConfig(image_size=512)).config.image_size == REQUIRED_IMAGE_SIZE
    assert REQUIRED_IMAGE_SIZE == 240
    assert PATCH_GRID_SIZE == 15
    # There is no image_size argument to pass, which is the honest signature:
    # unlike its siblings this backend has no say in the matter.
    assert "image_size" not in inspect.signature(WinCLIPModel.__init__).parameters


def test_preprocess_uses_clip_statistics_not_imagenet() -> None:
    """The silent one: ImageNet stats here would shift every image embedding.

    Nothing about the shapes or dtype would change, and no error would fire —
    the cosine similarity to both text prompts would just drift. Contrast with
    PatchCore, which must use ImageNet statistics, and EfficientAD, which must
    use none.
    """
    image = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)

    winclip = WinCLIPModel()._preprocess(image)  # noqa: SLF001 - the contract under test
    patchcore = PatchCoreModel(image_size=REQUIRED_IMAGE_SIZE)._preprocess(image)  # noqa: SLF001
    efficientad = EfficientADModel(image_size=256)._preprocess(image)  # noqa: SLF001

    assert winclip.shape == (1, 3, REQUIRED_IMAGE_SIZE, REQUIRED_IMAGE_SIZE)
    assert winclip.dtype == torch.float32
    # Normalized, so it leaves [0, 1] in both directions...
    assert float(winclip.min()) < 0.0
    assert float(winclip.max()) > 1.0
    # ...but not by the same amounts as ImageNet's statistics would produce.
    assert winclip.shape == patchcore.shape
    assert not torch.allclose(winclip, patchcore)
    # EfficientAD is unnormalized altogether, and works at a different size.
    assert efficientad.shape != winclip.shape


def test_preprocess_resizes_with_bicubic_like_open_clip() -> None:
    """CLIP's own preprocessing is bicubic, so ours is too.

    A small difference, and the only reason it is pinned in a test is that it is
    invisible otherwise: bilinear input produces a working model with slightly
    wrong embeddings.
    """
    assert WinCLIPModel._resize_mode == "bicubic"  # noqa: SLF001 - the hook under test
    assert PatchCoreModel._resize_mode == "bilinear"  # noqa: SLF001

    image = np.random.default_rng(2).integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
    model = WinCLIPModel()

    bicubic = model._preprocess(image)  # noqa: SLF001
    model._resize_mode = "bilinear"  # type: ignore[misc]
    try:
        bilinear = model._preprocess(image)  # noqa: SLF001
    finally:
        del model._resize_mode  # restore the class attribute

    assert not torch.allclose(bicubic, bilinear)


def test_bgr_and_rgb_preprocessing_are_channel_swaps_of_each_other() -> None:
    """Reading the same bytes as BGR flips channels, and only channels.

    Note that CLIP's per-channel statistics differ, so the swap is applied
    before normalization; asserting on the flipped *normalized* tensor would
    fail even on correct code.
    """
    model = WinCLIPModel()
    image = np.random.default_rng(1).integers(0, 256, size=(80, 80, 3), dtype=np.uint8)

    as_rgb = model._to_rgb_array(image, color_order="rgb")  # noqa: SLF001
    as_bgr = model._to_rgb_array(image, color_order="bgr")  # noqa: SLF001

    assert np.array_equal(as_bgr, as_rgb[:, :, ::-1])
