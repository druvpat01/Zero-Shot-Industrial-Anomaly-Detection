"""PatchCore anomaly detection, behind the project's own model interface.

This module is the only place allowed to ``import anomalib.models``. Everything
else goes through :class:`~app.models.base.AnomalyModel`.

How PatchCore works
===================
PatchCore is a *memory bank* method. It never trains a single weight: the
backbone is frozen ImageNet-pretrained WideResNet-50, and "training" means
memorising what normal looks like.

**1. Patch features from mid-level layers.** Each training image is pushed
through the backbone, and the activation maps of ``layer2`` and ``layer3`` are
kept. That choice is the whole trick, and it is a deliberate middle ground:

* ``layer1`` (stride 4) is still close to raw pixels — edges, corners, local
  colour gradients. Too generic to separate "scratch" from "specular highlight",
  and its 256x256/4 = 64x64 grid makes for an enormous memory bank.
* ``layer4`` (stride 32) is the most semantic, but it was trained to answer
  *"which of 1000 ImageNet classes is this?"*. It has learned to discard exactly
  the information we need — a bottle with a chipped rim is still, emphatically,
  a bottle — and its 8x8 grid is far too coarse to localise a defect.
* ``layer2`` and ``layer3`` (strides 8 and 16) sit where features are abstract
  enough to describe texture and local structure, yet not so abstract that they
  have thrown away appearance. They are also less biased towards ImageNet's
  object categories than ``layer4``, which matters because industrial parts look
  nothing like ImageNet photos.

The two maps are bilinearly aligned to the coarser grid and concatenated, so
every spatial position ends up with one descriptor carrying both scales. Local
average pooling (3x3) over the map gives each patch a small receptive-field
neighbourhood, making the descriptor tolerant to a pixel or two of misalignment.

**2. Coreset subsampling — why it is done.** A single 256x256 image yields
32x32 = 1024 patch descriptors of ~1536 dimensions. 200 training images is
therefore ~200k descriptors, and a realistic 1000-image line dataset is over a
million. Storing all of them costs gigabytes, and — worse — inference is a
nearest-neighbour search *against every one of them* for each of the 1024 test
patches. Kept whole, the bank makes PatchCore accurate and unusable.

The observation that rescues it: those descriptors are enormously redundant.
Most patches of a bottle are plain background or plain glass, and the bank holds
thousands of near-identical copies of each. What nearest-neighbour search
actually needs is *coverage* — that no region of normal-feature space is left
unrepresented — not density. So we keep a **coreset**: a small subset whose
maximum distance to any discarded point is as small as possible.

**3. How k-center-greedy works, conceptually.** Finding the true minimax coreset
is NP-hard, so PatchCore uses the standard greedy approximation, which gets
within a factor of 2:

1. Start with one arbitrary descriptor as the bank; record every other
   descriptor's distance to it.
2. Pick the descriptor that is *furthest* from everything selected so far — the
   point currently worst covered — and add it to the bank.
3. Update each remaining descriptor's "distance to the nearest selected point"
   (a running elementwise minimum against just the newly added centre, which is
   why the whole thing is cheap).
4. Repeat until the bank holds ``coreset_sampling_ratio`` of the original.

Because each step maximises the minimum distance, the selection deliberately
spreads out: it grabs the rare patches — the label edge, the neck highlight,
the cap thread — instead of the ten-thousandth copy of flat background. At 1%
the bank shrinks 100x with almost no AUROC loss, which is what makes PatchCore
practical. Anomalib runs the search on a random projection of the descriptors
(Johnson-Lindenstrauss), so distances are approximately preserved at a fraction
of the arithmetic.

**4. Scoring.** A test patch is anomalous to the degree that it is far from its
nearest neighbour in the bank. Those per-patch distances, reshaped to the
feature grid and upsampled, *are* the anomaly map; the image-level score is the
max patch distance, re-weighted by how isolated that patch's nearest neighbour
is among its own ``num_neighbors`` neighbours — a bank point that is itself in a
sparse region is weak evidence of normality.

Reference: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*
(CVPR 2022), https://arxiv.org/abs/2106.08265
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import lightning
import numpy as np
import torch
# anomalib subclasses Lightning's ModelCheckpoint and only recognises its own
# type when deciding whether to inject a default one. Importing Lightning's here
# would land us with two checkpoint callbacks and a Trainer misconfiguration.
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.engine import Engine
from anomalib.models import Patchcore

from app.data.transforms import normalize_image
from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig, get_model_config

__all__ = ["PatchCoreModel"]

logger = logging.getLogger(__name__)


class PatchCoreModel(AnomalyModel):
    """PatchCore wrapped as an :class:`~app.models.base.AnomalyModel`.

    See the module docstring for what the algorithm does and why ``layer2`` /
    ``layer3`` and coreset subsampling are the choices they are.

    Every constructor argument defaults to ``None`` and falls through to
    :class:`~app.models.config.ModelConfig`, which resolves it from the
    environment and ultimately from a hardcoded default
    (``backbone="wide_resnet50_2"``, ``coreset_sampling_ratio=0.1``,
    ``num_neighbors=9``, ``image_size=256``). Defaults are not repeated here, so
    there is exactly one place in the codebase to change one.

    The fitted Lightning module is cached on the instance: :meth:`predict` reuses
    it, and only ever touches disk on the first call after a bare construction.

    Args:
        category: Object category this instance is for. Also selects the default
            checkpoint filename.
        backbone: timm backbone name.
        coreset_sampling_ratio: Fraction of patch descriptors the memory bank
            keeps.
        num_neighbors: Neighbours used to re-weight the image-level score.
        image_size: Square resolution images are resized to.
        config: Pre-built config to start from. Defaults to the process-wide one.

    Example:
        >>> from app.data import DataModule                      # doctest: +SKIP
        >>> model = PatchCoreModel(category="bottle")            # doctest: +SKIP
        >>> model.train(DataModule(category="bottle"))           # doctest: +SKIP
        >>> result = model.predict(cv2.imread(path), color_order="bgr")
        >>> result.anomaly_score, result.is_defective            # doctest: +SKIP
        (0.83, True)
    """

    model_name = "patchcore"

    def __init__(
        self,
        category: str | None = None,
        backbone: str | None = None,
        coreset_sampling_ratio: float | None = None,
        num_neighbors: int | None = None,
        image_size: int | None = None,
        *,
        config: ModelConfig | None = None,
    ) -> None:
        base = config if config is not None else get_model_config()
        super().__init__(
            base.with_overrides(
                category=category,
                backbone=backbone,
                coreset_sampling_ratio=coreset_sampling_ratio,
                num_neighbors=num_neighbors,
                image_size=image_size,
            ),
        )

        # Cached across predict() calls. Rebuilding either of these costs a
        # backbone download check plus a full state_dict load, which would
        # dominate the ~50 ms of actual inference work.
        self._module: Patchcore | None = None
        self._engine: Engine | None = None
        self._checkpoint_path: Path | None = None
        self._device = torch.device("cpu")

    # -- properties ------------------------------------------------------------

    @property
    def category(self) -> str:
        """Object category this instance scores."""
        return self.config.category

    @property
    def checkpoint_path(self) -> Path:
        """Where :meth:`train` writes, and a bare :meth:`predict` reads from."""
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        return self.config.checkpoint_path(self.model_name, self.category)

    @property
    def is_trained(self) -> bool:
        """Whether a memory bank is loaded and non-empty."""
        return self._module is not None and self._module.model.memory_bank.numel() > 0

    @property
    def is_calibrated(self) -> bool:
        """Whether score normalization statistics were fitted.

        These come from the validation pass inside :meth:`train`. Without them
        scores are raw nearest-neighbour distances rather than ``[0, 1]``.
        """
        if self._module is None or self._module.post_processor is None:
            return False
        post = self._module.post_processor
        return not bool(post.image_min.isnan() or post.image_max.isnan())

    # -- construction ----------------------------------------------------------

    def _build_pre_processor(self) -> Any:
        """anomalib's resize/normalize pipeline, sized from our config.

        Kept a function of :attr:`config` rather than of the checkpoint, so a
        model always preprocesses at the resolution the process is configured
        for. :meth:`predict` does its own preprocessing regardless; this is what
        anomalib uses during fit and what gets baked into an exported model, so
        the two must not drift apart.
        """
        return Patchcore.configure_pre_processor(image_size=self.config.image_hw)

    def _build_module(self) -> Patchcore:
        """Instantiate the anomalib Lightning module from the resolved config."""
        return Patchcore(
            backbone=self.config.backbone,
            layers=self.config.layers,
            pre_trained=True,
            coreset_sampling_ratio=self.config.coreset_sampling_ratio,
            num_neighbors=self.config.num_neighbors,
            pre_processor=self._build_pre_processor(),
        )

    def _build_engine(self) -> Engine:
        """Create the Lightning trainer wrapper used for fitting.

        Supplying our own ``ModelCheckpoint`` is what keeps the artifact story
        simple: anomalib only injects its own when the callback list has none,
        so this claims the slot and the checkpoint lands at
        :attr:`checkpoint_path` instead of in a versioned
        ``results/Patchcore/MVTecAD/<category>/vN/weights/`` tree that grows a
        directory per run.
        """
        checkpoint = self.config.checkpoint_path(self.model_name, self.category)
        return Engine(
            callbacks=[
                ModelCheckpoint(
                    dirpath=checkpoint.parent,
                    filename=checkpoint.stem,
                    auto_insert_metric_name=False,
                    save_last=False,
                ),
            ],
            default_root_dir=self.config.results_dir,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            max_epochs=self.config.max_epochs,
            logger=False,
        )

    def _require_module(self) -> Patchcore:
        """Return the cached module, loading the default checkpoint if needed."""
        if self._module is None:
            checkpoint = self.checkpoint_path
            if not checkpoint.is_file():
                msg = (
                    f"{type(self).__name__} for category {self.category!r} is not trained. "
                    f"Call train(datamodule), load(path), or run "
                    f"`python scripts/train_patchcore.py --category {self.category}` "
                    f"(expected a checkpoint at {checkpoint})."
                )
                raise RuntimeError(msg)
            logger.info("No model in memory; loading %s", checkpoint)
            self.load(checkpoint)

        module = self._module
        if module is None or module.model.memory_bank.numel() == 0:
            msg = "PatchCore memory bank is empty; the model was constructed but never fitted."
            raise RuntimeError(msg)
        return module

    # -- AnomalyModel ----------------------------------------------------------

    def train(self, datamodule: Any) -> None:
        """Build the memory bank from the category's defect-free train split.

        Runs anomalib's ``Engine.fit``, which for PatchCore means one pass over
        the training images to collect patch descriptors, the coreset subsampling
        described in the module docstring, then a validation pass that fits the
        score normalization and adaptive threshold. The Lightning checkpoint is
        written to ``results/checkpoints/patchcore_<category>.ckpt``.

        Args:
            datamodule: An :class:`app.data.DataModule`. Its ``category`` wins
                over this instance's config, so training a model for a category
                it was not constructed for cannot silently mislabel the
                checkpoint.

        Raises:
            TypeError: If ``datamodule`` is not an :class:`app.data.DataModule`.
        """
        if not hasattr(datamodule, "for_anomalib_engine"):
            msg = (
                f"train() expects an app.data.DataModule, got {type(datamodule).__name__}. "
                "Passing an anomalib datamodule directly bypasses the project's data contract."
            )
            raise TypeError(msg)

        if datamodule.category != self.config.category:
            logger.info(
                "Datamodule category %r overrides configured category %r",
                datamodule.category,
                self.config.category,
            )
            self.config = self.config.with_overrides(category=datamodule.category)

        if self.config.max_epochs != 1:
            logger.warning(
                "max_epochs=%d requested, but PatchCore is a single-pass memory-bank model; "
                "anomalib pins training to 1 epoch.",
                self.config.max_epochs,
            )

        self._module = self._build_module()
        self._engine = self._build_engine()

        started = time.perf_counter()
        # anomalib's Lightning module needs anomalib's own batch type; see
        # DataModule.for_anomalib_engine for why that is scoped to this block.
        with datamodule.for_anomalib_engine() as anomalib_datamodule:
            self._engine.fit(model=self._module, datamodule=anomalib_datamodule)
        elapsed = time.perf_counter() - started

        self._device = self._module.device
        self._module.eval()

        bank = self._module.model.memory_bank
        logger.info(
            "Fitted %s on %r in %.1fs; memory bank %s (coreset ratio %.3f)",
            self.model_name,
            self.category,
            elapsed,
            tuple(bank.shape),
            self.config.coreset_sampling_ratio,
        )
        if not self.is_calibrated:
            logger.warning(
                "No validation data was seen during fit, so scores are un-normalized "
                "nearest-neighbour distances rather than [0, 1].",
            )

        self.save(self.config.checkpoint_path(self.model_name, self.category))

    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        """Score one raw image and return a heatmap at its original resolution.

        The image is converted to 3-channel RGB, resized to ``image_size``,
        normalized with ImageNet statistics, scored against the memory bank, and
        the resulting low-resolution anomaly map is bilinearly upsampled back to
        the input's ``H x W`` so it can be overlaid directly.

        Args:
            image: ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``
                array; ``uint8`` in ``[0, 255]`` or float.
            color_order: ``"rgb"`` or ``"bgr"``. Use ``"bgr"`` for anything from
                OpenCV.

        Returns:
            A :class:`ModelOutput` for this image.

        Raises:
            RuntimeError: If the model is neither trained nor loadable from
                :attr:`checkpoint_path`.
            GuardError: If the frame fails the input-quality guard (blur,
                exposure, resolution or aspect ratio) — see
                :meth:`~app.models.base.AnomalyModel._guard_frame`.
        """
        module = self._require_module()

        array = self._to_rgb_array(image, color_order=color_order)
        self._guard_frame(array)  # reject camera glitches / dirty lens / dead lighting before scoring
        height, width = array.shape[:2]
        tensor = self._to_model_input(array)

        with torch.no_grad():
            # Deliberately calls the inner torch model rather than
            # `module(tensor)`: the Lightning module's forward would run its own
            # pre-processor over an already-normalized tensor. Preprocessing is
            # this wrapper's contract, so it happens exactly once, here.
            raw = module.model(tensor.to(self._device))
            scored = module.post_processor(raw) if module.post_processor is not None else raw

        score = float(scored.pred_score.reshape(-1)[0])
        anomaly_map = self._to_input_resolution(scored.anomaly_map, height, width)

        return ModelOutput(
            anomaly_score=score,
            anomaly_map=anomaly_map,
            is_defective=score >= self.config.anomaly_threshold,
            model_name=self.model_name,
        )

    def save(self, path: str | Path) -> None:
        """Write a Lightning checkpoint containing weights, memory bank and calibration.

        Args:
            path: Destination ``.ckpt`` file. Parent directories are created.

        Raises:
            RuntimeError: If there is nothing to save yet.
        """
        module = self._require_module()
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        trainer = self._trainer_or_none()
        if trainer is not None:
            trainer.save_checkpoint(destination)
        else:
            # Loaded-then-resaved models have no trainer attached. Hand-roll the
            # three keys `LightningModule.load_from_checkpoint` reads, so both
            # paths produce a file `load()` accepts.
            torch.save(
                {
                    "pytorch-lightning_version": lightning.__version__,
                    "state_dict": module.state_dict(),
                    "hyper_parameters": self._portable_hparams(module),
                },
                destination,
            )

        self._checkpoint_path = destination
        size_mb = destination.stat().st_size / 1024**2
        logger.info("Saved %s checkpoint to %s (%.1f MB)", self.model_name, destination, size_mb)

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save` and cache it for inference.

        Args:
            path: Checkpoint file to load.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            msg = f"No checkpoint at {source}."
            raise FileNotFoundError(msg)

        # weights_only=False: anomalib pickles enum hyperparameters (PrecisionType)
        # into the checkpoint, which torch's restricted unpickler rejects. These
        # are our own artifacts, written by save() above.
        module = Patchcore.load_from_checkpoint(
            source,
            map_location=self._device,
            weights_only=False,
            # Overrides the checkpoint's saved value: preprocessing follows the
            # running config, not whatever resolution this file was fitted at.
            pre_processor=self._build_pre_processor(),
        )
        module.eval()

        self._module = module
        self._checkpoint_path = source
        logger.info(
            "Loaded %s from %s; memory bank %s, calibrated=%s",
            self.model_name,
            source,
            tuple(module.model.memory_bank.shape),
            self.is_calibrated,
        )

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _portable_hparams(module: Patchcore) -> dict[str, Any]:
        """Hyperparameters with the ``nn.Module`` ones replaced by ``True``.

        anomalib accepts ``True`` for ``pre_processor``/``post_processor``/
        ``evaluator``/``visualizer`` meaning "build the default", and
        :meth:`load` supplies the real pre-processor from config anyway. Pickling
        the live objects instead would double the file: a ``PreProcessor``
        registered as a Lightning callback holds a back-reference to the module,
        so it drags a second copy of the backbone into the checkpoint — invisible
        when Lightning writes it (``torch.save`` deduplicates shared storages in
        a single call) but very visible here, where the state dict is serialised
        separately.
        """
        return {key: True if isinstance(value, torch.nn.Module) else value for key, value in module.hparams.items()}

    def _trainer_or_none(self) -> Any:
        """The Lightning trainer from the last fit, or ``None`` if there was none."""
        if self._engine is None:
            return None
        try:
            return self._engine.trainer
        except Exception:  # noqa: BLE001 - anomalib raises a bespoke UnassignedError
            return None

    def _scale_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization; the frozen WideResNet expects nothing else.

        PatchCore's backbone is a bare timm model, so the statistics it was
        pretrained with have to be applied before it sees the batch. (EfficientAD
        normalizes inside its own PDN forward and therefore leaves the base
        class's identity hook alone.)
        """
        return normalize_image(tensor)
