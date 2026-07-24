"""EfficientAD anomaly detection, behind the project's own model interface.

Together with :mod:`app.models.patchcore`, one of only two modules allowed to
``import anomalib.models``. Everything else goes through
:class:`~app.models.base.AnomalyModel`.

How EfficientAD works
=====================
EfficientAD is a *student-teacher distillation* method, and that makes it
structurally the opposite of PatchCore. PatchCore trains nothing and remembers
everything: its "training" is filling a memory bank with normal patch
descriptors, and inference is a nearest-neighbour search against that bank.
EfficientAD remembers nothing and trains: it fits a small network on the normal
images, and inference is one forward pass.

**1. The teacher.** A Patch Description Network (PDN) — six layers, four
convolutions and two average pools — that has been distilled offline from a
WideResNet-101 on ImageNet. anomalib ships those weights; they are frozen, and
:meth:`EfficientADModel.train` downloads them on first use.

**2. The student.** An identically-shaped PDN, randomly initialised, trained to
regress the teacher's output on this category's defect-free images *only*. The
loss is a hard-mining L2: the highest-error fraction of feature elements is
what gets backpropagated, so the student is not allowed to spend its capacity
polishing the easy background it already matches.

**3. Why disagreement means "anomaly".** The student has seen only normal
images, so it learns the teacher's behaviour on normal appearance and nothing
else. On a defect — a texture the teacher was pretrained to describe but the
student was never shown — the student's regression has nothing to interpolate
from, and the two outputs diverge. The squared distance between them, per
spatial position, *is* the anomaly map.

Left alone, that argument has a hole: a student with enough capacity can learn
to imitate the teacher everywhere, including on inputs it never saw, which
would collapse the anomaly signal. EfficientAD closes it by penalising the
student for matching the teacher on random ImageNet (ImageNette) images during
training. The student is thereby pushed to be a *good* teacher-imitator on
bottles and a *bad* one on everything else — which is exactly the property the
detector needs. This is why :meth:`EfficientADModel.train` wants an ImageNette
copy on disk and PatchCore does not.

**4. The autoencoder branch.** The PDN's receptive field is local by
construction (point 5), so a student-teacher pair cannot see a defect that is
only anomalous *globally* — a correctly-textured part in the wrong place, a
missing component. A small autoencoder trained to reconstruct the teacher
output from a 64-dimensional bottleneck supplies that global view: its
bottleneck cannot encode a full logical layout, so it reconstructs the layout it
saw in training and disagrees with the student where the image departs from it.
The final map is the mean of the two, each rescaled by quantiles measured on
validation data so neither branch dominates.

Receptive field, and why 33x33 is the interesting number
========================================================
Every value in a PDN feature map depends on exactly a 33x33 pixel window of the
input, with a stride of 4. That is arithmetic, not an approximation — stacking
the small PDN's ``conv 4x4 -> pool 2x2 -> conv 4x4 -> pool 2x2 -> conv 3x3 ->
conv 4x4`` gives ``r = 33``, ``j = 4``. The consequence is a hard guarantee:
**a pixel more than 16 pixels away cannot influence a given feature.**

PatchCore has no comparable guarantee. Its descriptors come from ``layer2`` and
``layer3`` of a WideResNet-50 whose receptive fields are already nominally
larger than the 256x256 input, and are then 3x3 average-pooled on top. The
*effective* receptive field is smaller than the nominal one, but it is a soft,
Gaussian-ish falloff with no boundary — distant context does leak in.

Why that matters for false positives
------------------------------------
PatchCore's characteristic failure mode on MVTec-style data is false positives
at **object boundaries**. The mechanism is straightforward once the receptive
fields are in view:

* A patch on the silhouette of a part mixes foreground and background into one
  descriptor. How much of each depends on where the part sits in the frame.
* Industrial parts are not placed with sub-pixel repeatability. A bottle
  rotated a few degrees, or shifted five pixels, produces boundary descriptors
  that are a *different mixture* from any in the memory bank.
* Nearest-neighbour distance cannot distinguish "this mixture is unusual
  because the part is damaged" from "this mixture is unusual because the part
  moved". Both come back as a large distance, and the anomaly map lights up
  along the outline of a perfectly good part.

The bounded receptive field partially defuses this. A feature 17+ pixels inside
the part is *provably* uncontaminated by background, so a rigid shift of the
object translates the anomaly map rather than changing its values, and interior
features stay comparable regardless of placement. Contamination is confined to
a 16-pixel band along the silhouette instead of bleeding inward.

Note the honest scope of that claim: **partially**. Features that genuinely
straddle the boundary are still mixtures, and still noisy. What changes is that
the affected region is small, bounded and predictable, rather than diffuse. The
autoencoder branch, which is deliberately global, reintroduces some placement
sensitivity of its own. EfficientAD narrows this failure mode; it does not
delete it.

The accuracy-latency tradeoff
=============================
This is where the two models genuinely separate, and the reason the benchmark in
this project runs both:

* **Latency — and read this before quoting the paper.** The published claim is
  ~2 ms/image on an A100, against tens of milliseconds for PatchCore. That
  result does not transfer unexamined to this project, and measurement here
  says the *opposite*: on CPU at 256x256, batch 1, PatchCore scores a bottle in
  ~150 ms and EfficientAD in ~440 ms.

  Two things explain the inversion, and both are properties of the setup rather
  than of the algorithms:

  1. **The memory bank is tiny.** PatchCore's cost is a backbone forward *plus*
     a nearest-neighbour search, and only the search scales with the bank. At
     ``coreset_sampling_ratio=0.01`` on one MVTec category the bank is ~2k
     vectors, so the search is nearly free and PatchCore is paying for a
     WideResNet-50 forward and little else. Raise the ratio, or train on
     thousands of images, and that term is what grows.
  2. **The PDN is not cheap on a CPU.** It looks small — six layers — but it
     runs at near-input resolution with 128-to-384-channel convolutions, and it
     runs *twice* (teacher and student), plus the autoencoder. Those are
     exactly the wide, shallow, highly parallel shapes a GPU eats for free and
     a CPU does not.

  So the real claim is narrower than "EfficientAD is faster": its inference
  cost is **constant in the size of the training set**, where PatchCore's grows.
  That is the property that matters for a line accumulating reference images,
  and it is the axis the Step 5 benchmark should measure — on the hardware that
  will actually serve, because on this one the paper's ordering does not hold.
* **Memory.** A PatchCore bank is tens to hundreds of MB and scales with the
  number of training images. EfficientAD's weights are a fixed ~31 MB whatever
  it was trained on. (The ``.ckpt`` on disk is ~74 MB, because a Lightning
  checkpoint also carries Adam's optimizer state; that is training baggage, and
  what gets served is the 31 MB.)
* **Training.** Inverted. PatchCore fits in one pass over the data and no
  gradients; EfficientAD needs tens of thousands of optimizer steps to converge
  (the paper uses 70k) and an ImageNette download. A one-epoch EfficientAD fit,
  which is what this project's tests and quick runs use, is a smoke test rather
  than a converged model — expect it to separate defective from clean but not
  to reach published AUROC.
* **Accuracy.** At convergence the two are close on MVTec AD image AUROC, with
  EfficientAD clearly ahead on the *logical* anomalies of MVTec LOCO that its
  autoencoder branch exists for. Under-trained it still ranks well — a single
  epoch on ``bottle`` reaches ~0.97 image AUROC here — but the *scores* are
  another matter, and this is worth knowing before reading a benchmark table:
  anomalib min-max normalizes against the validation split, so one saturating
  image compresses every other score towards the 0.5 decision boundary. Ranking
  metrics (AUROC, AU-PRO) stay meaningful; a fixed 0.5 threshold does not.

The short version: PatchCore is the better model to fit, EfficientAD is the
better model to serve.

Reference: Batzner et al., *EfficientAD: Accurate Visual Anomaly Detection at
Millisecond-Level Latencies* (WACV 2024), https://arxiv.org/abs/2303.14535
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lightning
import numpy as np
import torch
# See the note in app/models/patchcore.py: anomalib only recognises its own
# ModelCheckpoint subclass when deciding whether to inject a default one.
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig, get_model_config

__all__ = ["EfficientADModel"]

logger = logging.getLogger(__name__)

#: EfficientAD's training step pairs each image with one ImageNette image and
#: hard-mines the loss over the batch, both of which the paper defines per
#: single image. anomalib enforces it rather than rescaling.
_REQUIRED_TRAIN_BATCH_SIZE = 1

#: The autoencoder branch downsamples by 32 through five stride-2 convolutions
#: and then applies a *valid* 8x8 convolution, so it needs at least 8x8 to
#: convolve over: ``floor(image_size / 32) >= 8``. Below this the model dies
#: mid-fit with "Kernel size can't be greater than actual input size", several
#: minutes in and nowhere near the setting that caused it. Unlike PatchCore,
#: whose fully-convolutional backbone genuinely runs at any resolution, this is
#: a fixed architecture: 256 is the floor and the size the paper trains at.
_MIN_IMAGE_SIZE = 256


class EfficientADModel(AnomalyModel):
    """EfficientAD wrapped as an :class:`~app.models.base.AnomalyModel`.

    Interface-identical to :class:`~app.models.patchcore.PatchCoreModel` — same
    ``train``/``predict``/``save``/``load``, same :class:`ModelOutput` — so the
    benchmark runner and the serving layer hold one or the other without
    knowing which. See the module docstring for how differently the two arrive
    at that output.

    Three differences are visible through the wrapper, and each is handled here
    rather than pushed onto the caller:

    * **Training is real.** PatchCore memorises in a single pass; this fits a
      network with gradients, honours ``max_epochs``, and at one epoch is not
      converged. :meth:`train` says so in the logs.
    * **The train loader is forced to batch size 1**, which the algorithm
      requires. The datamodule's configured batch size still applies to
      validation and test.
    * **Inputs are not ImageNet-normalized.** The PDN normalizes inside its own
      forward pass, so this wrapper leaves the base class's identity scaling
      hook alone and hands the network a ``[0, 1]`` batch. Normalizing twice is
      silent — no shape changes, no error, just a quietly worse model.

    Every constructor argument defaults to ``None`` and falls through to
    :class:`~app.models.config.ModelConfig` (``model_size="small"``,
    ``image_size=256``, ``imagenet_dir=data/imagenette``). Defaults are not
    repeated here, so there is exactly one place to change one.

    The fitted Lightning module is cached on the instance: :meth:`predict`
    reuses it, and only touches disk on the first call after a bare
    construction.

    Args:
        category: Object category this instance is for. Also selects the default
            checkpoint filename.
        model_size: PDN size, ``"small"`` or ``"medium"``. ``"small"`` is the
            default because it is the one that trains in reasonable time on a
            single modest GPU — ``"medium"`` roughly triples the parameter count
            for a fraction of a point of AUROC.
        image_size: Square resolution images are resized to. Must be at least
            256 — see :data:`_MIN_IMAGE_SIZE`; this is a fixed architecture, not
            a fully-convolutional one.
        imagenet_dir: ImageNette root for the distillation penalty. Downloaded
            (~1.5 GB) on the first :meth:`train` if absent; :meth:`predict`
            never touches it.
        config: Pre-built config to start from. Defaults to the process-wide one.

    Raises:
        ValueError: If ``image_size`` is below the autoencoder's 256-pixel floor.

    Example:
        >>> from app.data import DataModule                       # doctest: +SKIP
        >>> model = EfficientADModel(category="bottle")           # doctest: +SKIP
        >>> model.train(DataModule(category="bottle"))            # doctest: +SKIP
        >>> result = model.predict(cv2.imread(path), color_order="bgr")
        >>> result.anomaly_score, result.is_defective             # doctest: +SKIP
        (0.79, True)
    """

    model_name = "efficientad"

    def __init__(
        self,
        category: str | None = None,
        model_size: str | None = None,
        image_size: int | None = None,
        imagenet_dir: Path | str | None = None,
        *,
        config: ModelConfig | None = None,
    ) -> None:
        base = config if config is not None else get_model_config()
        super().__init__(
            base.with_overrides(
                category=category,
                model_size=model_size,
                image_size=image_size,
                imagenet_dir=imagenet_dir,
            ),
        )

        if self.config.image_size < _MIN_IMAGE_SIZE:
            msg = (
                f"EfficientAD needs image_size >= {_MIN_IMAGE_SIZE}, got {self.config.image_size}. "
                f"Its autoencoder branch downsamples by 32 and then applies a valid 8x8 convolution, "
                f"so anything smaller has nothing left to convolve over. PatchCore has no such floor; "
                f"if you are shrinking images to speed up a benchmark, the two models cannot share "
                f"that setting."
            )
            raise ValueError(msg)

        # Cached across predict() calls, as in PatchCoreModel: rebuilding costs a
        # full state_dict load, which would dominate the few ms of actual work.
        self._module: EfficientAd | None = None
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
        """Whether the teacher's channel statistics have been measured.

        The student's weights are always *present* — it is randomly initialised
        at construction — so parameter existence proves nothing here, unlike
        PatchCore's memory bank. ``mean_std`` is the honest marker: it is filled
        in ``on_train_start`` from a pass over the training set, and stays all
        zeros on a model that was only constructed.
        """
        if self._module is None:
            return False
        return bool(self._module.model.is_set(self._module.model.mean_std))

    @property
    def is_calibrated(self) -> bool:
        """Whether score normalization statistics were fitted.

        Same meaning as :attr:`PatchCoreModel.is_calibrated`: these come from
        the validation pass inside :meth:`train`, and without them scores are
        raw student-teacher distances rather than ``[0, 1]``.

        Distinct from :attr:`has_map_quantiles`, which is EfficientAD's own
        internal rescaling of its two branches against each other.
        """
        if self._module is None or self._module.post_processor is None:
            return False
        post = self._module.post_processor
        return not bool(post.image_min.isnan() or post.image_max.isnan())

    @property
    def has_map_quantiles(self) -> bool:
        """Whether the student/autoencoder map quantiles were measured.

        Fitted in ``on_validation_start`` from the *normal* validation images.
        Without them the two branches are averaged on their raw, incomparable
        scales and the combined map is dominated by whichever happens to be
        larger.
        """
        if self._module is None:
            return False
        return bool(self._module.model.is_set(self._module.model.quantiles))

    # -- construction ----------------------------------------------------------

    def _build_pre_processor(self) -> Any:
        """anomalib's resize pipeline, sized from our config.

        Deliberately *not* PatchCore's resize-plus-normalize: anomalib rejects a
        pre-processor containing ``Normalize`` for this model, because the PDN
        applies ImageNet statistics inside its own forward pass.
        """
        return EfficientAd.configure_pre_processor(image_size=self.config.image_hw)

    def _build_module(self) -> EfficientAd:
        """Instantiate the anomalib Lightning module from the resolved config."""
        return EfficientAd(
            imagenet_dir=self.config.imagenet_dir,
            model_size=self.config.model_size,
            pre_processor=self._build_pre_processor(),
        )

    def _build_engine(self) -> Engine:
        """Create the Lightning trainer wrapper used for fitting.

        Supplying our own ``ModelCheckpoint`` claims the slot anomalib would
        otherwise fill with its own, so the checkpoint lands at
        :attr:`checkpoint_path` rather than in a versioned
        ``results/EfficientAd/MVTecAD/<category>/vN/weights/`` tree.
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

    def _require_module(self) -> EfficientAd:
        """Return the cached module, loading the default checkpoint if needed."""
        if self._module is None:
            checkpoint = self.checkpoint_path
            if not checkpoint.is_file():
                msg = (
                    f"{type(self).__name__} for category {self.category!r} is not trained. "
                    f"Call train(datamodule), load(path), or run "
                    f"`python scripts/train_efficientad.py --category {self.category}` "
                    f"(expected a checkpoint at {checkpoint})."
                )
                raise RuntimeError(msg)
            logger.info("No model in memory; loading %s", checkpoint)
            self.load(checkpoint)

        module = self._module
        if module is None or not module.model.is_set(module.model.mean_std):
            msg = (
                "EfficientAD teacher statistics are unset; the model was constructed but never "
                "fitted. Its student is randomly initialised, so predictions would be noise."
            )
            raise RuntimeError(msg)
        return module

    @staticmethod
    @contextmanager
    def _single_image_train_batches(anomalib_datamodule: Any) -> Iterator[None]:
        """Force ``train_batch_size = 1`` for the duration of a fit.

        EfficientAD's training step pairs each image with one ImageNette image
        and hard-mines the loss over the batch; anomalib asserts the batch size
        rather than rescaling, and the assert fires in ``on_train_start``, after
        the teacher weights and ImageNette have already been fetched.

        Scoped to a context manager so a datamodule shared with a PatchCore run
        (or reused for evaluation) is handed back exactly as it arrived. Only
        the *train* loader is affected — validation and test keep the batch size
        the caller configured.
        """
        previous = anomalib_datamodule.train_batch_size
        anomalib_datamodule.train_batch_size = _REQUIRED_TRAIN_BATCH_SIZE
        try:
            yield
        finally:
            anomalib_datamodule.train_batch_size = previous

    # -- AnomalyModel ----------------------------------------------------------

    def train(self, datamodule: Any) -> None:
        """Distil a student against the frozen teacher on the defect-free train split.

        Runs anomalib's ``Engine.fit``, which for EfficientAD means: download the
        pretrained teacher and ImageNette if absent, measure the teacher's
        channel statistics over the training set, then ``max_epochs`` of
        gradient descent on the student and autoencoder, then a validation pass
        that fits both the map quantiles and the score normalization. The
        Lightning checkpoint is written to
        ``results/checkpoints/efficientad_<category>.ckpt``.

        Unlike PatchCore this is a genuine optimization, and one epoch is *not*
        convergence — see the module docstring's tradeoff section.

        Args:
            datamodule: An :class:`app.data.DataModule`. Its ``category`` wins
                over this instance's config, so training a model for a category
                it was not constructed for cannot silently mislabel the
                checkpoint. Its train batch size is overridden to 1 for the
                duration of the fit.

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

        if self.config.max_epochs == 1:
            logger.warning(
                "Training %s for a single epoch. EfficientAD is gradient-trained (the paper uses "
                "~70k steps); expect a working but under-converged model.",
                self.model_name,
            )
        if not Path(self.config.imagenet_dir).is_dir():
            logger.info(
                "ImageNette not found at %s; anomalib will download it (~1.5 GB). It is needed "
                "only to penalise the student for imitating the teacher off-distribution.",
                self.config.imagenet_dir,
            )

        self._module = self._build_module()
        self._engine = self._build_engine()

        started = time.perf_counter()
        # anomalib's Lightning module needs anomalib's own batch type; see
        # DataModule.for_anomalib_engine for why that is scoped to this block.
        with (
            datamodule.for_anomalib_engine() as anomalib_datamodule,
            self._single_image_train_batches(anomalib_datamodule),
        ):
            self._engine.fit(model=self._module, datamodule=anomalib_datamodule)
        elapsed = time.perf_counter() - started

        self._device = self._module.device
        self._module.eval()

        logger.info(
            "Fitted %s (%s PDN) on %r in %.1fs over %d epoch(s); quantiles=%s, calibrated=%s",
            self.model_name,
            self.config.model_size,
            self.category,
            elapsed,
            self.config.max_epochs,
            self.has_map_quantiles,
            self.is_calibrated,
        )
        if not self.is_calibrated:
            logger.warning(
                "No validation data was seen during fit, so scores are un-normalized "
                "student-teacher distances rather than [0, 1].",
            )

        self.save(self.config.checkpoint_path(self.model_name, self.category))

    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        """Score one raw image and return a heatmap at its original resolution.

        The image is converted to 3-channel RGB, resized to ``image_size``, and
        passed to the network as a ``[0, 1]`` batch — *without* ImageNet
        normalization, which the PDN applies itself. Student and autoencoder
        maps are combined, and the result is bilinearly resampled back to the
        input's ``H x W`` so it can be overlaid directly.

        Args:
            image: ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``
                array; ``uint8`` in ``[0, 255]`` or float.
            color_order: ``"rgb"`` or ``"bgr"``. Use ``"bgr"`` for anything from
                OpenCV.

        Returns:
            A :class:`ModelOutput` for this image, schema-identical to
            PatchCore's.

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
            # As in PatchCore, deliberately the inner torch model rather than
            # `module(tensor)`: the Lightning module's forward would re-run its
            # own pre-processor over a batch this wrapper has already prepared.
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
        """Write a Lightning checkpoint containing weights, statistics and calibration.

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

        # weights_only=False: anomalib pickles enum hyperparameters
        # (EfficientAdModelSize, PrecisionType) into the checkpoint, which
        # torch's restricted unpickler rejects. These are our own artifacts,
        # written by save() above.
        module = EfficientAd.load_from_checkpoint(
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
            "Loaded %s from %s; model_size=%s, quantiles=%s, calibrated=%s",
            self.model_name,
            source,
            self.config.model_size,
            self.has_map_quantiles,
            self.is_calibrated,
        )

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _portable_hparams(module: EfficientAd) -> dict[str, Any]:
        """Hyperparameters with the ``nn.Module`` ones replaced by ``True``.

        Same reasoning as :meth:`PatchCoreModel._portable_hparams`: anomalib
        accepts ``True`` for ``pre_processor``/``post_processor``/``evaluator``/
        ``visualizer`` meaning "build the default", and :meth:`load` supplies
        the real pre-processor from config anyway. Pickling the live objects
        instead would drag a second copy of the network into the file.
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
