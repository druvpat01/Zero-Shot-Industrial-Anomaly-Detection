"""WinCLIP zero-shot anomaly detection, behind the project's own model interface.

The third and last module allowed to ``import anomalib.models``. Everything else
goes through :class:`~app.models.base.AnomalyModel`.

Why this model is the interesting one
=====================================
PatchCore and EfficientAD disagree about almost everything — memory bank versus
distillation, remember-everything versus train-a-network — but they agree on the
premise: *show me a few hundred pictures of a good bottle and I will tell you
when one stops looking like them.* Both need a curated, verified, defect-free
training set per category before they can score a single frame.

WinCLIP does not. It needs the word ``"bottle"``.

That is the whole architectural claim of this project, so the rest of this
docstring is about how it can possibly be true, and where it stops being true.

(a) Where the zero-shot ability comes from
==========================================
CLIP is trained on ~400M image-caption pairs scraped from the web, with a
contrastive objective: an image encoder and a text encoder are pushed to produce
*the same* vector for an image and its caption, and different vectors for
mismatched pairs. Nothing about anomalies, defects or industrial inspection
appears anywhere in that training.

What comes out is a single embedding space that images and English sentences
share. And that is the leverage, because it converts *classification* into
*writing down the classes*. To build a normal-vs-defective classifier for
bottles, you do not fit anything; you encode two sentences —

    "a photo of a bottle without defect."
    "a photo of a damaged bottle."

— and ask which one the image embedding is closer to, by cosine similarity.
A softmax over the two similarities (temperature 0.07, the CLIP paper's value)
turns that into a probability, and *that probability is the anomaly score*.
Nothing was fitted; the decision boundary is the perpendicular bisector between
two sentences.

In practice a single pair of sentences is a fragile classifier — CLIP is
famously prompt-sensitive, and "a photo of a damaged bottle" carries connotations
of a *smashed* bottle rather than a chipped one. WinCLIP therefore builds a
**compositional prompt ensemble**: 7 normal states x 21 photographic templates
and 4 anomalous states x 21 templates, ~150 prompts in all, encoded and averaged
per class. Averaging over templates cancels out the incidental direction each
phrasing adds ("a blurry photo of...", "a jpeg corrupted photo of...") and leaves
the part they share, which is the concept. Two vectors survive: normal and
anomalous. See :mod:`anomalib.models.image.winclip.prompting`.

This is also the ceiling, and it is worth naming early: **the model can only
detect what its vocabulary can describe.** "Damaged bottle" is a concept CLIP has
seen thousands of captions for. "0.3 mm burr on the third thread of an M4 insert"
is not, and no amount of prompt engineering conjures a direction in embedding
space that was never trained.

(b) Sliding windows, and why one image embedding is not enough
==============================================================
The similarity above is computed on the embedding of the *whole image*, so it
yields one number per image — a classifier, not a segmenter. And it is a poor
classifier for our purpose: a bottle with a hairline crack is, overwhelmingly,
still a photo of a bottle. The defect occupies perhaps 1% of the pixels, and the
global embedding is dominated by the 99%.

WinCLIP's answer is to ask the same question about *local regions*, which is
where the "Win" comes from:

**1. The patch grid.** The backbone is ``ViT-B-16-plus-240``: 240x240 input,
16x16 patches, so a **15x15 grid** of patch tokens. Each token is a
position-aware descriptor of a 16x16 pixel region.

**2. Windows over the grid.** For each scale ``s`` in :attr:`scales` (default
``(2, 3)``), every ``s x s`` block of adjacent patches is one window — 196
windows at scale 2, 169 at scale 3, covering 32x32 and 48x48 pixels
respectively. The trick that makes this affordable is that a window is *not*
re-encoded as a cropped image. Instead the window's patch tokens are pooled
through the transformer's own attention with a **mask** that hides everything
outside the window, so the whole image's patch tokens are computed once and each
window reuses them. The naive alternative — crop, resize, encode — is 365 extra
ViT forward passes per frame. That reuse is the engineering contribution of the
paper.

A footnote with teeth: this is also the part that silently breaks. anomalib
drives the masked pooling by reaching into ``clip.visual``'s internals and
permuting the token tensor to length-first before calling the transformer stack
by hand. open_clip made that stack batch-first in 2.26, so on a newer open_clip
every window pools to the *same* vector, every window scores identically, and
the anomaly map comes back a uniform field — no exception, no warning, just a
model that appears to detect nothing. ``requirements.txt`` pins
``open-clip-torch<2.26.1`` for this reason, and
``test_anomaly_map_is_not_a_uniform_field`` in ``tests/test_winclip.py`` fails
loudly if the pin is ever lost.

**3. Scoring each window.** Every window embedding gets the same treatment as
the whole image: cosine similarity against the two averaged text embeddings,
softmaxed, take the "anomalous" probability. A window whose contents look like
"a photo of a damaged bottle" scores high whether or not the rest of the bottle
does.

**4. Harmonic aggregation back to pixels.** Each patch belongs to several
windows (a patch in the interior sits in ``s^2`` windows at scale ``s``), so the
per-window scores must be folded back onto the 15x15 grid. WinCLIP uses the
**harmonic mean** of the scores of the windows containing a patch, and the choice
is deliberate: the harmonic mean is dominated by the *smallest* term. A patch is
only called normal if *every* window containing it agrees it is normal — one
confidently-normal window is enough to veto. This is the conservative direction,
and it suppresses the failure mode where a large window containing one small
defect and a lot of clean background averages out to "fine".

**5. Across scales.** The three score maps — full image, scale 2, scale 3 — are
combined by harmonic mean again, then bilinearly upsampled from 15x15 to the
input resolution. Multiple scales matter because a defect's *apparent* size is
unknown a priori: a 2x2 window localises a chip tightly but has almost no
context to judge it against, and a 3x3 window has context but blurs a small
defect into its surroundings. Neither wins everywhere, so both vote.

The resulting anomaly map is a genuine pixel-level segmentation, produced
without one gradient step and without one training image.

(c) What ``k_shot=0`` actually buys
===================================
It is easy to read "zero-shot" as a benchmark curiosity. It is not; it is a
change in what the deployment *costs*, and this is the part worth being able to
argue in an interview.

* **No labelling, no curation, no waiting.** PatchCore and EfficientAD need a
  defect-free training set. Producing one is not a download — it is somebody on
  the line photographing a few hundred parts and *certifying* each is good. A
  single defective image that slips into ``train/good`` teaches the model that
  the defect is normal, and it does so silently.
* **Day-one coverage of a new SKU.** A line that changes product weekly cannot
  afford a per-SKU training cycle. WinCLIP scores a new part as soon as somebody
  types its name, so it can cover the gap while a per-SKU PatchCore is being
  collected.
* **A cold-start path, not just a model.** This is why the platform serves all
  three: WinCLIP from day zero, PatchCore once ~50 good images exist. The
  interesting middle is ``k_shot > 0``: hand WinCLIP 1-4 reference images and it
  additionally compares patch embeddings against *those* (see
  ``_compute_few_shot_scores``), which recovers much of the gap without anything
  resembling training. The wrapper supports it via :attr:`k_shot`, but this
  project's headline configuration is ``k_shot=0``.
* **A defect it has never seen is not special.** All three models are
  unsupervised, so none needs *defect* examples — but PatchCore's notion of
  "anomalous" is "far from my bank of this category's normals", which is only
  meaningful for the category the bank was built from. WinCLIP's is "closer to
  the word *damaged* than the word *flawless*", which needs no bank at all.

(d) Where it loses, and it does lose
====================================
The honest comparison, on MVTec AD ``bottle``-style categories with well-defined
defects:

* **Pixel-level AUROC is the weak spot.** The paper reports ~85% pixel AUROC
  zero-shot against PatchCore's ~98%. The mechanism is resolution, and it is
  structural rather than a tuning problem: every score in the map originates on a
  **15x15 grid**, so the finest thing WinCLIP can localise is one patch — a 16x16
  pixel block, upsampled to whatever the caller's frame is. PatchCore scores on a
  32x32 (or finer) feature grid *and* its descriptors come from a backbone whose
  mid-level layers were built to be spatially precise. Asked to outline a
  hairline crack, WinCLIP produces a warm blob in the right place; PatchCore
  produces the crack. For an operator being shown where to look, the blob is
  often enough. For an automated measurement of defect area, it is not.
* **Image-level AUROC is closer than you would guess** (~91% vs ~99% on MVTec
  AD) — the ranking is largely right even where the localisation is coarse.
* **It is prompt- and category-sensitive.** Categories CLIP has rich web
  vocabulary for (bottle, cable, zipper) do markedly better than the ones it does
  not (``metal_nut``, ``grid``). Performance therefore varies in a way that is
  hard to predict from anything except trying it — where PatchCore's accuracy is
  a fairly smooth function of how many good images you fed it.
* **It is by some distance the slowest of the three here.** A ViT-B/16-plus
  forward at 240x240 plus the masked pooling of ~365 windows measures ~3.9 s per
  image on this CPU (4 threads), against PatchCore's ~150 ms and EfficientAD's
  ~440 ms — and note it is doing that at 240x240 while the other two work at
  256x256. It also holds ~830 MB of CLIP weights resident. Those windows are the
  cost: they are what turns a whole-image classifier into a segmenter, and they
  are pure transformer arithmetic, which is to say exactly the shape a GPU
  absorbs and a CPU does not. The thing WinCLIP is cheap in is *data*, not
  compute, and a CPU deployment should expect to batch or to serve it behind the
  faster models rather than in front of them.

The short version, to set against the one at the end of
:mod:`app.models.efficientad`: PatchCore is the better model to fit, EfficientAD
is the better model to serve, and WinCLIP is the only one you can use before you
have anything to fit it on.

Reference: Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and
Segmentation* (CVPR 2023), https://arxiv.org/abs/2303.14814
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
# See the note in app/models/patchcore.py: anomalib only recognises its own
# ModelCheckpoint subclass when deciding whether to inject a default one.
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.engine import Engine
from anomalib.models import WinClip

from app.data.transforms import CLIP_MEAN, CLIP_STD, normalize_image
from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig, get_model_config

__all__ = ["WinCLIPModel"]

logger = logging.getLogger(__name__)

#: WinCLIP's input resolution, and not a hyperparameter. The backbone is
#: ``ViT-B-16-plus-240``, whose learned positional embeddings are a 15x15 grid of
#: 16x16 patches; feeding it anything else either fails outright or silently
#: interpolates the position table into a configuration it was never trained on.
#: :class:`WinCLIPModel` pins ``image_size`` to this so callers never have to
#: know that this backend disagrees with the other two about resolution.
REQUIRED_IMAGE_SIZE = 240

#: Side of the ViT patch grid, ``240 / 16``. The ceiling on how finely WinCLIP
#: can localise anything — see the module docstring, section (d).
PATCH_GRID_SIZE = REQUIRED_IMAGE_SIZE // 16

#: State-dict prefix of the CLIP backbone inside anomalib's ``WinClipModel``.
#: :meth:`WinCLIPModel.save` drops these keys; see its docstring for why.
_CLIP_WEIGHT_PREFIX = "clip."

#: Marker written into our own checkpoints so :meth:`WinCLIPModel.load` can tell
#: one from a Lightning checkpoint saved by a sibling wrapper.
_CHECKPOINT_FORMAT = "winclip-calibration-v1"

#: Qualname suffix of the forward hook anomalib's ``WinClipModel.encode_image``
#: registers on every call and never removes. :meth:`WinCLIPModel._clear_leaked_hooks`
#: strips these; see it for why it has to.
_LEAKED_HOOK_QUALNAME_SUFFIX = "get_feature_map.<locals>.hook"


class WinCLIPModel(AnomalyModel):
    """WinCLIP wrapped as an :class:`~app.models.base.AnomalyModel`.

    Interface-identical to :class:`~app.models.patchcore.PatchCoreModel` and
    :class:`~app.models.efficientad.EfficientADModel` — same
    ``train``/``predict``/``save``/``load``, same :class:`ModelOutput` — so the
    benchmark runner and the serving layer hold any of the three without knowing
    which. See the module docstring for how differently this one gets there.

    Four differences are visible through the wrapper, and each is absorbed here
    rather than pushed onto the caller:

    * **:meth:`predict` works on a bare instance.** No :meth:`train`, no
      checkpoint, no dataset — construct and score. This is the zero-shot claim,
      and ``tests/test_winclip.py`` asserts it by never calling :meth:`train`.
      Its siblings raise :class:`RuntimeError` in the same situation.
    * **:meth:`train` optimizes nothing.** It exists to fit *score calibration*
      against a labelled split, which is worth having and is not required to
      predict. See its docstring.
    * **``image_size`` is pinned to 240.** Not a preference —
      :data:`REQUIRED_IMAGE_SIZE` is fixed by the backbone. A configured value is
      overridden with a debug log rather than an error, because the process-wide
      default of 256 exists for the other two backends and should not make this
      one unconstructible.
    * **Inputs are CLIP-normalized and bicubic-resized**, not ImageNet-normalized
      and bilinear-resized. Both are properties of how open_clip preprocessed
      the images this ViT was trained on. Getting either wrong is silent: right
      shapes, right dtype, no error, quietly worse embeddings.

    Every constructor argument defaults to ``None`` and falls through to
    :class:`~app.models.config.ModelConfig` (``class_name=None`` meaning "derive
    from ``category``", ``k_shot=0``, ``scales=(2, 3)``). Defaults are not
    repeated here, so there is exactly one place in the codebase to change one.

    Building the module loads ~830 MB of CLIP weights (downloaded once by
    open_clip, then cached under ``~/.cache/clip``). It is therefore cached on
    the instance and built lazily on the first :meth:`train`, :meth:`predict` or
    :meth:`load` — never in ``__init__``, so constructing a
    :class:`WinCLIPModel` you may not end up calling costs nothing.

    Args:
        class_name: Noun used in the text prompts, e.g. ``"bottle"``. Defaults to
            :attr:`~app.models.config.ModelConfig.prompt_class_name`, which
            derives it from ``category``. Passing this alone also seeds
            ``category``, so ``WinCLIPModel("cable")`` reads the way it looks;
            pass ``category`` explicitly when the prompt noun and the dataset
            folder differ (``category="metal_nut"``, ``class_name="metal nut"``).
        k_shot: Normal reference images to compare against in addition to the
            prompts. ``0`` (the default) is pure zero-shot. Above ``0`` the
            references must come from somewhere, so :meth:`predict` on a bare
            few-shot instance raises rather than silently degrading to
            zero-shot — call :meth:`train` or :meth:`load` first.
        scales: Sliding-window edge lengths in ViT patches. ``(2, 3)`` is the
            paper's setting: 32x32 and 48x48 pixel windows over a 240x240 input.
        category: Dataset category, used for the checkpoint filename and checked
            against the datamodule in :meth:`train`. Defaults to ``class_name``
            if that was given, else to the configured category.
        config: Pre-built config to start from. Defaults to the process-wide one.

    Example:
        Note what is missing between construction and inference — there is no
        ``train()`` call, and there is nothing to put in one.

        >>> model = WinCLIPModel(class_name="bottle")               # doctest: +SKIP
        >>> result = model.predict(cv2.imread(path), color_order="bgr")
        >>> result.anomaly_score, result.model_name                 # doctest: +SKIP
        (0.5054, 'winclip')
    """

    model_name = "winclip"

    # CLIP's preprocessing pipeline resizes with bicubic interpolation, so its
    # ViT has only ever seen images that were downsampled that way. See the
    # hook's declaration in AnomalyModel.
    _resize_mode = "bicubic"

    def __init__(
        self,
        class_name: str | None = None,
        k_shot: int | None = None,
        scales: tuple[int, ...] | None = None,
        category: str | None = None,
        *,
        config: ModelConfig | None = None,
    ) -> None:
        base = config if config is not None else get_model_config()

        if base.image_size != REQUIRED_IMAGE_SIZE:
            # Debug rather than warning: 256 is the process-wide default for the
            # other two backends, so this fires on almost every construction and
            # is not something the user did wrong.
            logger.debug(
                "Pinning image_size %d -> %d; WinCLIP's ViT-B-16-plus-240 has a fixed input size.",
                base.image_size,
                REQUIRED_IMAGE_SIZE,
            )

        super().__init__(
            base.with_overrides(
                category=category or class_name,
                class_name=class_name,
                k_shot=k_shot,
                scales=scales,
                image_size=REQUIRED_IMAGE_SIZE,
            ),
        )

        # Cached across predict() calls. Rebuilding costs an ~830 MB weight load,
        # which dwarfs the ~4 s of actual inference.
        self._module: WinClip | None = None
        self._engine: Engine | None = None
        self._checkpoint_path: Path | None = None
        self._device = torch.device("cpu")

    # -- properties ------------------------------------------------------------

    @property
    def category(self) -> str:
        """Dataset category this instance scores."""
        return self.config.category

    @property
    def class_name(self) -> str:
        """Noun this instance puts in its prompts, e.g. ``"bottle"``.

        Distinct from :attr:`category`, which names a folder. They coincide for
        every single-word MVTec category and diverge for the compound ones — see
        :attr:`~app.models.config.ModelConfig.prompt_class_name`.
        """
        return self.config.prompt_class_name

    @property
    def k_shot(self) -> int:
        """Reference images consulted alongside the prompts; ``0`` is zero-shot."""
        return self.config.k_shot

    @property
    def scales(self) -> tuple[int, ...]:
        """Sliding-window edge lengths, in ViT patches."""
        return self.config.scales

    @property
    def is_zero_shot(self) -> bool:
        """Whether this instance needs no reference images at all."""
        return self.k_shot == 0

    @property
    def checkpoint_path(self) -> Path:
        """Where :meth:`train` writes calibration, and :meth:`predict` looks for it.

        Unlike its siblings, a missing file here is not an error: it costs
        calibration, not the ability to predict.
        """
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        return self.config.checkpoint_path(self.model_name, self.category)

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`predict` can run.

        ``True`` on a freshly constructed zero-shot instance, which is the whole
        point of this backend and the one place its answer to a base-class
        question differs from PatchCore's and EfficientAD's. There is nothing to
        fit: the text embeddings follow deterministically from
        :attr:`class_name`, and :meth:`predict` collects them on demand.

        Few-shot instances (``k_shot > 0``) do have state to establish — the
        reference images' visual embeddings — so this reports whether they have
        been collected yet.
        """
        if self.is_zero_shot:
            return True
        return self._module is not None and self._module.model.k_shot == self.k_shot

    @property
    def is_calibrated(self) -> bool:
        """Whether score normalization statistics were fitted by :meth:`train`.

        Same meaning as the sibling wrappers', but with much less riding on it.
        PatchCore and EfficientAD emit unbounded distances when uncalibrated, so
        a threshold means nothing without this. A WinCLIP score is a softmax
        probability over two text prompts, so it is *natively* in ``[0, 1]`` and
        ``0.5`` is already the "closer to damaged than to flawless" boundary —
        calibration sharpens the separation against a particular category, it
        does not make the number interpretable.
        """
        if self._module is None or self._module.post_processor is None:
            return False
        post = self._module.post_processor
        return not bool(post.image_min.isnan() or post.image_max.isnan())

    # -- construction ----------------------------------------------------------

    def _build_pre_processor(self) -> Any:
        """anomalib's resize/normalize pipeline for CLIP inputs.

        Used by ``Engine.test`` during :meth:`train`; :meth:`predict` does its own
        preprocessing (see :meth:`_scale_for_model`). anomalib ignores the
        ``image_size`` argument here and hardcodes 240, which is the same
        constraint :data:`REQUIRED_IMAGE_SIZE` encodes on our side.
        """
        return WinClip.configure_pre_processor(image_size=None)

    def _build_module(self) -> WinClip:
        """Instantiate the anomalib Lightning module from the resolved config.

        Loads the CLIP backbone, so this is the expensive call in the class.
        """
        started = time.perf_counter()
        module = WinClip(
            class_name=self.class_name,
            k_shot=self.k_shot,
            scales=self.scales,
            pre_processor=self._build_pre_processor(),
        )
        logger.info(
            "Loaded CLIP backbone (ViT-B-16-plus-240) in %.1fs for class_name=%r, k_shot=%d, scales=%s",
            time.perf_counter() - started,
            self.class_name,
            self.k_shot,
            tuple(self.scales),
        )
        return module

    def _build_engine(self) -> Engine:
        """Create the Lightning trainer wrapper used by :meth:`train`.

        Installs a **non-saving** ``ModelCheckpoint``, and the difference from the
        siblings is the whole point. They install one to *redirect* the fit
        checkpoint to a clean path; this one exists to *suppress* Lightning
        checkpointing altogether. Two anomalib behaviours make it necessary,
        both specific to this backend:

        * anomalib injects its own ``ModelCheckpoint`` when the callback list has
          none, and its zero-shot override deliberately saves at *validation*
          end (ordinary models only save at train end, and this model never
          trains). So "no callback" does not mean "no checkpoint" here — it means
          anomalib writes an ~830 MB Lightning dump of the CLIP backbone into a
          versioned ``results`` tree, which :meth:`save` exists precisely to
          avoid.
        * That save would not even succeed: anomalib's own
          ``WinClipModel.encode_image`` registers a forward hook that is a local
          closure and never removes it (see :meth:`_clear_leaked_hooks`), so by
          validation end the module carries an unpicklable object and
          ``torch.save`` raises ``AttributeError: Can't pickle local object``.

        ``save_top_k=0`` with ``save_last=False`` claims the slot anomalib checks
        and then writes nothing, sidestepping both.
        """
        checkpoint = self.config.checkpoint_path(self.model_name, self.category)
        return Engine(
            callbacks=[
                ModelCheckpoint(
                    dirpath=checkpoint.parent,
                    filename=checkpoint.stem,
                    save_top_k=0,  # write nothing; our save() persists the small calibration artifact
                    save_last=False,
                    auto_insert_metric_name=False,
                ),
            ],
            default_root_dir=self.config.results_dir,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            logger=False,
        )

    @staticmethod
    def _clear_leaked_hooks(module: WinClip) -> int:
        """Remove the forward hooks anomalib's ``encode_image`` leaks, and report how many.

        Every call to anomalib's ``WinClipModel.encode_image`` does
        ``clip.visual.patch_dropout.register_forward_hook(...)`` to grab an
        intermediate feature map, and never removes it. The hooks therefore
        accumulate one per forward pass, and each one's closure pins the ~5.7 MB
        feature map it captured — an unbounded leak across a long-lived model's
        :meth:`predict` calls, and the ``on_validation_batch_end`` inside
        :meth:`train` leaks one per test image on top.

        It is also why the module cannot be pickled while a hook is live (the
        closure is a local function), which is the other half of why
        :meth:`_build_engine` suppresses Lightning checkpointing.

        This is anomalib's bug, not ours, but it is on our inference path, so the
        wrapper cleans up after each forward rather than waiting for an upstream
        fix. Matching on the hook's qualname keeps it surgical — nothing else in
        the codebase registers hooks on that submodule, but a blanket clear would
        be a latent trap if that ever changed.

        Returns:
            The number of hooks removed, so a caller (or a test) can assert the
            leak is actually being contained.
        """
        removed = 0
        for submodule in module.model.modules():
            hooks = submodule._forward_hooks  # noqa: SLF001 - the leak lives in this private dict
            for handle_id, fn in list(hooks.items()):
                if getattr(fn, "__qualname__", "").endswith(_LEAKED_HOOK_QUALNAME_SUFFIX):
                    del hooks[handle_id]
                    removed += 1
        return removed

    def _require_module(self) -> WinClip:
        """Return the cached module, building or loading one if needed.

        The zero-shot counterpart to ``PatchCoreModel._require_module``, and
        pointedly not the same shape. That one raises when there is no
        checkpoint, because a PatchCore with an empty memory bank cannot score
        anything. Here a missing checkpoint costs calibration only, so the
        fallback is to build the model and collect its text embeddings — which
        is exactly what "zero-shot" means operationally.

        Raises:
            RuntimeError: Only for ``k_shot > 0``, where the reference images'
                embeddings genuinely cannot be conjured from a class name.
        """
        if self._module is None:
            checkpoint = self.checkpoint_path
            if checkpoint.is_file():
                logger.info("No model in memory; loading calibration from %s", checkpoint)
                self.load(checkpoint)
            elif self.is_zero_shot:
                logger.info(
                    "No calibration at %s; running uncalibrated zero-shot from the prompt ensemble for %r. "
                    "Scores are softmax probabilities over the normal/anomalous prompts, so they are still "
                    "in [0, 1]; run train() to sharpen them against this category.",
                    checkpoint,
                    self.class_name,
                )
                module = self._build_module()
                # Lightning would call this hook during a fit/test loop. Nothing
                # here is running one, so collect the text embeddings by hand.
                module.setup(stage="predict")
                module.eval()
                self._module = module
            else:
                msg = (
                    f"{type(self).__name__} for category {self.category!r} is configured with "
                    f"k_shot={self.k_shot}, so it needs {self.k_shot} reference image(s) it does not have. "
                    f"Call train(datamodule), load(path), or construct it with k_shot=0 for pure zero-shot "
                    f"(expected calibration at {checkpoint})."
                )
                raise RuntimeError(msg)

        module = self._module
        if module is None or not self._has_text_embeddings(module):
            msg = (
                f"WinCLIP has no text embeddings for {self.class_name!r}; the prompt ensemble was never "
                "encoded. This should not happen — the model collects them on construction."
            )
            raise RuntimeError(msg)
        return module

    @staticmethod
    def _has_text_embeddings(module: WinClip) -> bool:
        """Whether the prompt ensemble has been encoded.

        Reads the buffer rather than the ``text_embeddings`` property, which
        raises instead of returning empty.
        """
        return bool(module.model._text_embeddings.numel())  # noqa: SLF001 - the property raises when empty

    # -- AnomalyModel ----------------------------------------------------------

    def train(self, datamodule: Any) -> None:
        """Establish embeddings and fit score calibration. **No optimization happens.**

        Worth being precise about, because the method name is inherited and
        overpromises here. For a zero-shot model this is not training in any
        sense a gradient would recognise — no parameter in the CLIP backbone is
        touched, no loss is computed, no optimizer exists (anomalib's WinCLIP
        module returns ``None`` from ``configure_optimizers``). What it does:

        1. **Encode the prompt ensemble** for :attr:`class_name` — ~150 sentences
           through CLIP's text encoder, averaged into two vectors. This is the
           only "learning" involved and it is a deterministic function of a
           string.
        2. **Collect reference images**, if ``k_shot > 0``, from the datamodule's
           train split, and encode their patch embeddings.
        3. **Fit score normalization and an adaptive threshold** by running
           ``Engine.test`` over the labelled test split. This is why the method
           takes a datamodule at all: for a zero-shot model anomalib runs a
           validation pass before testing precisely to collect these statistics.

        Only step 3 needs data, and only step 3 is optional — :meth:`predict`
        works on an instance this was never called on, which is the property
        ``tests/test_winclip.py`` exists to pin down. What calling it buys is a
        score scale fitted to *this* category rather than to CLIP's general sense
        of "damaged", which typically widens the gap between clean and defective
        scores considerably.

        One caveat about step 3, and it applies equally to the sibling wrappers:
        the calibration is fitted on the labelled test split, so metrics measured
        on that same split afterwards are optimistic. It is the split anomalib's
        MVTec datamodule derives its validation set from, and the alternative —
        calibrating on nothing — is worse. Read benchmark numbers with that in
        mind.

        Args:
            datamodule: An :class:`app.data.DataModule`. Its ``category`` wins
                over this instance's config, so calibrating a model for a
                category it was not constructed for cannot silently mislabel the
                checkpoint. Note that the *prompt* is not re-derived from it:
                an explicit :attr:`class_name` survives.

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

        logger.info(
            "'Training' %s on %r means encoding the prompt ensemble for %r and fitting score "
            "normalization — there are no gradients and no optimizer in this call.",
            self.model_name,
            self.category,
            self.class_name,
        )

        self._module = self._build_module()
        self._engine = self._build_engine()

        started = time.perf_counter()
        # anomalib's Lightning module needs anomalib's own batch type; see
        # DataModule.for_anomalib_engine for why that is scoped to this block.
        with datamodule.for_anomalib_engine() as anomalib_datamodule:
            # Engine.test, not Engine.fit: WinCLIP has no training_step at all.
            # For a zero-shot model anomalib runs a validation pass first, and
            # that pass is what fills in the normalization statistics.
            self._engine.test(model=self._module, datamodule=anomalib_datamodule, verbose=False)
        elapsed = time.perf_counter() - started

        # The validation pass leaked one encode_image hook per test image; drop
        # them before the module is cached for inference or written by save().
        self._clear_leaked_hooks(self._module)
        self._device = self._module.device
        self._module.eval()

        logger.info(
            "Set up %s for %r in %.1fs; class_name=%r, k_shot=%d, scales=%s, calibrated=%s",
            self.model_name,
            self.category,
            elapsed,
            self.class_name,
            self.k_shot,
            tuple(self.scales),
            self.is_calibrated,
        )
        if not self.is_calibrated:
            logger.warning(
                "No validation data was seen, so scores are raw prompt-similarity probabilities. "
                "Still in [0, 1] — unlike PatchCore's, which would be meaningless here.",
            )

        self.save(self.config.checkpoint_path(self.model_name, self.category))

    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        """Score one raw image and return a heatmap at its original resolution.

        Runs with or without a prior :meth:`train` — see :attr:`is_trained`. The
        image is converted to 3-channel RGB, bicubic-resized to
        240x240 (:data:`REQUIRED_IMAGE_SIZE`, which is why this backend ignores
        the configured ``image_size``), CLIP-normalized, and pushed through the
        ViT once. Window embeddings at each scale are compared to the two text
        embeddings, harmonically aggregated onto the 15x15 patch grid, and the
        result is bilinearly resampled back to the input's ``H x W`` so it can be
        overlaid directly. Interpolating 15x15 up to 900x900 is the reason the
        map looks smooth where PatchCore's looks sharp; the detail was never
        there to lose.

        Args:
            image: ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``
                array; ``uint8`` in ``[0, 255]`` or float.
            color_order: ``"rgb"`` or ``"bgr"``. Use ``"bgr"`` for anything from
                OpenCV.

        Returns:
            A :class:`ModelOutput` for this image, schema-identical to the other
            two backends'.

        Raises:
            RuntimeError: Only when ``k_shot > 0`` and no reference embeddings
                have been established. A zero-shot instance never raises here.
        """
        module = self._require_module()

        array = self._to_rgb_array(image, color_order=color_order)
        height, width = array.shape[:2]
        tensor = self._to_model_input(array)

        with torch.no_grad():
            # As in the sibling wrappers, deliberately the inner torch model
            # rather than `module(tensor)`: the Lightning module's forward would
            # re-run its own pre-processor over a batch already prepared here.
            raw = module.model(tensor.to(self._device))
            # anomalib's encode_image leaks a forward hook on every call; strip
            # it now so a long-lived served model does not grow ~5.7 MB per frame.
            self._clear_leaked_hooks(module)
            # Guarded on is_calibrated, which the siblings need not do because
            # they cannot predict before a fit. anomalib's post-processor is a
            # no-op when its statistics are NaN, so this is belt-and-braces — it
            # keeps the uncalibrated path from depending on that.
            scored = module.post_processor(raw) if self.is_calibrated and module.post_processor else raw

        score = float(scored.pred_score.reshape(-1)[0])
        anomaly_map = self._to_input_resolution(scored.anomaly_map, height, width)

        return ModelOutput(
            anomaly_score=score,
            anomaly_map=anomaly_map,
            is_defective=score >= self.config.anomaly_threshold,
            model_name=self.model_name,
        )

    def save(self, path: str | Path) -> None:
        """Write the calibration artifact — deliberately *not* a full checkpoint.

        The sibling wrappers persist a Lightning checkpoint because their weights
        *are* the trained thing. Nothing here is. WinCLIP's ~830 MB of CLIP
        weights are a fixed public artifact that open_clip already caches on
        disk, and the text embeddings are a pure function of :attr:`class_name`
        and the prompt templates. Serialising them would produce an ~830 MB file
        per category whose contents are byte-identical apart from two 640-element
        vectors, and would tie a "zero-shot" model to a checkpoint it does not
        need.

        So this writes only what cannot be recomputed from the config:

        * the post-processor's normalization statistics and adaptive thresholds,
          which came from :meth:`train`'s pass over real data;
        * the WinCLIP module's own buffers minus the ``clip.*`` weights — the
          text embeddings (cheap, and they pin down the exact prompt ensemble
          this calibration was fitted against) and, for ``k_shot > 0``, the
          reference images' visual embeddings, which genuinely are unrecoverable
          without the images;
        * ``class_name``/``k_shot``/``scales``, so :meth:`load` can detect a file
          fitted under a different configuration.

        The result is a few hundred KB. It keeps the
        ``results/checkpoints/<model>_<category>.ckpt`` naming convention so the
        serving layer can locate it knowing only the model name and category.

        Args:
            path: Destination ``.ckpt`` file. Parent directories are created.

        Raises:
            RuntimeError: If there is nothing to save yet.
        """
        module = self._require_module()
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        winclip_state = {
            key: value
            for key, value in module.model.state_dict().items()
            if not key.startswith(_CLIP_WEIGHT_PREFIX)
        }
        torch.save(
            {
                "format": _CHECKPOINT_FORMAT,
                "model_name": self.model_name,
                "class_name": self.class_name,
                "category": self.category,
                "k_shot": self.k_shot,
                "scales": tuple(self.scales),
                "winclip_state": winclip_state,
                "post_processor_state": (
                    module.post_processor.state_dict() if module.post_processor is not None else None
                ),
            },
            destination,
        )

        self._checkpoint_path = destination
        size_kb = destination.stat().st_size / 1024
        logger.info(
            "Saved %s calibration to %s (%.0f KB; CLIP weights deliberately excluded)",
            self.model_name,
            destination,
            size_kb,
        )

    def load(self, path: str | Path) -> None:
        """Restore an artifact written by :meth:`save` and cache it for inference.

        Rebuilds the CLIP backbone from open_clip's cache — which is why this is
        slow despite the file being small — and then overlays the embeddings and
        calibration the file carries.

        Args:
            path: Checkpoint file to load.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``path`` is not a WinCLIP calibration artifact. A
                Lightning checkpoint from a sibling wrapper is the likely
                mistake, and it would otherwise fail much later with an opaque
                key error.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            msg = f"No checkpoint at {source}."
            raise FileNotFoundError(msg)

        # weights_only=False: the payload carries plain Python metadata
        # (class_name, scales) alongside the tensors. These are our own
        # artifacts, written by save() above.
        payload = torch.load(source, map_location=self._device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != _CHECKPOINT_FORMAT:
            msg = (
                f"{source} is not a WinCLIP calibration artifact (expected format {_CHECKPOINT_FORMAT!r}). "
                f"WinCLIP does not read Lightning checkpoints — see WinCLIPModel.save for why it does not "
                f"write them either."
            )
            raise ValueError(msg)

        saved_class_name = payload.get("class_name")
        saved_scales = tuple(payload.get("scales", self.scales))
        if saved_class_name != self.class_name or saved_scales != tuple(self.scales):
            # Not fatal: the file's embeddings win, and they are self-consistent.
            # But the config now describes something the loaded model is not.
            logger.warning(
                "Calibration in %s was fitted for class_name=%r scales=%s; this instance is configured for "
                "class_name=%r scales=%s. Using the file's.",
                source,
                saved_class_name,
                saved_scales,
                self.class_name,
                tuple(self.scales),
            )
            self.config = self.config.with_overrides(class_name=saved_class_name, scales=saved_scales)

        self.config = self.config.with_overrides(k_shot=payload.get("k_shot"))
        module = self._build_module()

        # strict=False because the CLIP weights were deliberately not saved; they
        # come from open_clip's own cache via _build_module. Anything *extra* in
        # the file is a real inconsistency, so that is still checked.
        incompatible = module.model.load_state_dict(payload["winclip_state"], strict=False)
        unexpected = tuple(incompatible.unexpected_keys)
        if unexpected:
            msg = f"{source} contains state this WinCLIP build has no home for: {unexpected}."
            raise ValueError(msg)
        missing = [key for key in incompatible.missing_keys if not key.startswith(_CLIP_WEIGHT_PREFIX)]
        if missing:
            logger.warning("Calibration in %s is missing %s; those keep their freshly built values.", source, missing)

        module.model.class_name = saved_class_name or self.class_name
        module.model.k_shot = int(payload.get("k_shot", 0))
        module.is_setup = True

        post_state = payload.get("post_processor_state")
        if post_state is not None and module.post_processor is not None:
            module.post_processor.load_state_dict(post_state)

        module.eval()
        self._module = module
        self._checkpoint_path = source
        logger.info(
            "Loaded %s from %s; class_name=%r, k_shot=%d, calibrated=%s",
            self.model_name,
            source,
            self.class_name,
            self.k_shot,
            self.is_calibrated,
        )

    # -- internals -------------------------------------------------------------

    def _scale_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply CLIP's normalization statistics, not ImageNet's.

        The two are close enough that swapping them produces no error and no
        visible artifact — just a systematic shift in every image embedding,
        which moves its cosine similarity to *both* text embeddings and quietly
        rescales the score. PatchCore normalizes with ImageNet statistics here
        because its backbone is a timm model; EfficientAD leaves the hook alone
        because its PDN normalizes internally. Three backends, three answers, one
        override each.

        The clamp handles bicubic resizing's overshoot at saturated pixels: it
        can land a hair outside ``[0, 1]``, which bilinear-with-antialias does
        not do to the same degree.
        """
        return normalize_image(tensor.clamp(0.0, 1.0), mean=CLIP_MEAN, std=CLIP_STD)
