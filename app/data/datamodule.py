"""Thin wrapper around anomalib's MVTec AD datamodule.

Why this wrapper exists
-----------------------
anomalib is an implementation detail of *how* we load inspection images, not
part of this project's domain model. This module is the single place in the
codebase that is allowed to ``import anomalib.data`` — everything downstream
(models, evaluation, serving) talks to :class:`DataModule` and receives
:class:`~app.data.batch.DefectBatch` objects instead.

That buys three things:

1. **Swapping datasets is local.** Pointing at VisA, a bespoke MVTec-style
   folder tree, or a customer's own images means editing this file only; no
   business logic changes.
2. **No type leakage.** anomalib's dataloaders natively yield
   ``anomalib.data.ImageBatch``. Returning those would couple every consumer to
   anomalib's dataclass API (``gt_label``, ``gt_mask``, ...) even without an
   explicit import. We install a collate function that converts each batch into
   a plain :class:`DefectBatch` *inside the worker process*, so the objects
   crossing this boundary contain nothing but ``torch`` tensors and strings.
3. **A stable preprocessing contract.** ``image_size`` is enforced here and
   verified with :func:`~app.data.transforms.validate_image_shape`, so a
   mis-sized batch fails at the data layer rather than deep inside a model.

Example:
    >>> dm = DataModule(category="bottle", image_size=256, batch_size=8)
    >>> dm.setup()                                  # doctest: +SKIP
    >>> batch = next(iter(dm.test_dataloader()))    # doctest: +SKIP
    >>> batch.image.shape, batch.label.shape        # doctest: +SKIP
    (torch.Size([8, 3, 256, 256]), torch.Size([8]))
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from anomalib.data import ImageBatch, MVTecAD
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from app.data.batch import DefectBatch
from app.data.transforms import validate_image_shape

__all__ = ["DEFAULT_CATEGORY", "DEFAULT_DATA_ROOT", "DEFAULT_IMAGE_SIZE", "DataModule"]

# Repo-root-anchored so the datamodule finds the dataset regardless of the
# working directory a script, test or server happens to be launched from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT: Path = _REPO_ROOT / "data" / "MVTecAD"
DEFAULT_CATEGORY = "bottle"
DEFAULT_IMAGE_SIZE = 256


def _to_defect_batch(batch: ImageBatch, expected_size: int) -> DefectBatch:
    """Convert an anomalib ``ImageBatch`` into a framework-agnostic batch.

    Also the enforcement point for the image-size contract: anything that got
    past the augmentation pipeline at the wrong resolution fails here, with the
    dataset paths still in scope for a useful error message.
    """
    # `.as_subclass` strips the torchvision tv_tensor subclass, leaving plain
    # tensors that pickle and print predictably outside the anomalib stack.
    image = batch.image.as_subclass(torch.Tensor).to(torch.float32)
    validate_image_shape(image, expected_size, name="batch image")
    # Antialiased bilinear resizing overshoots by ~1e-7 at saturated pixels, so
    # clamp to make the documented [0, 1] contract exactly true. In-place is
    # safe here: `batch` is collated by the caller and discarded immediately.
    image.clamp_(0.0, 1.0)

    label = torch.as_tensor(batch.gt_label).as_subclass(torch.Tensor).to(torch.int64)

    mask = batch.gt_mask
    if mask is not None:
        mask = mask.as_subclass(torch.Tensor).to(torch.uint8)

    return DefectBatch(
        image=image,
        label=label,
        mask=mask,
        image_path=tuple(batch.image_path or ()),
        mask_path=tuple(path or "" for path in (batch.mask_path or ())),
    )


class _PlainBatchCollate:
    """Collate ``ImageItem``s with anomalib, then hand back a ``DefectBatch``.

    Defined as a module-level class (rather than a closure or lambda) because
    ``DataLoader`` workers pickle the collate function when ``num_workers > 0``.
    """

    def __init__(self, expected_size: int) -> None:
        self.expected_size = expected_size

    def __call__(self, items: Sequence[Any]) -> DefectBatch:
        return _to_defect_batch(ImageBatch.collate(list(items)), self.expected_size)


class DataModule:
    """Load MVTec-AD-style defect data without exposing anomalib to callers.

    Args:
        category: MVTec AD object category, e.g. ``"bottle"``.
        image_size: Square resolution every image and mask is resized to.
        batch_size: Batch size for the train loader; also used for eval unless
            ``eval_batch_size`` is given.
        root: Directory holding ``<category>/{train,test,ground_truth}``.
            Defaults to ``<repo>/data/MVTecAD``, which is what
            ``scripts/download_dataset.py`` populates.
        num_workers: Dataloader worker processes. ``0`` (default) keeps things
            deterministic and avoids worker start-up cost on small categories.
        eval_batch_size: Batch size for test/val loaders. Defaults to
            ``batch_size``.
        seed: Seed for anomalib's train/val/test splitting.
    """

    def __init__(
        self,
        category: str = DEFAULT_CATEGORY,
        image_size: int = DEFAULT_IMAGE_SIZE,
        batch_size: int = 8,
        root: Path | str = DEFAULT_DATA_ROOT,
        num_workers: int = 0,
        eval_batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        if image_size <= 0:
            msg = f"image_size must be positive, got {image_size}."
            raise ValueError(msg)
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}."
            raise ValueError(msg)

        self.category = category
        self.image_size = image_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.root = Path(root)
        self.num_workers = num_workers
        self.seed = seed
        self._is_setup = False

        # anomalib 2.x has no `image_size` argument: resolution is controlled by
        # the augmentation pipeline (or, during training, by the model's own
        # pre-processor). Resizing here rather than after collation also keeps
        # worker shared-memory usage proportional to `image_size`, not to the
        # 900x900 MVTec source images.
        self._datamodule = MVTecAD(
            root=self.root,
            category=self.category,
            train_batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            augmentations=Resize((self.image_size, self.image_size), antialias=True),
            seed=self.seed,
        )
        # Replaces anomalib's own collate on every split, so no ImageBatch ever
        # escapes this module. See the module docstring, point 2.
        self._datamodule.external_collate_fn = _PlainBatchCollate(self.image_size)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(category={self.category!r}, image_size={self.image_size}, "
            f"batch_size={self.batch_size}, root={str(self.root)!r})"
        )

    # -- lifecycle ------------------------------------------------------------

    @property
    def category_dir(self) -> Path:
        """Directory the images for this category are read from."""
        return self.root / self.category

    @property
    def is_setup(self) -> bool:
        """Whether :meth:`setup` has already run."""
        return self._is_setup

    @property
    def anomalib_datamodule(self) -> MVTecAD:
        """Escape hatch: the underlying anomalib datamodule.

        anomalib's ``Engine`` needs a real ``LightningDataModule`` to fit models
        such as PatchCore. Training entrypoints may reach for this; ordinary
        business logic must not, or the decoupling above is pointless.

        Note that its dataloaders still yield :class:`DefectBatch`. To drive
        ``Engine.fit`` you want :meth:`for_anomalib_engine` instead.
        """
        return self._datamodule

    @contextmanager
    def for_anomalib_engine(self) -> Iterator[MVTecAD]:
        """Yield the anomalib datamodule with anomalib's *native* collate restored.

        The plain-batch conversion above is what we want everywhere in our own
        code, but anomalib's Lightning modules are not consumers of ours: their
        ``validation_step`` calls ``batch.update(...)`` and their pre-processor
        writes back to ``batch.gt_mask``, both of which are ``ImageBatch`` API.
        Feeding them a :class:`DefectBatch` fails with an ``AttributeError``
        several minutes into a fit.

        So training — and only training — opts back into anomalib's own batch
        type, scoped to a ``with`` block so no long-lived object is left in that
        state:

            >>> with dm.for_anomalib_engine() as anomalib_dm:   # doctest: +SKIP
            ...     engine.fit(model=module, datamodule=anomalib_dm)

        Yields:
            The underlying :class:`~anomalib.data.MVTecAD` datamodule.
        """
        self._ensure_setup()
        previous = self._datamodule.external_collate_fn
        self._datamodule.external_collate_fn = None
        try:
            yield self._datamodule
        finally:
            self._datamodule.external_collate_fn = previous

    def prepare_data(self) -> None:
        """Ensure the dataset is present, downloading via anomalib if not.

        ``scripts/download_dataset.py`` is the normal way to populate the data
        directory (it fetches a single category from the HuggingFace mirror).
        This is the last-resort path, and pulls the full multi-category MVTec AD
        archive.
        """
        self._datamodule.prepare_data()

    def setup(self, stage: str | None = None) -> DataModule:
        """Build the train/val/test datasets.

        Args:
            stage: Optional Lightning stage hint (``"fit"``, ``"test"``, ...);
                anomalib creates all three subsets regardless, because the
                validation set is derived from the test set.

        Returns:
            ``self``, so ``dm = DataModule(...).setup()`` reads naturally.

        Raises:
            FileNotFoundError: If the category directory does not exist. Raised
                here rather than letting anomalib fail on an empty glob, so the
                fix (run the download script) is obvious.
        """
        if not self.category_dir.is_dir():
            msg = (
                f"No data found at {self.category_dir}. Run "
                f"`python scripts/download_dataset.py --category {self.category}` first."
            )
            raise FileNotFoundError(msg)

        self._datamodule.setup(stage)
        self._is_setup = True
        return self

    def _ensure_setup(self) -> None:
        if not self._is_setup:
            self.setup()

    # -- dataloaders ----------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Shuffled loader over ``train/good`` yielding :class:`DefectBatch`."""
        self._ensure_setup()
        return self._datamodule.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        """Validation loader yielding :class:`DefectBatch`.

        With anomalib's MVTec AD defaults the validation split mirrors the test
        split, since unsupervised anomaly detection has no labelled val set.
        """
        self._ensure_setup()
        return self._datamodule.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Loader over normal + defective test images yielding :class:`DefectBatch`."""
        self._ensure_setup()
        return self._datamodule.test_dataloader()

    # -- introspection --------------------------------------------------------

    def num_samples(self) -> dict[str, int]:
        """Sample counts per split, useful for logging and sanity checks."""
        self._ensure_setup()
        return {
            "train": len(self._datamodule.train_data),
            "val": len(self._datamodule.val_data),
            "test": len(self._datamodule.test_data),
        }
