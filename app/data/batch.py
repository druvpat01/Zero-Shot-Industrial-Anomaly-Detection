"""Framework-agnostic batch container handed to the rest of the codebase.

Why this exists
---------------
``app.data.datamodule`` builds on anomalib, but anomalib's dataloaders yield
``anomalib.data.ImageBatch`` objects. If we passed those straight through, every
consumer (models, evaluation, serving) would be coupled to anomalib's dataclass
API even without an ``import anomalib`` line — attribute names like ``gt_label``
and ``gt_mask`` would leak into business logic, and swapping the data source
would ripple across the project. :class:`DefectBatch` is the narrow contract
instead: three tensors plus provenance, nothing else.

It behaves both like a dataclass (``batch.image``) and like a read-only mapping
(``batch["image"]``, ``dict(batch)``), because downstream code and tests
naturally reach for either.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch

__all__ = ["BATCH_KEYS", "DefectBatch"]

#: Canonical field order. ``image``/``label``/``mask`` are the contract every
#: consumer relies on; the ``*_path`` fields are provenance for reporting.
BATCH_KEYS: tuple[str, ...] = ("image", "label", "mask", "image_path", "mask_path")


@dataclass(frozen=True)
class DefectBatch:
    """One batch of industrial inspection images.

    Attributes:
        image: ``(N, C, H, W)`` ``float32`` tensor in ``[0, 1]``.
        label: ``(N,)`` ``int64`` tensor — ``0`` normal, ``1`` anomalous.
        mask: ``(N, H, W)`` ``uint8`` segmentation ground truth, or ``None``
            when the dataset is classification-only. Normal samples carry an
            all-zero mask rather than ``None``, so the tensor is always
            batch-aligned with ``image``.
        image_path: Source path of each image, in batch order.
        mask_path: Source path of each ground-truth mask; empty string where a
            sample has no mask on disk.
    """

    image: torch.Tensor
    label: torch.Tensor
    mask: torch.Tensor | None = None
    image_path: tuple[str, ...] = ()
    mask_path: tuple[str, ...] = ()

    # -- mapping-style access -------------------------------------------------

    @staticmethod
    def keys() -> tuple[str, ...]:
        """Return the batch field names (enables ``dict(batch)``)."""
        return BATCH_KEYS

    def __getitem__(self, key: str) -> object:
        """Look a field up by name, e.g. ``batch["image"]``."""
        if key not in BATCH_KEYS:
            msg = f"Unknown batch key {key!r}; expected one of {BATCH_KEYS}."
            raise KeyError(msg)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return key in BATCH_KEYS

    def __iter__(self) -> Iterator[str]:
        return iter(BATCH_KEYS)

    def as_dict(self) -> dict[str, object]:
        """Return a plain ``dict`` copy of the batch."""
        return {key: getattr(self, key) for key in BATCH_KEYS}

    # -- convenience ----------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return int(self.image.shape[0])

    def __len__(self) -> int:
        """Number of *samples* (not fields) — batches are iterated by sample."""
        return self.batch_size

    @property
    def has_mask(self) -> bool:
        """Whether pixel-level ground truth is available for this batch."""
        return self.mask is not None

    def to(self, device: torch.device | str) -> DefectBatch:
        """Return a copy with all tensors moved to ``device``."""
        return replace(
            self,
            image=self.image.to(device),
            label=self.label.to(device),
            mask=None if self.mask is None else self.mask.to(device),
        )
