"""Shared image transforms for the defect-detection pipeline.

Every stage that touches pixels — dataset loading, the FastAPI inference
endpoint, evaluation scripts — goes through these helpers so that an image is
preprocessed identically no matter how it entered the system. Keeping the
ImageNet statistics in one place also means a backbone swap is a one-line
change here rather than a grep across the codebase.

Conventions used throughout:
    * Tensors are ``float32`` in ``[0, 1]`` after :func:`to_tensor`.
    * Images are channel-first: ``(C, H, W)`` or batched ``(N, C, H, W)``.
    * Normalization is applied *after* resizing, never before.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "denormalize_image",
    "normalize_image",
    "to_tensor",
    "validate_image_shape",
]

# Statistics of the ImageNet-1k training set. Every torchvision/timm backbone we
# use (ResNet, WideResNet, ViT) was pretrained with these, so features degrade
# noticeably if inputs are normalized with anything else.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

ImageLike = Image.Image | np.ndarray | torch.Tensor


def to_tensor(image: ImageLike) -> torch.Tensor:
    """Convert an image to a channel-first ``float32`` tensor in ``[0, 1]``.

    Accepts the three representations that reach us in practice: a PIL image
    (file uploads), a NumPy array (OpenCV frames, which are ``H x W x C``), or a
    tensor that is already channel-first (dataloader output).

    Args:
        image: PIL image, ``(H, W)``/``(H, W, C)`` array, or ``(C, H, W)`` /
            ``(N, C, H, W)`` tensor.

    Returns:
        ``float32`` tensor scaled to ``[0, 1]``, channel-first, with a channel
        dimension always present.

    Raises:
        TypeError: If ``image`` is not one of the supported types.
        ValueError: If the array/tensor rank is not supported.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # grayscale mask or single-channel image
            image = image[:, :, None]
        if image.ndim != 3:
            msg = f"Expected a 2D or 3D array for an image, got shape {image.shape}."
            raise ValueError(msg)
        array = np.ascontiguousarray(image)
        if not array.flags.writeable:
            # PIL-backed buffers are read-only, and torch.from_numpy would alias
            # them into a tensor it believes is mutable.
            array = array.copy()
        # NumPy images are H x W x C; torch wants C x H x W.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        # Drop any tensor subclass (e.g. torchvision tv_tensors) so callers
        # always get a plain torch.Tensor back.
        tensor = image.as_subclass(torch.Tensor)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim not in {3, 4}:
            msg = f"Expected a 2D, 3D or 4D tensor for an image, got shape {tuple(tensor.shape)}."
            raise ValueError(msg)
    else:
        msg = f"Unsupported image type {type(image)!r}; expected PIL.Image, numpy.ndarray or torch.Tensor."
        raise TypeError(msg)

    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255.0)
    return tensor.to(torch.float32)


def normalize_image(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Normalize a ``[0, 1]`` image tensor with per-channel mean/std.

    Args:
        image: ``(C, H, W)`` or ``(N, C, H, W)`` float tensor in ``[0, 1]``.
        mean: Per-channel means. Defaults to ImageNet statistics.
        std: Per-channel standard deviations. Defaults to ImageNet statistics.

    Returns:
        A new tensor, ``(image - mean) / std``. The input is not modified.

    Raises:
        ValueError: If the rank is wrong, the channel count does not match the
            supplied statistics, or any ``std`` entry is zero.
    """
    if image.ndim not in {3, 4}:
        msg = f"normalize_image expects (C, H, W) or (N, C, H, W), got shape {tuple(image.shape)}."
        raise ValueError(msg)

    channels = image.shape[-3]
    if len(mean) != channels or len(std) != channels:
        msg = f"Image has {channels} channel(s) but mean/std describe {len(mean)}/{len(std)}."
        raise ValueError(msg)
    if any(s == 0 for s in std):
        msg = f"std must not contain zeros, got {tuple(std)}."
        raise ValueError(msg)

    image = image.as_subclass(torch.Tensor).to(torch.float32)
    shape = (channels, 1, 1)
    mean_t = torch.as_tensor(tuple(mean), dtype=image.dtype, device=image.device).view(shape)
    std_t = torch.as_tensor(tuple(std), dtype=image.dtype, device=image.device).view(shape)
    return (image - mean_t) / std_t


def denormalize_image(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert :func:`normalize_image`, clamped back to ``[0, 1]``.

    Used when overlaying anomaly heatmaps on the original image for the
    visualization endpoints.
    """
    channels = image.shape[-3]
    image = image.as_subclass(torch.Tensor).to(torch.float32)
    shape = (channels, 1, 1)
    mean_t = torch.as_tensor(tuple(mean), dtype=image.dtype, device=image.device).view(shape)
    std_t = torch.as_tensor(tuple(std), dtype=image.dtype, device=image.device).view(shape)
    return (image * std_t + mean_t).clamp_(0.0, 1.0)


def validate_image_shape(
    tensor: torch.Tensor,
    expected_size: int | tuple[int, int],
    *,
    expected_channels: int | None = 3,
    name: str = "image",
) -> torch.Tensor:
    """Assert that ``tensor`` is a well-formed image of ``expected_size``.

    This is the guard rail used at every boundary in the codebase (dataloader
    collation, model input, serving handlers). It raises :class:`ValueError`
    rather than using a bare ``assert`` so the check survives ``python -O`` and
    produces a message a caller can actually act on.

    Args:
        tensor: ``(C, H, W)`` or ``(N, C, H, W)`` image tensor.
        expected_size: Square size as an ``int``, or an explicit ``(H, W)``.
        expected_channels: Required channel count, or ``None`` to skip the
            check (masks, for instance, are single-channel).
        name: Label used in the error message to identify the offending tensor.

    Returns:
        The tensor unchanged, so the call can be used inline.

    Raises:
        TypeError: If ``tensor`` is not a :class:`torch.Tensor`.
        ValueError: If the rank, channel count or spatial size is wrong.
    """
    if not isinstance(tensor, torch.Tensor):
        msg = f"{name} must be a torch.Tensor, got {type(tensor)!r}."
        raise TypeError(msg)

    if tensor.ndim not in {3, 4}:
        msg = f"{name} must be (C, H, W) or (N, C, H, W), got shape {tuple(tensor.shape)}."
        raise ValueError(msg)

    expected_hw = (expected_size, expected_size) if isinstance(expected_size, int) else tuple(expected_size)
    if len(expected_hw) != 2:
        msg = f"expected_size must be an int or an (H, W) pair, got {expected_size!r}."
        raise ValueError(msg)

    if expected_channels is not None and tensor.shape[-3] != expected_channels:
        msg = f"{name} must have {expected_channels} channel(s), got {tensor.shape[-3]} (shape {tuple(tensor.shape)})."
        raise ValueError(msg)

    actual_hw = tuple(tensor.shape[-2:])
    if actual_hw != expected_hw:
        msg = f"{name} must be {expected_hw[0]}x{expected_hw[1]}, got {actual_hw[0]}x{actual_hw[1]}."
        raise ValueError(msg)

    return tensor
