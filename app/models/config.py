"""Single source of truth for model hyperparameters.

Why this module exists
----------------------
Every tunable number the model layer needs lives here, as a validated pydantic
model with a hardcoded default. Model wrappers, training scripts, benchmarks and
the serving layer all read :class:`ModelConfig` instead of calling
``os.getenv`` themselves. That gives us:

* **One place to look.** ``coreset_sampling_ratio=0.1`` appears exactly once in
  the codebase; nothing else is allowed to spell a magic number inline.
* **Validation at the boundary.** A typo'd ``CORESET_SAMPLING_RATIO=10`` fails
  immediately with a readable pydantic error, rather than silently sampling the
  entire embedding set and exhausting memory an hour into a run.
* **Layered overrides.** Hardcoded default -> ``.env`` file -> process
  environment -> explicit keyword argument, each layer beating the one before
  it. A CLI flag therefore wins over the environment without any extra plumbing.

Resolution order is implemented by :meth:`ModelConfig.from_env`; the env var
backing each field is declared on the field itself, so adding a knob means
editing one line.

Example:
    >>> config = ModelConfig.from_env()                      # doctest: +SKIP
    >>> config.backbone                                      # doctest: +SKIP
    'wide_resnet50_2'
    >>> config.with_overrides(image_size=128).image_size      # doctest: +SKIP
    128
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ENV_PREFIXLESS_VARS", "ModelConfig", "get_model_config"]

# Repo-root-anchored so paths resolve identically from a test, a script or the
# API server, whatever the working directory happens to be.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"

#: Mirrors ``anomalib.models.image.efficient_ad.torch_model.EfficientAdModelSize``.
#: Spelled out rather than imported so loading a config never drags in anomalib.
_EFFICIENTAD_MODEL_SIZES: frozenset[str] = frozenset({"small", "medium"})


def _env(name: str) -> dict[str, str]:
    """Declare which environment variable backs a field."""
    return {"env": name}


class ModelConfig(BaseModel):
    """Validated hyperparameters shared by every anomaly-detection model wrapper.

    Instances are frozen: a config is read once and passed around, never mutated
    from under a running model. Use :meth:`with_overrides` to derive a variant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- dataset ---------------------------------------------------------------

    category: str = Field(
        default="bottle",
        description="MVTec-style object category to train/serve.",
        json_schema_extra=_env("DEFAULT_CATEGORY"),
    )
    data_root: Path = Field(
        default=_REPO_ROOT / "data" / "MVTecAD",
        description="Directory holding <category>/{train,test,ground_truth}.",
        json_schema_extra=_env("DATA_ROOT"),
    )

    # -- backbone / feature extraction ----------------------------------------

    backbone: str = Field(
        default="wide_resnet50_2",
        description="timm backbone used as the frozen patch-feature extractor.",
        json_schema_extra=_env("MODEL_BACKBONE"),
    )
    layers: tuple[str, ...] = Field(
        default=("layer2", "layer3"),
        min_length=1,
        description="Backbone layers whose activations form the patch embedding.",
        json_schema_extra=_env("MODEL_LAYERS"),
    )

    # -- PatchCore memory bank -------------------------------------------------

    coreset_sampling_ratio: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Fraction of training patch embeddings kept by k-center-greedy.",
        json_schema_extra=_env("CORESET_SAMPLING_RATIO"),
    )
    num_neighbors: int = Field(
        default=9,
        ge=1,
        description="Neighbours consulted when re-weighting the nearest-patch distance.",
        json_schema_extra=_env("NUM_NEIGHBORS"),
    )

    # -- EfficientAD student-teacher -------------------------------------------

    model_size: str = Field(
        default="small",
        description="EfficientAD patch description network size: small | medium.",
        json_schema_extra=_env("MODEL_SIZE"),
    )
    imagenet_dir: Path = Field(
        default=_REPO_ROOT / "data" / "imagenette",
        description="ImageNette root used by EfficientAD's distillation penalty.",
        json_schema_extra=_env("IMAGENET_DIR"),
    )

    # -- WinCLIP zero-shot -----------------------------------------------------

    class_name: str | None = Field(
        default=None,
        description="Noun WinCLIP puts in its text prompts; None derives it from `category`.",
        json_schema_extra=_env("WINCLIP_CLASS_NAME"),
    )
    k_shot: int = Field(
        default=0,
        ge=0,
        description="Normal reference images WinCLIP may consult; 0 is pure zero-shot.",
        json_schema_extra=_env("K_SHOT"),
    )
    scales: tuple[int, ...] = Field(
        default=(2, 3),
        min_length=1,
        description="Sliding-window sizes, in ViT patches, WinCLIP aggregates scores over.",
        json_schema_extra=_env("WINCLIP_SCALES"),
    )

    # -- inference -------------------------------------------------------------

    image_size: int = Field(
        default=256,
        ge=32,
        description="Square resolution every image is resized to before inference.",
        json_schema_extra=_env("IMAGE_SIZE"),
    )
    anomaly_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized score at or above which a frame is called defective.",
        json_schema_extra=_env("ANOMALY_THRESHOLD"),
    )

    # -- training --------------------------------------------------------------

    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for the train loader (eval reuses it).",
        json_schema_extra=_env("TRAIN_BATCH_SIZE"),
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="Dataloader worker processes; 0 keeps runs deterministic.",
        json_schema_extra=_env("NUM_WORKERS"),
    )
    max_epochs: int = Field(
        default=1,
        ge=1,
        description="Training epochs. PatchCore is single-pass and pins this to 1; EfficientAD honours it.",
        json_schema_extra=_env("MAX_EPOCHS"),
    )
    accelerator: str = Field(
        default="auto",
        description="Lightning accelerator: auto | cpu | gpu | mps.",
        json_schema_extra=_env("ACCELERATOR"),
    )
    devices: int = Field(
        default=1,
        ge=1,
        description="Device count. PatchCore's memory bank is single-device only.",
        json_schema_extra=_env("DEVICES"),
    )
    seed: int | None = Field(
        default=None,
        description="Seed for dataset splitting; None leaves anomalib's default.",
        json_schema_extra=_env("SEED"),
    )

    # -- artifacts -------------------------------------------------------------

    results_dir: Path = Field(
        default=_REPO_ROOT / "results",
        description="Root for run artifacts (logs, metrics, checkpoints).",
        json_schema_extra=_env("RESULTS_DIR"),
    )
    checkpoint_dir: Path = Field(
        default=_REPO_ROOT / "results" / "checkpoints",
        description="Directory trained checkpoints are written to.",
        json_schema_extra=_env("CHECKPOINT_DIR"),
    )

    # -- parsing ---------------------------------------------------------------

    @field_validator("layers", mode="before")
    @classmethod
    def _split_layers(cls, value: Any) -> Any:
        """Accept ``MODEL_LAYERS="layer2,layer3"`` as well as a real sequence."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("scales", mode="before")
    @classmethod
    def _split_scales(cls, value: Any) -> Any:
        """Accept ``WINCLIP_SCALES="2,3"`` as well as a real sequence of ints."""
        if isinstance(value, str):
            return tuple(int(part.strip()) for part in value.split(",") if part.strip())
        return value

    @field_validator("scales")
    @classmethod
    def _check_scales(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject window sizes that cannot tile WinCLIP's 15x15 patch grid.

        A scale is a window edge measured in ViT patches, so 1 is a single patch
        (no context at all, which the full-image scale already covers) and
        anything past the grid produces zero windows and an empty score map.
        """
        if any(scale < 2 for scale in value):
            msg = f"window scales must be at least 2 patches, got {value}"
            raise ValueError(msg)
        return value

    @field_validator("category", "backbone", "accelerator")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "must not be blank"
            raise ValueError(msg)
        return value.strip()

    @field_validator("model_size")
    @classmethod
    def _check_model_size(cls, value: str) -> str:
        """Reject sizes anomalib's ``EfficientAdModelSize`` does not know.

        Caught here rather than inside ``EfficientAd.__init__`` so a typo fails
        at config load, not twenty minutes into a fit.
        """
        size = value.strip().lower()
        if size not in _EFFICIENTAD_MODEL_SIZES:
            msg = f"must be one of {sorted(_EFFICIENTAD_MODEL_SIZES)}, got {value!r}"
            raise ValueError(msg)
        return size

    # -- loading ---------------------------------------------------------------

    @classmethod
    def env_var_for(cls, field_name: str) -> str | None:
        """Return the environment variable backing ``field_name``, if any."""
        field = cls.model_fields[field_name]
        extra = field.json_schema_extra
        return extra.get("env") if isinstance(extra, Mapping) else None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        use_dotenv: bool = True,
        **overrides: Any,
    ) -> ModelConfig:
        """Build a config from defaults, then ``.env``, then env vars, then kwargs.

        Args:
            environ: Environment mapping to read. Defaults to ``os.environ``.
            use_dotenv: Whether to load ``<repo>/.env`` first. Values already
                present in the real environment are never clobbered, so an
                exported variable still beats the file.
            **overrides: Explicit values that win over everything else.
                ``None`` values are ignored, which lets callers forward optional
                CLI arguments straight through without null-checking each one.

        Returns:
            A validated, frozen :class:`ModelConfig`.
        """
        if use_dotenv and environ is None and _DOTENV_PATH.is_file():
            load_dotenv(_DOTENV_PATH, override=False)

        source = os.environ if environ is None else environ

        values: dict[str, Any] = {}
        for name in cls.model_fields:
            env_name = cls.env_var_for(name)
            if env_name is None:
                continue
            raw = source.get(env_name)
            # Treat an empty string as "unset" so `FOO=` in a .env file falls
            # back to the default instead of failing validation.
            if raw is not None and raw.strip() != "":
                values[name] = raw

        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> ModelConfig:
        """Return a revalidated copy with ``overrides`` applied (``None`` ignored)."""
        merged = self.model_dump()
        merged.update({key: value for key, value in overrides.items() if value is not None})
        return type(self)(**merged)

    # -- derived paths ---------------------------------------------------------

    def checkpoint_path(self, model_name: str, category: str | None = None) -> Path:
        """Canonical checkpoint location, e.g. ``results/checkpoints/patchcore_bottle.ckpt``.

        Keeping the naming convention here (rather than in each wrapper) means
        the serving layer can locate a checkpoint knowing only the model name
        and category.
        """
        return self.checkpoint_dir / f"{model_name}_{category or self.category}.ckpt"

    @property
    def image_hw(self) -> tuple[int, int]:
        """``image_size`` as the ``(height, width)`` pair torchvision expects."""
        return (self.image_size, self.image_size)

    @property
    def prompt_class_name(self) -> str:
        """The noun WinCLIP puts in its prompts: :attr:`class_name`, else :attr:`category`.

        When it falls through to the category, underscores become spaces. MVTec's
        folder names are identifiers (``metal_nut``, ``pill``) and CLIP's text
        encoder was trained on prose: ``"a photo of a metal_nut."`` byte-pair
        encodes into fragments it has never seen arranged that way, while
        ``"a photo of a metal nut."`` is an ordinary English phrase whose
        embedding sits where the visual embedding of an actual metal nut does.
        The substitution is a no-op on single-word categories and free accuracy
        on the compound ones.
        """
        return self.class_name or self.category.replace("_", " ")


#: Environment variables this module reads, for documentation and tests.
ENV_PREFIXLESS_VARS: tuple[str, ...] = tuple(
    env for env in (ModelConfig.env_var_for(name) for name in ModelConfig.model_fields) if env is not None
)

_cached: ModelConfig | None = None


def get_model_config(*, refresh: bool = False, **overrides: Any) -> ModelConfig:
    """Return the process-wide config, loading it from the environment once.

    Args:
        refresh: Re-read the environment instead of using the cached instance.
            Tests that monkeypatch env vars need this.
        **overrides: Applied on top of the resolved config. Passing any override
            returns a derived instance and leaves the cache untouched.
    """
    global _cached  # noqa: PLW0603 - module-level memoisation of a frozen value
    if _cached is None or refresh:
        _cached = ModelConfig.from_env()
    return _cached.with_overrides(**overrides) if overrides else _cached
