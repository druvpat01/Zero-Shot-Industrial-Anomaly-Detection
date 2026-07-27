"""Serve an exported ONNX model behind the project's :class:`AnomalyModel` interface.

Why this exists
---------------
:mod:`scripts.export_onnx` turns a trained PatchCore or EfficientAD checkpoint
into a ``.onnx`` graph. This module is what *runs* one at inference time, and it
is deliberately an :class:`~app.models.base.AnomalyModel` like the PyTorch
wrappers: same ``predict(image) -> ModelOutput`` contract, same input-quality
guard, same "heatmap comes back at the caller's resolution" promise. The serving
layer therefore holds an :class:`ONNXRunner` or a
:class:`~app.models.patchcore.PatchCoreModel` without knowing which — flipping
``MODEL_BACKEND`` from ``patchcore`` to ``patchcore-onnx`` is a factory change,
not a route change.

What the exported graph already does, and why preprocessing is thinner here
---------------------------------------------------------------------------
anomalib exports the *whole* Lightning module's ``forward``, not the bare inner
network:

    input image ─▶ pre-processor (resize, + ImageNet-normalize for PatchCore)
                ─▶ model
                ─▶ post-processor (min-max score/​map normalization, thresholds)
                ─▶ (pred_score, pred_label, anomaly_map, pred_mask)

So resize, normalization and calibration all live *inside* the graph. That is
the one place this wrapper diverges from :class:`~app.models.patchcore.PatchCoreModel`:
the PyTorch wrapper hands its backbone an ImageNet-normalized tensor and calls
``module.model`` directly, but the ONNX graph normalizes internally, so feeding
it a normalized batch would normalize twice. :meth:`_scale_for_model` is
therefore the identity — this wrapper hands the graph a plain ``[0, 1]`` batch
and lets the baked-in pre-processor do the rest. Verified end to end: the ONNX
image-level score reproduces the PyTorch wrapper's to ~1e-6.

The graph's input is a fixed ``(N, 3, S, S)`` where ``S`` is the resolution the
checkpoint was exported at; the ``.onnx`` file, not the process config, is the
source of truth for ``S``, so the runner reads it back from the session and
resizes to match (reconciling :attr:`config` to it, with a warning if they
disagree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from app.models.base import AnomalyModel, ModelOutput
from app.models.config import ModelConfig, get_model_config
from app.observability.logging_config import get_logger

__all__ = ["DEFAULT_EXPORTED_DIR", "ONNXRunner", "onnx_artifact_path"]

log = get_logger(__name__)

# Repo-root-anchored so the default export location resolves the same from a
# script, a test or the API server, whatever the working directory is.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where :mod:`scripts.export_onnx` writes, mirroring anomalib's own
#: ``<root>/weights/onnx`` layout so ``engine.export(export_root=...)`` and this
#: module agree on one convention rather than two.
DEFAULT_EXPORTED_DIR: Path = _REPO_ROOT / "results" / "exported"
_ONNX_SUBDIR = Path("weights") / "onnx"

#: The output tensors of anomalib's exported ``forward``. Only the first two are
#: needed to build a :class:`ModelOutput`; the graph also emits ``pred_label``
#: and ``pred_mask`` (its own thresholded views), which this wrapper ignores in
#: favour of applying :attr:`~app.models.config.ModelConfig.anomaly_threshold`
#: to the score itself, exactly as the PyTorch wrappers do.
_SCORE_OUTPUT = "pred_score"
_MAP_OUTPUT = "anomaly_map"


def onnx_artifact_path(
    model_name: str,
    precision: str = "fp32",
    exported_dir: Path | str | None = None,
) -> Path:
    """Canonical location of an exported ``.onnx`` file.

    Keeping the naming convention here (rather than duplicated across the export
    script, the benchmark and the tests) means every caller locates an artifact
    knowing only the model name and precision.

    Args:
        model_name: Backend the graph was exported from, e.g. ``"patchcore"``.
        precision: ``"fp32"`` for the plain export, or ``"int8"`` for the
            dynamically-quantized one. ``fp32`` gets no suffix so the file keeps
            the bare name anomalib's ``engine.export`` writes.
        exported_dir: Export root. Defaults to :data:`DEFAULT_EXPORTED_DIR`.

    Returns:
        e.g. ``results/exported/weights/onnx/patchcore_int8.onnx``.
    """
    base = Path(exported_dir) if exported_dir is not None else DEFAULT_EXPORTED_DIR
    suffix = "" if precision == "fp32" else f"_{precision}"
    return base / _ONNX_SUBDIR / f"{model_name}{suffix}.onnx"


class ONNXRunner(AnomalyModel):
    """Run an exported ONNX anomaly model through onnxruntime on CPU.

    Interface-identical to the PyTorch wrappers: ``predict`` takes a raw frame
    and returns a :class:`ModelOutput` whose ``anomaly_map`` matches the frame's
    resolution, and the same input-quality guard runs first. Unlike them there
    is no training lifecycle — an :class:`ONNXRunner` is inference-only, built
    from a ``.onnx`` file that :mod:`scripts.export_onnx` produced — so
    :meth:`train`, :meth:`save` and :meth:`load` raise :class:`NotImplementedError`.

    Args:
        onnx_path: Path to the ``.onnx`` file to serve.
        model_name: Value carried on every :class:`ModelOutput.model_name`.
            Defaults to the file stem (``"patchcore"``, ``"patchcore_int8"``),
            which is descriptive and keeps distinct backends distinguishable in
            logs and benchmark tables.
        config: Config supplying ``anomaly_threshold`` (and a fallback input
            size). Defaults to the process-wide one. Its ``image_size`` is
            reconciled to the graph's actual input resolution.
        providers: onnxruntime execution providers. Defaults to CPU only, which
            is the backend this whole export path exists to beat PyTorch on.
        num_threads: Intra-op thread count for the session. ``None`` leaves
            onnxruntime's default, which on a hyperthreaded CPU tends to spawn
            one thread per *logical* core and oversubscribe against a co-resident
            torch thread pool — measurably slower. Pin it to the *physical* core
            count (what a co-timed PyTorch model uses) for a fair, faster run;
            the latency benchmark does exactly this.

    Raises:
        FileNotFoundError: If ``onnx_path`` does not exist.

    Example:
        >>> from app.models.onnx_runner import ONNXRunner, onnx_artifact_path
        >>> runner = ONNXRunner(onnx_artifact_path("patchcore"))       # doctest: +SKIP
        >>> runner.predict(cv2.imread(path), color_order="bgr").anomaly_score
        0.83
    """

    model_name = "onnx"

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        model_name: str | None = None,
        config: ModelConfig | None = None,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(config if config is not None else get_model_config())

        self._path = Path(onnx_path).expanduser()
        if not self._path.is_file():
            msg = (
                f"No ONNX model at {self._path}. Export one first with "
                f"`python scripts/export_onnx.py` (writes to {DEFAULT_EXPORTED_DIR})."
            )
            raise FileNotFoundError(msg)

        self._providers = providers or ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        if num_threads is not None:
            # Sequential graph, so inter-op parallelism buys nothing; one intra-op
            # pool sized to the physical cores is the fast, contention-free setup.
            session_options.intra_op_num_threads = num_threads
            session_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=session_options,
            providers=self._providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._verify_outputs()

        self.model_name = model_name or self._path.stem
        self._reconcile_input_size()

        log.info(
            "model_checkpoint_loaded",
            backend="onnx",
            model_name=self.model_name,
            checkpoint=str(self._path),
            input_size="x".join(str(d) for d in self.config.image_hw),
            providers=self._session.get_providers(),
        )

    # -- introspection ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Always ``True``: an :class:`ONNXRunner` only exists once loaded."""
        return True

    @property
    def is_calibrated(self) -> bool:
        """Whether scores are ``[0, 1]``.

        anomalib bakes the post-processor's min-max normalization into the
        exported graph, so a model exported from a calibrated checkpoint (which
        is the only kind :mod:`scripts.export_onnx` produces) emits calibrated
        scores. There is no way to read the fitted statistics back out of the
        ``.onnx`` file, so this reflects that export invariant rather than an
        introspection of the graph.
        """
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_name={self.model_name!r}, path={self._path.name!r})"

    # -- AnomalyModel ----------------------------------------------------------

    def predict(self, image: np.ndarray, *, color_order: str = "rgb") -> ModelOutput:
        """Score one raw image and return a heatmap at its original resolution.

        Identical contract to :meth:`PatchCoreModel.predict`: the frame is
        converted to 3-channel RGB, quality-guarded, resized to the graph's
        input resolution as a plain ``[0, 1]`` batch (the graph normalizes and
        calibrates internally), run through onnxruntime, and the anomaly map is
        bilinearly upsampled back to the input's ``H x W``.

        Args:
            image: ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``
                array; ``uint8`` in ``[0, 255]`` or float.
            color_order: ``"rgb"`` or ``"bgr"``. Use ``"bgr"`` for OpenCV frames.

        Returns:
            A :class:`ModelOutput`, schema-identical to the PyTorch wrappers'.

        Raises:
            GuardError: If the frame fails the input-quality guard — see
                :meth:`~app.models.base.AnomalyModel._guard_frame`.
        """
        feed, height, width = self._to_input_batch(image, color_order=color_order)
        outputs = self._session.run(self._output_names, feed)
        named = dict(zip(self._output_names, outputs, strict=True))

        score = float(np.asarray(named[_SCORE_OUTPUT]).reshape(-1)[0])
        raw_map = torch.from_numpy(np.asarray(named[_MAP_OUTPUT], dtype=np.float32))
        anomaly_map = self._to_input_resolution(raw_map, height, width)

        return ModelOutput(
            anomaly_score=score,
            anomaly_map=anomaly_map,
            is_defective=score >= self.config.anomaly_threshold,
            model_name=self.model_name,
        )

    def train(self, datamodule: object) -> None:  # noqa: ARG002 - interface method
        """Unsupported: an ONNX graph is a frozen export, not a trainable model."""
        msg = (
            "ONNXRunner is inference-only. Train a checkpoint with the PyTorch wrapper, "
            "then export it with `python scripts/export_onnx.py`."
        )
        raise NotImplementedError(msg)

    def save(self, path: str | Path) -> None:  # noqa: ARG002 - interface method
        """Unsupported: the ``.onnx`` file *is* the saved artifact."""
        msg = "ONNXRunner has nothing to save; the .onnx file passed to the constructor is the artifact."
        raise NotImplementedError(msg)

    def load(self, path: str | Path) -> None:  # noqa: ARG002 - interface method
        """Unsupported: pass the ``.onnx`` path to the constructor instead."""
        msg = "ONNXRunner loads in its constructor; build a new ONNXRunner(path) rather than calling load()."
        raise NotImplementedError(msg)

    # -- internals -------------------------------------------------------------

    def _scale_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Identity: the exported graph normalizes internally.

        The PyTorch PatchCore wrapper overrides this to apply ImageNet statistics
        before its bare backbone. Here the anomalib pre-processor is *inside* the
        graph, so the network is fed a plain ``[0, 1]`` batch and normalizing
        here would do it twice — silently, with no shape change, just a worse
        score. See the module docstring.
        """
        return tensor

    def _verify_outputs(self) -> None:
        """Fail loudly at construction if the graph is not an anomalib export."""
        missing = [name for name in (_SCORE_OUTPUT, _MAP_OUTPUT) if name not in self._output_names]
        if missing:
            msg = (
                f"{self._path.name} is missing expected output(s) {missing}; got {self._output_names}. "
                f"This does not look like an anomalib-exported anomaly model."
            )
            raise ValueError(msg)

    def _reconcile_input_size(self) -> None:
        """Point :attr:`config`'s ``image_size`` at the graph's real input resolution.

        The ``.onnx`` file is the source of truth for the resolution its
        pre-processor was traced at; the running config may have been built for a
        different one. When the graph declares a static square input we adopt it
        (warning if it differs from the config), so :meth:`_to_model_input`
        resizes to exactly what the fixed input tensor expects rather than
        failing the ``session.run`` with a shape mismatch. A dynamic input shape
        leaves the config untouched.
        """
        height, width = self._session.get_inputs()[0].shape[-2:]
        if not (isinstance(height, int) and isinstance(width, int) and height == width > 0):
            return
        if height != self.config.image_size:
            log.warning(
                "ONNX graph %s expects %dx%d input; overriding config image_size=%d to match.",
                self._path.name,
                height,
                width,
                self.config.image_size,
            )
            self.config = self.config.with_overrides(image_size=height)

    def _to_input_batch(self, image: np.ndarray, *, color_order: str = "rgb") -> tuple[dict[str, Any], int, int]:
        """Preprocess a raw frame into the session feed dict, plus its ``(H, W)``.

        Factored out of :meth:`predict` so the latency benchmark can time
        ``session.run`` in isolation from the NumPy/resize preprocessing, without
        reaching into private preprocessing internals.
        """
        array = self._to_rgb_array(image, color_order=color_order)
        self._guard_frame(array)  # reject camera glitches / dirty lens / dead lighting before scoring
        height, width = array.shape[:2]
        tensor = self._to_model_input(array)  # (1, 3, S, S) in [0, 1]; graph owns normalization
        return {self._input_name: tensor.numpy()}, height, width
