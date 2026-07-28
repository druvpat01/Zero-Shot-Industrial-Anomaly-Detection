"""The process's model cache: one loaded model per ``(backend, category)``, built on demand.

Why loading is lazy
===================
This process can serve five backends across any number of categories. Loading
them all at startup would mean a WideResNet-50 memory bank, an EfficientAD
student-teacher pair, ~830 MB of CLIP weights and two onnxruntime sessions *per
category* — tens of seconds and gigabytes before the first request, most of it
for backends nobody is going to ask for.

Worse, it breaks the thing a container orchestrator actually needs: a health
check that answers immediately. A pod that takes 40 s to report ready is a pod
that gets killed and restarted by a liveness probe, forever. So ``GET /health``
touches nothing here, and a model is built the first time somebody asks for it.

The cost is that the first request for a backend pays its load. That is real and
it is why :meth:`ModelRegistry.get_model` is worth reading twice — everything in
it exists to make that cost happen *once*, and to make it visible when it does.

Cache keys, concurrency, and warm-up
====================================
* **Key.** ``(backend, category)``. Category matters even where it looks like it
  should not: PatchCore's memory bank is per-category by construction, and
  WinCLIP's prompt ensemble is a function of the category noun. The ONNX
  artifacts are the one place the key is finer than the artifact (see
  :meth:`ModelRegistry._load_onnx`).
* **Concurrency.** FastAPI runs synchronous handlers in a thread pool, so two
  requests for the same cold backend genuinely race. A single global lock would
  serialise them correctly but would also make a request for PatchCore wait out
  somebody else's 30-second CLIP load. So there is a short-lived global lock over
  the *lock table*, and a per-key lock over the load itself: concurrent loads of
  different backends proceed in parallel, concurrent loads of the same one
  collapse into one.
* **Warm-up.** After loading, each model scores one synthetic frame before it is
  cached. This is deliberate and it buys two things. First, honest latency: the
  lazily-built pieces (WinCLIP's CLIP backbone, cuDNN/MKL kernel selection,
  PatchCore's first nearest-neighbour allocation) are paid at load time instead
  of landing inside the first real request's ``latency_ms``. Second, honest
  errors: a checkpoint that exists but is corrupt fails here, as a 503
  ``model_not_ready``, rather than as a 500 on a request that looked fine.

Falling back to ONNX
====================
A category can have an exported graph but no PyTorch checkpoint — that is the
normal state of a deployment image, which ships the ``.onnx`` files and leaves
the multi-hundred-MB checkpoints on the training box. Rather than answer 503 for
``patchcore`` when ``patchcore.onnx`` is sitting right there, the registry serves
the export and says so, both in the log and in the returned model's
``model_name`` (``onnx_patchcore``, never ``patchcore``). The response therefore
tells the caller what actually scored their frame; a silent substitution would
make a latency or accuracy regression impossible to explain.

Per-model state that is not the model
=====================================
Two things besides the weights are keyed here, because this is the only object in
the process that knows which models exist and outlives any one request:

* **A drift monitor per** ``(model_name, category)``
  (:meth:`ModelRegistry.monitor_for`). Note the key: the *resolved* model name,
  not the requested backend. It matches how ``images_processed_total`` attributes
  a score, so an ONNX fallback accumulates its own distribution rather than
  silently mixing into the PyTorch backend's — which matters precisely because
  the two produce slightly different scores from the same weights, and a monitor
  that pooled them would report that difference as drift.
* **Threshold overrides per** ``(backend, category)``
  (:meth:`ModelRegistry.set_threshold`). ``POST /calibrate`` fits a new operating
  point and applies it to the loaded model by swapping its (frozen)
  :class:`~app.models.config.ModelConfig` for a revalidated copy. The override is
  recorded here as well as applied, so a model evicted and rebuilt comes back
  with the calibrated threshold rather than reverting to ``ANOMALY_THRESHOLD``.
  It lives in memory only: a restart returns to the configured default, which is
  a real limitation and is why ``docs/evaluation.md`` says to write a fitted
  threshold into the environment once it is trusted.

Surviving a restart
===================
Everything above is per-process, so a restart starts cold and the first request
after a deploy pays a load it did not ask for. :mod:`app.serving.model_cache`
takes the edge off: each load writes ``{backend, category, checkpoint_path,
loaded_at}`` to Redis with an hour's TTL, and :meth:`ModelRegistry.warm_from_cache`
reads that list back and rebuilds the same working set. The API calls it from a
background thread at startup, so ``/health`` still answers immediately and the
laziness contract above is intact — the difference is only that the warm-up is
now speculative and free rather than charged to whoever asked first.

The cache holds *metadata*, never weights, and is optional in the strong sense:
if Redis is unreachable the registry logs a warning and behaves exactly as it did
before it existed. That module's docstring has the reasoning.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from app.evaluation.drift import DEFAULT_WINDOW_SIZE, ScoreDistributionMonitor
from app.models.base import AnomalyModel
from app.models.config import ModelConfig, get_model_config
from app.models.efficientad import EfficientADModel
from app.models.onnx_runner import ONNXRunner, onnx_artifact_path
from app.models.patchcore import PatchCoreModel
from app.models.winclip import WinCLIPModel
from app.observability.logging_config import get_logger
from app.observability.metrics import set_models_loaded
from app.serving.model_cache import CheckpointCache, LoadedModel, get_checkpoint_cache

__all__ = ["ModelNotReadyError", "ModelRegistry", "ThresholdOutOfRangeError", "get_registry"]

log = get_logger(__name__)

#: Backends backed by a PyTorch wrapper -> the class that implements them.
_TORCH_BACKENDS: dict[str, Callable[..., AnomalyModel]] = {
    "patchcore": PatchCoreModel,
    "efficientad": EfficientADModel,
    "winclip": WinCLIPModel,
}

#: ONNX backend name -> the model name its exported graph was written under.
_ONNX_BACKENDS: dict[str, str] = {
    "onnx_patchcore": "patchcore",
    "onnx_efficientad": "efficientad",
}

#: Backends that need a fitted checkpoint before they can score anything.
#: WinCLIP is pointedly absent: it derives its text embeddings from the category
#: name, so a bare instance is already able to predict. That is the zero-shot
#: claim, and here is where it shows up as a difference in code.
_NEEDS_CHECKPOINT = frozenset({"patchcore", "efficientad"})

#: Backends that can serve with nothing on disk. The complement of "needs an
#: artifact", and not the same set as the complement of :data:`_NEEDS_CHECKPOINT`
#: — the ONNX backends need no *checkpoint* but very much need their exported
#: graph, and conflating the two would have ``/models`` advertise a backend that
#: 503s the moment anybody uses it.
_ZERO_SHOT_BACKENDS = frozenset({"winclip"})

#: Edge of the synthetic frame used to warm a freshly loaded model. Large enough
#: to clear the guard's ``min_resolution`` (64) with room to spare, small enough
#: that the warm-up is one cheap forward pass.
_WARMUP_EDGE = 256


class ModelNotReadyError(RuntimeError):
    """Raised when a backend cannot be served for a category.

    Handled in :mod:`app.serving.main` as a 503 carrying ``{"detail":
    "model_not_ready", "backend": ...}``. 503 rather than 404 or 500 on purpose:
    the request was well-formed and the endpoint exists, the *server* is simply
    not in a state to answer it yet, and that changes the moment somebody trains
    or copies in an artifact. It is the status a caller should retry on.

    Attributes:
        backend: The requested backend name.
        category: The requested category.
        detail: Human-readable explanation, logged and (unlike a traceback) safe
            to surface — it names the file that was missing, which is the one
            thing an operator needs.
    """

    def __init__(self, backend: str, category: str, detail: str) -> None:
        self.backend = backend
        self.category = category
        self.detail = detail
        super().__init__(f"{backend!r} is not ready for category {category!r}: {detail}")


class ThresholdOutOfRangeError(ValueError):
    """Raised when a calibrated threshold cannot be represented in the config.

    :attr:`~app.models.config.ModelConfig.anomaly_threshold` is bounded to
    ``[0, 1]`` because it is documented as a *normalized* score, and calibrated
    backends emit exactly that. An **uncalibrated** model emits raw distances
    instead — PatchCore without a validation pass scores in the single digits —
    so a threshold fitted to its scores lands outside the range and cannot be
    installed.

    Reported as a 422 rather than a 500: the calibration set was fine and the
    arithmetic was fine, but the model this was aimed at does not have an
    operating point on the ``[0, 1]`` scale. The fix is to calibrate the model
    itself (train it with a validation split so ``is_calibrated`` is true), not
    to retry the request.

    Attributes:
        backend: The backend the threshold was aimed at.
        category: The category it was aimed at.
        threshold: The value that could not be installed.
        detail: Human-readable explanation, safe to return to the caller.
    """

    def __init__(self, backend: str, category: str, threshold: float, detail: str) -> None:
        self.backend = backend
        self.category = category
        self.threshold = float(threshold)
        self.detail = detail
        super().__init__(detail)


class ModelRegistry:
    """Lazily builds and caches one :class:`AnomalyModel` per ``(backend, category)``.

    Args:
        config: Base config every model is built from, with ``category``
            overridden per request. Defaults to the process-wide
            :func:`~app.models.config.get_model_config`, resolved lazily so an
            environment change before the first request is still picked up.
        exported_dir: Root the ONNX artifacts are looked up under. Defaults to
            :data:`~app.models.onnx_runner.DEFAULT_EXPORTED_DIR`. Injectable
            mainly so tests can point at a directory they control.
        warmup: Whether to score one synthetic frame at load time. On by default
            — see the module docstring. Tests that only care about *which* model
            comes back turn it off to keep the suite quick.
        drift_window_size: Rolling window each
            :class:`~app.evaluation.drift.ScoreDistributionMonitor` keeps. Passed
            through rather than read from the environment here so a test can make
            a monitor fill in a handful of scores.
        cache: Where the "these models were loaded" records go. Defaults to the
            process-wide :func:`~app.serving.model_cache.get_checkpoint_cache`.
            Inject ``CheckpointCache(url="")`` to keep a test off any real Redis.

    Example:
        >>> registry = ModelRegistry()                              # doctest: +SKIP
        >>> model = registry.get_model("patchcore", "bottle")       # doctest: +SKIP
        >>> registry.loaded_keys()                                  # doctest: +SKIP
        ['patchcore:bottle']
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        exported_dir: Path | str | None = None,
        warmup: bool = True,
        drift_window_size: int = DEFAULT_WINDOW_SIZE,
        cache: CheckpointCache | None = None,
    ) -> None:
        self._config = config
        self._exported_dir = Path(exported_dir) if exported_dir is not None else None
        self._warmup = warmup
        self._drift_window_size = drift_window_size
        self._cache = cache if cache is not None else get_checkpoint_cache()

        self._models: dict[tuple[str, str], AnomalyModel] = {}
        # Guards every dict below it, and nothing else. Held for microseconds.
        self._table_lock = threading.Lock()
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        #: (model_name, category) -> monitor. Keyed by the *resolved* model name;
        #: see the module docstring for why that differs from the model cache's key.
        self._monitors: dict[tuple[str, str], ScoreDistributionMonitor] = {}
        #: (backend, category) -> calibrated anomaly_threshold, survives a reload.
        self._thresholds: dict[tuple[str, str], float] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(loaded={self.loaded_keys()})"

    # -- configuration ---------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """The base config, resolved from the environment on first use."""
        if self._config is None:
            self._config = get_model_config()
        return self._config

    def _config_for(self, category: str) -> ModelConfig:
        """The base config retargeted at ``category``."""
        return self.config.with_overrides(category=category)

    # -- introspection ---------------------------------------------------------

    def loaded_keys(self) -> list[str]:
        """Resident models as sorted ``"<backend>:<category>"`` strings."""
        with self._table_lock:
            keys = list(self._models)
        return sorted(f"{backend}:{category}" for backend, category in keys)

    def is_loaded(self, backend: str, category: str) -> bool:
        """Whether this backend/category pair is already in memory."""
        with self._table_lock:
            return (backend.strip().lower(), category) in self._models

    def clear(self) -> None:
        """Drop every cached model, drift monitor and threshold override.

        Frees the weights; used by tests. The monitors and overrides go with them
        rather than surviving, because all three are per-model state and a
        registry that reported a drift summary for a model it no longer holds
        would be lying about where the numbers came from.
        """
        with self._table_lock:
            self._models.clear()
            self._monitors.clear()
            self._thresholds.clear()
            remaining = len(self._models)
        # The unload half of ``models_loaded_count``. Published from inside the
        # only two methods that change the dict's size, so the gauge cannot drift
        # away from the truth the way an inc/dec pair around a load that raised
        # would.
        set_models_loaded(remaining)

    def describe(self, category: str) -> list[dict[str, object]]:
        """Report every backend's availability for ``category``, loading nothing.

        Answered entirely from the filesystem, which is what makes ``GET
        /models`` cheap enough to poll: it reports whether the artifact a backend
        *would* load exists, not whether it loads.

        Returns:
            One dict per backend, in :data:`~app.serving.schemas.MODEL_BACKENDS`
            order, shaped for :class:`~app.serving.schemas.ModelInfo`.
        """
        rows: list[dict[str, object]] = []
        for backend in (*_TORCH_BACKENDS, *_ONNX_BACKENDS):
            artifact, detail = self._artifact_for(backend, category)
            rows.append(
                {
                    "backend": backend,
                    "available": artifact is not None or backend in _ZERO_SHOT_BACKENDS,
                    "loaded": self.is_loaded(backend, category),
                    "artifact": None if artifact is None else str(artifact),
                    "detail": detail,
                },
            )
        return rows

    def _artifact_for(self, backend: str, category: str) -> tuple[Path | None, str | None]:
        """The file a backend would serve from, plus a note about how it got there.

        Returns ``(None, reason)`` when nothing on disk can serve it. WinCLIP
        returns ``(None, "zero-shot ...")`` — no artifact, but available anyway,
        which is exactly the distinction the two return values exist to carry.
        """
        config = self._config_for(category)

        if backend in _ONNX_BACKENDS:
            path = self._onnx_path(_ONNX_BACKENDS[backend])
            if path.is_file():
                return path, None
            return None, f"no exported graph at {path}; run `python scripts/export_onnx.py`"

        if backend in _ZERO_SHOT_BACKENDS:
            calibration = config.checkpoint_path(backend, category)
            if calibration.is_file():
                return calibration, "calibrated"
            return None, "zero-shot; no artifact needed"

        checkpoint = config.checkpoint_path(backend, category)
        if checkpoint.is_file():
            return checkpoint, None

        fallback = self._onnx_path(backend)
        if fallback.is_file():
            return fallback, f"no checkpoint at {checkpoint}; would fall back to the ONNX export"
        return None, (
            f"no checkpoint at {checkpoint} and no ONNX export at {fallback}; "
            f"run `python scripts/train_{backend}.py --category {category}`"
        )

    def _onnx_path(self, model_name: str) -> Path:
        """Where the FP32 export for ``model_name`` lives."""
        return onnx_artifact_path(model_name, "fp32", self._exported_dir)

    # -- loading ---------------------------------------------------------------

    def get_model(self, backend: str, category: str) -> AnomalyModel:
        """Return a ready-to-score model, loading and caching it on first request.

        Args:
            backend: One of :data:`~app.serving.schemas.MODEL_BACKENDS`.
            category: MVTec-style category, e.g. ``"bottle"``.

        Returns:
            A loaded, warmed :class:`~app.models.base.AnomalyModel`. Note that
            its ``model_name`` may not equal ``backend``: a PyTorch backend with
            no checkpoint is served by its ONNX export, which names itself
            accordingly.

        Raises:
            ValueError: If ``backend`` is not a known name. Callers coming
                through the API cannot reach this — the schema rejects unknown
                backends first — so it means a programming error.
            ModelNotReadyError: If no artifact can serve this pair.
        """
        name = backend.strip().lower()
        if name not in _TORCH_BACKENDS and name not in _ONNX_BACKENDS:
            known = sorted((*_TORCH_BACKENDS, *_ONNX_BACKENDS))
            msg = f"Unknown backend {backend!r}; expected one of {known}."
            raise ValueError(msg)

        key = (name, category)
        cached = self._models.get(key)
        if cached is not None:
            return cached

        # Per-key lock, taken outside the table lock: the load below can run for
        # tens of seconds, and holding the table lock for it would block every
        # other backend's cache *reads*.
        with self._table_lock:
            lock = self._key_locks.setdefault(key, threading.Lock())

        with lock:
            # Re-check inside the lock: whoever we queued behind was very likely
            # loading exactly this model, and their result is as good as ours.
            cached = self._models.get(key)
            if cached is not None:
                return cached

            # Resolved before the load so the log line can name the file even
            # when the load fails: "which artifact was it reaching for" is the
            # first question a ModelNotReadyError raises.
            artifact, artifact_detail = self._artifact_for(name, category)

            started = time.perf_counter()
            model = self._load(name, category)
            # Before the warm-up and before caching: a threshold fitted by an
            # earlier POST /calibrate has to be back in place the moment this
            # model is reachable, or the first request after an eviction is
            # judged against ANOMALY_THRESHOLD instead of the calibrated point.
            self._apply_threshold_override(model, key)
            if self._warmup:
                self._warm(model, name, category)
            elapsed = time.perf_counter() - started

            with self._table_lock:
                self._models[key] = model
                loaded_count = len(self._models)
            set_models_loaded(loaded_count)

            # The model-load event. `model_name` is reported separately from
            # `backend` on purpose — they differ exactly when a PyTorch backend
            # fell back to its ONNX export, and that substitution is the single
            # most common explanation for "the numbers changed and nothing was
            # deployed". A field that is usually redundant earns its place by
            # being unmissable on the day it is not.
            log.info(
                "model_loaded",
                backend=name,
                category=category,
                model_name=model.model_name,
                checkpoint=None if artifact is None else str(artifact),
                checkpoint_detail=artifact_detail,
                duration_seconds=round(elapsed, 3),
                calibrated=model.is_calibrated,
                models_loaded=loaded_count,
            )

            # After the model is cached and the event is logged, because it is
            # the least important thing here: a cache write that fails must not
            # turn a load that succeeded into a request that failed. The method
            # is total (see app/serving/model_cache.py) so there is nothing to
            # catch, but the ordering is the part that would still be right if
            # that ever stopped being true.
            self._cache.record(name, category, None if artifact is None else str(artifact))
            return model

    def _load(self, backend: str, category: str) -> AnomalyModel:
        """Build one model. Assumes the per-key lock is held."""
        if backend in _ONNX_BACKENDS:
            return self._load_onnx(backend, category)
        return self._load_torch(backend, category)

    def _load_onnx(self, backend: str, category: str) -> AnomalyModel:
        """Serve an explicitly requested ONNX export.

        One wrinkle worth naming rather than hiding: the export path is
        ``<exported>/weights/onnx/<model>.onnx``, with no category in it —
        :mod:`scripts.export_onnx` writes one graph per model, from whichever
        category's checkpoint it was pointed at. So this registry's per-category
        cache key is finer than the artifact it loads, and asking for
        ``onnx_patchcore`` on a category the graph was not exported from will
        quietly score against the wrong memory bank. The cache key stays
        per-category anyway (it costs one extra session and keeps the key
        uniform), but a multi-category ONNX deployment needs the category in the
        export filename first.
        """
        path = self._onnx_path(_ONNX_BACKENDS[backend])
        if not path.is_file():
            raise ModelNotReadyError(
                backend,
                category,
                f"no exported graph at {path}; run `python scripts/export_onnx.py`",
            )
        return self._build_onnx(path, backend, category)

    def _load_torch(self, backend: str, category: str) -> AnomalyModel:
        """Build a PyTorch-backed model, falling back to its ONNX export if needed."""
        config = self._config_for(category)
        model = _TORCH_BACKENDS[backend](config=config)

        if backend in _ZERO_SHOT_BACKENDS:
            # WinCLIP. It loads its own calibration artifact on first use if one
            # exists and runs zero-shot if it does not, so there is nothing to
            # check here — which is the whole point of the backend.
            return model

        checkpoint = config.checkpoint_path(backend, category)
        if checkpoint.is_file():
            try:
                model.load(checkpoint)
            except Exception as exc:
                # A checkpoint that exists but will not load is a not-ready
                # server, not a failed request: same 503, same retry advice.
                log.error(
                    "model_load_failed",
                    backend=backend,
                    category=category,
                    checkpoint=str(checkpoint),
                    error=type(exc).__name__,
                    exc_info=True,
                )
                raise ModelNotReadyError(
                    backend,
                    category,
                    f"checkpoint {checkpoint} exists but failed to load: {type(exc).__name__}",
                ) from exc
            return model

        fallback = self._onnx_path(backend)
        if fallback.is_file():
            log.warning(
                "model_backend_substituted",
                backend=backend,
                category=category,
                checkpoint=str(checkpoint),
                fallback=str(fallback),
                served_as=f"onnx_{backend}",
                reason="no checkpoint on disk; serving the ONNX export instead",
            )
            return self._build_onnx(fallback, f"onnx_{backend}", category)

        raise ModelNotReadyError(
            backend,
            category,
            f"no checkpoint at {checkpoint} and no ONNX export at {fallback}; "
            f"run `python scripts/train_{backend}.py --category {category}`",
        )

    def _build_onnx(self, path: Path, model_name: str, category: str) -> AnomalyModel:
        """Open an onnxruntime session, reporting a failure as not-ready.

        ``model_name`` is set explicitly rather than left to default to the file
        stem. Two reasons: the response and the benchmark table then say
        ``onnx_patchcore``, which is what actually ran; and
        :class:`~app.evaluation.benchmark.BenchmarkRunner` keys its results by
        ``model_name`` and rejects duplicates, so a run comparing ``patchcore``
        against ``onnx_patchcore`` would otherwise collide on the stem.
        """
        try:
            return ONNXRunner(path, model_name=model_name, config=self._config_for(category))
        except Exception as exc:
            log.error(
                "onnx_session_failed",
                model_name=model_name,
                category=category,
                artifact=str(path),
                error=type(exc).__name__,
                exc_info=True,
            )
            raise ModelNotReadyError(
                model_name,
                category,
                f"{path} exists but could not be loaded: {type(exc).__name__}",
            ) from exc

    def _warm(self, model: AnomalyModel, backend: str, category: str) -> None:
        """Score one synthetic frame so the first real request does not pay for it.

        The frame is seeded uniform noise: sharp (high Laplacian variance), mid-
        exposed and square, so it clears every input-quality check without
        needing a file on disk. Its *score* is meaningless and discarded — what
        matters is that every lazily-initialised code path downstream has run
        once.
        """
        frame = np.random.default_rng(0).integers(0, 256, size=(_WARMUP_EDGE, _WARMUP_EDGE, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            model.predict(frame, color_order="rgb")
        except Exception as exc:
            log.error(
                "model_warmup_failed",
                backend=backend,
                category=category,
                error=type(exc).__name__,
                exc_info=True,
            )
            raise ModelNotReadyError(
                backend,
                category,
                f"the model loaded but failed its warm-up inference: {type(exc).__name__}",
            ) from exc
        log.debug(
            "model_warmed",
            backend=backend,
            category=category,
            duration_seconds=round(time.perf_counter() - started, 3),
        )

    # -- restoring the working set across a restart -----------------------------

    @property
    def cache(self) -> CheckpointCache:
        """The checkpoint-metadata cache backing :meth:`warm_from_cache`."""
        return self._cache

    def previously_loaded(self) -> list[LoadedModel]:
        """What the cache says was resident before this process started.

        Reads only — it loads nothing and is safe to call from a health or
        diagnostic path. Entries already resident in *this* process are filtered
        out, so on a warm process the list is what is still missing rather than
        what exists.

        Returns:
            Records oldest-load-first, possibly empty. Empty is the normal answer
            for a first-ever start, for a Redis that is down, and for a service
            that has been idle longer than the cache's TTL — all three are fine,
            and none of them is distinguishable here on purpose: the caller's
            behaviour is the same.
        """
        return [entry for entry in self._cache.entries() if not self.is_loaded(entry.backend, entry.category)]

    def warm_from_cache(self, stop: threading.Event | None = None) -> list[str]:
        """Rebuild the working set the cache remembers. Returns the keys it loaded.

        Called from a background thread at startup (see
        :func:`app.serving.main.lifespan`), which is the only way this can exist
        without breaking the promise ``/health`` makes: the loads happen while
        the service is already answering, so a slow warm-up delays nothing and a
        failed one degrades nothing.

        Every failure mode is swallowed for the same reason. A record whose
        artifact has been deleted since — the normal consequence of a deploy that
        replaced a checkpoint — is dropped from the cache rather than retried on
        every future restart, and anything else is logged and stepped over so one
        bad entry cannot cost the rest of the set.

        Args:
            stop: Checked between models, so a shutdown that arrives partway
                through stops after the current load rather than being ignored
                for the remaining ones. A single load cannot be interrupted —
                there is no safe point inside a torch deserialisation — which is
                exactly why the granularity is per model and why the caller
                joins with a timeout rather than unconditionally.

        Returns:
            The keys actually loaded, which may be shorter than the cache's list
            if ``stop`` was set or an artifact had gone missing.
        """
        pending = self.previously_loaded()
        if not pending:
            return []

        log.info(
            "model_warm_start",
            source="checkpoint_cache",
            mode=self._cache.mode,
            models=[entry.key for entry in pending],
        )
        started = time.perf_counter()
        warmed: list[str] = []
        for index, entry in enumerate(pending):
            if stop is not None and stop.is_set():
                # `remaining` is sliced by position rather than by len(warmed):
                # a record that was skipped for a missing artifact advances the
                # loop without adding to `warmed`, and reporting it as still
                # pending would be a lie about what shutdown interrupted.
                log.info(
                    "model_warm_cancelled",
                    warmed=warmed,
                    remaining=[item.key for item in pending[index:]],
                    reason="shutdown requested",
                )
                break
            try:
                self.get_model(entry.backend, entry.category)
            except ModelNotReadyError as exc:
                # The artifact went away between the two runs. Forget it: a
                # record that cannot be reloaded is a failure paid at every
                # restart from here to the end of time.
                log.warning(
                    "model_warm_skipped",
                    backend=entry.backend,
                    category=entry.category,
                    checkpoint=entry.checkpoint_path,
                    reason=exc.detail,
                    action="dropped from the checkpoint cache",
                )
                self._cache.forget(entry.backend, entry.category)
            except Exception as exc:  # noqa: BLE001 - a speculative warm-up may not take the process down
                log.error(
                    "model_warm_failed",
                    backend=entry.backend,
                    category=entry.category,
                    error=type(exc).__name__,
                    exc_info=True,
                )
            else:
                warmed.append(entry.key)

        log.info(
            "model_warm_complete",
            warmed=warmed,
            requested=len(pending),
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        return warmed

    # -- drift monitoring ------------------------------------------------------

    def monitor_for(self, model_name: str, category: str) -> ScoreDistributionMonitor:
        """The drift monitor for one ``(model_name, category)``, created on first use.

        Args:
            model_name: The *resolved* name from
                :attr:`~app.models.base.ModelOutput.model_name` — what actually
                scored the frame, so an ONNX fallback gets its own monitor. See
                the module docstring.
            category: The category the frame belongs to.

        Returns:
            The monitor, which is shared: two concurrent ``/predict`` calls for
            the same pair record into the same window, which is the point.
        """
        key = (model_name, category)
        with self._table_lock:
            monitor = self._monitors.get(key)
            if monitor is None:
                monitor = ScoreDistributionMonitor(window_size=self._drift_window_size)
                self._monitors[key] = monitor
        return monitor

    def monitors(self) -> dict[tuple[str, str], ScoreDistributionMonitor]:
        """Every active monitor, keyed by ``(model_name, category)``.

        A snapshot of the dict, not the dict itself, so ``GET /drift`` can
        iterate while the prediction path keeps creating monitors. The monitors
        it hands out are the live ones — each is independently thread-safe.
        """
        with self._table_lock:
            return dict(self._monitors)

    # -- calibration -----------------------------------------------------------

    def set_threshold(self, backend: str, category: str, threshold: float) -> float:
        """Install a calibrated decision threshold, and return the one it replaced.

        Applied by swapping the model's frozen
        :class:`~app.models.config.ModelConfig` for a revalidated copy rather
        than mutating it: the config is frozen precisely so nothing can change
        under a running model, and every backend reads
        ``self.config.anomaly_threshold`` at predict time, so the next request
        sees the new value and no in-flight one sees a half-applied change.

        Loading the model first is deliberate. Recording an override against a
        backend that cannot be served would report success for a change nobody
        will ever see; this way a missing artifact fails as
        :class:`ModelNotReadyError` — the same 503 ``/predict`` gives — before
        anything is recorded.

        Args:
            backend: One of :data:`~app.serving.schemas.MODEL_BACKENDS`.
            category: MVTec-style category.
            threshold: The new operating point. A frame is defective when
                ``anomaly_score >= threshold``.

        Returns:
            The threshold that was in force before this call.

        Raises:
            ModelNotReadyError: If no artifact can serve this pair.
            ThresholdOutOfRangeError: If the value is outside the ``[0, 1]``
                range ``anomaly_threshold`` is bounded to.
        """
        name = backend.strip().lower()
        model = self.get_model(name, category)
        previous = float(model.config.anomaly_threshold)
        value = float(threshold)

        try:
            # pydantic revalidates the whole config here, so the [0, 1] bound on
            # anomaly_threshold is enforced by the same declaration that
            # documents it rather than by a second copy of the rule.
            updated = model.config.with_overrides(anomaly_threshold=value)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            detail = (
                f"threshold {value:.6g} is outside the [0, 1] range anomaly_threshold accepts. "
                f"{model.model_name} reports is_calibrated={model.is_calibrated}; an uncalibrated "
                "backend emits raw distances, so a threshold fitted to its scores has no place on "
                "the normalized scale. Train it with a validation split, or compare raw scores "
                "outside the service."
            )
            raise ThresholdOutOfRangeError(name, category, value, detail) from exc

        model.config = updated
        with self._table_lock:
            self._thresholds[(name, category)] = value

        log.info(
            "threshold_updated",
            backend=name,
            category=category,
            model_name=model.model_name,
            previous_threshold=round(previous, 6),
            threshold=round(value, 6),
            calibrated=model.is_calibrated,
            persistence="in-memory only; lost on restart",
        )
        return previous

    def threshold_overrides(self) -> dict[tuple[str, str], float]:
        """Every calibrated threshold currently in force, keyed by ``(backend, category)``."""
        with self._table_lock:
            return dict(self._thresholds)

    def _apply_threshold_override(self, model: AnomalyModel, key: tuple[str, str]) -> None:
        """Re-apply a recorded threshold to a model that was just built."""
        with self._table_lock:
            override = self._thresholds.get(key)
        if override is None:
            return
        model.config = model.config.with_overrides(anomaly_threshold=override)
        log.info(
            "threshold_override_reapplied",
            backend=key[0],
            category=key[1],
            model_name=model.model_name,
            threshold=round(override, 6),
        )


#: The process-wide registry. Reached through :func:`get_registry` rather than
#: imported directly, so FastAPI's dependency injection can substitute another
#: one in tests without any handler knowing.
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """FastAPI dependency returning the process-wide :class:`ModelRegistry`."""
    return _registry
