"""Run every model over a category's test set and score them on the same footing.

:class:`BenchmarkRunner` is the piece that turns "the model runs" into a table of
numbers. It holds a list of :class:`~app.models.base.AnomalyModel` instances and
a :class:`~app.data.DataModule`, pushes every test image through each model's
``predict``, and hands the collected scores and heatmaps to
:mod:`app.evaluation.metrics`.

Two design choices are worth stating, because both are what make the comparison
fair:

* **The runner only knows :class:`AnomalyModel`.** It calls ``predict`` and reads
  :class:`~app.models.base.ModelOutput`; it never asks whether it is holding
  PatchCore, EfficientAD or WinCLIP. That is the payoff of the wrapper layer —
  three algorithms that share nothing internally are scored by identical code.
* **Every model sees byte-identical inputs.** Each image is drawn once from the
  datamodule, at one ``image_size``, and passed to all models. Differences in the
  table are therefore differences between *models*, not between preprocessing
  paths — the ground-truth mask a heatmap is compared against is the very tensor
  the image was scored at, so pixels line up exactly.

Results are written to ``results/benchmark_<category>_<timestamp>.json``. The
timestamp is deliberate: runs accumulate rather than overwrite, so a checkpoint
change or a hyperparameter sweep leaves a dated trail instead of clobbering the
last number.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.data import DataModule
from app.evaluation.metrics import au_pro, f1_at_best_threshold, image_auroc, pixel_auroc
from app.models.base import AnomalyModel

__all__ = ["BenchmarkResult", "BenchmarkRunner"]

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """One model's scores over one test set, plus the run metadata behind them.

    The four headline metrics are the same four the CLI table prints. The counts
    and timing are context: a benchmark with no defective images, or one that
    scored ``nan`` on the pixel metrics because masks were absent, should be
    obvious from the JSON without re-running anything.
    """

    model_name: str
    image_auroc: float
    pixel_auroc: float
    au_pro: float
    best_f1: float
    best_f1_threshold: float
    num_images: int
    num_defective: int
    num_normal: int
    num_masked: int
    elapsed_seconds: float
    seconds_per_image: float

    def as_dict(self) -> dict[str, object]:
        """Plain-``dict`` form for JSON serialization."""
        return {
            "model_name": self.model_name,
            "image_auroc": self.image_auroc,
            "pixel_auroc": self.pixel_auroc,
            "au_pro": self.au_pro,
            "best_f1": self.best_f1,
            "best_f1_threshold": self.best_f1_threshold,
            "num_images": self.num_images,
            "num_defective": self.num_defective,
            "num_normal": self.num_normal,
            "num_masked": self.num_masked,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_image": self.seconds_per_image,
        }


@dataclass
class _Predictions:
    """Everything one model produced over the test set, held until metrics run."""

    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    pred_maps: list[np.ndarray] = field(default_factory=list)
    gt_masks: list[np.ndarray] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def has_masks(self) -> bool:
        """Whether pixel-level ground truth was available for every image."""
        return len(self.gt_masks) == len(self.scores) and len(self.gt_masks) > 0


class BenchmarkRunner:
    """Score a set of models on a datamodule's test split and persist the results.

    Args:
        models: The :class:`~app.models.base.AnomalyModel` instances to compare.
            Each must already be trained or loaded (WinCLIP zero-shot needs
            neither); the runner does not train them — see
            ``scripts/run_benchmark.py`` for the orchestration that does.
        datamodule: The :class:`~app.data.DataModule` whose ``test_dataloader``
            supplies the images and ground truth.
        results_dir: Where the JSON report is written. Defaults to ``results/``.

    Example:
        >>> runner = BenchmarkRunner([patchcore, efficientad, winclip], dm)  # doctest: +SKIP
        >>> results = runner.run()                                           # doctest: +SKIP
        >>> results["patchcore"]["image_auroc"]                              # doctest: +SKIP
        0.994
    """

    def __init__(
        self,
        models: list[AnomalyModel],
        datamodule: DataModule,
        results_dir: Path | str | None = None,
    ) -> None:
        if not models:
            msg = "BenchmarkRunner needs at least one model to benchmark."
            raise ValueError(msg)
        names = [m.model_name for m in models]
        if len(set(names)) != len(names):
            msg = f"model_name must be unique within a run; results are keyed by it, got {names}."
            raise ValueError(msg)

        self.models = models
        self.datamodule = datamodule
        self.results_dir = Path(results_dir) if results_dir is not None else Path("results")

    # -- running --------------------------------------------------------------

    def run(self, *, save: bool = True) -> dict[str, dict[str, object]]:
        """Benchmark every model and return a ``{model_name: metrics}`` dict.

        The test set is materialised once into memory (raw images plus masks) and
        replayed for each model, so every model is scored on the identical
        sequence of frames regardless of dataloader shuffling.

        Args:
            save: Whether to write the JSON report. ``True`` in normal use;
                tests pass ``False``.

        Returns:
            A dict keyed by ``model_name``, each value the metrics dict from
            :meth:`BenchmarkResult.as_dict`.
        """
        images, labels, masks = self._materialise_test_set()
        logger.info(
            "Benchmarking %d model(s) on %r: %d images (%d defective, %d normal), %d with masks.",
            len(self.models),
            self.datamodule.category,
            len(images),
            int(sum(labels)),
            len(labels) - int(sum(labels)),
            sum(m is not None for m in masks),
        )

        results: dict[str, BenchmarkResult] = {}
        for model in self.models:
            logger.info("Scoring %s ...", model.model_name)
            predictions = self._collect_predictions(model, images, labels, masks)
            results[model.model_name] = self._score(model.model_name, predictions)
            r = results[model.model_name]
            logger.info(
                "%s: img-AUROC=%.4f px-AUROC=%.4f AU-PRO=%.4f best-F1=%.4f (%.2fs, %.2fs/img)",
                r.model_name,
                r.image_auroc,
                r.pixel_auroc,
                r.au_pro,
                r.best_f1,
                r.elapsed_seconds,
                r.seconds_per_image,
            )

        payload = self._build_payload(results)
        if save:
            self._save(payload)
        return {name: result.as_dict() for name, result in results.items()}

    def _materialise_test_set(
        self,
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray | None]]:
        """Pull the whole test split into memory as raw RGB frames plus masks.

        Each dataloader batch is a :class:`~app.data.batch.DefectBatch` of
        ``[0, 1]`` float tensors at ``image_size``. We convert each image to an
        ``(H, W, 3)`` RGB array — the raw form ``AnomalyModel.predict`` takes —
        and keep the ground-truth mask at the *same* resolution, so a heatmap and
        its mask are directly comparable with no rescaling anywhere.
        """
        images: list[np.ndarray] = []
        labels: list[int] = []
        masks: list[np.ndarray | None] = []

        for batch in self.datamodule.test_dataloader():
            batch_images = batch.image.detach().cpu().numpy()  # (N, C, H, W)
            batch_labels = batch.label.detach().cpu().numpy()  # (N,)
            batch_masks = (
                batch.mask.detach().cpu().numpy() if batch.mask is not None else [None] * len(batch_labels)
            )
            for i in range(len(batch_labels)):
                # (C, H, W) [0, 1] float -> (H, W, C) RGB, exactly what predict() expects.
                images.append(np.ascontiguousarray(batch_images[i].transpose(1, 2, 0)))
                labels.append(int(batch_labels[i]))
                masks.append(None if batch_masks[i] is None else (np.asarray(batch_masks[i]) > 0).astype(np.uint8))

        return images, labels, masks

    def _collect_predictions(
        self,
        model: AnomalyModel,
        images: list[np.ndarray],
        labels: list[int],
        masks: list[np.ndarray | None],
    ) -> _Predictions:
        """Run one model over the materialised test set, timing the inference.

        Timing brackets only ``predict`` — not the one-off dataset load — so
        ``seconds_per_image`` is a fair per-model inference cost, which is the
        first thing Step 7's latency work will want to sanity-check against.
        """
        predictions = _Predictions()
        started = time.perf_counter()
        for image, label, mask in zip(images, labels, masks, strict=True):
            # The frames come from the datamodule already in RGB order.
            output = model.predict(image, color_order="rgb")
            predictions.scores.append(float(output.anomaly_score))
            predictions.labels.append(int(label))
            if mask is not None:
                predictions.pred_maps.append(np.asarray(output.anomaly_map, dtype=np.float32))
                predictions.gt_masks.append(mask)
        predictions.elapsed = time.perf_counter() - started
        return predictions

    def _score(self, model_name: str, predictions: _Predictions) -> BenchmarkResult:
        """Turn one model's raw predictions into the four metrics plus metadata."""
        num_images = len(predictions.scores)
        num_defective = int(sum(predictions.labels))
        best_f1, best_threshold = f1_at_best_threshold(predictions.labels, predictions.scores)

        if predictions.has_masks:
            px_auroc = pixel_auroc(predictions.gt_masks, predictions.pred_maps)
            region_au_pro = au_pro(predictions.gt_masks, predictions.pred_maps)
        else:
            logger.warning("%s: no ground-truth masks available; pixel metrics reported as nan.", model_name)
            px_auroc = float("nan")
            region_au_pro = float("nan")

        return BenchmarkResult(
            model_name=model_name,
            image_auroc=image_auroc(predictions.labels, predictions.scores),
            pixel_auroc=px_auroc,
            au_pro=region_au_pro,
            best_f1=best_f1,
            best_f1_threshold=best_threshold,
            num_images=num_images,
            num_defective=num_defective,
            num_normal=num_images - num_defective,
            num_masked=len(predictions.gt_masks),
            elapsed_seconds=predictions.elapsed,
            seconds_per_image=predictions.elapsed / num_images if num_images else float("nan"),
        )

    # -- persistence ----------------------------------------------------------

    def _build_payload(self, results: dict[str, BenchmarkResult]) -> dict[str, object]:
        """Assemble the JSON document: run context first, then per-model metrics."""
        return {
            "category": self.datamodule.category,
            "image_size": self.datamodule.image_size,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": [m.model_name for m in self.models],
            "results": {name: result.as_dict() for name, result in results.items()},
        }

    def _save(self, payload: dict[str, object]) -> Path:
        """Write the report to ``results/benchmark_<category>_<timestamp>.json``.

        The filename carries a compact UTC timestamp so runs never collide;
        nothing is ever overwritten.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = self.results_dir / f"benchmark_{self.datamodule.category}_{stamp}.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved benchmark report to %s", destination)
        return destination
