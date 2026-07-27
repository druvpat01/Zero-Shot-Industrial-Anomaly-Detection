"""Score-distribution drift detection: catching the failure that does not raise.

The failure this exists to catch
================================
:mod:`app.guardrails` catches a bad *frame* — a defocused lens, a dead light, a
truncated read. It cannot catch a bad *model*, because the model is behaving
exactly as designed: it takes a perfectly good frame and returns a perfectly
well-formed score. The score is simply no longer meaningful.

That happens for reasons nobody deploys against:

* **The product line changes.** A supplier switches to a slightly different
  plastic, the moulding temperature is retuned, a new SKU joins the line. Every
  frame is now marginally out-of-distribution relative to the nominal set
  PatchCore's memory bank was built from, so every frame scores a little higher.
* **The camera ages.** Sensor gain drifts, a lens coating hazes, the LED ring
  loses output over thousands of hours. The images stay sharp enough to clear the
  guard and different enough to move the scores.
* **Somebody changed something.** A new checkpoint, a re-export to ONNX, an
  ``IMAGE_SIZE`` bump. The deployment is fine; the operating point is not.

In all three the model keeps answering, ``/health`` stays green, latency is
unchanged, and the ``images_processed_total{result="defective"}`` rate quietly
moves. By the time anyone notices, a shift's worth of parts has been graded
against a threshold that no longer means what it meant when it was fitted. This
is the silent failure that :class:`ScoreDistributionMonitor` exists to make loud.

Why a Kolmogorov-Smirnov test, and what it actually measures
===========================================================
The obvious monitor is "alert if the mean score moves". That is a bad monitor for
this data, and the reason is worth stating: it assumes the thing that changes is
the *centre* of a distribution whose shape you already know. Anomaly scores
violate both halves. They are not Gaussian — they are bounded below, heavily
right-skewed, and typically bimodal (a dense cluster of nominal parts plus a thin
tail of defects), and the shifts that matter are often changes in *shape* rather
than location: the defect tail thickening while the nominal mode sits exactly
where it was, which moves the mean by almost nothing.

The two-sample **Kolmogorov-Smirnov** test (:func:`scipy.stats.ks_2samp`) asks a
question that needs none of those assumptions. Build the empirical cumulative
distribution function (ECDF) of each sample — for the reference window and for
the current window — and take the **largest vertical gap between the two curves**:

.. code-block:: text

    1.0 |                    ..--=========           <- reference ECDF
        |                 ..-'   |
        |              .-'       | D = the KS statistic:
        |           .-'          |     the largest gap between the curves
        |        .-'    ,--======#=========
        |     .-'    ,-'         |                   <- current ECDF
    0.0 |__.-'____,-'____________|________________
          low score                      high score

That maximum gap is the KS statistic ``D``. It is a *distribution-free* measure:
its sampling distribution under the null hypothesis ("both samples were drawn
from the same underlying distribution") depends only on the two sample sizes, not
on what the underlying distribution looks like. So the test is valid on skewed,
bimodal, bounded anomaly scores without anybody having to assume they are normal
— which is exactly why it is the right test here and a t-test is not.

The p-value is what the test reports, and it is not "the probability the model
drifted"
-------------------------------------------------------------------------------
``p`` answers a precise and narrower question: **if the two windows really had
been drawn from the same distribution, how often would random sampling alone
produce a gap at least as large as the one observed?**

So ``p = 0.30`` means a gap this big happens in 30% of samples from an unchanged
process — utterly unremarkable, and evidence of nothing. ``p = 0.001`` means a
gap this big happens once in a thousand samples of an unchanged process, so
either something rare just happened or the assumption that nothing changed is
wrong. Below :data:`DEFAULT_KS_DRIFT_THRESHOLD` (0.05, the field's conventional
line, overridable with ``KS_DRIFT_THRESHOLD``) this module calls it drift.

Three consequences of that definition, all of which matter operationally:

1. **0.05 buys a 5% false-alarm rate, by construction.** On a genuinely
   unchanged line, one in twenty comparisons crosses the line by chance. That is
   the *definition* of the threshold, not a defect in it. It is also why the
   right response to a single ``ks_drift`` is to look, not to halt a line — see
   below — and why lowering ``KS_DRIFT_THRESHOLD`` to 0.01 is a reasonable
   choice for a deployment that checks often.
2. **A large p-value is not proof of stability.** Failing to detect drift is not
   the same as detecting stability; with a small window there may simply not be
   enough data to see it. That is why :attr:`ScoreDistributionMonitor.min_samples`
   exists and why a short window reports "no verdict" rather than "no drift".
3. **Statistical significance is not operational significance.** With a large
   enough window, KS will eventually flag a shift far too small to change a
   single verdict. ``p`` says *whether* the distribution moved; the percentiles
   in :meth:`ScoreDistributionMonitor.get_summary` say *by how much*, and only
   the second answers "does this matter". Report both, always.

What a production system does when this fires
=============================================
Escalating, and deliberately not starting at "stop the line" — a monitor whose
only action is drastic gets muted within a week:

1. **Alert, with the numbers attached.** Page nobody; raise a warning on the
   ops channel carrying the p-value *and* the summary percentiles, so the person
   reading it can tell a 0.002-point shift in p50 from a 0.3-point one. Cross-
   check ``guard_rejections_total`` first: a drift that arrives together with a
   rising ``blurry`` rate is a fouling lens, not a model problem, and it is fixed
   with a cloth.
2. **Flag the affected window for human review.** The frames scored since the
   drift began were graded against an operating point that may no longer hold.
   They are not necessarily wrong — but they are the batch to re-inspect, and
   knowing *which* batch is most of the value of having timestamped the drift.
3. **Recalibrate the operating point.** Often the model is fine and only the
   threshold is stale — the scores shifted uniformly and the decision boundary
   needs to move with them. That is a labelled calibration set and a call to
   :func:`app.evaluation.calibration.find_optimal_threshold`, minutes of work,
   and it is why ``POST /calibrate`` exists next door.
4. **Retrain or re-fit the reference.** If recalibration cannot recover the F1 —
   the shape changed, not just the location — the nominal set itself is stale.
   PatchCore's memory bank has to be rebuilt from current production frames.
   This is the expensive answer and it should be the last one reached for.
5. **Only then, gate the line** — fall back to 100% human inspection for the
   affected category. Correct when the alternative is shipping scrap, and far too
   expensive to trigger on a 5%-of-the-time false alarm, which is the whole
   reason steps 1-4 come first.

Scope, and what is deliberately not here
========================================
This monitors the *output* distribution, which is the cheap and general signal:
it needs no labels, no ground truth and no second model, and it catches input
drift, model drift and configuration drift alike because all three land in the
same place. It cannot tell you *which* of those it saw — that is what step 1's
cross-check against the guard metrics is for.

There is deliberately **no new Prometheus series** for drift. The label set that
would make one useful is ``(model, category)``, which is exactly the pair
``images_processed_total`` already carries, and the summary this module produces
is five numbers per monitor that change slowly — a scrape-shaped answer to a
question nobody asks at scrape frequency. ``GET /drift`` serves it on demand
instead, and :mod:`app.observability.metrics` keeps its rule that a metric is a
permanent commitment rather than a convenient place to put a number.

Example:
    >>> monitor = ScoreDistributionMonitor(window_size=500)
    >>> monitor.set_reference([0.10, 0.12, 0.09, 0.11] * 25)
    >>> for score in [0.80, 0.82, 0.79, 0.81] * 25:
    ...     monitor.record_score(score)
    >>> monitor.is_drifted()
    (True, 'ks_drift')
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Mapping

import numpy as np
from scipy.stats import ks_2samp

from app.observability.logging_config import get_logger

__all__ = [
    "DEFAULT_KS_DRIFT_THRESHOLD",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_WINDOW_SIZE",
    "KS_DRIFT_THRESHOLD_VAR",
    "REASON_KS_DRIFT",
    "ScoreDistributionMonitor",
    "ks_drift_threshold",
]

log = get_logger(__name__)

#: Scores retained in the rolling window. 500 frames is roughly ten minutes of a
#: line running at one part per second — long enough that the ECDF is stable and
#: the KS test has power, short enough that a shift is visible within a shift.
DEFAULT_WINDOW_SIZE = 500

#: p-value at or below which the two windows are called different. 0.05 is the
#: field's conventional line and it is a *choice*, not a fact: it accepts a 5%
#: false-alarm rate on an unchanged process in exchange for detecting a real
#: change quickly. See the module docstring.
DEFAULT_KS_DRIFT_THRESHOLD = 0.05

#: Environment variable overriding :data:`DEFAULT_KS_DRIFT_THRESHOLD`.
KS_DRIFT_THRESHOLD_VAR = "KS_DRIFT_THRESHOLD"

#: Samples each window needs before a verdict is offered at all. Below this the
#: KS test is not *wrong*, it is merely powerless — with a handful of points the
#: largest ECDF gap is dominated by sampling noise and ``p`` stays comfortably
#: above any threshold no matter how far the distribution has moved. Reporting
#: "no verdict" is honest; reporting "no drift" from six samples is not.
DEFAULT_MIN_SAMPLES = 30

#: The one reason :meth:`ScoreDistributionMonitor.is_drifted` reports. A slug
#: rather than a sentence, matching :class:`~app.guardrails.GuardResult.reason`,
#: so an alert rule can match on it.
REASON_KS_DRIFT = "ks_drift"


def ks_drift_threshold(environ: Mapping[str, str] | None = None) -> float:
    """Resolve the drift p-value threshold from the environment.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``. A blank or missing
            variable falls back to :data:`DEFAULT_KS_DRIFT_THRESHOLD`, matching
            :meth:`app.guardrails.GuardConfig.from_env`.

    Returns:
        The threshold, in ``(0, 1)``.

    Raises:
        ValueError: If the variable is set but is not a float in ``(0, 1)``. A
            p-value threshold of 0 can never fire and one of 1 always fires, so
            both bounds are excluded — a typo that silently disables the monitor
            is the failure mode this check exists to prevent.
    """
    source = os.environ if environ is None else environ
    raw = source.get(KS_DRIFT_THRESHOLD_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_KS_DRIFT_THRESHOLD

    try:
        value = float(raw.strip())
    except ValueError as exc:
        msg = f"{KS_DRIFT_THRESHOLD_VAR}={raw!r} is not a number."
        raise ValueError(msg) from exc

    if not 0.0 < value < 1.0:
        msg = f"{KS_DRIFT_THRESHOLD_VAR} must be in (0, 1), got {value}."
        raise ValueError(msg)
    return value


class ScoreDistributionMonitor:
    """A rolling window of anomaly scores, compared against a reference window.

    One instance watches one score stream — in this service, one
    ``(model, category)`` pair, created and held by
    :class:`~app.serving.model_registry.ModelRegistry`. Scores go in one at a
    time from the prediction path; verdicts and summaries come out on demand
    from ``GET /drift``.

    **Thread-safe.** FastAPI runs this app's synchronous handlers in a thread
    pool, so ``/predict`` calls :meth:`record_score` from several workers at once
    while ``/drift`` reads the window from another. Every method takes a lock
    held for microseconds — long enough to keep a summary from being computed
    over a window that is being appended to, short enough never to be contended
    in a way that matters against a forward pass.

    Args:
        window_size: Scores retained. The window is a ``deque`` with this
            ``maxlen``, so the oldest score is evicted on overflow and the
            monitor's memory is bounded regardless of uptime.
        threshold: p-value below which drift is declared. Defaults to
            :func:`ks_drift_threshold`, resolved at construction so a deployment
            sets it once in the environment.
        min_samples: Scores each window needs before a verdict is offered.

    Raises:
        ValueError: On a non-positive ``window_size`` or ``min_samples``, or a
            ``threshold`` outside ``(0, 1)``.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        *,
        threshold: float | None = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        if window_size < 1:
            msg = f"window_size must be >= 1, got {window_size}."
            raise ValueError(msg)
        if min_samples < 1:
            msg = f"min_samples must be >= 1, got {min_samples}."
            raise ValueError(msg)

        resolved = ks_drift_threshold() if threshold is None else float(threshold)
        if not 0.0 < resolved < 1.0:
            msg = f"threshold must be in (0, 1), got {resolved}."
            raise ValueError(msg)

        self.window_size = int(window_size)
        self.threshold = resolved
        self.min_samples = int(min_samples)

        self._window: deque[float] = deque(maxlen=self.window_size)
        self._reference: np.ndarray = np.empty(0, dtype=float)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(window_size={self.window_size}, "
            f"count={self.count}, reference_size={self.reference_size}, "
            f"threshold={self.threshold})"
        )

    def __len__(self) -> int:
        return self.count

    # -- state -----------------------------------------------------------------

    @property
    def count(self) -> int:
        """Scores currently in the rolling window."""
        with self._lock:
            return len(self._window)

    @property
    def reference_size(self) -> int:
        """Scores in the reference window; ``0`` until :meth:`set_reference`."""
        with self._lock:
            return int(self._reference.size)

    @property
    def has_reference(self) -> bool:
        """Whether a reference distribution has been set."""
        return self.reference_size > 0

    def record_score(self, score: float) -> None:
        """Add one anomaly score to the rolling window.

        Called from ``POST /predict`` after every scored frame. O(1), and the
        eviction of the oldest score is the ``deque``'s, so this stays cheap
        enough to sit on the request path without being conditional.

        Args:
            score: The image-level anomaly score.

        Raises:
            ValueError: If ``score`` is not finite. A NaN entering the window
                poisons every subsequent KS test and percentile silently, which
                is precisely the class of failure this module exists to catch, so
                it is refused at the door. Unreachable from the prediction path:
                :class:`~app.models.base.ModelOutput` already rejects a
                non-finite score at construction.
        """
        value = float(score)
        if not np.isfinite(value):
            msg = f"score must be finite, got {score!r}."
            raise ValueError(msg)
        with self._lock:
            self._window.append(value)

    def set_reference(self, scores: list[float]) -> None:
        """Install the distribution the rolling window is judged against.

        The reference is "what normal looked like when this operating point was
        chosen", so the natural source is the same labelled calibration set that
        fitted the decision threshold — which is why ``POST /calibrate`` sets
        both in one call.

        There is deliberately **no automatic reference**. Seeding it from the
        first N production scores would be convenient and would compare the
        window against itself: a line that was already drifting when the process
        started would establish the drifted state as normal and report health
        forever. A reference is a claim about what good looks like, and a claim
        has to be made by somebody.

        Args:
            scores: The reference sample. Copied, so a later mutation of the
                caller's list cannot move the baseline underneath a live monitor.

        Raises:
            ValueError: If ``scores`` is empty or holds a non-finite value.
        """
        array = np.asarray(list(scores), dtype=float)
        if array.size == 0:
            msg = "set_reference() needs at least one score, got an empty sequence."
            raise ValueError(msg)
        if not np.all(np.isfinite(array)):
            msg = "set_reference() got a non-finite score; the reference must be usable for a KS test."
            raise ValueError(msg)

        if array.size < self.min_samples:
            # Not fatal — a caller may have only a small calibration set — but
            # `is_drifted` will withhold a verdict until it grows, and silently
            # answering "no drift" forever is the worst possible outcome.
            log.warning(
                "drift_reference_undersized",
                reference_size=int(array.size),
                min_samples=self.min_samples,
                impact="is_drifted() returns no verdict until the reference has at least min_samples scores",
            )

        with self._lock:
            self._reference = array

        log.info("drift_reference_set", reference_size=int(array.size), threshold=self.threshold)

    def reset_window(self) -> None:
        """Empty the rolling window, leaving the reference in place.

        What a production system calls after acting on a drift alert: the
        operating point has been re-fitted or the lens has been cleaned, and the
        scores from before the fix should not keep the alarm ringing.
        """
        with self._lock:
            self._window.clear()

    # -- verdict ---------------------------------------------------------------

    def ks_p_value(self) -> float | None:
        """Two-sample KS p-value for the current window against the reference.

        Returns:
            The p-value, or ``None`` when no verdict is available — no reference
            has been set, or either window holds fewer than
            :attr:`min_samples` scores. ``None`` means *not enough evidence to
            say*, which is a third state and not a synonym for "no drift"; see
            the module docstring.
        """
        with self._lock:
            reference = self._reference
            current = np.fromiter(self._window, dtype=float, count=len(self._window))

        if reference.size < self.min_samples or current.size < self.min_samples:
            return None
        return float(ks_2samp(reference, current).pvalue)

    def is_drifted(self) -> tuple[bool, str | None]:
        """Whether the current window's distribution differs from the reference.

        Returns:
            ``(True, "ks_drift")`` when the KS p-value is below
            :attr:`threshold`, and ``(False, None)`` otherwise — including when
            there is not yet enough data for a verdict, because a monitor that
            has just started must not report drift it cannot see. Use
            :meth:`ks_p_value` (``None``) or :attr:`has_reference` to tell the
            two ``(False, None)`` cases apart; ``GET /drift`` reports both.
        """
        p_value = self.ks_p_value()
        if p_value is None:
            return (False, None)
        if p_value < self.threshold:
            return (True, REASON_KS_DRIFT)
        return (False, None)

    def get_summary(self) -> dict[str, float | None]:
        """Descriptive statistics for the current window.

        The companion to :meth:`is_drifted`, and the reason both are reported
        together: the p-value says *whether* the distribution moved, these say
        *by how much*, and only the second answers whether an operator should
        care. The percentiles are the useful part — p50 is where the bulk of
        nominal parts sit, and p90 tracks the defect tail, so a p90 that climbs
        while p50 holds is a line producing more defects rather than a model that
        has come unstuck.

        Returns:
            ``count`` (an ``int``-valued float count of scores in the window),
            plus ``mean``, ``std``, ``p10``, ``p50`` and ``p90``. The five
            statistics are ``None`` on an empty window rather than ``nan``, so
            the JSON ``GET /drift`` returns is valid without a serializer
            special case. ``std`` is the population standard deviation
            (``ddof=0``): this is a description of the window in hand, not an
            estimate of a wider population's spread.
        """
        with self._lock:
            values = np.fromiter(self._window, dtype=float, count=len(self._window))

        if values.size == 0:
            return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}

        p10, p50, p90 = (float(value) for value in np.percentile(values, [10, 50, 90]))
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }
