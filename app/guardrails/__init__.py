"""Input-quality guardrails: the gate every frame clears before a model scores it.

This package is the seam between "an array arrived" and "a model was asked to
score it". :class:`~app.guardrails.quality.FrameGuard` runs the cheap,
model-independent checks that catch the failures a production inspection line
actually produces — a dropped frame, a fouled lens, a dead light — before they
reach code that would turn them into a plausible-looking anomaly score.

Two things are exported for the prediction path:

* :data:`guard` — a process-wide :class:`FrameGuard`, configured from the
  environment once at import. Every model wrapper's ``predict`` calls it, so the
  gate is applied identically no matter which backend serves the frame. Sharing
  one instance is safe: :class:`FrameGuard` is stateless apart from its frozen
  config.
* :class:`GuardError` — raised when a frame is rejected, carrying the
  :class:`GuardResult` so the caller can log the reason and metrics.

Everything else (:class:`FrameGuard`, :class:`GuardResult`, :class:`GuardConfig`)
is re-exported for callers that want to build a guard with bespoke thresholds
rather than use the shared one.
"""

from app.guardrails.quality import (
    FrameGuard,
    GuardConfig,
    GuardError,
    GuardResult,
)

#: Process-wide guard used by the model wrappers. Reads its thresholds from the
#: environment at import time; construct a :class:`FrameGuard` explicitly to
#: override them.
guard = FrameGuard()

__all__ = [
    "FrameGuard",
    "GuardConfig",
    "GuardError",
    "GuardResult",
    "guard",
]
