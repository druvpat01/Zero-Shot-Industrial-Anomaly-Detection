"""Observability: the record of what the service did, for whoever has to ask later.

Four questions, four modules
============================
Each of these answers a question the others cannot, which is why there are four
rather than one:

===============================  =========================================================
Module                           The question
===============================  =========================================================
:mod:`~app.observability.logging_config`
                                 *What happened in this request?* Structured events,
                                 one per interesting decision, carrying the ids that
                                 join them to everything else.
:mod:`~app.observability.metrics`
                                 *What is happening across all requests?* Five
                                 Prometheus series, aggregate and cheap, with no
                                 per-request cardinality anywhere in them.
:mod:`~app.observability.tracing`
                                 *Where did the milliseconds go?* One span per
                                 pipeline stage, so a latency regression can be
                                 attributed instead of guessed at.
:mod:`~app.observability.audit_log`
                                 *Who ran the expensive, privacy-relevant thing?*
                                 Append-only, independent of log level, readable
                                 years later.
===============================  =========================================================

The first three are telemetry: rotated, sampled, filtered, and safe to lose. The
audit trail is a record, and its own module docstring argues at length why
keeping it apart from the log is not merely tidiness. That distinction is the
one design decision in this package worth defending, and it is the reason
``results/audit.jsonl`` is not simply a log level.

Joining them up
===============
``request_id`` and ``trace_id`` are bound once by
:class:`~app.observability.middleware.ObservabilityMiddleware` and appear on
every log record and every span for that request. ``model_backend`` and
``category`` are bound by the prediction path and are *also* the label names in
:mod:`~app.observability.metrics`, deliberately — a spike on a dashboard filters
to a model and a category, and those same two words select the matching log
lines without a translation step.
"""

from app.observability.audit_log import (
    AUDIT_LOG_PATH,
    AuditEntry,
    get_audit_log,
    record_benchmark,
    record_calibration,
)
from app.observability.logging_config import (
    bind_log_context,
    clear_log_context,
    configure_logging,
    get_logger,
)
from app.observability.metrics import (
    CONTENT_TYPE_LATEST,
    RESULT_DEFECTIVE,
    RESULT_NORMAL,
    RESULT_REJECTED,
    observe_inference,
    record_guard_rejection,
    record_image_processed,
    render_latest,
    set_models_loaded,
)
from app.observability.tracing import (
    STAGE_GUARD,
    STAGE_MODEL_INFERENCE,
    STAGE_MODEL_LOAD,
    STAGE_POSTPROCESS,
    STAGE_PREPROCESS,
    configure_tracing,
    current_trace_id,
    stage_span,
)

__all__ = [
    "AUDIT_LOG_PATH",
    "CONTENT_TYPE_LATEST",
    "RESULT_DEFECTIVE",
    "RESULT_NORMAL",
    "RESULT_REJECTED",
    "STAGE_GUARD",
    "STAGE_MODEL_INFERENCE",
    "STAGE_MODEL_LOAD",
    "STAGE_POSTPROCESS",
    "STAGE_PREPROCESS",
    "AuditEntry",
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "configure_tracing",
    "current_trace_id",
    "get_audit_log",
    "get_logger",
    "observe_inference",
    "record_benchmark",
    "record_calibration",
    "record_guard_rejection",
    "record_image_processed",
    "render_latest",
    "set_models_loaded",
    "stage_span",
]
