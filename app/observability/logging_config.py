"""Structured logging: one configuration, applied once, that every logger obeys.

What changes, and what deliberately does not
============================================
Before this module the app logged through the standard library —
``logger.info("Loaded backend %r for %r in %.1fs", ...)`` — which is fine to read
in a terminal and nearly useless in aggregate. "How many cold loads did
``winclip`` take yesterday" is a regex over a human sentence, and the answer
changes the day somebody rewords the message.

:func:`configure_logging` replaces the *rendering*, not the call sites. Every
existing ``logging.getLogger(__name__)`` in the codebase keeps working and gets
routed through the same structlog processor chain as native structlog calls, so
the two produce identical output and carry identical context. That is what
``foreign_pre_chain`` below buys, and it is why this landed as ~200 lines of
configuration instead of a rewrite of every module: the call sites that *benefit*
from key-value fields (the prediction path, the guard, the registry, the audit
trail) were converted; the ones that are genuinely prose (a nan-metric warning
from :mod:`app.evaluation.metrics`) stayed as they were and are structured
anyway by virtue of passing through here.

Two formats, and which one you get
==================================
``LOG_FORMAT`` picks the renderer:

=================  ==========================================================
``LOG_FORMAT``     Output
=================  ==========================================================
``console``        Colour-keyed human-readable lines. The default, because the
                   overwhelmingly common case is somebody running ``make serve``
                   in a terminal and a wall of JSON helps nobody.
``json``           One JSON object per line, for a log shipper. What a container
                   should set.
=================  ==========================================================

Anything else falls back to ``console`` with a warning rather than raising: a
typo in a deployment's environment must not be the reason a server fails to
start, and a server that logs in the wrong *format* is still a server that logs.

The context every record carries
================================
Four fields are attached without any call site asking for them:

* ``timestamp`` — ISO-8601, UTC, milliseconds. Explicitly UTC because the first
  thing anybody does with two logs is line them up, and local time makes that a
  puzzle.
* ``level`` — the standard name, lowercased by structlog.
* ``request_id`` — bound per request by
  :class:`~app.observability.middleware.ObservabilityMiddleware`. Records emitted
  outside a request (startup, a CLI script, a background thread) get
  :data:`NO_REQUEST_ID` rather than nothing at all, so a consumer can select on
  the field without special-casing its absence.
* ``model_backend`` and ``category`` — bound by the prediction path once it knows
  them, absent otherwise. These are the two labels that make a log searchable in
  the same terms as the Prometheus metrics in
  :mod:`app.observability.metrics`, which is the point of choosing them.

The mechanism is :mod:`structlog.contextvars`, not a global or a thread-local.
This app runs synchronous handlers in Starlette's thread pool, and a
:class:`contextvars.ContextVar` is the one carrier that survives that hop
correctly — anyio copies the calling context into the worker thread, so a
``request_id`` bound in the ASGI middleware is still bound inside a handler
running on another thread, and two concurrent requests cannot see each other's.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Literal, MutableMapping

import structlog

__all__ = [
    "NO_REQUEST_ID",
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "get_logger",
    "is_configured",
    "resolve_log_format",
]

LogFormat = Literal["console", "json"]

#: ``request_id`` for a record emitted outside any HTTP request — startup, a CLI
#: script, a background thread. A sentinel rather than an absent key so every
#: record has the same shape and a query does not need an "or missing" clause.
NO_REQUEST_ID = "-"

#: Environment variable selecting the renderer. See the module docstring.
_FORMAT_VAR = "LOG_FORMAT"

#: Environment variable selecting the threshold. Shared with the pre-structlog
#: configuration and documented in ``.env.example``.
_LEVEL_VAR = "LOG_LEVEL"

_DEFAULT_FORMAT: LogFormat = "console"
_DEFAULT_LEVEL = "INFO"

#: Loggers uvicorn configures for itself. It installs its own handlers and sets
#: ``propagate=False``, which would leave its startup banner and access log
#: rendering in uvicorn's format while everything else rendered in ours — the
#: single most confusing possible outcome of a logging change. They are stripped
#: and re-pointed at the root handler in :func:`_reroute_third_party_loggers`.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")

#: Guards :data:`_configured`. ``configure_logging`` is called from FastAPI
#: startup, from tests and potentially from a script that also imports the app,
#: and reconfiguring the root logger from two threads at once is how you end up
#: with duplicated handlers and every line printed twice.
_configure_lock = threading.Lock()
_configured = False


def resolve_log_format(value: str | None = None) -> LogFormat:
    """Normalise a ``LOG_FORMAT`` value to one this module can render.

    Args:
        value: The raw setting. ``None`` reads :data:`_FORMAT_VAR` from the
            environment; blank falls back to the default.

    Returns:
        ``"json"`` or ``"console"``. An unrecognised value returns ``"console"``
        and warns — see the module docstring for why this does not raise.
    """
    raw = (value if value is not None else os.getenv(_FORMAT_VAR, "")).strip().lower()
    if not raw:
        return _DEFAULT_FORMAT
    if raw in ("json", "console"):
        return raw  # type: ignore[return-value]

    # Cannot use a structlog logger here: this runs *during* configuration.
    logging.getLogger(__name__).warning(
        "Unrecognised %s=%r; expected 'json' or 'console'. Falling back to %r.",
        _FORMAT_VAR,
        raw,
        _DEFAULT_FORMAT,
    )
    return _DEFAULT_FORMAT


def _resolve_level(value: str | int | None = None) -> int:
    """Normalise a ``LOG_LEVEL`` value to a :mod:`logging` integer level."""
    if isinstance(value, int):
        return value
    raw = (value if value is not None else os.getenv(_LEVEL_VAR, "")).strip().upper()
    if not raw:
        raw = _DEFAULT_LEVEL
    resolved = logging.getLevelName(raw)
    if isinstance(resolved, int):
        return resolved
    logging.getLogger(__name__).warning("Unrecognised %s=%r; using %s.", _LEVEL_VAR, raw, _DEFAULT_LEVEL)
    return logging.INFO


class _StderrHandler(logging.StreamHandler):
    """A :class:`logging.StreamHandler` that resolves ``sys.stderr`` at emit time.

    ``StreamHandler(stream=sys.stderr)`` — and ``StreamHandler()``, which does the
    same thing internally — stores the stream *object* at construction. That is
    wrong for any process that later rebinds ``sys.stderr``, and three do it
    routinely: :func:`contextlib.redirect_stderr`, pytest (which installs a fresh
    capture object for each of a test's setup/call/teardown phases), and CLI
    wrappers that tee output. In every case the handler keeps writing to a stream
    nobody is reading any more, and logging silently disappears.

    Resolving the attribute on each write costs one dictionary lookup per record
    and makes the handler follow the rebinding. This mirrors what the standard
    library does for its own last-resort handler.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        # StreamHandler.__init__ assigns to self.stream; swallow it rather than
        # let it shadow the property with an instance attribute.
        pass


def _ensure_request_id(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Give every record a ``request_id``, real or :data:`NO_REQUEST_ID`.

    Runs after :func:`structlog.contextvars.merge_contextvars`, so a bound id is
    already present and is left alone. Only the unbound case is filled in.
    """
    event_dict.setdefault("request_id", NO_REQUEST_ID)
    return event_dict


def _drop_noisy_extras(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Strip library-specific ``extra`` keys that are noise once rendered.

    uvicorn attaches ``color_message`` to its startup records: the same message
    again, with ANSI escape codes embedded. :class:`structlog.stdlib.ExtraAdder`
    faithfully promotes it to a field, where it is a duplicate of ``event`` with
    terminal control characters in it — unreadable in a file and actively
    corrupting in a log viewer that does not strip them.

    Dropping it is worth the special case because ``ExtraAdder`` is otherwise
    exactly what is wanted: it is how any library's structured ``extra`` reaches
    the output, and turning it off to avoid this one key would cost far more than
    naming the key.
    """
    event_dict.pop("color_message", None)
    return event_dict


def _shared_processors() -> list[Any]:
    """The processor chain applied to structlog and stdlib records alike.

    Order matters and is the reason this is a function rather than a constant:

    1. ``merge_contextvars`` first, so ``request_id``/``model_backend``/
       ``category`` are in the dict before anything can act on them.
    2. Level and logger name, which the stdlib bridge cannot supply itself.
    3. ``_ensure_request_id`` after the merge, so it only fills a genuine gap.
    4. ``PositionalArgumentsFormatter`` before the renderer, so the ``%s``-style
       call sites still in the model and evaluation modules interpolate rather
       than emitting a ``positional_args`` field full of unformatted tuples.
       Those call sites are prose — "no defect regions found in the ground
       truth; returning nan" has no useful key-value decomposition — and
       rewriting them into event names would have made them worse, so the
       formatter exists to let both styles coexist under one renderer.
    5. Timestamp last of the enrichers, so it reflects when the record was
       *emitted* rather than when a slow upstream processor started.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _ensure_request_id,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
    ]


def _build_renderer(log_format: LogFormat) -> Any:
    """The final processor: JSON for shippers, colour-keyed lines for humans."""
    if log_format == "json":
        # sort_keys keeps the field order stable across records, which matters
        # more than it sounds: a diff of two log files is otherwise noise.
        return structlog.processors.JSONRenderer(sort_keys=True)
    # Colours only when stderr is a terminal. A captured or redirected stream
    # gets plain text rather than ANSI escapes, which are unreadable in a file
    # and actively corrupt a CI log.
    return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())


def configure_logging(
    *,
    log_format: str | None = None,
    level: str | int | None = None,
    force: bool = False,
) -> LogFormat:
    """Configure structlog and the standard library to render identically. Idempotent.

    Called once from the FastAPI lifespan in :mod:`app.serving.main`, and
    available to scripts and tests that want the same output without starting a
    server.

    Both logging systems are pointed at a single :class:`logging.StreamHandler`
    on stderr whose formatter is a :class:`structlog.stdlib.ProcessorFormatter`.
    That is the whole trick: a record from ``logging.getLogger(...)`` enters the
    formatter, runs the same chain as a native structlog event via
    ``foreign_pre_chain``, and comes out the other side in the same shape. There
    is exactly one place output is rendered, so the two can never drift.

    stderr, not stdout, on purpose. A container's stdout is where a CLI script
    writes its *report* (see ``scripts/``), and interleaving log lines into a
    table somebody is trying to read — or into a JSON document being piped to
    ``jq`` — is a self-inflicted wound. Both streams still land in the same place
    under Docker's logging driver. The handler is a :class:`_StderrHandler`,
    which follows a later rebinding of ``sys.stderr`` rather than pinning the
    object it saw at startup — see that class for why that distinction is not
    theoretical.

    Args:
        log_format: ``"json"`` or ``"console"``. Defaults to ``LOG_FORMAT``.
        level: Threshold, as a name or a :mod:`logging` integer. Defaults to
            ``LOG_LEVEL``, then ``INFO``.
        force: Reconfigure even if this has already run. Off by default so a
            second call — a test importing the app after a script configured it —
            is a no-op rather than a second handler on the root logger.

    Returns:
        The format that was applied, so a caller can log which one it got.
    """
    global _configured  # noqa: PLW0603 - module-level "has this run" flag

    with _configure_lock:
        if _configured and not force:
            return resolve_log_format(log_format)

        resolved_format = resolve_log_format(log_format)
        resolved_level = _resolve_level(level)
        shared = _shared_processors()

        structlog.configure(
            processors=[
                *shared,
                # Hands the event dict to the stdlib formatter below rather than
                # rendering here. Without this, structlog and stdlib records
                # would take two different paths to the same stream.
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            # Bound loggers are created per call site and cached on the module;
            # caching them makes get_logger() cheap enough to call at import.
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                # Strips the bookkeeping keys ProcessorFormatter adds; must be
                # the last thing before the renderer or they leak into output.
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _build_renderer(resolved_format),
            ],
            # Applied to records that did *not* come from structlog — every
            # `logging.getLogger(__name__)` call site in the codebase.
            foreign_pre_chain=[
                *shared,
                structlog.stdlib.ExtraAdder(),
                _drop_noisy_extras,
                structlog.processors.format_exc_info,
            ],
        )

        handler = _StderrHandler()
        handler.setFormatter(formatter)

        root = logging.getLogger()
        # Replace rather than append. Re-running this (force=True, or a test
        # flipping the format) must not double every line.
        for existing in list(root.handlers):
            root.removeHandler(existing)
        root.addHandler(handler)
        root.setLevel(resolved_level)

        _reroute_third_party_loggers(resolved_level)

        _configured = True

    return resolved_format


def _reroute_third_party_loggers(level: int) -> None:
    """Make uvicorn's loggers propagate to the root handler instead of their own.

    uvicorn calls ``logging.config.dictConfig`` at startup with handlers of its
    own and ``propagate=False``. Left alone, the access log would render in
    uvicorn's format and everything else in ours. Clearing the handlers and
    turning propagation back on routes them through the same formatter.
    """
    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
        third_party.setLevel(level)


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run in this process."""
    return _configured


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, configuring logging first if nobody has yet.

    The lazy configuration is what makes this safe to call at module import.
    Importing :mod:`app.serving.model_registry` from a script that never starts
    the FastAPI app still produces formatted output rather than the stdlib's
    "no handlers could be found" silence.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A bound logger. Calls take an event string and arbitrary keyword fields:
        ``log.info("model_loaded", backend="patchcore", duration_s=1.2)``.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Per-request / per-operation context
# ---------------------------------------------------------------------------


def bind_log_context(**fields: Any) -> None:
    """Attach fields to every subsequent log record in this context.

    Used by the middleware for ``request_id`` and ``trace_id``, and by the
    prediction path for ``model_backend`` and ``category``. ``None`` values are
    dropped rather than bound, so an optional field is simply absent instead of
    appearing as ``null`` on every record.
    """
    present = {key: value for key, value in fields.items() if value is not None}
    if present:
        structlog.contextvars.bind_contextvars(**present)


def clear_log_context() -> None:
    """Drop everything bound in this context.

    Called at the *start* of each request rather than only at the end. A worker
    thread is reused across requests, and while Starlette copies the context per
    request, clearing first makes the isolation a property of this code rather
    than an assumption about the server.
    """
    structlog.contextvars.clear_contextvars()
