"""The per-request seam: where a request acquires an id, a trace and a log context.

One middleware, three jobs
==========================
Every log line, every span and every response header that identifies a request
originates here:

1. **A request id.** Taken from an inbound ``X-Request-Id`` if the caller sent
   one, generated otherwise, and echoed back on the response. Reusing the
   caller's id is what lets an inspection line correlate "frame 41 823 came back
   wrong" with this service's logs without anybody grepping timestamps.
2. **A root span**, with the four pipeline stages from
   :mod:`app.observability.tracing` hanging off it, continuing an upstream trace
   when the caller sent a ``traceparent``.
3. **A log context**, so ``request_id`` and ``trace_id`` appear on every record
   the request emits — including records from library code and from the
   exception handlers, neither of which could plausibly pass them down by hand.

Why a raw ASGI middleware rather than ``BaseHTTPMiddleware``
============================================================
This is written against the ASGI interface directly, and that is load-bearing
rather than stylistic. Starlette's :class:`BaseHTTPMiddleware` runs the
downstream app in a *separate anyio task*: a :class:`~contextvars.ContextVar`
bound in the middleware is copied into that task, so bindings do reach the
handler, but the copy is one-way and the isolation depends on a detail of how
the task group is constructed. A raw middleware runs in the same context as the
handler, so the contextvar bindings this module makes and the ones the handler
adds live in one place, and unbinding at the end genuinely unbinds.

It matters here more than it would elsewhere because this app's handlers are
synchronous and run in Starlette's thread pool. That hop is exactly where a
thread-local would break and a contextvar does not — anyio copies the calling
context into the worker thread — and keeping everything in one task keeps that
guarantee simple enough to reason about.

What is deliberately *not* traced
=================================
``/metrics`` produces no span. Prometheus scrapes it every 15 seconds forever,
and a trace backend filled with identical scrape spans is a trace backend nobody
opens. It still gets a request id, because a scrape that fails is worth being
able to find in the log.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, MutableMapping

from opentelemetry import propagate, trace

from app.observability.logging_config import bind_log_context, clear_log_context, get_logger
from app.observability.tracing import current_trace_id, get_tracer

__all__ = ["REQUEST_ID_HEADER", "TRACE_ID_HEADER", "ObservabilityMiddleware"]

log = get_logger(__name__)

#: Header the request id is read from and echoed back on. ``X-Request-Id`` is
#: the de-facto convention across load balancers and proxies, so an id assigned
#: upstream is usually already sitting in it.
REQUEST_ID_HEADER = "x-request-id"

#: Header the trace id is echoed back on. Not ``traceparent``: that is the W3C
#: *propagation* header for outbound calls and has its own format. This is a
#: convenience for a human reading a response with ``curl -i`` who wants to look
#: the trace up.
TRACE_ID_HEADER = "x-trace-id"

#: Paths that get a request id but no span. See the module docstring.
_UNTRACED_PATHS = frozenset({"/metrics"})

#: Longest inbound request id that will be trusted. An id is echoed into a
#: response header and written to every log line the request produces, so an
#: unbounded one is a log-injection and header-size problem handed straight to us
#: by the caller. 200 characters is far past any real id (a UUID is 36).
_MAX_REQUEST_ID_LENGTH = 200

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class ObservabilityMiddleware:
    """Bind a request id, a trace and a log context around every HTTP request.

    Args:
        app: The next ASGI application in the chain.

    Example:
        >>> app.add_middleware(ObservabilityMiddleware)   # doctest: +SKIP
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket scopes pass straight through: neither has a
        # request id to bind, and the lifespan scope in particular is where
        # configure_logging() itself runs.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        request_id = _resolve_request_id(headers.get(REQUEST_ID_HEADER))
        path = scope.get("path", "")
        method = scope.get("method", "")

        # Cleared first, not just on the way out: a worker thread is reused, and
        # starting from a known-empty context makes isolation a property of this
        # code rather than an assumption about the server.
        clear_log_context()
        bind_log_context(request_id=request_id)

        if path in _UNTRACED_PATHS:
            try:
                await self.app(scope, receive, _with_headers(send, request_id, trace_id=None))
            finally:
                clear_log_context()
            return

        # Continues the caller's trace when they sent a `traceparent`, and starts
        # a fresh one when they did not. Extraction is what makes this service a
        # participant in a distributed trace rather than the root of a lonely one.
        parent = propagate.extract(headers)
        span_name = f"{method} {scope.get('root_path', '')}{path}"

        with get_tracer().start_as_current_span(span_name, context=parent, kind=trace.SpanKind.SERVER) as span:
            trace_id = current_trace_id()
            bind_log_context(trace_id=trace_id)
            span.set_attribute("http.request.method", method)
            span.set_attribute("url.path", path)
            span.set_attribute("request_id", request_id)

            status_holder: dict[str, int] = {}

            async def send_wrapper(message: MutableMapping[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    status_holder["status"] = message["status"]
                    span.set_attribute("http.response.status_code", message["status"])
                await send(message)

            try:
                await self.app(scope, receive, _with_headers(send_wrapper, request_id, trace_id))
            finally:
                # Nothing catches here. Starlette's ServerErrorMiddleware sits
                # *outside* this one and turns an escaped exception into a 500,
                # and `start_as_current_span` has already recorded it on the span
                # and set the status to ERROR on its way out. Adding a handler
                # would only duplicate the app's own logging.
                log.info(
                    "request_completed",
                    method=method,
                    path=path,
                    status_code=status_holder.get("status"),
                )
                clear_log_context()


def _headers(scope: Scope) -> dict[str, str]:
    """Inbound headers as a lowercased ``str -> str`` dict.

    ASGI delivers headers as a list of raw byte pairs, with names already
    lowercased by the spec. Latin-1 is the correct decoding for HTTP/1.1 header
    values, and ``errors="replace"`` means a malformed byte produces a mangled
    header rather than a 500 from the middleware.
    """
    return {
        name.decode("latin-1"): value.decode("latin-1", errors="replace") for name, value in scope.get("headers", [])
    }


def _resolve_request_id(inbound: str | None) -> str:
    """Reuse the caller's request id if it is usable, otherwise mint one.

    An inbound id is *sanitised*, not trusted: control characters are stripped
    (an id ends up in log lines, and a newline in one is a forged log record) and
    the length is capped at :data:`_MAX_REQUEST_ID_LENGTH`. An id that is empty
    after that is replaced rather than rejected — the request is fine, the header
    was not.
    """
    if inbound:
        cleaned = "".join(char for char in inbound.strip() if char.isprintable())[:_MAX_REQUEST_ID_LENGTH]
        if cleaned:
            return cleaned
    return uuid.uuid4().hex


def _with_headers(send: Send, request_id: str, trace_id: str | None) -> Send:
    """Wrap ``send`` so the response carries the request and trace ids.

    Appended to the outbound headers on ``http.response.start``, which is the one
    message that has them. Doing it here rather than in a route handler means it
    holds for *every* response, including the ones no handler produced: a 422
    from the validation handler, a 500 from Starlette's error middleware, a 404
    from the router.
    """

    async def wrapped(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1")))
            if trace_id is not None:
                headers.append((TRACE_ID_HEADER.encode("latin-1"), trace_id.encode("latin-1")))
            message = {**message, "headers": headers}
        await send(message)

    return wrapped
