"""Which models this process had loaded, remembered outside the process.

Why remember at all
===================
:mod:`app.serving.model_registry` is deliberately lazy: nothing is loaded until
somebody asks for it, so ``GET /health`` answers in milliseconds and an
orchestrator never kills a container for taking half a minute to become ready.
The cost is that the first request after *every* restart pays a load — tens of
seconds for PatchCore's memory bank, more for WinCLIP's CLIP weights — and on a
rolling deploy that cost lands on production traffic once per replica.

The fix is not to load everything at startup; that is the problem laziness
exists to avoid. It is to remember *which* models were worth having. This module
keeps that list outside the process, so a restarted API can rebuild exactly the
working set it had — in the background, while it is already answering — instead
of waiting for an operator to re-warm it by hand.

What is stored, and what is not
===============================
Four fields per model: ``backend``, ``category``, ``checkpoint_path`` and
``loaded_at``. **Not the weights.** A PatchCore checkpoint is 221 MB and its
ONNX export 362 MB; pushing those through Redis would be slower than reading the
file already sitting on the volume, and it would put a cache eviction between
the service and its ability to answer at all. What is cached is a note saying
"this pair was in use, and this is the artifact it came from". The load path is
untouched and still reads from disk.

Records expire after an hour (:data:`DEFAULT_TTL_SECONDS`) and are written once,
at load time, rather than refreshed on every prediction. So the restored set is
one that was *recently established*: a model loaded at 09:00 and used all day is
forgotten by 10:00. That bias is deliberate. Re-warming is speculative work, and
an hour is where speculating on the strength of a single old fact should stop —
a deployment that wants a longer memory should raise the TTL, not refresh on
read, which would turn every ``/predict`` into a Redis write.

Redis is optional, always
=========================
Every method here is total: it does the thing, or it logs and carries on against
an in-process dict. "Which models to speculatively re-warm" is the definition of
non-essential state, and an inspection line stopping because a cache is
unreachable would be an outage manufactured entirely out of an optimisation. So:

* nothing here raises — the guarded calls catch :class:`Exception`, not
  ``RedisError``, because a bad URL raises ``ValueError`` and a dead DNS entry
  raises ``OSError``, and "never hard-fail" has to mean all of them;
* a failure demotes the cache to the fallback dict and is not retried for
  :data:`RETRY_AFTER_SECONDS`, so a Redis that is down costs one timeout rather
  than one per model load;
* the fallback dict implements the same TTL, so the two modes differ *only* in
  surviving a restart — which is the entire point of the Redis one, and worth
  keeping as the single visible difference.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

from app.observability.logging_config import get_logger

try:  # pragma: no cover - the package is a declared dependency; this is belt and braces
    import redis
except ImportError:  # pragma: no cover - an install without it still serves, without the cache
    redis = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_KEY_PREFIX",
    "DEFAULT_TTL_SECONDS",
    "REDIS_URL_VAR",
    "RETRY_AFTER_SECONDS",
    "CheckpointCache",
    "LoadedModel",
    "get_checkpoint_cache",
]

log = get_logger(__name__)

#: Environment variable carrying the connection URL, e.g.
#: ``redis://redis:6379/0`` under docker-compose. Unset or empty means "no
#: Redis": the cache runs on its fallback dict and says so once, at DEBUG. That
#: is the correct behaviour for `make serve` on a laptop, where a warning about
#: an optional service nobody asked for is noise.
REDIS_URL_VAR = "REDIS_URL"

#: Read like :mod:`app.serving.auth` and :mod:`app.models.config` do — the real
#: environment wins, so a container injecting ``REDIS_URL`` is not overridden by
#: a stale ``.env`` baked into an image.
_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

#: Lifetime of one record. See the module docstring for why it is not refreshed.
DEFAULT_TTL_SECONDS = 3600

#: Key namespace. Prefixed rather than bare so this can share a Redis with
#: anything else without a collision, and so :meth:`CheckpointCache.entries` can
#: scan for exactly its own keys.
DEFAULT_KEY_PREFIX = "defect-detection:model:"

#: Connect *and* command timeout. Sub-second on purpose: this call sits inside a
#: model load, and the whole design says a missing cache must cost approximately
#: nothing. The default (no timeout) would hang a load until the kernel gave up.
SOCKET_TIMEOUT_SECONDS = 0.5

#: How long the cache stays demoted after a failure before probing Redis again.
#: Long enough that a Redis outage is one timeout every 30 s rather than one per
#: load; short enough that a Redis coming back is picked up without a restart.
RETRY_AFTER_SECONDS = 30.0


def _redact(url: str) -> str:
    """``redis://user:pw@host:6379/0`` -> ``redis://user:***@host:6379/0``.

    The URL is logged — it is the first thing anybody debugging a cache miss
    wants — and a Redis URL is a place people put passwords.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable>"
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    return urlunsplit((parts.scheme, f"{user}:***@{host}", parts.path, parts.query, parts.fragment))


@dataclass(frozen=True)
class LoadedModel:
    """One cached record: a model that was resident when it was written.

    Attributes:
        backend: The requested backend, e.g. ``"patchcore"``. This is the
            *request-side* name, because it is what has to be replayed to
            reload the model — an ONNX substitution is a property of what is on
            disk now, and is re-decided at load time rather than pinned here.
        category: MVTec-style category, e.g. ``"bottle"``.
        checkpoint_path: The artifact the load resolved to, or ``None`` for a
            zero-shot backend that needed none. Recorded for diagnosis rather
            than for reloading: it answers "which file was this serving?" after
            a deploy swapped one out.
        loaded_at: ISO 8601 UTC timestamp, second resolution. A string rather
            than an epoch float so that ``redis-cli GET`` is readable by a human
            at 3 a.m.
    """

    backend: str
    category: str
    checkpoint_path: str | None
    loaded_at: str

    @property
    def key(self) -> str:
        """The ``"<backend>:<category>"`` form the registry reports models in."""
        return f"{self.backend}:{self.category}"

    def to_json(self) -> str:
        return json.dumps(
            {
                "backend": self.backend,
                "category": self.category,
                "checkpoint_path": self.checkpoint_path,
                "loaded_at": self.loaded_at,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> LoadedModel | None:
        """Parse a stored record, returning ``None`` if it is not one.

        Anything can write to a Redis. A record that does not parse is dropped
        with a warning rather than raising, because the caller is a startup
        warm-up loop and one bad key must not stop it reading the rest.
        """
        try:
            payload = json.loads(raw)
            return cls(
                backend=str(payload["backend"]),
                category=str(payload["category"]),
                checkpoint_path=None if payload.get("checkpoint_path") is None else str(payload["checkpoint_path"]),
                loaded_at=str(payload.get("loaded_at", "")),
            )
        except (TypeError, ValueError, KeyError) as exc:
            log.warning("checkpoint_cache_record_invalid", error=type(exc).__name__, raw=raw[:200])
            return None


class CheckpointCache:
    """A TTL'd record of loaded models, in Redis when there is one and in a dict when there is not.

    Args:
        url: Connection URL. ``None`` resolves :data:`REDIS_URL_VAR` from the
            environment (and ``.env``) on first use; the empty string forces
            fallback mode without touching the environment, which is how tests
            keep themselves off a developer's real Redis.
        ttl_seconds: Record lifetime. Applied identically in both modes.
        key_prefix: Key namespace; see :data:`DEFAULT_KEY_PREFIX`.
        use_dotenv: Whether resolving the URL may read ``<repo>/.env``.

    Example:
        >>> cache = CheckpointCache(url="")                      # doctest: +SKIP
        >>> cache.record("patchcore", "bottle", "results/checkpoints/patchcore_bottle.ckpt")
        >>> [entry.key for entry in cache.entries()]             # doctest: +SKIP
        ['patchcore:bottle']
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        use_dotenv: bool = True,
    ) -> None:
        self._explicit_url = url
        self._use_dotenv = use_dotenv
        self._ttl_seconds = int(ttl_seconds)
        self._key_prefix = key_prefix

        # Guards the client handle, the cooldown deadline and the fallback dict.
        # Held only around dict work and the connect probe, never around a Redis
        # command — a command holding this lock would serialise every model load
        # in the process behind one slow socket.
        self._lock = threading.Lock()
        self._client: object | None = None
        self._probe_after = 0.0
        self._announced_disabled = False
        #: Why the cache is not on Redis, for :meth:`status` to report. A short
        #: slug, never the URL — see that method for why the distinction matters.
        self._disabled_reason: str | None = None
        #: key -> (payload, monotonic expiry). The fallback's whole job is to
        #: behave like the Redis one, TTL included.
        self._fallback: dict[str, tuple[str, float]] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode!r}, ttl_seconds={self._ttl_seconds})"

    @property
    def mode(self) -> str:
        """``"redis"`` when a live client is held, ``"memory"`` otherwise.

        Reported in the startup log so "why did my warm set not survive?" is
        answerable from the logs alone. Does not itself attempt a connection:
        asking a cache what mode it is in must not be the thing that connects it.
        """
        with self._lock:
            return "redis" if self._client is not None else "memory"

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def status(self) -> dict[str, object]:
        """Connection state, cheap enough for a health endpoint to poll.

        Unlike :attr:`mode`, this *does* connect — it is the difference between
        "no client is held" and "Redis is unreachable", and a status field that
        cannot tell those apart reports a healthy stack as degraded for the whole
        window before the first model load. The cost is bounded by the same two
        constants every other caller relies on: an established client is a lock
        and an attribute read, and an absent one is at most one connect attempt
        per :data:`RETRY_AFTER_SECONDS`, capped at
        :data:`SOCKET_TIMEOUT_SECONDS`. A 10 s dashboard refresh therefore costs
        nothing in the normal case and 0.5 s twice a minute in the worst one.

        Returns:
            ``backend`` (``"redis"``/``"memory"``), ``connected``,
            ``ttl_seconds``, and ``detail`` — a short reason when not connected.

        Note:
            The URL is deliberately **not** included. This is surfaced on
            unauthenticated ``GET /health``, whose disclosure is bounded to
            liveness by design, and an internal hostname is a free piece of the
            deployment's map. ``checkpoint_cache_connected`` in the logs carries
            the (redacted) URL for anyone who is entitled to it.
        """
        client = self._client_or_none()
        with self._lock:
            reason = self._disabled_reason
        return {
            "backend": "redis" if client is not None else "memory",
            "connected": client is not None,
            "ttl_seconds": self._ttl_seconds,
            "detail": None if client is not None else (reason or "not connected"),
        }

    # -- connection ------------------------------------------------------------

    def _resolve_url(self) -> str:
        if self._explicit_url is not None:
            return self._explicit_url.strip()
        if self._use_dotenv and _DOTENV_PATH.is_file():
            load_dotenv(_DOTENV_PATH, override=False)
        return (os.environ.get(REDIS_URL_VAR) or "").strip()

    def _client_or_none(self) -> object | None:
        """The live client, connecting once if the cooldown has elapsed.

        Returns ``None`` in every unhappy case — no URL, no ``redis`` package,
        a refused connection, or a demotion that has not yet timed out — and the
        caller falls through to the dict. It never raises, which is what lets
        every public method here be total.
        """
        with self._lock:
            if self._client is not None:
                return self._client
            now = time.monotonic()
            if now < self._probe_after:
                return None
            self._probe_after = now + RETRY_AFTER_SECONDS

            if redis is None:
                self._announce_disabled("the `redis` package is not installed")
                return None
            url = self._resolve_url()
            if not url:
                self._announce_disabled(f"{REDIS_URL_VAR} is unset")
                return None

            try:
                client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
                    socket_timeout=SOCKET_TIMEOUT_SECONDS,
                )
                # from_url is lazy — it validates the URL and returns. PING is
                # what actually proves there is a server, and doing it here means
                # a dead Redis is discovered by the probe rather than by the
                # first write, where it would be one failure per model load.
                client.ping()
            except Exception as exc:  # noqa: BLE001 - see the module docstring: this must not raise
                log.warning(
                    "checkpoint_cache_unavailable",
                    url=_redact(url),
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                    fallback="in-process dict",
                    impact="model loads still work; the warm set will not survive a restart",
                    retry_in_seconds=RETRY_AFTER_SECONDS,
                )
                # The exception *type*, not its message: a connection error's
                # detail is a host and port, and this is read back out through
                # an unauthenticated endpoint. See `status`.
                self._disabled_reason = f"unreachable ({type(exc).__name__})"
                return None

            self._client = client
            self._announced_disabled = False
            self._disabled_reason = None
            log.info(
                "checkpoint_cache_connected",
                url=_redact(url),
                ttl_seconds=self._ttl_seconds,
                key_prefix=self._key_prefix,
            )
            return client

    def _announce_disabled(self, reason: str) -> None:
        """Say once, at DEBUG, that there is no Redis configured. Assumes the lock is held.

        DEBUG and once: running without Redis is a supported configuration (it is
        what ``make serve`` does), so it is not a warning, and repeating it every
        30 s would bury the logs that matter.
        """
        self._disabled_reason = reason
        if self._announced_disabled:
            return
        self._announced_disabled = True
        log.debug("checkpoint_cache_disabled", reason=reason, fallback="in-process dict")

    def _demote(self, operation: str, exc: Exception) -> None:
        """Drop a client that failed mid-command and start the cooldown."""
        with self._lock:
            self._client = None
            self._probe_after = time.monotonic() + RETRY_AFTER_SECONDS
            self._disabled_reason = f"{operation} failed ({type(exc).__name__})"
        log.warning(
            "checkpoint_cache_command_failed",
            operation=operation,
            error=type(exc).__name__,
            detail=str(exc)[:200],
            fallback="in-process dict",
            retry_in_seconds=RETRY_AFTER_SECONDS,
        )

    # -- records ---------------------------------------------------------------

    def _key(self, backend: str, category: str) -> str:
        return f"{self._key_prefix}{backend}:{category}"

    def record(self, backend: str, category: str, checkpoint_path: str | None = None) -> LoadedModel:
        """Note that ``(backend, category)`` is loaded, with a :data:`DEFAULT_TTL_SECONDS` lifetime.

        Called from the model load path, once per load rather than once per
        request — see the module docstring.

        Returns:
            The record as written, so a caller can log exactly what it stored.
        """
        entry = LoadedModel(
            backend=backend,
            category=category,
            checkpoint_path=checkpoint_path,
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        key = self._key(backend, category)
        payload = entry.to_json()

        client = self._client_or_none()
        if client is not None:
            try:
                client.set(key, payload, ex=self._ttl_seconds)
                return entry
            except Exception as exc:  # noqa: BLE001 - never fail a model load over a cache write
                self._demote("set", exc)

        with self._lock:
            self._fallback[key] = (payload, time.monotonic() + self._ttl_seconds)
        return entry

    def entries(self) -> list[LoadedModel]:
        """Every unexpired record, oldest load first.

        Ordered so a warm-up loop rebuilds the working set in the order it was
        originally established, which is the closest thing available to "most
        likely to be asked for first".

        Returns:
            Records, possibly empty. Never raises: a Redis that fails mid-scan
            demotes the cache and the fallback dict answers instead.
        """
        client = self._client_or_none()
        if client is not None:
            try:
                found: list[LoadedModel] = []
                # SCAN, not KEYS: this may share a Redis with anything else, and
                # KEYS blocks the server for the length of the keyspace.
                for key in client.scan_iter(match=f"{self._key_prefix}*", count=100):
                    raw = client.get(key)
                    if raw is None:  # expired between the scan and the read
                        continue
                    entry = LoadedModel.from_json(raw)
                    if entry is not None:
                        found.append(entry)
                return sorted(found, key=lambda item: (item.loaded_at, item.key))
            except Exception as exc:  # noqa: BLE001
                self._demote("scan", exc)

        now = time.monotonic()
        with self._lock:
            for key in [key for key, (_, expiry) in self._fallback.items() if expiry <= now]:
                del self._fallback[key]
            payloads = [payload for payload, _ in self._fallback.values()]
        parsed = [entry for entry in (LoadedModel.from_json(payload) for payload in payloads) if entry is not None]
        return sorted(parsed, key=lambda item: (item.loaded_at, item.key))

    def forget(self, backend: str, category: str) -> None:
        """Drop one record.

        Used when a re-warm discovers the artifact is gone: keeping a record that
        cannot be reloaded means paying its failure on every restart forever.
        """
        key = self._key(backend, category)
        client = self._client_or_none()
        if client is not None:
            try:
                client.delete(key)
            except Exception as exc:  # noqa: BLE001
                self._demote("delete", exc)
        with self._lock:
            self._fallback.pop(key, None)


#: The process-wide cache, reached through :func:`get_checkpoint_cache` so a test
#: can substitute one without a registry knowing. Constructed eagerly but
#: connected lazily — importing this module must not open a socket.
_cache: CheckpointCache | None = None
_cache_lock = threading.Lock()


def get_checkpoint_cache() -> CheckpointCache:
    """The process-wide :class:`CheckpointCache`."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = CheckpointCache()
        return _cache
