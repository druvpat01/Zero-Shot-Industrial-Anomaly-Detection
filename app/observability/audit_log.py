"""The audit trail: an append-only record of who ran a benchmark, and what they got.

Why this is not the application log
===================================
The application log answers *what is the server doing* — it is written for
debugging, rotated aggressively, sampled when it gets expensive, and filtered
down to WARNING in production because nobody wants a gigabyte of INFO. Every one
of those properties is fatal for an audit trail, which answers a different
question: *who did the expensive, privacy-relevant thing, when, and what did they
learn from it*. That record has to survive log-level changes, be greppable years
later, and be readable by a person who has never seen this codebase.

So it gets its own file, its own format and its own module. ``results/audit.jsonl``
is newline-delimited JSON: one self-contained object per line, appended and never
rewritten. That format is chosen for the properties an audit file needs rather
than for elegance — a truncated write damages one line instead of invalidating
the document (as it would for a top-level JSON array), ``tail -f`` works, ``wc -l``
counts events, and ``jq`` reads it without loading the history into memory.

What is audited
===============
Two operations, distinguished by the ``event`` field:

* ``benchmark`` — the *expensive, privacy-relevant read*. Minutes of CPU per
  call, and a response describing the customer's test data.
* ``calibration`` — the *cheap, consequential write*. ``POST /calibrate`` moves
  the threshold at which the service calls a part defective, for every
  subsequent request from every caller, until the process restarts. It costs
  milliseconds and cannot leak a dataset, so neither of the arguments for
  auditing ``/benchmark`` applies; it is audited for the opposite reason. When
  someone asks in three weeks why the scrap rate stepped on a Tuesday, the
  answer is one line of this file, and the application log that would otherwise
  hold it has long since rotated. A change to how a machine grades parts should
  outlive the debugging of it.

What is recorded, and the two things that are not
=================================================
Each entry carries the timestamp, the *hashed* caller identity and role, the
requested category and backends, how long the operation took, and the ``metrics``
the caller received — for a benchmark, its results; for a calibration, the old
and new thresholds and what the new one achieves. That last field is the point:
an audit trail that says "someone ran a benchmark" is nearly useless, whereas one
that says "this key obtained these numbers over this category's test split"
answers the question an incident review actually asks.

Two omissions are deliberate:

* **The raw API key is never written.** Only the ``hmac-sha256:`` identity from
  :func:`app.serving.auth.hash_api_key`. An audit file is copied around and read
  widely, and one that hands out working credentials is a liability rather than a
  control.
* **Denied requests are not here.** A 403 is rejected by the dependency in
  :mod:`app.serving.auth` before any handler runs, so it lands in the application
  log (at WARNING, with the same hashed identity) and not in this file. That is a
  real gap and ``docs/security.md`` says so.

Failure policy: this fails *open*
=================================
If the audit write fails — full disk, read-only mount, a permissions change —
:func:`record_benchmark` logs the failure at ERROR and returns. It does not turn
a benchmark the caller successfully paid for into a 500. That is the right
trade-off for an evaluation endpoint on an inspection service and the wrong one
for a regulated system, where "cannot audit" must mean "must not serve"; the
choice is called out in ``docs/security.md`` rather than left for someone to
discover from the code.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.observability.logging_config import get_logger

__all__ = [
    "AUDIT_LOG_PATH",
    "DEFAULT_TAIL",
    "AuditEntry",
    "get_audit_log",
    "record_benchmark",
    "record_calibration",
]

log = get_logger(__name__)

#: Repo-root-anchored, so the trail lands in the same place whether the server
#: was started by ``make serve``, a container entrypoint or a test.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where entries are appended. Override with ``AUDIT_LOG_PATH`` — a deployment
#: will usually point this at a volume that outlives the container.
AUDIT_LOG_PATH: Path = Path(os.getenv("AUDIT_LOG_PATH", "").strip() or _REPO_ROOT / "results" / "audit.jsonl")

#: How many entries :func:`get_audit_log` returns when not told otherwise.
DEFAULT_TAIL = 50

#: File mode applied when the trail is first created. Entries name who ran what
#: and carry the metrics they received, so the file is at least as sensitive as
#: the endpoint it audits; there is no reason for it to be world-readable.
_FILE_MODE = 0o600

#: Serialises appends. FastAPI runs this app's synchronous handlers in a thread
#: pool, so two ``/benchmark`` calls really can finish at once. ``O_APPEND``
#: already makes a single small write atomic on POSIX, but the lock makes that a
#: property of this code rather than of the platform it happens to run on.
_write_lock = threading.Lock()

#: ``outcome`` for a run that completed and returned metrics.
OUTCOME_OK = "ok"


@dataclass(frozen=True)
class AuditEntry:
    """One audited event, and the exact shape of one line in the file.

    Attributes:
        timestamp: UTC ISO-8601, second resolution, with an explicit offset.
            Naive local timestamps are how an audit trail becomes unusable the
            first time it is read on a machine in another timezone.
        event: What happened — ``"benchmark"`` or ``"calibration"``. The field is
            why a second audited operation needed neither a second file nor a
            schema migration; ``metrics`` carries whatever that operation's
            payload happens to be.
        caller: Hashed key identity from :func:`app.serving.auth.hash_api_key`.
        role: The role that key holds, so a privilege escalation is visible in
            the trail itself rather than only by cross-referencing the config.
        category: The category whose test split was read.
        models: The backends the caller asked for, as requested — not as
            resolved, so an ONNX fallback shows up as a difference between this
            field and the keys of ``metrics``.
        duration_seconds: Wall-clock cost of the run. The resource-consumption
            side of the record: this is what makes an abusive caller visible.
        outcome: ``"ok"``, or ``"failed:<ExceptionType>"`` when the run raised.
        metrics: ``{model_name: metrics}`` exactly as returned to the caller,
            or ``{}`` on failure. What the caller *learned*, which is the part an
            incident review cares about.
    """

    timestamp: str
    event: str
    caller: str
    role: str
    category: str
    models: list[str]
    duration_seconds: float
    outcome: str
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Plain-``dict`` form, which is also the on-disk field order."""
        return asdict(self)

    def as_json_line(self) -> str:
        """The entry as one line of JSON, with no embedded newlines.

        ``ensure_ascii`` is left on: a category or model name arriving with a
        non-ASCII character must not be able to change the file's encoding
        assumptions halfway through, and an audit file is read by whatever is to
        hand rather than by a careful reader.
        """
        return json.dumps(self.as_dict(), default=str)


def record_benchmark(
    *,
    caller: str,
    role: str,
    category: str,
    models: Sequence[str],
    duration_seconds: float,
    metrics: Mapping[str, Any] | None = None,
    outcome: str = OUTCOME_OK,
    path: Path | str | None = None,
) -> AuditEntry:
    """Append one ``benchmark`` entry to the audit trail.

    Args:
        caller: Hashed key identity — :attr:`app.serving.auth.Principal.key_id`.
            Never a raw key; nothing here validates that, so callers must not
            hand one over.
        role: The caller's role, e.g. ``"operator"``.
        category: Category the benchmark ran against.
        models: Backends as the caller requested them.
        duration_seconds: Wall-clock duration of the run.
        metrics: ``{model_name: metrics}`` as returned to the caller. ``None``
            is recorded as ``{}``, which is the normal state for a failed run.
        outcome: :data:`OUTCOME_OK`, or ``"failed:<ExceptionType>"``.
        path: Destination file. Defaults to :data:`AUDIT_LOG_PATH`; injectable so
            a test writes somewhere it owns rather than into the real trail.

    Returns:
        The :class:`AuditEntry` that was written — returned even if the write
        itself failed, because the caller may want to log it.
    """
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        event="benchmark",
        caller=caller,
        role=role,
        category=category,
        models=list(models),
        duration_seconds=round(float(duration_seconds), 3),
        outcome=outcome,
        metrics=dict(metrics or {}),
    )
    _append(entry, path)
    return entry


def record_calibration(
    *,
    caller: str,
    role: str,
    category: str,
    model_name: str,
    duration_seconds: float,
    metrics: Mapping[str, Any] | None = None,
    outcome: str = OUTCOME_OK,
    path: Path | str | None = None,
) -> AuditEntry:
    """Append one ``calibration`` entry: who moved the decision threshold, and to what.

    Uses the same :class:`AuditEntry` shape as :func:`record_benchmark` rather
    than a variant, which is what the ``event`` field was for. ``models`` holds
    the single model that was updated, and ``metrics`` holds the change —
    ``threshold``, ``previous_threshold``, the metric maximised and what it
    achieves — so one line answers "what was it before, what is it now, and on
    what evidence".

    Args:
        caller: Hashed key identity — :attr:`app.serving.auth.Principal.key_id`.
        role: The caller's role.
        category: Category whose threshold was calibrated.
        model_name: The resolved model that was updated.
        duration_seconds: Wall-clock duration of the request.
        metrics: The change, as returned to the caller.
        outcome: :data:`OUTCOME_OK`, or ``"failed:<ExceptionType>"``. A failed
            calibration is recorded too: a rejected calibration set is an
            operator trying to move the threshold and being refused, which is
            exactly as worth knowing as a successful one.
        path: Destination file. Defaults to :data:`AUDIT_LOG_PATH`.

    Returns:
        The :class:`AuditEntry` that was written.
    """
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        event="calibration",
        caller=caller,
        role=role,
        category=category,
        models=[model_name],
        duration_seconds=round(float(duration_seconds), 3),
        outcome=outcome,
        metrics=dict(metrics or {}),
    )
    _append(entry, path)
    return entry


def _append(entry: AuditEntry, path: Path | str | None = None) -> None:
    """Write one line, or explain loudly why it could not. Never raises.

    See the module docstring for why this fails open rather than propagating.
    """
    destination = Path(path) if path is not None else AUDIT_LOG_PATH
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            is_new = not destination.exists()
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(entry.as_json_line() + "\n")
            if is_new:
                destination.chmod(_FILE_MODE)
    except OSError:
        # ERROR, with the destination, because this is the failure that leaves a
        # served request unrecorded — the one thing this module exists to
        # prevent. `exc_info` carries the OSError (ENOSPC, EROFS, EACCES), which
        # is the difference between "the disk is full" and "the mount went
        # read-only" and therefore between two different fixes.
        log.error(
            "audit_write_failed",
            # `audit_event`, not `event`: structlog names a bound logger's first
            # positional parameter `event`, so passing one as a keyword raises
            # "got multiple values for argument 'event'". The field is renamed
            # rather than dropped — which of the audited operations this was is
            # the point of recording it at all.
            audit_event=entry.event,
            caller=entry.caller,
            role=entry.role,
            category=entry.category,
            destination=str(destination),
            impact="request was served but is NOT in the audit trail",
            exc_info=True,
        )
    else:
        # `caller` is the hashed identity from app.serving.auth.hash_api_key —
        # the same value written into the file — so a log line and an audit entry
        # can be joined on it without either one carrying a usable credential.
        log.info(
            "audit_written",
            # `audit_event`, not `event`: structlog names a bound logger's first
            # positional parameter `event`, so passing one as a keyword raises
            # "got multiple values for argument 'event'". The field is renamed
            # rather than dropped — which of the audited operations this was is
            # the point of recording it at all.
            audit_event=entry.event,
            caller=entry.caller,
            role=entry.role,
            category=entry.category,
            models=entry.models,
            outcome=entry.outcome,
            duration_seconds=entry.duration_seconds,
            destination=str(destination),
        )


def get_audit_log(limit: int = DEFAULT_TAIL, *, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return the last ``limit`` entries, oldest first.

    Reading is as tolerant as writing is strict. A missing file is an empty
    trail, not an error — that is the state of every deployment before its first
    benchmark. A line that will not parse is skipped with a warning rather than
    raising: the usual cause is a write interrupted by a crash, and refusing to
    show the *other* 400 entries because the last one is half-written is exactly
    the wrong behaviour for a forensic tool.

    Args:
        limit: Maximum entries to return, counting back from the newest.
            Non-positive returns nothing.
        path: File to read. Defaults to :data:`AUDIT_LOG_PATH`.

    Returns:
        Parsed entries in the order they were written.
    """
    source = Path(path) if path is not None else AUDIT_LOG_PATH
    if limit <= 0 or not source.is_file():
        return []

    entries: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("audit_line_unparseable", line_number=number, source=str(source))

    return entries[-limit:]
