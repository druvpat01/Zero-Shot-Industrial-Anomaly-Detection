"""Observability: the record of what the service did, for whoever has to ask later.

Two audiences, and they want different files. The application log (plain
``logging``, configured by whatever starts the server) is for whoever is
debugging *now*: verbose, rotated, filtered by level. The **audit trail** in
:mod:`app.observability.audit_log` is for whoever is asking months later who ran
the expensive, privacy-relevant operation and what they got back — so it is
append-only, self-describing, and independent of log level.

Keeping the two apart is the point; :mod:`app.observability.audit_log`'s
docstring argues the case.
"""

from app.observability.audit_log import (
    AUDIT_LOG_PATH,
    AuditEntry,
    get_audit_log,
    record_benchmark,
)

__all__ = [
    "AUDIT_LOG_PATH",
    "AuditEntry",
    "get_audit_log",
    "record_benchmark",
]
