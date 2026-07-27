"""API-key access control: who may call which endpoint, as a FastAPI dependency.

What is being protected, and from what
======================================
Not every endpoint in this service costs the same to serve. ``GET /health``
touches nothing. ``POST /predict`` scores one frame. ``POST /benchmark`` runs
every requested model over a category's *entire* test split — minutes of CPU,
gigabytes of resident weights, and a thread-pool worker held for the whole run
(see the endpoint's docstring in :mod:`app.serving.main`). An unauthenticated
``/benchmark`` is therefore a denial-of-service primitive with a JSON body: a
handful of concurrent requests will starve the inference path that a production
line is actually waiting on.

There is a quieter problem too. The benchmark's response is a set of metrics
computed over the customer's test data, and metrics leak properties of the data
they were computed on — how many images are in the split, how many are
defective, how separable the defects are. That is not the images themselves, but
it is more than an anonymous caller should be able to enumerate. See
``docs/security.md`` for the longer version of this argument.

Two roles, and why only two
===========================
* ``viewer`` — may score frames (``POST /predict``) and read ``GET /health``.
  This is the inspection line: the thing that submits images all day.
* ``operator`` — everything a viewer may do, plus the expensive and
  introspective endpoints: ``POST /benchmark`` and ``GET /models``.

Roles nest rather than sit side by side (:data:`_ROLE_GRANTS`), so an operator
key works on every viewer route without being listed twice. Two roles is the
smallest split that separates "runs the line" from "administers the service",
and inventing more before there is a second consumer would be fiction.

The failure modes, and the status code each gets
================================================
=========================================  ======  =============================
Condition                                  Status  ``detail``
=========================================  ======  =============================
No ``X-API-Key`` header                    401     ``missing_api_key``
Header present, key not recognised         401     ``invalid_api_key``
Key recognised, role insufficient          403     ``insufficient_role``
No keys configured on the server at all    503     ``auth_not_configured``
=========================================  ======  =============================

The 401/403 split is the meaningful one: 401 says *I do not know who you are*,
403 says *I know exactly who you are and the answer is still no*. A caller
holding a viewer key that gets a 401 from ``/benchmark`` would reasonably go
looking for a typo in their key; the 403 tells them the truth, which is that they
need a different key.

The last row is the one worth arguing about, and it is deliberate: a server with
no keys configured **refuses gated requests** rather than serving them openly. A
missing environment variable is the single most likely way this protection gets
turned off by accident, and failing open would mean a deployment that quietly has
no access control at all while every endpoint returns 200. 503 also names the
right owner — the problem is the server's configuration, not the caller's
request.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

__all__ = [
    "API_KEY_HEADER",
    "AuthConfig",
    "Principal",
    "Role",
    "get_auth_config",
    "hash_api_key",
    "require_operator",
    "require_role",
    "require_viewer",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In production, replace with OAuth2/JWT and a proper identity provider. .env
# key storage is portfolio-scope only.
#
# Concretely, what that means here: the keys below are long-lived bearer secrets
# read from a file on the server's disk. They cannot be rotated without a
# restart, cannot be revoked individually without editing that file, carry no
# expiry, and are compared by this process rather than validated against an
# issuer. A real deployment puts them in a secrets manager (or drops API keys
# entirely for OIDC-issued JWTs), scopes them per tenant, and gets rotation and
# revocation from the identity provider instead of from an editor. Everything
# below the dependency boundary — the roles, the 401/403 split, the audit trail
# in app/observability/audit_log.py — survives that swap unchanged; only the
# credential check itself changes.
# ---------------------------------------------------------------------------

#: Repo-root-anchored so ``.env`` resolves the same from a test, a script or the
#: API server, whatever the working directory happens to be. Matches
#: :mod:`app.models.config`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"

#: Header the key is read from. Not ``Authorization``: this is not a bearer token
#: in any standardised sense, and pretending otherwise would invite a client to
#: send a JWT and expect it to be validated as one.
API_KEY_HEADER = "X-API-Key"

Role = Literal["viewer", "operator"]

#: role held -> roles it satisfies. The one place the hierarchy is written down;
#: adding a role means adding a row, not auditing every route.
_ROLE_GRANTS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"viewer"}),
    "operator": frozenset({"viewer", "operator"}),
}

#: Environment variables the keys are read from, as comma-separated lists.
_VIEWER_KEYS_VAR = "VIEWER_API_KEYS"
_OPERATOR_KEYS_VAR = "OPERATOR_API_KEYS"

#: Optional pepper mixed into :func:`hash_api_key`. Empty by default, which is
#: honest rather than ideal — see the function's docstring.
_PEPPER_VAR = "API_KEY_HASH_PEPPER"

#: Characters of the digest kept in a key id. 16 hex chars is 64 bits: far too
#: wide to collide across the handful of keys a deployment has, and short enough
#: that a log line or an audit entry stays readable.
_KEY_ID_LENGTH = 16

#: ``auto_error=False`` because FastAPI's built-in 403-on-missing-header is the
#: wrong answer twice over: wrong status (a missing credential is a 401) and
#: wrong body (a bare string, not this API's ``{"detail": "<slug>"}`` contract).
#: Declaring the scheme anyway is what puts the padlock on ``/docs``.
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def hash_api_key(api_key: str) -> str:
    """Return a stable, non-reversible identity for ``api_key``.

    This is what gets logged and written to the audit trail. The raw key never
    goes anywhere but memory: a log file is copied, shipped to a log aggregator
    and read by people who should not thereby acquire the ability to call the
    API, and an audit trail that hands out working credentials is worse than no
    audit trail.

    HMAC-SHA-256 keyed by ``API_KEY_HASH_PEPPER``, truncated to
    :data:`_KEY_ID_LENGTH` hex characters. With no pepper set — the default —
    this is deterministic across processes, which is what makes an audit entry
    correlatable with a log line, and equally what makes it *reversible for a
    guessable key*: an attacker holding the audit log can hash candidate keys
    until one matches. Setting a pepper closes that, at the cost of making key
    ids incomparable across deployments that use different peppers.

    Args:
        api_key: The raw key as presented by the caller.

    Returns:
        A short identity string such as ``"hmac-sha256:1f3a9c02b8e7d456"``.
    """
    pepper = os.getenv(_PEPPER_VAR, "").encode("utf-8")
    digest = hmac.new(pepper, api_key.encode("utf-8"), sha256).hexdigest()
    return f"hmac-sha256:{digest[:_KEY_ID_LENGTH]}"


@dataclass(frozen=True)
class Principal:
    """Who is making this request, once the key has been recognised.

    Deliberately does *not* carry the raw key. Handlers receive a
    :class:`Principal`, log it and audit it, and there is no path from one back
    to a usable credential — so an accidental ``logger.info("%s", principal)``
    cannot leak anything.

    Attributes:
        key_id: Hashed identity from :func:`hash_api_key`.
        role: The role the key was configured with.
    """

    key_id: str
    role: Role

    def __str__(self) -> str:
        return f"{self.role}/{self.key_id}"


@dataclass(frozen=True)
class AuthConfig:
    """The configured keys, per role.

    Built from the environment by :meth:`from_env` and reached through
    :func:`get_auth_config`, which is a FastAPI dependency precisely so a test
    can substitute a config it controls instead of writing keys into the real
    environment.

    Attributes:
        viewer_keys: Keys granted the ``viewer`` role.
        operator_keys: Keys granted the ``operator`` role.
    """

    viewer_keys: tuple[str, ...] = ()
    operator_keys: tuple[str, ...] = ()

    @property
    def is_configured(self) -> bool:
        """Whether any key at all exists. False means every gated route 503s."""
        return bool(self.viewer_keys or self.operator_keys)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> AuthConfig:
        """Read ``VIEWER_API_KEYS`` and ``OPERATOR_API_KEYS`` as comma-separated lists.

        Args:
            environ: Mapping to read. Defaults to ``os.environ``.
            use_dotenv: Whether to load ``<repo>/.env`` first. Values already
                exported in the real environment win, matching
                :meth:`app.models.config.ModelConfig.from_env` — so a container
                that injects ``OPERATOR_API_KEYS`` is not overridden by a stale
                ``.env`` left in the image.

        Returns:
            An :class:`AuthConfig`. Blank entries and surrounding whitespace are
            dropped, so ``VIEWER_API_KEYS="a, b,"`` yields two keys.
        """
        if use_dotenv and environ is None and _DOTENV_PATH.is_file():
            load_dotenv(_DOTENV_PATH, override=False)

        source = os.environ if environ is None else environ
        config = cls(
            viewer_keys=_split_keys(source.get(_VIEWER_KEYS_VAR)),
            operator_keys=_split_keys(source.get(_OPERATOR_KEYS_VAR)),
        )

        if not config.is_configured:
            logger.error(
                "No API keys configured (%s / %s are unset or blank). Every authenticated "
                "endpoint will refuse requests with 503 auth_not_configured; only /health "
                "will answer. Set them in .env — see .env.example.",
                _VIEWER_KEYS_VAR,
                _OPERATOR_KEYS_VAR,
            )
        else:
            # Counts, never the keys. Enough to spot "I set the wrong variable"
            # from a startup log without putting a credential in one.
            logger.info(
                "Auth configured: %d viewer key(s), %d operator key(s).",
                len(config.viewer_keys),
                len(config.operator_keys),
            )
        return config

    def principal_for(self, api_key: str) -> Principal | None:
        """Resolve a presented key to a :class:`Principal`, or ``None`` if unknown.

        Operator keys are checked first, so a key listed under both roles gets
        the higher privilege rather than depending on iteration order.

        Comparison is :func:`hmac.compare_digest` rather than ``==`` or a dict
        lookup. Both of those short-circuit on the first differing byte, which
        makes the time to reject a candidate a function of how much of the real
        key it got right — enough, over many requests, to recover a key one byte
        at a time. The loop does still return early on a *match*, which leaks
        only which position in a list of a handful of keys matched, and that is
        not a secret.
        """
        for role, keys in (("operator", self.operator_keys), ("viewer", self.viewer_keys)):
            for known in keys:
                if hmac.compare_digest(api_key, known):
                    return Principal(key_id=hash_api_key(api_key), role=role)  # type: ignore[arg-type]
        return None


def _split_keys(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated key list, dropping blanks and whitespace."""
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


#: Process-wide config, resolved from the environment on first use. Memoised so
#: every request does not re-read ``.env`` off disk.
_cached: AuthConfig | None = None


def get_auth_config() -> AuthConfig:
    """FastAPI dependency returning the process-wide :class:`AuthConfig`.

    Takes no arguments on purpose: FastAPI reads a dependency's signature and
    turns any plain parameter into a *query parameter*, so a ``refresh`` flag
    here would appear on every gated endpoint's public contract. Callers that
    want a fresh read (scripts, tests) build one with
    :meth:`AuthConfig.from_env` or override this dependency.
    """
    global _cached  # noqa: PLW0603 - module-level memoisation of a frozen value
    if _cached is None:
        _cached = AuthConfig.from_env()
    return _cached


def require_role(role: Role) -> Callable[..., Principal]:
    """Build a dependency that admits ``role`` and anything that outranks it.

    Args:
        role: The minimum role a caller must hold.

    Returns:
        A FastAPI dependency yielding the authenticated :class:`Principal`.
        Handlers that take it get the caller's identity for free, which is what
        lets ``/benchmark`` write an audit entry naming who ran it.

    Example:
        >>> @app.post("/benchmark")                                  # doctest: +SKIP
        ... def benchmark(principal: Principal = Depends(require_operator)): ...
    """
    if role not in _ROLE_GRANTS:
        msg = f"Unknown role {role!r}; expected one of {sorted(_ROLE_GRANTS)}."
        raise ValueError(msg)

    def dependency(
        api_key: str | None = Security(_api_key_scheme),
        config: AuthConfig = Depends(get_auth_config),
    ) -> Principal:
        return _authorize(api_key, config, required=role)

    # So OpenAPI and any dependency-graph dump name the guard rather than
    # showing three identical `dependency` entries.
    dependency.__name__ = f"require_{role}"
    dependency.__doc__ = f"Admit callers holding the {role!r} role or higher."
    return dependency


def _authorize(api_key: str | None, config: AuthConfig, *, required: Role) -> Principal:
    """The whole check, in one place, so both roles fail identically.

    Raises:
        HTTPException: 503 if the server has no keys configured, 401 if the
            caller presented no key or an unrecognised one, 403 if their key is
            valid but outranked by the endpoint.
    """
    if not config.is_configured:
        # Logged every time rather than once: this is a broken deployment, and
        # the line is what tells an operator why their API is answering 503.
        logger.error("Rejecting an authenticated request: no API keys are configured on this server.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="auth_not_configured",
        )

    if api_key is None or not api_key.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_api_key")

    principal = config.principal_for(api_key.strip())
    if principal is None:
        # The presented key is hashed, not printed. An unrecognised key is very
        # often a *valid* key for another environment — logging it verbatim is
        # how a staging credential ends up in a production log aggregator.
        logger.warning("Rejected an unknown API key (%s).", hash_api_key(api_key.strip()))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")

    if required not in _ROLE_GRANTS[principal.role]:
        logger.warning(
            "Denied %s: role %r does not grant %r.",
            principal,
            principal.role,
            required,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient_role")

    return principal


#: The two guards the routes actually use. Built once at import so FastAPI sees
#: the same callable object on every route that shares a role.
require_viewer = require_role("viewer")
require_operator = require_role("operator")
