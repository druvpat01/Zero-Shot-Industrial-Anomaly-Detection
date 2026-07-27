#!/usr/bin/env python3
"""End-to-end demo of the serving layer, over real HTTP.

Usage::

    python scripts/run_api_demo.py                          # bottle, patchcore
    python scripts/run_api_demo.py --backend onnx_patchcore  # the exported graph
    python scripts/run_api_demo.py --backend winclip         # zero-shot, slow
    python scripts/run_api_demo.py --base-url http://localhost:8000   # already running

What it does, and why it is a subprocess
========================================
Starts ``uvicorn app.serving.main:app`` on a free port, waits for ``/health`` to
answer, sends five real MVTec bottle images — a mix of clean and defective —
through ``POST /predict``, prints what came back, and asserts the defective ones
scored higher than the clean ones.

The server runs as a real child process rather than in a thread or through
Starlette's ``TestClient``, and that is the point of the script: ``TestClient``
short-circuits the ASGI interface in-process, so it cannot catch anything that
breaks between "the handler returns" and "bytes arrive over a socket" — JSON
serialisation of a NumPy scalar, a response body larger than a buffer, a worker
that dies on import. ``tests/test_api.py`` covers the handlers; this covers the
deployment. It is also why the health poll below allows a generous timeout: the
child pays the full ``import torch, anomalib`` cost before it can answer, which
is tens of seconds on a cold filesystem cache.

The assertion is a smoke test, not a metric
===========================================
Five images say nothing about AUROC — ``scripts/run_benchmark.py`` measures that
properly over the full test split. What "every defect scored above every clean
part" catches is the class of bug that makes a served model *look* fine and be
wrong: a channel-order slip in the API's decode path, a stale checkpoint, an
ONNX graph exported from a different category. All of those leave a model that
still returns plausible numbers in the right range, and all of them collapse the
separation this checks.

Authentication
==============
``/models`` and ``/predict`` are gated (see ``docs/security.md``), so the demo
needs a key. By default it reads the *same* ``.env`` the child server will read
and uses the first operator key it finds, which means a working ``.env`` is the
only setup step; ``--api-key`` overrides it, and is the way to watch the demo
fail with a 401 or a 403 on purpose.

Exits non-zero if the assertion fails, so it can be a CI step.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Allow `python scripts/run_api_demo.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.datamodule import DEFAULT_DATA_ROOT  # noqa: E402
from app.serving.auth import API_KEY_HEADER, AuthConfig  # noqa: E402
from app.serving.schemas import MODEL_BACKENDS  # noqa: E402

logger = logging.getLogger("run_api_demo")

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Defect folders to draw from, in preference order. MVTec splits a category's
#: defects by type; taking from several exercises more than one failure mode.
_DEFECT_SUBDIRS = ("broken_large", "broken_small", "contamination")

#: The demo's sample: three defective frames against two clean ones. Weighted
#: toward defects because they are the harder call — a model that flags nothing
#: passes a clean-only demo perfectly.
_NUM_DEFECTIVE = 3
_NUM_CLEAN = 2

#: How long to wait for the child server to answer /health. Generous because the
#: first request into a fresh process imports torch and anomalib.
_STARTUP_TIMEOUT_SECONDS = 180.0

#: Per-request timeout. WinCLIP is the reason it is minutes rather than seconds:
#: a cold load is ~830 MB of CLIP weights plus a warm-up pass.
_REQUEST_TIMEOUT_SECONDS = 600.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default="bottle", help="MVTec AD category (default: %(default)s).")
    parser.add_argument(
        "--backend",
        default="patchcore",
        choices=MODEL_BACKENDS,
        help="Model backend to demo (default: %(default)s).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface the demo server binds (default: %(default)s).")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: a free one).")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Talk to an already-running server at this URL instead of starting one.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "X-API-Key to send. Defaults to the first OPERATOR_API_KEYS entry in the "
            "environment or .env; pass a viewer key to watch /models 403 instead."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for this script (default: %(default)s).",
    )
    parser.add_argument(
        "--server-log",
        action="store_true",
        help="Stream the child server's log to this terminal (it is suppressed by default).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def collect_samples(data_root: Path, category: str) -> list[tuple[Path, bool]]:
    """Pick the demo's five frames: ``(path, is_defective)``, defects first.

    Deterministic — sorted filenames, fixed folders — so two runs of the demo
    are comparable and a regression shows up as a changed number rather than a
    changed sample.
    """
    test_dir = data_root / category / "test"
    if not test_dir.is_dir():
        msg = (
            f"No test images at {test_dir}. Run "
            f"`python scripts/download_dataset.py --category {category}` first."
        )
        raise SystemExit(msg)

    defective: list[Path] = []
    for subdir in _DEFECT_SUBDIRS:
        for path in sorted((test_dir / subdir).glob("*.png")):
            defective.append(path)
            break  # one per defect type, so the sample spans failure modes
    defective = defective[:_NUM_DEFECTIVE]

    clean = sorted((test_dir / "good").glob("*.png"))[:_NUM_CLEAN]

    if len(defective) < _NUM_DEFECTIVE or len(clean) < _NUM_CLEAN:
        msg = (
            f"Need {_NUM_DEFECTIVE} defective and {_NUM_CLEAN} clean images under {test_dir}; "
            f"found {len(defective)} and {len(clean)}."
        )
        raise SystemExit(msg)

    return [(path, True) for path in defective] + [(path, False) for path in clean]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _free_port(host: str) -> int:
    """Ask the OS for an unused port, so parallel runs cannot collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def start_server(host: str, port: int, *, stream_log: bool) -> subprocess.Popen:
    """Launch ``uvicorn app.serving.main:app`` as a child process."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.serving.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    logger.info("Starting server: %s", " ".join(command))
    # PYTHONPATH so the child imports `app` from the repo without an install,
    # matching how this script was itself invoked.
    env = {**os.environ, "PYTHONPATH": f"{_REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    sink = None if stream_log else subprocess.DEVNULL
    return subprocess.Popen(command, cwd=_REPO_ROOT, env=env, stdout=sink, stderr=sink)


def wait_for_health(base_url: str, process: subprocess.Popen | None, timeout: float) -> dict:
    """Poll ``GET /health`` until it answers, or give up.

    Also watches the child for an early exit: a server that died on import will
    never answer, and reporting *that* beats a three-minute timeout with no
    explanation.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            msg = (
                f"The server exited with code {process.returncode} before becoming healthy. "
                f"Re-run with --server-log to see why."
            )
            raise SystemExit(msg)
        try:
            return _get(f"{base_url}/health", timeout=5.0)
        except (URLError, HTTPError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.5)
    msg = f"Server at {base_url} did not become healthy within {timeout:.0f}s (last error: {last_error})."
    raise SystemExit(msg)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _default_api_key() -> str | None:
    """The first operator key the environment (or ``.env``) offers, else a viewer key.

    Reading the server's own configuration rather than taking a flag means the
    demo works straight after ``cp .env.example .env`` — the child server and
    this script agree by construction, instead of by the operator remembering to
    paste the same string twice.
    """
    config = AuthConfig.from_env()
    for keys in (config.operator_keys, config.viewer_keys):
        if keys:
            return keys[0]
    return None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {API_KEY_HEADER: api_key} if api_key else {}


def _get(url: str, *, timeout: float, api_key: str | None = None) -> dict:
    request = Request(url, headers=_auth_headers(api_key), method="GET")  # noqa: S310 - fixed localhost URL
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _post(url: str, payload: dict, *, timeout: float, api_key: str | None = None) -> tuple[int, dict]:
    """POST JSON and return ``(status, body)``, treating an error status as data.

    The API's failure modes are structured JSON, so a 401, 422 or 503 is something
    to print rather than an exception to raise — ``urlopen`` disagrees, hence the
    ``HTTPError`` catch.
    """
    request = Request(  # noqa: S310 - fixed localhost URL
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **_auth_headers(api_key)},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo(
    base_url: str,
    category: str,
    backend: str,
    samples: list[tuple[Path, bool]],
    api_key: str | None = None,
) -> int:
    """Score every sample, print the table, and check the separation. Returns an exit code."""
    print(f"\nServer     : {base_url}")
    print(f"Category   : {category}")
    print(f"Backend    : {backend}")
    print(f"Images     : {len(samples)} ({sum(is_bad for _, is_bad in samples)} defective, "
          f"{sum(not is_bad for _, is_bad in samples)} clean)")

    try:
        rows = _get(f"{base_url}/models?category={category}", timeout=30.0, api_key=api_key)
    except HTTPError as exc:
        # 401/403 here means the key is missing or is a viewer's. Worth naming,
        # because the alternative is a stack trace that says only "HTTP Error".
        print(
            f"\nGET /models -> HTTP {exc.code}: {json.loads(exc.read() or b'{}')}. "
            f"/models needs an operator key — set OPERATOR_API_KEYS in .env or pass --api-key.",
        )
        return 1

    available = {row["backend"]: row for row in rows}
    row = available.get(backend, {})
    if not row.get("available", False):
        print(f"\n{backend!r} is not available for {category!r}: {row.get('detail')}")
        return 1

    header = f"{'Image':<34} | {'Truth':<9} | {'Score':>7} | {'Defective':<9} | {'Latency':>9} | {'Model'}"
    print()
    print(header)
    print("-" * len(header))

    scores: dict[bool, list[float]] = {True: [], False: []}
    for path, is_defective in samples:
        payload = {
            "category": category,
            "model_backend": backend,
            "image_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        status_code, body = _post(
            f"{base_url}/predict",
            payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            api_key=api_key,
        )

        label = f"{path.parent.name}/{path.name}"
        if status_code != 200:
            print(f"{label:<34} | {'defect' if is_defective else 'clean':<9} | HTTP {status_code}: {body}")
            return 1

        scores[is_defective].append(float(body["anomaly_score"]))
        print(
            f"{label:<34} | "
            f"{('defect' if is_defective else 'clean'):<9} | "
            f"{body['anomaly_score']:>7.4f} | "
            f"{str(body['is_defective']):<9} | "
            f"{body['latency_ms']:>7.1f}ms | "
            f"{body['model_name']}",
        )

    return _report_separation(scores)


def _report_separation(scores: dict[bool, list[float]]) -> int:
    """Assert every defect outscored every clean part, and explain either outcome."""
    worst_defect = min(scores[True])
    best_clean = max(scores[False])
    margin = worst_defect - best_clean

    print()
    print(f"Lowest defective score : {worst_defect:.4f}")
    print(f"Highest clean score    : {best_clean:.4f}")
    print(f"Margin                 : {margin:+.4f}")

    if margin <= 0:
        print(
            "\nFAILED: at least one clean frame scored at or above a defective one. "
            "On this sample that points at the serving path (channel order, a stale "
            "checkpoint, an ONNX graph exported from another category) rather than at "
            "model accuracy — run scripts/run_benchmark.py to separate the two.",
        )
        return 1

    print("\nOK: every defective image scored above every clean one.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)-8s %(name)s: %(message)s")
    # The table below goes to stdout and the log lines to stderr. Piped to a file
    # or a pager, stdout switches to block buffering and the two arrive badly out
    # of order — the "stopping server" line lands above the results it follows.
    sys.stdout.reconfigure(line_buffering=True)

    samples = collect_samples(args.data_root, args.category)

    api_key = args.api_key or _default_api_key()
    if api_key is None:
        msg = (
            "No API key available. /models and /predict are authenticated: set "
            "OPERATOR_API_KEYS in .env (see .env.example) or pass --api-key."
        )
        raise SystemExit(msg)

    process: subprocess.Popen | None = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        port = args.port or _free_port(args.host)
        base_url = f"http://{args.host}:{port}"
        process = start_server(args.host, port, stream_log=args.server_log)

    try:
        health = wait_for_health(base_url, process, _STARTUP_TIMEOUT_SECONDS)
        # Empty at this point, and that is the assertion: startup loaded nothing.
        print(f"\n/health -> {health}")
        exit_code = run_demo(base_url, args.category, args.backend, samples, api_key=api_key)
        print(f"/health -> {_get(f'{base_url}/health', timeout=30.0)}")
        return exit_code
    finally:
        if process is not None:
            logger.info("Stopping server (pid %d)", process.pid)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged child
                logger.warning("Server did not stop on SIGTERM; killing it.")
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
