# syntax=docker/dockerfile:1
#
# The defect-detection API, built in two stages.
#
#   docker build -f docker/Dockerfile -t defect-detection-api .
#
# Note the build context is the repo root, not docker/ — the image needs
# `app/`, and a Dockerfile that lives beside the thing it builds is worth less
# than one that can see the source. docker-compose.yml sets `context: .` for the
# same reason.
#
# Where the size goes
# ===================
# A naive `pip install -r requirements.txt` on top of python:3.11-slim produces
# an image over 6 GB, and almost none of it is this project. `torch` on PyPI is
# the CUDA build: it drags in ~2.7 GB of `nvidia-*` wheels (cuDNN, cuBLAS, NCCL)
# that a CPU inference container will never open, and a further ~700 MB of
# CUDA-linked libtorch. So stage 1 installs torch and torchvision from PyTorch's
# CPU wheel index *first*, and the later `pip install -r requirements.txt` finds
# them already satisfied and leaves them alone. That single ordering is worth
# more than every other size measure here combined.
#
# The rest is ordinary hygiene: no build toolchain in the runtime stage (it is
# needed to compile a couple of sdists and is then thrown away with the builder),
# no pip cache, no dev dependencies — requirements-dev.txt carries pytest, ruff,
# onnx and datasets, and none of them are installed here. Measured result is
# comfortably under the 2 GB budget; `docker images defect-detection-api` is the
# check, and the assertion in stage 1 is what stops a CUDA wheel quietly
# reintroducing itself when a dependency bumps its torch pin.
#
# What is *not* in the image
# ==========================
# No weights, no dataset. Checkpoints are hundreds of megabytes each and change
# on a different schedule from the code; they arrive on the `./results` volume
# (see docker-compose.yml), which is also why the registry's ONNX fallback
# exists — a deployment can ship exported graphs and no checkpoints at all.

ARG PYTHON_VERSION=3.11

# PyTorch's CPU-only wheel index. Overridable at build time
# (`--build-arg TORCH_INDEX_URL=https://pypi.org/simple`) for a GPU image, which
# will not fit in 2 GB and is not what this Dockerfile is for.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# UID/GID the container runs as. 1000 is the first non-system user on Debian and
# on almost every developer's laptop, which is what makes the bind-mounted
# ./results writable without a chown or a privileged container.
ARG APP_UID=1000
ARG APP_GID=1000


# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve and install every runtime dependency into a venv.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

# A handful of dependencies still publish sdists for some platforms, and a
# compiler missing at install time surfaces as a wall of gcc output rather than
# a clear error. It costs ~250 MB, all of which stays in this stage.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# A venv rather than the system site-packages, so stage 2 can take the whole
# dependency set as one self-contained directory and nothing else.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# CPU torch first — see the header. Unpinned on purpose: the index only carries
# `+cpu` builds, so "latest" here cannot accidentally mean "latest CUDA".
RUN pip install --index-url "${TORCH_INDEX_URL}" torch torchvision

COPY requirements.txt ./
RUN pip install -r requirements.txt

# The 2 GB budget as an assertion rather than a comment. If a dependency ever
# pins an exact torch version, pip will silently replace the CPU build above
# with the CUDA one from PyPI and the image will quadruple; this fails the build
# instead, at the layer that caused it.
RUN python <<'PY'
import sys
import torch

if "+cpu" not in torch.__version__:
    sys.exit(
        f"Expected a CPU-only torch build, got {torch.__version__!r}. Something re-resolved "
        "torch from PyPI, which means the CUDA wheels are back and this image is about to be "
        "4 GB larger than its budget. Look for a dependency pinning an exact torch version."
    )
print(f"torch {torch.__version__}: CPU build confirmed")
PY

# pip compiles .pyc alongside every .py it installs. They are regenerated on
# demand and this venv is read-only at runtime, so they are ~200 MB of files
# nothing will ever open.
RUN find /opt/venv -depth -type d -name '__pycache__' -exec rm -rf {} +

# ~390 MB more that inference will never open. Each line is a specific claim
# rather than a blanket `rm -rf`, because over-trimming a venv produces failures
# that surface hours later as an ImportError inside a request:
#
#   * strip(1) over every shared object — ~200 MB, nearly all of it the symbol
#     table of libtorch_cpu.so (415 MB before, 235 MB after). --strip-unneeded
#     keeps everything the dynamic linker relocates against; what goes is debug
#     and local symbols, which cost a stack trace some names and nothing else.
#   * torch/test and torch/include — gtest binaries and the C++ headers for
#     compiling an extension against libtorch, ~145 MB. This image compiles
#     nothing.
#   * torch/bin — more gtest binaries and a bundled protoc, ~50 MB. Note the one
#     exception: torch_shm_manager stays, because `import torch` resolves its
#     path eagerly and raises RuntimeError when it is missing. That is a
#     memorable way to learn where the line is.
#
# The runtime stage imports the whole stack afterwards, so an over-trim fails
# the build rather than a request.
RUN set -eu; \
    find /opt/venv \( -name '*.so' -o -name '*.so.*' \) -type f -print0 \
        | xargs -0 -r strip --strip-unneeded 2>/dev/null || true; \
    TORCH_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')"; \
    rm -rf "${TORCH_DIR}/test" "${TORCH_DIR}/include"; \
    find "${TORCH_DIR}/bin" -type f ! -name 'torch_shm_manager' -delete


# ---------------------------------------------------------------------------
# Stage 2 — runtime: the venv, the app, and nothing that built them.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_UID
ARG APP_GID

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache

# Non-root, and no login shell: nothing in this image should ever be a session.
RUN groupadd --gid "${APP_GID}" appuser \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# app/ and scripts/ separately rather than `COPY . .`: the context is filtered by
# .dockerignore, but naming what goes in means a new top-level directory does not
# join the image by default.
COPY --chown=${APP_UID}:${APP_GID} app/ ./app/
COPY --chown=${APP_UID}:${APP_GID} scripts/ ./scripts/
COPY --chown=${APP_UID}:${APP_GID} requirements.txt README.md ./

# Mount points, created here so the container starts cleanly when nothing is
# mounted over them — an image that only works under docker-compose is an image
# that cannot be tested on its own. `.cache` is where torch and huggingface_hub
# put downloaded backbone weights (WinCLIP's CLIP, PatchCore's WideResNet); give
# it a volume or every restart re-downloads them.
RUN mkdir -p /app/data /app/results /home/appuser/.cache \
 && chown -R "${APP_UID}:${APP_GID}" /app /home/appuser

# Fail the build, not the first request, when a shared library is missing from
# the slim base or a trim in stage 1 went too far. This is the check that
# catches the opencv-python/-headless mistake (libGL.so.1) and anything like it.
# It imports *and runs* a forward pass, because a stripped libtorch that imports
# cleanly and segfaults on the first matmul is exactly the failure this is here
# to make impossible.
RUN python <<'PY'
import cv2, fastapi, onnxruntime, redis, torch
import anomalib
import torchvision

with torch.no_grad():
    model = torchvision.models.resnet18(weights=None).eval()
    logits = model(torch.randn(1, 3, 64, 64))
print(f"runtime ok: torch {torch.__version__}, anomalib {anomalib.__version__}, "
      f"cv2 {cv2.__version__}, onnxruntime {onnxruntime.__version__}, forward {tuple(logits.shape)}")
PY

USER appuser

EXPOSE 8000

# GET /health is the right probe precisely because it loads nothing: it reports
# liveness in milliseconds whether or not a model is resident, so a container
# warming a 30-second CLIP backbone in the background still passes. Written with
# urllib rather than curl to avoid installing an HTTP client for one line.
#
# --start-period covers the interpreter's cold start: `import torch, anomalib`
# is 15-30 s before uvicorn binds, and failures inside the start period do not
# count against --retries.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)"]

# One worker. Each uvicorn worker keeps its own Prometheus counters and its own
# model cache, so a second one doubles the memory and makes a scrape hit
# whichever process the OS picks — see PROMETHEUS_MULTIPROC_DIR in .env.example
# before raising this.
CMD ["uvicorn", "app.serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
