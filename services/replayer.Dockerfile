# syntax=docker/dockerfile:1
# M5 CPU-only image (PLAN §5): the main + serve groups from uv.lock, without the gnn group.
# A Debian glibc base like Modal's debian_slim, so math.log1p reproduces the bundle's bits: the
# codename's glibc must equal serving/metadata.json -> platform.libc_version (2.36 = bookworm).
# scorer.Dockerfile and replayer.Dockerfile differ only in EXPOSE and CMD (shared layers).
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
# The code runs from /app/src via PYTHONPATH: no build backend, no README in the context.
RUN uv sync --frozen --no-default-groups --group serve --no-install-project
COPY src ./src
COPY configs ./configs
# Fails the build on a missing import or a torch install; prints versions, libc, log1p digest.
RUN python -m aml.serving.scorer --self-check
CMD ["python", "-m", "aml.streaming.replayer"]
