# The public spinf-serve image: spinf-serve from PyPI on python:3.12-slim, nothing else baked in. The model is downloaded from
# the Hugging Face Hub at first start into /root/.cache (mount a volume there to keep it, the compile caches and the official
# packs between runs). torch carries its own CUDA 13 libraries: the host needs an NVIDIA driver supporting CUDA 13 and the
# NVIDIA Container Toolkit, nothing more. gcc at run time: the kernels compile on the host's GPU at first start, then are cached.
#
#   docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve
#   docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve spinf/gemma-4-31b-fp8 --token s3cret
#
# Arguments after the image name are those of `spinf-serve serve` (docs/configuration.md); the environment works too
# (SPINF_MODEL, SPINF_TOKEN, SPINF_PACKS, the limits). Build: docker build --build-arg SPINF_SERVE_VERSION=0.1.2 -t spinf-serve .
FROM python:3.12-slim-bookworm
ARG SPINF_SERVE_VERSION=0.1.2
# expandable_segments: the allocator grows its segments instead of keeping fragments, which a 24 GB card needs at 20 GiB allocated
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 \
    HF_HOME=/root/.cache/huggingface \
    SPINF_MODEL=spinf/gemma-4-12b-fp8 SPINF_HOST=0.0.0.0 SPINF_PORT=8000 \
    PATH=/opt/spinf/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
RUN uv venv /opt/spinf && UV_COMPILE_BYTECODE=1 uv pip install --python /opt/spinf/bin/python --no-cache \
    "spinf-serve[server]==${SPINF_SERVE_VERSION}"
LABEL org.opencontainers.image.title="spinf-serve" \
      org.opencontainers.image.description="The open-source inference engine for decision models: one input read once, hundreds of questions scored in parallel on one GPU" \
      org.opencontainers.image.source="https://github.com/spinfinc/spinf-serve" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${SPINF_SERVE_VERSION}"
VOLUME /root/.cache
EXPOSE 8000
# ready = the model is loaded and the warm-up done: a first start downloads the model (minutes) and compiles (two more)
HEALTHCHECK --interval=15s --timeout=3s --start-period=20m --retries=3 CMD ["curl", "-fs", "http://127.0.0.1:8000/readyz"]
ENTRYPOINT ["spinf-serve", "serve"]
