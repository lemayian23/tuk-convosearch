# TUK-ConvoSearch API image.
#
# Two decisions that matter for a small Kenyan VPS:
#   1. CPU-only torch. The default PyPI wheel drags ~2GB of CUDA libraries
#      onto a box that has no GPU.
#   2. The MiniLM embedding model is downloaded at build time and baked into
#      the image, so first boot does not depend on Hugging Face being
#      reachable from the data centre.
#
# Location: deploy/Dockerfile

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /srv/app

# libgomp1 is required by faiss; curl by the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image (needs network at build only).
# Deliberately before the source COPY: code changes should not re-download
# ~100MB of model weights.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY scripts ./scripts
COPY pyproject.toml ./

# Never run the API as root.
RUN useradd --create-home --uid 10001 tuk \
    && mkdir -p /srv/app/data /srv/app/corpus \
    && chown -R tuk:tuk /srv/app \
    && chown -R tuk:tuk /opt/models
USER tuk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
