# syntax=docker/dockerfile:1
# Official LightX2V runtime; immutable digest from Docker Hub's official 26052801 CUDA 13 build.
FROM lightx2v/lightx2v@sha256:a7e92ffa9c9a5320f8b68dc5b733abbcf56062d13d472548bb2f11769359af97

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/imagine-hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    IMAGINE_GLOBAL_MOUNT=/workspace-global IMAGINE_DOWNLOAD_MODELS=0 \
    LIGHTWAN_MODEL_DIR=/opt/imagine/models/Wan2.2-I2V-A14B \
    IMAGINE_VIDEO_DATA=/tmp/imagine-video-worker/jobs \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 PROFILING_DEBUG_LEVEL=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git util-linux tini libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/imagine-video-worker
ENV PYTHONPATH=/opt/LightX2V:/opt/imagine-video-worker
COPY config/ ./config/
COPY requirements.txt ./requirements.txt
RUN git init /opt/LightX2V && cd /opt/LightX2V && \
    git remote add origin https://github.com/ModelTC/LightX2V.git && \
    git fetch --depth 1 origin tag 0.6.0 && git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "94f0f19459e66f61a723db7690f1707f308599ef" && \
    rm -rf .git docs examples scripts tests lightx2v_kernel && \
    python -m pip install -r /opt/imagine-video-worker/requirements.txt && \
    python -c "import fastapi, gguf, huggingface_hub, prometheus_client, sentencepiece, uvicorn; from PIL import Image" && \
    command -v ffmpeg && command -v ffprobe

ARG IMAGINE_SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.revision=$IMAGINE_SOURCE_COMMIT \
      org.opencontainers.image.source=https://github.com/ChristianCollaguazo/imagine-video-worker
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
RUN bash -n docker/entrypoint.sh && bash -n scripts/start.sh && \
    python -m py_compile app/main.py app/lightwan.py docker/stage_models.py

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/opt/imagine-video-worker/docker/entrypoint.sh"]
