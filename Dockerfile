# syntax=docker/dockerfile:1
# Reuse the exact CUDA, PyTorch, ComfyUI and FFmpeg runtime validated in MVP 1.
FROM christiancollaguazo/imagine-video-worker:turbo-v1@sha256:79d667cff51630335cdfe5ba898e17d2c924e15e6888f7b23dd3b6e9201cc83d
ARG IMAGINE_SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.revision=$IMAGINE_SOURCE_COMMIT \
      org.opencontainers.image.source=https://github.com/ChristianCollaguazo/imagine-video-worker
ENV IMAGINE_DEBUG_METRICS=0 IMAGINE_H3_VRAM_PROFILE=auto IMAGINE_H3_AUDIO=1 IMAGINE_DOWNLOAD_MODELS=0
WORKDIR /opt/imagine-video-worker
COPY config/model-manifest.json ./config/model-manifest.json
COPY app/ ./app/
COPY scripts/start.sh ./scripts/start.sh
COPY docker/ ./docker/
RUN bash -n docker/entrypoint.sh && bash -n scripts/start.sh && \
    .venv-comfy/bin/python -m compileall -q app docker
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/opt/imagine-video-worker/docker/entrypoint.sh"]
