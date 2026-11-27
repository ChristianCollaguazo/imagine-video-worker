# syntax=docker/dockerfile:1
# Reuse the exact CUDA, PyTorch, ComfyUI and FFmpeg runtime validated in MVP 1.
FROM christiancollaguazo/imagine-video-worker:turbo-v1@sha256:79d667cff51630335cdfe5ba898e17d2c924e15e6888f7b23dd3b6e9201cc83d

ARG IMAGINE_SOURCE_COMMIT=unknown

# Community-built SageAttention wheel.
ARG SAGEATTENTION_WHEEL_URL="https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312/resolve/8109741f23e8e065178bb26710f543e501f5f54f/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
ARG SAGEATTENTION_WHEEL_SHA256="4a47a462059bf09b7d10be68f93a0a462dd6184068298fc6ff10e94751fa6dd4"

LABEL org.opencontainers.image.revision=$IMAGINE_SOURCE_COMMIT \
      org.opencontainers.image.source=https://github.com/ChristianCollaguazo/imagine-video-worker

ENV IMAGINE_DEBUG_METRICS=0 \
    IMAGINE_H3_VRAM_PROFILE=auto \
    IMAGINE_H3_AUDIO=1 \
    IMAGINE_MODEL_STORAGE_MODE=local \
    IMAGINE_DOWNLOAD_MODELS=0 \
    IMAGINE_GLOBAL_MOUNT=/workspace-global \
    IMAGINE_SAGE_ATTENTION=on \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /opt/imagine-video-worker

COPY config/model-manifest.json ./config/model-manifest.json
COPY config/comfy-paths.yaml ./config/comfy-paths.yaml
COPY app/ ./app/
COPY custom_nodes/ ./custom_nodes/
COPY video_export.py h3_runtime.py h3_diagnostics.py ./
COPY scripts/start.sh scripts/check_models.py scripts/stop.sh scripts/preflight_export.py scripts/benchmark_video_export.py scripts/benchmark_h3.py ./scripts/
COPY docker/ ./docker/

RUN set -eux; \
    node_dir="$PWD/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-FirstBlockCache"; \
    mkdir -p "$(dirname "$node_dir")"; \
    if [ ! -d "$node_dir" ]; then \
      git clone --filter=blob:none https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache.git "$node_dir"; \
    fi; \
    git -C "$node_dir" checkout --detach d474755360cc1784df1b7c933ca28195e55e3445; \
    test "$(git -C "$node_dir" rev-parse HEAD)" = d474755360cc1784df1b7c933ca28195e55e3445; \
    test -n "$SAGEATTENTION_WHEEL_URL" && test -n "$SAGEATTENTION_WHEEL_SHA256" || { echo 'SageAttention wheel URL and SHA256 build args are required' >&2; exit 1; }; \
    wheel_name="$(.venv-comfy/bin/python -c 'from urllib.parse import urlparse; import os,sys; print(os.path.basename(urlparse(sys.argv[1]).path))' "$SAGEATTENTION_WHEEL_URL")"; \
    case "$wheel_name" in *.whl) ;; *) echo 'SageAttention URL must point to a .whl file' >&2; exit 1 ;; esac; \
    curl --fail --location --silent --show-error "$SAGEATTENTION_WHEEL_URL" -o "/tmp/$wheel_name"; \
    echo "$SAGEATTENTION_WHEEL_SHA256  /tmp/$wheel_name" | sha256sum --check --status; \
    .venv-comfy/bin/python -m pip install --no-deps "/tmp/$wheel_name"; \
    rm "/tmp/$wheel_name"; \
    bash -n docker/entrypoint.sh && bash -n scripts/start.sh && \
    .venv-comfy/bin/python -m compileall -q app custom_nodes docker scripts video_export.py h3_runtime.py h3_diagnostics.py

ARG BUILDARCH
ARG TARGETARCH
RUN set -eu; \
    .venv-comfy/bin/python -c 'import av, json, runpy; version = runpy.run_path("ComfyUI/comfyui_version.py")["__version__"]; assert "v" + version == json.load(open("config/model-manifest.json"))["comfyui"]["ref"]; assert callable(av.VideoFrame.from_numpy_buffer); av.Codec("h264_nvenc", "w"); av.Codec("libx264", "w")'; \
    command -v ffmpeg; \
    command -v ffprobe; \
    if [ "$BUILDARCH" = "$TARGETARCH" ]; then \
      .venv-comfy/bin/python scripts/preflight_export.py --cpu-smoke; \
      .venv-comfy/bin/python scripts/benchmark_video_export.py --width 160 --height 256 --seconds 1 --encoder libx264; \
    else \
      echo 'Cross-build: run the export smoke tests on a native host.'; \
    fi

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/opt/imagine-video-worker/docker/entrypoint.sh"]
