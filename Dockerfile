# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/imagine-hf IMAGINE_GLOBAL_MOUNT=/workspace-global \
    IMAGINE_DOWNLOAD_MODELS=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git ca-certificates curl ffmpeg \
    build-essential procps util-linux tini libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/imagine-video-worker
COPY config/ ./config/
COPY requirements.txt ./requirements.txt
# Installation only: GPU availability is checked at container startup, not build.
RUN python3 -m venv .venv-comfy && \
    .venv-comfy/bin/python -m pip install --upgrade pip && \
    python3 -c 'import json,subprocess; m=json.load(open("config/model-manifest.json")); r=m["runtime"]; subprocess.check_call(["git","clone","--depth","1","--branch",m["comfyui"]["ref"],m["comfyui"]["repository"],"ComfyUI"]); subprocess.check_call([".venv-comfy/bin/python","-m","pip","install",*[f"{k}=={r[k]}" for k in ("torch","torchvision","torchaudio")],"--index-url",r["torch_index_url"]])' && \
    .venv-comfy/bin/python -m pip install -r ComfyUI/requirements.txt -r requirements.txt huggingface_hub && \
    .venv-comfy/bin/python -m pip check
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
RUN bash -n docker/entrypoint.sh && .venv-comfy/bin/python -m py_compile docker/stage_models.py
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/opt/imagine-video-worker/docker/entrypoint.sh"]
