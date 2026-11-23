#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MANIFEST="$ROOT/config/model-manifest.json"
COMFY_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["comfyui"]["ref"])' "$MANIFEST")"
TORCH_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["torch"])' "$MANIFEST")"
TORCHVISION_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["torchvision"])' "$MANIFEST")"
TORCHAUDIO_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["torchaudio"])' "$MANIFEST")"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["torch_index_url"])' "$MANIFEST")}"

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap.sh [--install-runtime] [--download-models]

No action runs by default. --install-runtime installs the pinned ComfyUI source
and Python environment on a CUDA Linux host. --download-models downloads the
large BF16 checkpoint files into the persistent ComfyUI models directory.
Review the model license and ensure the provider's budget is sufficient first.
USAGE
}

install_runtime=0
download_models=0
for arg in "$@"; do
  case "$arg" in
    --install-runtime) install_runtime=1 ;;
    --download-models) download_models=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ "$install_runtime" -eq 0 && "$download_models" -eq 0 ]]; then
  usage
  exit 0
fi

if [[ "$install_runtime" -eq 1 || "$download_models" -eq 1 ]]; then
  python3 "$ROOT/scripts/preflight.py"
fi

if [[ "$install_runtime" -eq 1 ]]; then
  command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
  command -v python3 >/dev/null || { echo "Python 3 is required" >&2; exit 1; }
  python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
  if [[ ! -d "$ROOT/ComfyUI/.git" ]]; then
    git clone --depth 1 --branch "$COMFY_REF" https://github.com/Comfy-Org/ComfyUI.git "$ROOT/ComfyUI"
  fi
  python3 -m venv "$ROOT/.venv-comfy"
  "$ROOT/.venv-comfy/bin/python" -m pip install --upgrade pip
  # CUDA 13 wheels support Blackwell and Hopper; versions/index come from the manifest.
  "$ROOT/.venv-comfy/bin/python" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" --index-url "$TORCH_INDEX_URL"
  "$ROOT/.venv-comfy/bin/python" -m pip install -r "$ROOT/ComfyUI/requirements.txt"
  "$ROOT/.venv-comfy/bin/python" -m pip install -r "$ROOT/requirements.txt"
  "$ROOT/.venv-comfy/bin/python" -m pip install "huggingface_hub[cli]"
  command -v ffmpeg >/dev/null || { echo "Install ffmpeg before starting the worker" >&2; exit 1; }
  command -v ffprobe >/dev/null || { echo "Install ffprobe before starting the worker" >&2; exit 1; }
  "$ROOT/.venv-comfy/bin/python" -c 'import torch, torchaudio; assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"; assert torch.cuda.is_bf16_supported(), "GPU does not report BF16 support"; print(torch.cuda.get_device_name(0), "BF16 ready; torchaudio", torchaudio.__version__)'
fi

if [[ "$download_models" -eq 1 ]]; then
  [[ -d "$ROOT/ComfyUI/models" ]] || { echo "Run with --install-runtime first" >&2; exit 1; }
  [[ -x "$ROOT/.venv-comfy/bin/hf" ]] || { echo "Run with --install-runtime first" >&2; exit 1; }
  cd "$ROOT"
  "$ROOT/.venv-comfy/bin/python" scripts/download_models.py
fi
