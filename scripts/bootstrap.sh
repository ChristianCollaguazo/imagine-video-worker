#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap.sh [--preflight] [--download-models]

This branch runs in the LightX2V CUDA Docker image. It does not install a host
runtime. --preflight checks the RunPod GPU and Global Volume without downloading
weights. --download-models populates only imagine-lightwan/models after setting
IMAGINE_DOWNLOAD_MODELS=1; the normal Docker entrypoint performs this on first
startup when that variable is enabled.
USAGE
}

action=""
for arg in "$@"; do
  case "$arg" in
    --preflight|--download-models)
      [[ -z "$action" ]] || { usage >&2; exit 2; }
      action="$arg"
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

case "$action" in
  --preflight) python "$ROOT/scripts/preflight.py" ;;
  --download-models) python "$ROOT/scripts/download_models.py" ;;
  "") usage ;;
esac
