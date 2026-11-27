#!/usr/bin/env bash
set -euo pipefail
cd /opt/imagine-video-worker
benchmark_mode="${IMAGINE_H3_BENCHMARK_MODE:-0}"
case "$benchmark_mode" in
  0) : "${IMAGINE_VIDEO_TOKEN:?Configure IMAGINE_VIDEO_TOKEN in RunPod Secrets}" ;;
  1) ;;
  *) echo "IMAGINE_H3_BENCHMARK_MODE must be 0 or 1" >&2; exit 1 ;;
esac
export IMAGINE_VIDEO_TOKEN
model_storage_mode="${IMAGINE_MODEL_STORAGE_MODE:-local}"
case "$model_storage_mode" in
  local|download) ;;
  *) echo "IMAGINE_MODEL_STORAGE_MODE must be local or download" >&2; exit 1 ;;
esac
if [[ "$model_storage_mode" == "download" && "${IMAGINE_DOWNLOAD_MODELS:-0}" != "1" ]]; then
  echo "IMAGINE_MODEL_STORAGE_MODE=download requires IMAGINE_DOWNLOAD_MODELS=1" >&2
  exit 1
fi
boot_started=$SECONDS
export IMAGINE_VIDEO_RUN_ROOT=/tmp/imagine-video-worker
# Fail fast on incompatible GPU before copying large weights.
.venv-comfy/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
if [[ "${IMAGINE_REQUIRE_NVENC:-0}" == "1" ]]; then
  .venv-comfy/bin/python scripts/preflight_export.py
fi
if .venv-comfy/bin/python docker/stage_models.py; then
  :
else
  stage_status=$?
  if [[ "$model_storage_mode" == "download" ]]; then
    echo "Pinned model download failed. Check local disk space and Hugging Face access; no Global Volume fallback was attempted." >&2
  else
    echo "H3 model preparation failed; verify the model volume and staging configuration." >&2
  fi
  exit "$stage_status"
fi
cleanup() {
  bash scripts/stop.sh || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$benchmark_mode" == "1" ]]; then
  echo "Model staging completed. Benchmark mode is ready for loopback ComfyUI."
  sleep infinity & wait $!
  exit
fi
bash scripts/start.sh
if [[ "${IMAGINE_DEBUG_METRICS:-0}" == "1" ]]; then
  printf 'imagine_metric {"event":"container_ready","seconds":%s}\n' "$((SECONDS-boot_started))"
fi
# The original start.sh backgrounds both services; keep container alive and
# exit if either process disappears. Tini reaps orphaned child processes.
while true; do
  for service in comfyui worker; do
    pid=$(cat "$IMAGINE_VIDEO_RUN_ROOT/pids/$service.pid")
    if ! kill -0 "$pid" 2>/dev/null || [[ $(ps -p "$pid" -o stat=) == Z* ]]; then
      if [[ "$service" == "comfyui" && "$model_storage_mode" == "download" ]]; then
        echo "ComfyUI exited after local model download mode; check local disk and downloaded model files. No storage fallback was applied." >&2
      else
        echo "$service exited; stopping container" >&2
      fi
      exit 1
    fi
  done
  sleep 3 & wait $!
done
