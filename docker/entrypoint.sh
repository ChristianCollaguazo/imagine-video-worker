#!/usr/bin/env bash
set -euo pipefail
cd /opt/imagine-video-worker
: "${IMAGINE_VIDEO_TOKEN:?Configure IMAGINE_VIDEO_TOKEN in RunPod Secrets}"
export IMAGINE_VIDEO_TOKEN
[[ "${IMAGINE_DOWNLOAD_MODELS:-0}" == 0 ]] || { echo "This image requires pre-staged H3 models" >&2; exit 1; }
boot_started=$SECONDS
export IMAGINE_VIDEO_RUN_ROOT=/tmp/imagine-video-worker
# Fail fast on incompatible GPU before copying large weights.
.venv-comfy/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
stage_started=$SECONDS
.venv-comfy/bin/python docker/stage_models.py
if [[ "${IMAGINE_DEBUG_METRICS:-0}" == "1" ]]; then
  printf 'imagine_metric {"event":"model_staging_finished","seconds":%s}\n' "$((SECONDS-stage_started))"
fi
cleanup() {
  bash scripts/stop.sh || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
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
      echo "$service exited; stopping container" >&2
      exit 1
    fi
  done
  sleep 3 & wait $!
done
