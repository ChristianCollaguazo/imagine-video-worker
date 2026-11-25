#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${IMAGINE_VIDEO_RUN_ROOT:-/tmp/imagine-video-worker}"
[[ "$RUN_ROOT" == /tmp/* ]] || { echo "IMAGINE_VIDEO_RUN_ROOT must stay under /tmp" >&2; exit 1; }
export COMFY_ROOT="$ROOT/ComfyUI"
export COMFY_INPUT_DIR="$RUN_ROOT/comfy/input"
export COMFY_OUTPUT_DIR="$RUN_ROOT/comfy/output"
export COMFY_TEMP_DIR="$RUN_ROOT/comfy/temp"
export IMAGINE_VIDEO_DATA="$RUN_ROOT/jobs"
comfy_args=()
case "${IMAGINE_H3_VRAM_PROFILE:-auto}" in
  auto) ;;
  high)
    "${ROOT}/.venv-comfy/bin/python" -c 'import torch; assert torch.cuda.get_device_properties(0).total_memory >= 90 * 1024**3, "High VRAM profile requires at least 90 GiB VRAM"'
    comfy_args+=(--highvram --reserve-vram 12)
    ;;
  *) echo "IMAGINE_H3_VRAM_PROFILE must be auto or high" >&2; exit 1 ;;
esac
case "${IMAGINE_H3_AUDIO:-1}" in
  0|1) ;;
  *) echo "IMAGINE_H3_AUDIO must be 0 or 1" >&2; exit 1 ;;
esac
case "${IMAGINE_SAGE_ATTENTION:-on}" in
  auto|on|off) ;;
  *) echo "IMAGINE_SAGE_ATTENTION must be auto, on, or off" >&2; exit 1 ;;
esac

: "${IMAGINE_VIDEO_TOKEN:?Set IMAGINE_VIDEO_TOKEN in the process environment}"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 1; }
"$ROOT/.venv-comfy/bin/python" -c 'import torch, torchaudio; assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"; assert torch.cuda.is_bf16_supported(), "GPU does not report BF16 support"'
attention_status="$(PYTHONPATH="$ROOT" "$ROOT/.venv-comfy/bin/python" -m app.attention_backend)"
attention_backend="$(printf '%s' "$attention_status" | "$ROOT/.venv-comfy/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["backend"])')"
export IMAGINE_SELECTED_ATTENTION_BACKEND="$attention_backend"
if [[ "$attention_backend" == sage ]]; then comfy_args+=(--use-sage-attention); fi
printf 'imagine_attention %s\n' "$attention_status"
"$ROOT/.venv-comfy/bin/python" -c 'import socket; a=socket.socket(); a.bind(("127.0.0.1",8188)); b=socket.socket(); b.bind(("0.0.0.0",8000)); a.close(); b.close()' || { echo "A required service port is already in use" >&2; exit 1; }
mkdir -p "$COMFY_INPUT_DIR" "$COMFY_OUTPUT_DIR" "$COMFY_TEMP_DIR" "$IMAGINE_VIDEO_DATA" "$RUN_ROOT/pids"
chmod 700 "$RUN_ROOT" "$RUN_ROOT/jobs" "$RUN_ROOT/pids"

process_matches() {
  local name="$1" pid="$2" command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$name" == "comfyui" ]]; then
    [[ "$command_line" == *"$COMFY_ROOT/main.py"* ]]
  else
    [[ "$command_line" == *"uvicorn app.main:app"* && "$command_line" == *"$ROOT/.venv-comfy/bin/uvicorn"* ]]
  fi
}

for name in worker comfyui; do
  pidfile="$RUN_ROOT/pids/$name.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile")"
    if kill -0 "$old_pid" 2>/dev/null; then
      if process_matches "$name" "$old_pid"; then
        echo "$name is already running (PID $old_pid)" >&2
      else
        echo "Stale $name PID file points at unrelated PID $old_pid; refusing to touch it" >&2
      fi
      exit 1
    fi
    rm -f "$pidfile"
  fi
done

comfy_pid=""
worker_pid=""
cleanup_on_error() {
  result=$?
  if [[ "$result" -ne 0 ]]; then
    for pid in "$worker_pid" "$comfy_pid"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        if [[ "$pid" == "$worker_pid" ]]; then owned_name=worker; else owned_name=comfyui; fi
        if process_matches "$owned_name" "$pid"; then kill -TERM "$pid" 2>/dev/null || true; fi
      fi
    done
    [[ -n "$worker_pid" ]] && wait "$worker_pid" 2>/dev/null || true
    [[ -n "$comfy_pid" ]] && wait "$comfy_pid" 2>/dev/null || true
    rm -f "$RUN_ROOT/pids/worker.pid" "$RUN_ROOT/pids/comfyui.pid"
  fi
}
trap cleanup_on_error EXIT

"$ROOT/.venv-comfy/bin/python" "$ROOT/scripts/check_models.py"

cd "$ROOT"
# Only allow fixed, upstream technical events on stdout. Comfy errors and
# payload-derived strings stay out of container logs.
log_comfy() {
  local line
  while IFS= read -r line; do
    case "$line" in
      'Using pytorch attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"pytorch_sdpa"}' ;;
      'Using sage attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"sage"}' ;;
      'Using Flash Attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"flash"}' ;;
      'Using xformers attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"xformers"}' ;;
      'Using Comfy Kitchen attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"comfy_kitchen"}' ;;
      'Using split optimization for attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"split"}' ;;
      'Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention') echo 'imagine_metric {"event":"comfy_attention_selected","backend":"sub_quad"}' ;;
    esac
  done
}
run_comfy() {
  exec "$ROOT/.venv-comfy/bin/python" "$COMFY_ROOT/main.py" \
    --base-directory "$COMFY_ROOT" "${comfy_args[@]}" \
    --input-directory "$COMFY_INPUT_DIR" \
    --output-directory "$COMFY_OUTPUT_DIR" \
    --temp-directory "$RUN_ROOT/comfy" \
    --database-url sqlite:///:memory: \
    --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-metadata
}
comfy_start_started=$SECONDS
if [[ "${IMAGINE_DEBUG_METRICS:-0}" == "1" ]]; then
  run_comfy > >(log_comfy) 2>&1 &
else
  run_comfy >/dev/null 2>&1 &
fi
comfy_pid=$!
echo "$comfy_pid" > "$RUN_ROOT/pids/comfyui.pid"

healthy=0
for _ in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8188/system_stats >/dev/null; then healthy=1; break; fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then echo "ComfyUI did not become healthy" >&2; exit 1; fi
if [[ "${IMAGINE_DEBUG_METRICS:-0}" == "1" ]]; then
  printf 'imagine_metric {"event":"comfy_available","seconds":%s}\n' "$((SECONDS-comfy_start_started))"
fi

# Provider ingress must terminate TLS and expose only this service on HTTPS port 8000.
api_start_started=$SECONDS
PYTHONPATH="$ROOT" "$ROOT/.venv-comfy/bin/uvicorn" app.main:app \
  --host 0.0.0.0 --port 8000 --no-access-log 2>/dev/null &
worker_pid=$!
echo "$worker_pid" > "$RUN_ROOT/pids/worker.pid"

worker_health() {
  printf 'url = "http://127.0.0.1:8000/v1/health"\nheader = "Authorization: Bearer %s"\nfail\nsilent\nshow-error\n' \
    "$IMAGINE_VIDEO_TOKEN" | curl --config - --output /dev/null --max-time 10
}

healthy=0
for _ in $(seq 1 30); do
  if worker_health; then healthy=1; break; fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then
  echo "Worker did not become healthy; services stopped" >&2
  exit 1
fi
if [[ "${IMAGINE_DEBUG_METRICS:-0}" == "1" ]]; then
  printf 'imagine_metric {"event":"api_available","seconds":%s}\n' "$((SECONDS-api_start_started))"
fi
trap - EXIT
echo "Worker ready at http://127.0.0.1:8000/v1/health (expose only through HTTPS provider ingress)."
