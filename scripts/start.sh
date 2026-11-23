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

: "${IMAGINE_VIDEO_TOKEN:?Set IMAGINE_VIDEO_TOKEN in the process environment}"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 1; }
"$ROOT/.venv-comfy/bin/python" -c 'import torch, torchaudio; assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"; assert torch.cuda.is_bf16_supported(), "GPU does not report BF16 support"'
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
# Prompts and images are transient; do not write ComfyUI/worker logs or prompt metadata to disk.
"$ROOT/.venv-comfy/bin/python" "$COMFY_ROOT/main.py" \
  --base-directory "$COMFY_ROOT" \
  --input-directory "$COMFY_INPUT_DIR" \
  --output-directory "$COMFY_OUTPUT_DIR" \
  --temp-directory "$RUN_ROOT/comfy" \
  --database-url sqlite:///:memory: \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-metadata \
  >/dev/null 2>&1 &
comfy_pid=$!
echo "$comfy_pid" > "$RUN_ROOT/pids/comfyui.pid"

healthy=0
for _ in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8188/system_stats >/dev/null; then healthy=1; break; fi
  sleep 1
done
if [[ "$healthy" -ne 1 ]]; then echo "ComfyUI did not become healthy" >&2; exit 1; fi

# Provider ingress must terminate TLS and expose only this service on HTTPS port 8000.
PYTHONPATH="$ROOT" "$ROOT/.venv-comfy/bin/uvicorn" app.main:app \
  --host 0.0.0.0 --port 8000 >/dev/null 2>&1 &
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
trap - EXIT
echo "Worker ready at http://127.0.0.1:8000/v1/health (expose only through HTTPS provider ingress)."
