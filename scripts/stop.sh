#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${IMAGINE_VIDEO_RUN_ROOT:-/tmp/imagine-video-worker}"
[[ "$RUN_ROOT" == /tmp/* ]] || { echo "IMAGINE_VIDEO_RUN_ROOT must stay under /tmp" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY_ROOT="$ROOT/ComfyUI"
process_matches() {
  local name="$1" pid="$2" command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$name" == "comfyui" ]]; then
    [[ "$command_line" == *"$COMFY_ROOT/main.py"* ]]
  else
    [[ "$command_line" == *"uvicorn app.main:app"* && "$command_line" == *"$ROOT/.venv-comfy/bin/uvicorn"* ]]
  fi
}
failed=0
for name in worker comfyui; do
  pidfile="$RUN_ROOT/pids/$name.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pidfile"
      continue
    fi
    if ! process_matches "$name" "$pid"; then
      echo "Stale $name PID file points at unrelated PID $pid; refusing to touch it" >&2
      failed=1
      continue
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$pid" 2>/dev/null || ! process_matches "$name" "$pid"; then break; fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null && process_matches "$name" "$pid"; then
      echo "$name (PID $pid) did not exit; transient files left intact" >&2
      failed=1
      continue
    fi
    rm -f "$pidfile"
  fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi
"$ROOT/.venv-comfy/bin/python" -c 'import socket; a=socket.socket(); a.bind(("127.0.0.1",8188)); b=socket.socket(); b.bind(("0.0.0.0",8000)); a.close(); b.close()' || { echo "A service port remains in use; transient files left intact" >&2; exit 1; }
# Purge only this service's ephemeral inputs, outputs, temporary media and jobs.
for directory in "$RUN_ROOT/comfy/input" "$RUN_ROOT/comfy/output" "$RUN_ROOT/comfy/temp" "$RUN_ROOT/jobs"; do
  if [[ -d "$directory" ]]; then
    find "$directory" -mindepth 1 -delete
  fi
done
