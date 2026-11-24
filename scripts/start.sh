#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${IMAGINE_VIDEO_RUN_ROOT:-/tmp/imagine-video-worker}"
[[ "$RUN_ROOT" == /tmp/* ]] || { echo "IMAGINE_VIDEO_RUN_ROOT must stay under /tmp" >&2; exit 1; }
: "${IMAGINE_VIDEO_TOKEN:?Set IMAGINE_VIDEO_TOKEN in the process environment}"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"'
python -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0",8000)); s.close()' || { echo "Port 8000 is already in use" >&2; exit 1; }

mkdir -p "$RUN_ROOT/pids" "${IMAGINE_VIDEO_DATA:-$RUN_ROOT/jobs}"
chmod 700 "$RUN_ROOT" "$RUN_ROOT/pids" "${IMAGINE_VIDEO_DATA:-$RUN_ROOT/jobs}"
pidfile="$RUN_ROOT/pids/worker.pid"
if [[ -f "$pidfile" ]]; then
  old_pid="$(cat "$pidfile")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Worker PID file points at a live process ($old_pid); refusing duplicate start" >&2
    exit 1
  fi
  rm -f "$pidfile"
fi

cd "$ROOT"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log 2>/dev/null &
worker_pid=$!
echo "$worker_pid" > "$pidfile"
cleanup() {
  if kill -0 "$worker_pid" 2>/dev/null; then
    kill -TERM "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}
trap cleanup EXIT TERM INT
wait "$worker_pid"
