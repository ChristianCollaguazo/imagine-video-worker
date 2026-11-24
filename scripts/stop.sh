#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${IMAGINE_VIDEO_RUN_ROOT:-/tmp/imagine-video-worker}"
[[ "$RUN_ROOT" == /tmp/* ]] || { echo "IMAGINE_VIDEO_RUN_ROOT must stay under /tmp" >&2; exit 1; }
pidfile="$RUN_ROOT/pids/worker.pid"
if [[ ! -f "$pidfile" ]]; then
  echo "LightWan API is not running"
  exit 0
fi
pid="$(cat "$pidfile")"
command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
if [[ "$command_line" == *"uvicorn app.main:app"* ]]; then
  kill -TERM "$pid"
  echo "Sent TERM to LightWan API process"
else
  echo "PID file points to an unrelated or stopped process; refusing to signal it" >&2
  exit 1
fi
