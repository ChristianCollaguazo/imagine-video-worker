"""Read-only CUDA and disk preflight for the pinned H3 runtime."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "config/model-manifest.json").read_text())
if sys.argv[1:] not in ([], ["--turbo-only"], ["--include-turbo"]):
    raise SystemExit("Usage: preflight.py [--turbo-only|--include-turbo]")
if not shutil.which("nvidia-smi"):
    raise SystemExit("nvidia-smi was not found; use a CUDA Linux host with an NVIDIA GPU")
try:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
except subprocess.CalledProcessError as exc:
    raise SystemExit("nvidia-smi failed; check the host driver/GPU") from exc
print("GPU / VRAM / driver:")
print(gpu.stdout.strip())
if shutil.which("free"):
    subprocess.run(["free", "-h"], check=False)
subprocess.run(["df", "-h", str(root)], check=False)

items = manifest.get("optional_models", []) if sys.argv[1:] == ["--turbo-only"] else manifest["models"]
if sys.argv[1:] == ["--include-turbo"]:
    items = list(items) + manifest.get("optional_models", [])
needed = sum(
    item["bytes_approx"] for item in items
    if not (root / "ComfyUI/models" / item["path"]).is_file()
)
# The optional-only path needs room for the LoRA and its download cache; the
# full install also reserves space for the Python environment and output scratch.
reserve = 5 * 1024**3 if sys.argv[1:] == ["--turbo-only"] else 40 * 1024**3
required = needed + reserve
free = shutil.disk_usage(root).free
print(f"Additional model payload: {needed / 1e9:.1f} GB; free space required: {required / 1e9:.1f} GB")
if free < required:
    raise SystemExit("Insufficient free disk space; attach/expand persistent storage before continuing")
print("Preflight passed. This does not validate generation speed or VRAM fit.")
