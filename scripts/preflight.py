"""Read-only RunPod GPU, volume and storage preflight for LightWan."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "config/model-manifest.json").read_text())
mount = Path(os.environ.get("IMAGINE_GLOBAL_MOUNT", "/workspace-global"))
models = mount / "imagine-lightwan" / "models" / manifest["model_root"]
if not shutil.which("nvidia-smi"):
    raise SystemExit("nvidia-smi was not found; run this check on a CUDA Linux host")
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
)
print("GPU / VRAM / driver:")
print(gpu.stdout.strip())
if not shutil.which("mountpoint") or subprocess.run(["mountpoint", "-q", str(mount)]).returncode:
    raise SystemExit(f"No Global Volume mounted at {mount}")

missing = sum(
    item.get("bytes", item.get("bytes_approx", 0))
    for item in manifest["models"]
    if not (models / item["path"]).is_file()
)
available = shutil.disk_usage(mount).free
print(f"LightWan model payload missing from volume: {missing / 1e9:.2f} GB")
print(f"Global Volume free: {available / 1e9:.2f} GB")
if missing > available:
    raise SystemExit("Insufficient Global Volume space for LightWan model payload")
print(f"All LightWan model files total: {manifest['download_bytes'] / 1e9:.2f} GB")
print("The H3 model tree is not read or modified by this preflight.")
