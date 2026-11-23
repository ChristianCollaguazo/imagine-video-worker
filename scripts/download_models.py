"""Download manifest-pinned H3 files and verify official SHA-256 digests."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "config/model-manifest.json").read_text())
model_root = root / "ComfyUI/models"
for item in manifest["models"]:
    destination = model_root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        hf_cli = root / ".venv-comfy/bin/hf"
        subprocess.run([
            str(hf_cli), "download", "Comfy-Org/MiniMax-H3", item["path"],
            "--revision", manifest["revision"], "--local-dir", str(model_root),
        ], check=True)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != item["sha256"].lower():
        raise SystemExit(f"SHA-256 mismatch for {destination.name}: {actual}")
    print(f"verified {destination.name}")
