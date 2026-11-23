"""Fail fast if a manifest model file is missing; download script checks hashes."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "config/model-manifest.json").read_text())
missing = [
    item["name"] for item in manifest["models"]
    if not (root / "ComfyUI/models" / item["path"]).is_file()
]
if missing:
    raise SystemExit("Missing model files: " + ", ".join(missing))
