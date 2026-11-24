"""Check the local staged LightWan model stack against the manifest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "config/model-manifest.json").read_text())
model_root = Path(os.environ.get("LIGHTWAN_MODEL_DIR", "/opt/imagine/models/Wan2.2-I2V-A14B"))
missing: list[str] = []
invalid: list[str] = []
for item in manifest["models"]:
    path = model_root / item["path"]
    if not path.is_file():
        missing.append(item["path"])
    else:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                invalid.append(item["path"])
if missing or invalid:
    if missing:
        print("Missing files: " + ", ".join(missing))
    if invalid:
        print("SHA-256 mismatch: " + ", ".join(invalid))
    raise SystemExit(1)
print(f"Verified {len(manifest['models'])} LightWan model files")
