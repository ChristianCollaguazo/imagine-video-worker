"""Verify and stage the five existing H3 files onto local POSIX disk."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "ComfyUI" / "models"
MOUNT = Path(os.environ.get("IMAGINE_GLOBAL_MOUNT", "/workspace-global"))
REMOTE = MOUNT / "imagine-h3-turbo" / "models"
BLOCK_SIZE = 8 * 1024**2


def valid(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected


def stage(remote, dest, expected):
    partial = dest.with_name(dest.name + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with remote.open("rb") as source, partial.open("xb") as target:
            while block := source.read(BLOCK_SIZE):
                digest.update(block)
                target.write(block)
        if digest.hexdigest() != expected:
            raise SystemExit(f"Corrupt Global Volume model: {remote}")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def main():
    if not MOUNT.is_absolute() or MOUNT.resolve() == Path("/"):
        raise SystemExit("Invalid IMAGINE_GLOBAL_MOUNT")
    if subprocess.run(["mountpoint", "-q", str(MOUNT)]).returncode:
        raise SystemExit(f"No volume mounted at {MOUNT}; check RunPod mount path")
    manifest = json.loads((ROOT / "config/model-manifest.json").read_text())
    turbo = [item for item in manifest["optional_models"] if item["id"] == "fl2v_turbo_lora_8step"]
    if len(turbo) != 1:
        raise SystemExit("Expected the pinned H3 Turbo LoRA")
    models = []
    for item in manifest["models"] + turbo:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("Invalid manifest model path")
        dest = LOCAL / relative
        remote = REMOTE / relative
        if not remote.is_file():
            raise SystemExit(f"Missing persistent H3 model: {remote}")
        if not valid(dest, item["sha256"]):
            dest.unlink(missing_ok=True)
            dest.with_name(dest.name + ".part").unlink(missing_ok=True)
            models.append((relative, remote, dest, item["sha256"]))

    LOCAL.mkdir(parents=True, exist_ok=True)
    pending = sum(remote.stat().st_size for _, remote, _, _ in models)
    if shutil.disk_usage(LOCAL).free < pending + 10 * 1024**3:
        raise SystemExit("Insufficient local container disk for model staging and scratch")
    for relative, remote, dest, digest in models:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Staging H3 model: {relative}", flush=True)
        stage(remote, dest, digest)
    print("All H3 Turbo weights staged and verified locally", flush=True)


if __name__ == "__main__":
    main()
