"""Stage only the pinned LightWan I2V runtime files from the Global Volume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path(os.environ.get("IMAGINE_GLOBAL_MOUNT", "/workspace-global"))
MODEL_ROOT_NAME = "Wan2.2-I2V-A14B"
LOCAL = Path(os.environ.get("LIGHTWAN_MODEL_DIR", "/opt/imagine/models/Wan2.2-I2V-A14B"))
REMOTE = MOUNT / "imagine-lightwan" / "models" / MODEL_ROOT_NAME
DOWNLOAD = os.environ.get("IMAGINE_DOWNLOAD_MODELS", "0") == "1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_valid(path: Path, item: dict) -> bool:
    if not path.is_file():
        return False
    if item.get("sha256"):
        return digest(path) == item["sha256"].lower()
    expected = item.get("bytes")
    if expected is not None:
        return path.stat().st_size == expected
    expected = item.get("bytes_approx")
    return expected is not None and abs(path.stat().st_size - expected) <= expected * 0.02


def copy_verified(source: Path, destination: Path, item: dict) -> None:
    partial = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(source, partial)
        if not is_valid(partial, item):
            raise SystemExit(f"Integrity check failed for {item['path']}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_verified(item: dict, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(
        repo_id=item["repo"],
        filename=item["name"],
        revision=item["revision"],
        cache_dir="/tmp/imagine-hf",
    ))
    if not is_valid(cached, item):
        raise SystemExit(f"Downloaded file failed integrity check: {item['path']}")
    copy_verified(cached, destination, item)


def main() -> None:
    if not MOUNT.is_absolute() or MOUNT == Path("/"):
        raise SystemExit("Invalid IMAGINE_GLOBAL_MOUNT")
    if subprocess.run(["mountpoint", "-q", str(MOUNT)]).returncode:
        raise SystemExit(f"No volume mounted at {MOUNT}; check RunPod mount path")

    manifest = json.loads((ROOT / "config/model-manifest.json").read_text())
    if manifest.get("model_root") != MODEL_ROOT_NAME or len(manifest.get("models", [])) != 9:
        raise SystemExit("Unexpected LightWan manifest; refusing to stage other model families")
    if any(not item.get("sha256") and not item.get("bytes") for item in manifest["models"]):
        raise SystemExit("Every model object needs a pinned SHA-256 or exact size")

    # Keep staging on local disk. The Global Volume contains only model files.
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    pending = sum(
        item.get("bytes", item.get("bytes_approx", 0))
        for item in manifest["models"]
        if not is_valid(LOCAL / item["path"], item)
    )
    local_download_cache = pending if DOWNLOAD else 0
    if shutil.disk_usage(LOCAL.parent).free < pending + local_download_cache + 12 * 1024**3:
        raise SystemExit("Insufficient local container disk for weights and inference scratch")
    global_pending = sum(
        item.get("bytes", item.get("bytes_approx", 0))
        for item in manifest["models"]
        if not (REMOTE / item["path"]).is_file()
    )
    if global_pending > shutil.disk_usage(MOUNT).free:
        raise SystemExit("Insufficient Global Volume space for the LightWan model files")

    LOCAL.mkdir(parents=True, exist_ok=True)
    for item in manifest["models"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit("Invalid model path in manifest")
        local_file, global_file = LOCAL / rel, REMOTE / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if not is_valid(local_file, item):
            local_file.unlink(missing_ok=True)
            if global_file.is_file():
                print(f"Staging verified model from Global Volume: {rel}", flush=True)
                copy_verified(global_file, local_file, item)
            elif DOWNLOAD:
                print(f"Downloading pinned model object: {rel}", flush=True)
                download_verified(item, local_file)
            else:
                raise SystemExit(f"Missing {global_file}; set IMAGINE_DOWNLOAD_MODELS=1 for initial population")

        if not global_file.is_file():
            if not DOWNLOAD:
                raise SystemExit(f"Missing persistent copy {global_file}; enable initial population")
            print(f"Persisting verified model object: {rel}", flush=True)
            global_file.parent.mkdir(parents=True, exist_ok=True)
            # RunPod Global Volumes can be object-backed; write sequentially, then
            # validate the stored size. The next pod startup re-hashes it on staging.
            with local_file.open("rb") as source, global_file.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024**2)
            expected = item.get("bytes", item.get("bytes_approx", 0))
            if item.get("bytes") is not None and global_file.stat().st_size != expected:
                raise SystemExit(f"Persistent copy size mismatch: {rel}")
            if item.get("bytes_approx") is not None and abs(global_file.stat().st_size - expected) > expected * 0.02:
                raise SystemExit(f"Persistent copy size mismatch: {rel}")
        print(f"Ready: {rel}", flush=True)

    shutil.rmtree("/tmp/imagine-hf", ignore_errors=True)
    print("All LightWan model files staged and verified", flush=True)


if __name__ == "__main__":
    main()
