"""Stage verified H3 Turbo model files onto local POSIX disk; persist only weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "ComfyUI" / "models"
MOUNT = Path(os.environ.get("IMAGINE_GLOBAL_MOUNT", "/workspace-global"))
REMOTE = MOUNT / "imagine-h3-turbo" / "models"
DOWNLOAD = os.environ.get("IMAGINE_DOWNLOAD_MODELS", "0") == "1"
STORAGE_MODE = os.environ.get("IMAGINE_MODEL_STORAGE_MODE", "local").strip().lower()


def valid(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() == expected.lower()


def main():
    if STORAGE_MODE not in {"local", "download"}:
        raise SystemExit("IMAGINE_MODEL_STORAGE_MODE must be local or download")
    if STORAGE_MODE == "download" and not DOWNLOAD:
        raise SystemExit("IMAGINE_MODEL_STORAGE_MODE=download requires IMAGINE_DOWNLOAD_MODELS=1")

    manifest = json.loads((ROOT / "config/model-manifest.json").read_text())
    shared = manifest["models"]
    turbo = [x for x in manifest.get("optional_models", []) if x["id"] == "fl2v_turbo_lora_8step"]
    if len(turbo) != 1 or not any("pruned_int8_convrot" in x["path"] for x in shared):
        raise SystemExit("Expected quantized H3 + 8-step Turbo manifest; refusing other models")
    models = shared + turbo
    if any(not x.get("sha256") or not x.get("repo") or not x.get("revision") for x in models):
        raise SystemExit("Manifest must pin repo, revision and SHA-256 for each model")

    if STORAGE_MODE == "download":
        if __package__:
            from .model_downloader import download_pinned_models
        else:
            from model_downloader import download_pinned_models
        try:
            result = download_pinned_models(models, LOCAL)
        except (RuntimeError, OSError) as exc:
            raise SystemExit(f"Turbo model download failed: {exc}") from None
        result.update({"storage_mode": "download", "verification_mode": "full"})
        if os.environ.get("IMAGINE_DEBUG_METRICS", "0") == "1":
            print("imagine_metric " + json.dumps({"event": "model_staging_finished", **result}, sort_keys=True), flush=True)
        print("All Turbo weights downloaded and SHA-256 verified locally", flush=True)
        return

    if not MOUNT.is_absolute() or MOUNT == Path("/"):
        raise SystemExit("Invalid IMAGINE_GLOBAL_MOUNT")
    if subprocess.run(["mountpoint", "-q", str(MOUNT)]).returncode:
        raise SystemExit(f"No volume mounted at {MOUNT}; check RunPod mount path")
    LOCAL.mkdir(parents=True, exist_ok=True)
    pending = sum(x["bytes_approx"] for x in models if not (LOCAL / x["path"]).is_file())
    if shutil.disk_usage(LOCAL).free < pending + 10 * 1024**3:
        raise SystemExit("Insufficient local container disk for weights + scratch; increase disk")
    for item in models:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit("Invalid manifest model path")
        dest = LOCAL / rel
        remote = REMOTE / rel
        digest = item["sha256"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not valid(dest, digest):
            dest.unlink(missing_ok=True)
            if remote.is_file():
                print(f"Staging from Global Volume: {rel}", flush=True)
                partial = dest.with_name(dest.name + ".part")
                try:
                    shutil.copyfile(remote, partial)
                    if not valid(partial, digest):
                        raise SystemExit(f"Corrupt Global Volume object: {remote}. Remove that object explicitly before retrying.")
                    partial.replace(dest)
                finally:
                    partial.unlink(missing_ok=True)
            elif DOWNLOAD:
                from huggingface_hub import hf_hub_download
                print(f"Downloading pinned model: {rel}", flush=True)
                path = Path(hf_hub_download(repo_id=item["repo"], filename=str(rel), revision=item["revision"], local_dir=LOCAL))
                if path != dest or not valid(dest, digest):
                    dest.unlink(missing_ok=True)
                    raise SystemExit(f"Downloaded file failed verification: {rel}")
            else:
                raise SystemExit(f"Missing {remote}; set IMAGINE_DOWNLOAD_MODELS=1 for initial population")
        if not remote.is_file():
            if not DOWNLOAD:
                raise SystemExit(f"Missing persistent copy {remote}; enable initial population")
            print(f"Saving verified model to Global Volume: {rel}", flush=True)
            remote.parent.mkdir(parents=True, exist_ok=True)
            # Sequential whole-file write; no rename or locking on object-backed mount.
            with dest.open("rb") as src, remote.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
            if remote.stat().st_size != dest.stat().st_size:
                raise SystemExit(f"Persistent copy size mismatch: {rel}")
        print(f"Ready: {rel}", flush=True)
    print("All Turbo weights staged and verified locally", flush=True)


if __name__ == "__main__":
    main()
