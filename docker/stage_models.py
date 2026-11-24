"""Prepare H3 weights as local copies or direct Global Volume symlinks.

Verification is strict by default. The optional cached mode trusts only a
separate immutable receipt created after a prior complete SHA-256 check and an
exact match of the pinned manifest plus available size and mtime metadata.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "ComfyUI" / "models"
MOUNT = Path(os.environ.get("IMAGINE_GLOBAL_MOUNT", "/workspace-global"))
REMOTE = MOUNT / "imagine-h3-turbo" / "models"
MANIFEST_PATH = ROOT / "config" / "model-manifest.json"
RECEIPT_DIR = MOUNT / "imagine-h3-turbo" / ".verification-receipts"
BLOCK_SIZE = 8 * 1024**2
RECEIPT_LIMIT = 1024 * 1024
MIN_SCRATCH_BYTES = 10 * 1024**3
STORAGE_MODES = {"local", "direct"}
VERIFY_MODES = {"full", "cached", "size"}


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        while block := source.read(BLOCK_SIZE):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def _manifest_digest(manifest_bytes: bytes) -> str:
    return hashlib.sha256(manifest_bytes).hexdigest()


def _metadata(path: Path) -> dict[str, int] | None:
    """Return stat fields meaningful enough for an opt-in verification cache."""
    try:
        info = path.stat()
    except OSError:
        return None
    fields = {"size": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)}
    # Some object-backed mounts report placeholder timestamps.
    if fields["size"] <= 0 or fields["mtime_ns"] <= 0:
        return None
    return fields


def _check_size(path: Path, expected: int, label: str) -> int:
    try:
        actual = path.stat().st_size
    except OSError:
        raise RuntimeError(f"Cannot stat {label}: {path}") from None
    if actual != expected:
        raise RuntimeError(f"{label} size mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _cache_key(manifest_sha: str, item: dict, metadata: dict[str, int]) -> tuple:
    return (manifest_sha, item["path"], item["revision"], item["sha256"].lower(), int(item["bytes"]), metadata["size"], metadata["mtime_ns"])


def _read_receipts(directory: Path, manifest_sha: str) -> set[tuple]:
    receipts = set()
    try:
        paths = list(directory.glob(f"{manifest_sha}-*.json"))
    except OSError:
        return receipts
    if len(paths) > 2048:
        return receipts
    for path in paths:
        try:
            if path.stat().st_size > 16 * 1024:
                continue
            row = json.loads(path.read_text())
            if not isinstance(row, dict):
                continue
            if row.get("schema") != 1 or row.get("manifest_sha256") != manifest_sha:
                continue
            metadata = row.get("metadata")
            if (not isinstance(metadata, dict)
                    or not isinstance(row.get("path"), str)
                    or not isinstance(row.get("revision"), str)
                    or not isinstance(row.get("sha256"), str)
                    or len(row["sha256"]) != 64
                    or not isinstance(row.get("bytes"), int)
                    or not isinstance(metadata.get("size"), int)
                    or not isinstance(metadata.get("mtime_ns"), int)):
                continue
            item = {key: row[key] for key in ("path", "revision", "sha256", "bytes")}
            receipts.add(_cache_key(manifest_sha, item, metadata))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # Ignore partial, malformed or unreadable immutable receipts.
            continue
    return receipts


def _write_receipt(directory: Path, manifest_sha: str, item: dict, metadata: dict[str, int] | None) -> bool:
    if metadata is None:
        return False
    row = {
        "schema": 1,
        "manifest_sha256": manifest_sha,
        "path": item["path"],
        "revision": item["revision"],
        "sha256": item["sha256"].lower(),
        "bytes": int(item["bytes"]),
        "metadata": metadata,
    }
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    try:
        mount_root = directory.parents[1]
        if subprocess.run(["mountpoint", "-q", str(mount_root)], check=False).returncode:
            return False
        directory.mkdir(parents=True, exist_ok=True)
        # Each receipt is a new sequential object; never append or rename on GV.
        filename = f"{manifest_sha}-{hashlib.sha256(item['path'].encode()).hexdigest()[:12]}-{uuid.uuid4().hex}.json"
        with (directory / filename).open("xb") as output:
            output.write(payload)
        return True
    except OSError:
        return False


def _check_mount(mount: Path) -> None:
    if not mount.is_absolute() or mount.resolve() == Path("/") or mount.is_symlink():
        raise RuntimeError("Invalid IMAGINE_GLOBAL_MOUNT; expected an absolute mounted volume path")
    if subprocess.run(["mountpoint", "-q", str(mount)], check=False).returncode:
        raise RuntimeError(f"No Global Volume mounted at {mount}; verify the RunPod mount path")
    if not mount.is_dir() or not os.access(mount, os.R_OK | os.X_OK):
        raise RuntimeError(f"Global Volume path is not readable: {mount}")


def _safe_local_dest(local_root: Path, relative: Path) -> Path:
    if local_root.is_symlink():
        raise RuntimeError(f"Local model root must not be a symlink: {local_root}")
    try:
        local_root.resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise RuntimeError(f"Local model root resolves outside the worker: {local_root}") from None
    dest = local_root / relative
    nearest = dest.parent
    while nearest != local_root and not nearest.exists():
        nearest = nearest.parent
    try:
        nearest.resolve().relative_to(local_root.resolve())
    except ValueError:
        raise RuntimeError(f"Local model path escapes ComfyUI models root: {dest}") from None
    local_root.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.parent.resolve().relative_to(local_root.resolve())
    except ValueError:
        raise RuntimeError(f"Local model path escapes ComfyUI models root: {dest}") from None
    return dest


def _manifest_models(manifest: dict) -> list[dict]:
    turbo = [entry for entry in manifest.get("optional_models", []) if entry.get("id") == "fl2v_turbo_lora_8step"]
    if len(turbo) != 1 or not any("pruned_int8_convrot" in entry.get("path", "") for entry in manifest.get("models", [])):
        raise RuntimeError("Expected the pinned H3 INT8 model and 8-step Turbo LoRA manifest")
    items = manifest["models"] + turbo
    for item in items:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError("Manifest contains an invalid model path")
        if not item.get("repo") or not item.get("revision") or not item.get("sha256"):
            raise RuntimeError(f"Manifest metadata is incomplete for {item.get('id', 'unknown model')}")
        size = item.get("bytes")
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Manifest exact byte size is missing for {relative}")
    return items


def _source_path(remote_root: Path, relative: Path) -> Path:
    candidate = remote_root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(remote_root.resolve(strict=True))
    except (OSError, ValueError):
        raise RuntimeError(f"Missing or unsafe Global Volume model: {candidate}") from None
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise RuntimeError(f"Global Volume model is not a regular file: {candidate}")
    return resolved


def _verify_source(path: Path, item: dict, manifest_sha: str, verify_mode: str, receipts: set[tuple], receipt_path: Path | None, manifest_path: Path, metrics: dict) -> tuple[str, dict[str, int] | None]:
    _check_size(path, item["bytes"], "Global Volume model")
    metadata = _metadata(path)
    if metadata is None:
        metrics["metadata_unreliable"] += 1
    key = _cache_key(manifest_sha, item, metadata) if metadata else None
    if verify_mode == "cached" and key in receipts:
        metrics["files_cached"] += 1
        return "cached", metadata
    if verify_mode == "size":
        metrics["files_size_only"] += 1
        return "size", metadata
    started = time.perf_counter()
    digest, total = _sha256_file(path)
    metrics["verification_seconds"] += time.perf_counter() - started
    metrics["bytes_hashed"] += total
    if total != item["bytes"] or digest != item["sha256"].lower():
        raise RuntimeError(f"Global Volume model failed full SHA-256 verification: {item['path']}")
    after = _metadata(path)
    if metadata is not None and after != metadata:
        raise RuntimeError(f"Global Volume model metadata changed during verification: {item['path']}")
    metrics["files_full"] += 1
    if after and verify_mode == "cached" and receipt_path:
        if _manifest_digest(manifest_path.read_bytes()) != manifest_sha:
            raise RuntimeError("H3 model manifest changed during verification; refusing to cache the receipt")
        if _write_receipt(receipt_path, manifest_sha, item, after):
            metrics["receipts_written"] += 1
    return "full", after


def _mmap_probe(path: Path) -> None:
    """Check basic mmap support without materializing the whole model in RAM."""
    try:
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size <= 0:
                raise OSError("empty file")
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                _ = mapped[0]
                _ = mapped[-1]
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Global Volume model cannot be opened and memory-mapped: {path.name} ({exc}); retry with IMAGINE_MODEL_STORAGE_MODE=local") from None


def _link_direct(remote: Path, dest: Path) -> None:
    if dest.is_symlink():
        if dest.resolve() == remote.resolve():
            return
        dest.unlink()
    elif dest.exists():
        raise RuntimeError(f"Direct mode refuses to replace a local model file at {dest}; use local mode or clean this container path")
    os.symlink(str(remote), dest)


def _copy_and_hash(remote: Path, dest: Path, item: dict, metrics: dict) -> dict[str, int] | None:
    partial = dest.with_name(dest.name + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total = 0
    read_seconds = 0.0
    hash_seconds = 0.0
    write_seconds = 0.0
    try:
        with remote.open("rb") as source, partial.open("xb") as target:
            source_start = os.fstat(source.fileno())
            while True:
                read_started = time.perf_counter()
                block = source.read(BLOCK_SIZE)
                read_seconds += time.perf_counter() - read_started
                if not block:
                    break
                total += len(block)
                hash_started = time.perf_counter()
                digest.update(block)
                hash_seconds += time.perf_counter() - hash_started
                write_started = time.perf_counter()
                target.write(block)
                write_seconds += time.perf_counter() - write_started
            source_end = os.fstat(source.fileno())
        metrics["bytes_copied"] += total
        metrics["bytes_hashed"] += total
        metrics["copy_read_seconds"] += read_seconds
        metrics["copy_hash_seconds"] += hash_seconds
        metrics["copy_write_seconds"] += write_seconds
        if total != item["bytes"] or digest.hexdigest() != item["sha256"].lower():
            raise RuntimeError(f"Copied model failed full SHA-256 verification: {item['path']}")
        start_metadata = {"size": int(source_start.st_size), "mtime_ns": int(source_start.st_mtime_ns)}
        end_metadata = {"size": int(source_end.st_size), "mtime_ns": int(source_end.st_mtime_ns)}
        if start_metadata != end_metadata:
            raise RuntimeError(f"Global Volume model changed during copy: {item['path']}")
        path_metadata = _metadata(remote)
        if path_metadata and path_metadata != end_metadata:
            raise RuntimeError(f"Global Volume model path changed during copy: {item['path']}")
        partial.replace(dest)
        if end_metadata["size"] == item["bytes"] and end_metadata["mtime_ns"] > 0:
            return end_metadata
        return None
    finally:
        partial.unlink(missing_ok=True)


def prepare(*, storage_mode: str | None = None, verification_mode: str | None = None,
            mount: Path | None = None, remote_root: Path | None = None,
            local_root: Path | None = None, manifest_path: Path | None = None,
            receipt_path: Path | None = None, mount_check=None) -> dict:
    started = time.perf_counter()
    storage_requested = storage_mode or os.environ.get("IMAGINE_MODEL_STORAGE_MODE", "local").strip().lower()
    verify_requested = verification_mode or os.environ.get("IMAGINE_MODEL_VERIFY_MODE", "full").strip().lower()
    if storage_requested not in STORAGE_MODES:
        raise RuntimeError("IMAGINE_MODEL_STORAGE_MODE must be local or direct; auto is intentionally unsupported")
    if verify_requested not in VERIFY_MODES:
        raise RuntimeError("IMAGINE_MODEL_VERIFY_MODE must be full, cached, or size")
    mount = mount or MOUNT
    remote_root = remote_root or (mount / "imagine-h3-turbo" / "models")
    local_root = local_root or LOCAL
    manifest_path = manifest_path or MANIFEST_PATH
    receipt_path = receipt_path or (mount / "imagine-h3-turbo" / ".verification-receipts")
    metrics = {
        "mount_check_seconds": 0.0, "verification_seconds": 0.0,
        "copy_read_seconds": 0.0, "copy_hash_seconds": 0.0,
        "copy_write_seconds": 0.0, "copy_seconds": 0.0,
        "route_setup_seconds": 0.0, "bytes_copied": 0,
        "bytes_hashed": 0, "files_full": 0, "files_cached": 0,
        "files_size_only": 0, "metadata_unreliable": 0,
        "mmap_probe_seconds": 0.0, "receipts_written": 0,
    }
    mount_started = time.perf_counter()
    (mount_check or _check_mount)(mount)
    metrics["mount_check_seconds"] = time.perf_counter() - mount_started
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read H3 model manifest: {exc}") from None
    items = _manifest_models(manifest)
    manifest_sha = _manifest_digest(manifest_bytes)
    receipt_usable = False
    if verify_requested == "cached":
        try:
            volume_root = mount.resolve(strict=True)
            models_root = (mount / "imagine-h3-turbo").resolve(strict=True)
            models_root.relative_to(volume_root)
            receipt_usable = (receipt_path.parent.resolve(strict=True) == models_root
                              and not receipt_path.is_symlink()
                              and (not receipt_path.exists() or receipt_path.is_dir()))
        except (OSError, ValueError):
            receipt_usable = False
    receipts = _read_receipts(receipt_path, manifest_sha) if verify_requested == "cached" and receipt_usable else set()
    try:
        remote_root.resolve(strict=True).relative_to(mount.resolve(strict=True))
    except (OSError, ValueError):
        raise RuntimeError("H3 models directory must resolve inside IMAGINE_GLOBAL_MOUNT") from None
    available_sources = []
    for item in items:
        remote = _source_path(remote_root, Path(item["path"]))
        available_sources.append((item, remote))

    effective_modes = set()
    if storage_requested == "local":
        pending = 0
        for item, remote in available_sources:
            dest = _safe_local_dest(local_root, Path(item["path"]))
            if dest.is_symlink():
                raise RuntimeError(f"Local mode refuses to follow a model symlink: {dest}; clean the path first")
            if dest.is_file():
                check_started = time.perf_counter()
                if verify_requested == "size":
                    _check_size(dest, item["bytes"], "Local H3 model")
                    metrics["files_size_only"] += 1
                    effective_modes.add("size")
                else:
                    digest, total = _sha256_file(dest)
                    metrics["bytes_hashed"] += total
                    if total != item["bytes"] or digest != item["sha256"].lower():
                        dest.unlink()
                        pending += item["bytes"]
                    else:
                        metrics["files_full"] += 1
                        effective_modes.add("full")
                metrics["verification_seconds"] += time.perf_counter() - check_started
            else:
                pending += item["bytes"]
        if shutil.disk_usage(local_root).free < pending + MIN_SCRATCH_BYTES:
            raise RuntimeError("Insufficient local container disk for H3 weights and scratch space")
        for item, remote in available_sources:
            dest = _safe_local_dest(local_root, Path(item["path"]))
            if dest.is_file() and not dest.is_symlink():
                continue
            copy_started = time.perf_counter()
            verified_metadata = _copy_and_hash(remote, dest, item, metrics)
            metrics["copy_seconds"] += time.perf_counter() - copy_started
            effective_modes.add("full")
            metrics["files_full"] += 1
            if verify_requested == "cached" and receipt_usable:
                if _manifest_digest(manifest_path.read_bytes()) != manifest_sha:
                    raise RuntimeError("H3 model manifest changed during copy; refusing to cache the receipt")
                if _write_receipt(receipt_path, manifest_sha, item, verified_metadata):
                    metrics["receipts_written"] += 1
                elif verified_metadata is None:
                    metrics["metadata_unreliable"] += 1
    else:
        for item, remote in available_sources:
            verification_started = time.perf_counter()
            effective, _ = _verify_source(remote, item, manifest_sha, verify_requested, receipts, receipt_path if receipt_usable else None, manifest_path, metrics)
            if effective != "full":
                metrics["verification_seconds"] += time.perf_counter() - verification_started
            effective_modes.add(effective)
            mmap_started = time.perf_counter()
            _mmap_probe(remote)
            metrics["mmap_probe_seconds"] += time.perf_counter() - mmap_started
            route_started = time.perf_counter()
            _link_direct(remote, _safe_local_dest(local_root, Path(item["path"])))
            metrics["route_setup_seconds"] += time.perf_counter() - route_started

    if _manifest_digest(manifest_path.read_bytes()) != manifest_sha:
        raise RuntimeError("H3 model manifest changed during preparation; restart staging")
    metrics["seconds"] = time.perf_counter() - started
    metrics["storage_mode_requested"] = storage_requested
    metrics["storage_mode"] = storage_requested
    metrics["verification_mode_requested"] = verify_requested
    metrics["verification_mode"] = "+".join(sorted(effective_modes)) if effective_modes else verify_requested
    metrics["manifest_sha256"] = manifest_sha
    return metrics


def main() -> None:
    try:
        result = prepare()
    except Exception as exc:
        print(f"H3 model preparation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    if os.environ.get("IMAGINE_DEBUG_METRICS", "0") == "1":
        print("imagine_metric " + json.dumps({"event": "model_staging_finished", **result}, sort_keys=True), flush=True)
    print(f"H3 models ready ({result['storage_mode']} storage; {result['verification_mode']} verification)", flush=True)


if __name__ == "__main__":
    main()
