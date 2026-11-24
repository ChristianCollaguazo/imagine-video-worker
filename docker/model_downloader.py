"""Pinned, verified Hugging Face downloads into local model storage.

Uses huggingface_hub's local_dir implementation for resumable transfers. Each
file is downloaded to a temporary directory on the same filesystem, checked
against exact size and SHA-256, then atomically moved into the ComfyUI model
tree. The helper never reads or writes a persistent Global Volume.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time
from typing import Callable

BLOCK_SIZE = 8 * 1024**2
RETRIES = 3
RETRY_BASE_SECONDS = 2
SCRATCH_RESERVE_BYTES = 10 * 1024**3


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        while block := source.read(BLOCK_SIZE):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def _safe_destination(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError("Model manifest contains an unsafe local path")
    if root.is_symlink():
        raise RuntimeError(f"Local model root must not be a symlink: {root}")
    candidate = root / relative
    nearest = candidate.parent
    while nearest != root and not nearest.exists():
        nearest = nearest.parent
    try:
        nearest.resolve().relative_to(root.resolve())
    except ValueError:
        raise RuntimeError(f"Model destination escapes local model root: {candidate}") from None
    root.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError:
        raise RuntimeError(f"Model destination escapes local model root: {candidate}") from None
    if candidate.is_symlink():
        raise RuntimeError(f"Refusing to overwrite a symlinked local model path: {candidate}")
    return candidate


def _hub_download(item: dict, local_dir: Path, force_download: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is required for download storage mode") from None
    token = os.environ.get("HF_TOKEN")
    result = hf_hub_download(
        repo_id=item["repo"],
        filename=item["path"],
        revision=item["revision"],
        local_dir=str(local_dir),
        force_download=force_download,
        token=token if token else False,
    )
    return Path(result)


def _validate_item(item: dict) -> None:
    if not isinstance(item.get("repo"), str) or not item["repo"]:
        raise RuntimeError("Pinned model is missing its Hugging Face repository")
    revision = item.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise RuntimeError(f"Pinned model revision must be a commit SHA: {item.get('path', 'unknown')}")
    digest = item.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise RuntimeError(f"Pinned model SHA-256 is invalid: {item.get('path', 'unknown')}")
    size = item.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"Pinned model exact byte size is missing: {item.get('path', 'unknown')}")


def download_pinned_models(
    items: list[dict],
    local_root: Path,
    *,
    download_file: Callable[[dict, Path, bool], Path] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    scratch_reserve_bytes: int = SCRATCH_RESERVE_BYTES,
    metric_callback: Callable[[str, dict], None] | None = None,
) -> dict:
    """Download only manifest items, reusing valid files and hashing each new one.

    The injection points exist for tiny offline tests. Production defaults use
    the image's pinned `huggingface_hub` package and bounded retry waits.
    """
    started = time.perf_counter()
    downloader = download_file or _hub_download
    local_root.mkdir(parents=True, exist_ok=True)
    if local_root.is_symlink():
        raise RuntimeError(f"Local model root must not be a symlink: {local_root}")

    missing_bytes = 0
    missing: list[dict] = []
    metrics = {
        "download_seconds": 0.0,
        "verification_seconds": 0.0,
        "bytes_downloaded": 0,
        "bytes_hashed": 0,
        "files_downloaded": 0,
        "files_reused": 0,
        "download_attempts": 0,
    }
    destinations: dict[str, Path] = {}
    invalid_destinations: list[Path] = []
    for item in items:
        _validate_item(item)
        relative = Path(item["path"])
        dest = _safe_destination(local_root, relative)
        destinations[item["path"]] = dest
        if dest.is_file() and not dest.is_symlink():
            if dest.stat().st_size == item["bytes"]:
                check_started = time.perf_counter()
                digest, total = _sha256_file(dest)
                metrics["verification_seconds"] += time.perf_counter() - check_started
                metrics["bytes_hashed"] += total
                if total == item["bytes"] and digest == item["sha256"].lower():
                    metrics["files_reused"] += 1
                    continue
            invalid_destinations.append(dest)
        elif dest.exists():
            raise RuntimeError(f"Pinned local model path is not a regular file: {dest}")
        missing.append(item)
        missing_bytes += item["bytes"]

    required_free = missing_bytes + scratch_reserve_bytes
    free_bytes = shutil.disk_usage(local_root).free
    if free_bytes < required_free:
        raise RuntimeError(
            f"Insufficient local disk for pinned H3 downloads: need {required_free} bytes free "
            f"(missing models plus reserve), have {free_bytes}"
        )
    for dest in invalid_destinations:
        dest.unlink()
    if not missing:
        metrics["seconds"] = time.perf_counter() - started
        return metrics

    stage_root = local_root / ".imagine-hf-download-stage"
    if stage_root.is_symlink():
        raise RuntimeError(f"Download staging path must not be a symlink: {stage_root}")
    stage_root.mkdir(parents=True, exist_ok=True)
    complete = False
    try:
        for item in missing:
            print(f"Downloading pinned H3 model: {item['path']}", flush=True)
            dest = destinations[item["path"]]
            stage_file = _safe_destination(stage_root, Path(item["path"]))
            force = False
            succeeded = False
            last_error = ""
            file_started = time.perf_counter()
            print(f"Pinned model download started: {item.get('id', item['path'])}", flush=True)
            for attempt in range(1, RETRIES + 1):
                metrics["download_attempts"] += 1
                download_started = time.perf_counter()
                try:
                    downloaded = downloader(item, stage_root, force)
                except Exception as exc:
                    last_error = type(exc).__name__
                    if attempt < RETRIES:
                        sleep_fn(RETRY_BASE_SECONDS ** attempt)
                    continue
                metrics["download_seconds"] += time.perf_counter() - download_started
                try:
                    downloaded.resolve().relative_to(stage_root.resolve())
                except ValueError:
                    raise RuntimeError("Hugging Face returned a path outside local staging") from None
                if downloaded.is_symlink() or not downloaded.is_file():
                    raise RuntimeError("Hugging Face did not return a regular staged model file")
                if downloaded != stage_file:
                    raise RuntimeError("Hugging Face returned an unexpected model path")
                check_started = time.perf_counter()
                exact_size = downloaded.stat().st_size
                digest, total = _sha256_file(downloaded)
                metrics["verification_seconds"] += time.perf_counter() - check_started
                metrics["bytes_hashed"] += total
                if exact_size != item["bytes"] or total != item["bytes"] or digest != item["sha256"].lower():
                    downloaded.unlink(missing_ok=True)
                    force = True
                    last_error = "exact size or SHA-256 mismatch"
                    if attempt < RETRIES:
                        sleep_fn(RETRY_BASE_SECONDS ** attempt)
                    continue
                if downloaded.stat().st_dev != dest.parent.stat().st_dev:
                    raise RuntimeError("Staging and destination are on different filesystems; atomic publication is unavailable")
                # Same-filesystem replacement publishes only a fully verified file.
                os.replace(downloaded, dest)
                metrics["bytes_downloaded"] += total
                metrics["files_downloaded"] += 1
                succeeded = True
                break
            if not succeeded:
                raise RuntimeError(
                    f"Could not download pinned model {item.get('id', item['path'])} after {RETRIES} attempts "
                    f"({last_error}); check Hugging Face connectivity or access"
                )
            if metric_callback:
                metric_callback("model_download_finished", {"model": item.get("id", item["path"]), "seconds": time.perf_counter() - file_started, "bytes": item["bytes"]})
            print(f"Pinned model ready: {item.get('id', item['path'])}", flush=True)
        complete = True
    finally:
        # On failure keep only resumable model download state. On full success,
        # remove all HF metadata and staging files from local model directories.
        if complete:
            shutil.rmtree(stage_root, ignore_errors=True)

    metrics["seconds"] = time.perf_counter() - started
    return metrics
