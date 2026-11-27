"""Ephemeral, single-slot API for private image-to-video generation."""
from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from h3_diagnostics import HISTORY_KEY, STAGES as COMFY_NODE_LABELS
from h3_runtime import vae_output_mode

from .comfy import ComfyClient, ModelLoaderError, new_client_id
from .lora_catalog import available_lora_names, resolve_lora_name, resolve_lora_stack
from .workflow import (
    BUFFERED_VAE_NODE,
    DEFAULT_STEPS,
    DEFAULT_RESOLUTION,
    FIRST_BLOCK_CACHE_NODE,
    SUPPORTED_DURATIONS,
    SUPPORTED_RESOLUTIONS,
    VIDEO_FPS,
    TURBO_LORA,
    build_workflow,
    canvas_for_image,
    first_block_cache_settings,
    sampling_steps,
)

TOKEN = os.environ.get("IMAGINE_VIDEO_TOKEN", "")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_WS = os.environ.get("COMFY_WS", COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws")
WORKER_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", str(WORKER_ROOT / "ComfyUI")))
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", str(COMFY_ROOT / "input")))
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", str(COMFY_ROOT / "output")))
COMFY_TEMP_DIR = Path(os.environ.get("COMFY_TEMP_DIR", str(COMFY_ROOT / "temp")))
DATA_ROOT = Path(os.environ.get("IMAGINE_VIDEO_DATA", "/tmp/imagine-video-worker"))
MAX_UPLOAD_BYTES = int(os.environ.get("IMAGINE_VIDEO_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_LORA_UPLOAD_BYTES = int(os.environ.get("IMAGINE_VIDEO_MAX_LORA_UPLOAD_BYTES", str(4 * 1024 * 1024 * 1024)))
JOB_TIMEOUT_SECONDS = max(60, int(os.environ.get("IMAGINE_VIDEO_JOB_TIMEOUT_SECONDS", str(90 * 60))))
MAX_PIXELS = 40_000_000
RETENTION_SECONDS = 15 * 60
DEBUG_METRICS = os.environ.get("IMAGINE_DEBUG_METRICS", "0") == "1"
AUDIO = os.environ.get("IMAGINE_H3_AUDIO", "1") == "1"
FIRST_BLOCK_CACHE = first_block_cache_settings()
ATTENTION_BACKEND_SELECTED = os.environ.get("IMAGINE_SELECTED_ATTENTION_BACKEND", "unverified")
VRAM_PROFILE = os.environ.get("IMAGINE_H3_VRAM_PROFILE", "auto")
VAE_OUTPUT = vae_output_mode()
_generation_count = 0


def _metric(event: str, *, always: bool = False, **values):
    """Emit technical-only diagnostics; never include request or credential data."""
    if DEBUG_METRICS or always:
        print("imagine_metric " + json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _first_block_cache_diagnostics():
    return {
        "first_block_cache_mode": FIRST_BLOCK_CACHE.mode,
        "first_block_cache_preset": FIRST_BLOCK_CACHE.preset,
        "first_block_cache_threshold": FIRST_BLOCK_CACHE.threshold,
        "first_block_cache_max_consecutive_hits": FIRST_BLOCK_CACHE.max_consecutive_hits,
    }


def _report_export_metrics(node_output: dict, generation: int):
    records = node_output.get("video_export_metrics", [])
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        return
    record = records[0]
    if record.get("encoder") == "libx264" and record.get("hardware_fallback") is True:
        _metric("video_encoder_fallback", always=True, generation=generation, encoder="libx264")
    if not DEBUG_METRICS:
        return
    values = {}

    def numbers(source, keys):
        return {key: value for key in keys if type(value := source.get(key)) in (int, float) and 0 <= value < 1e12}

    values.update(numbers(record, ("seconds", "fps", "frame_count", "duration_seconds", "throughput_fps")))
    for key in ("tensor_shape", "tensor_strides"):
        dimensions = record.get(key)
        if isinstance(dimensions, list) and len(dimensions) == 4 and all(type(n) is int and 0 <= n < 2**48 for n in dimensions):
            values[key] = dimensions
    for key, allowed in (
        ("encoder", ("h264_nvenc", "libx264")),
        ("tensor_dtype", ("torch.float32", "torch.float64", "torch.float16", "torch.bfloat16")),
    ):
        if record.get(key) in allowed:
            values[key] = record[key]
    device = record.get("tensor_device")
    if isinstance(device, str):
        kind, _, index = device.partition(":")
        if kind in {"cpu", "cuda", "mps", "xpu"}:
            values["tensor_device"] = kind
            if index.isascii() and index.isdecimal() and len(index) <= 3:
                values["tensor_device_index"] = int(index)
    if type(record.get("hardware_fallback")) is bool:
        values["hardware_fallback"] = record["hardware_fallback"]
    stages = record.get("stages_seconds")
    if isinstance(stages, dict):
        values["stages_seconds"] = numbers(stages, (
            "frame_preparation", "device_to_cpu", "frame_wrapping", "pixel_conversion", "encoding", "muxing", "finalization",
        ))
    _metric("video_export", generation=generation, **values)


async def _collect_video(job, info, result_path):
    output_root = COMFY_OUTPUT_DIR.resolve()
    job_directory = output_root / "imagine_h3" / job["id"]
    source = (output_root / info.get("subfolder", "") / info["filename"]).resolve()
    if info.get("type") == "output" and source.parent == job_directory and source.is_file():
        try:
            source.replace(result_path)
            return "local_move"
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    await comfy.download(info, result_path, job["cancel_event"].is_set)
    return "http_stream"


async def _validate_video(path, frame_count, duration, expect_audio, width, height):
    probe = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height,nb_frames,avg_frame_rate,start_pts,duration_ts,time_base",
        "-of", "json", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await asyncio.wait_for(probe.communicate(), timeout=10)
    finally:
        if probe.returncode is None:
            probe.kill()
            await probe.wait()
    streams = json.loads(output)["streams"] if probe.returncode == 0 else []
    videos = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    if len(videos) != 1 or len(audio) != int(expect_audio):
        raise RuntimeError("Video did not match the requested stream layout")
    video = videos[0]
    if (video.get("width"), video.get("height")) != (width, height):
        raise RuntimeError("Video did not match the requested resolution")
    if (video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p"
            or video.get("nb_frames") != str(frame_count) or video.get("start_pts") != 0
            or Fraction(video["avg_frame_rate"]) != VIDEO_FPS
            or video["duration_ts"] * Fraction(video["time_base"]) != duration):
        raise RuntimeError("Video did not match the requested duration, native frame count and FPS")
    if audio and (audio[0]["codec_name"] != "aac" or audio[0].get("start_pts") != 0
                  or not 0 < audio[0]["duration_ts"] * Fraction(audio[0]["time_base"]) <= duration):
        raise RuntimeError("Audio did not match the requested video timeline")


async def _sample_gpu_peak(stop: asyncio.Event, peaks: dict[str, int | None]):
    """Track per-job sampled peaks in MiB, including cgroup v2 total RAM."""
    while not stop.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                output, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise
            values = [int(line.strip()) for line in output.decode().splitlines() if line.strip().isdigit()]
            if values:
                peaks["vram_mb"] = max(peaks["vram_mb"] or 0, sum(values))
        except (OSError, ValueError, asyncio.TimeoutError):
            pass
        try:
            current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip()) // (1024 * 1024)
            peaks["container_ram_mb"] = max(peaks["container_ram_mb"] or 0, current)
        except (OSError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass

app = FastAPI(title="Imagine Video Worker", docs_url=None, redoc_url=None)
comfy = ComfyClient(COMFY_URL, COMFY_WS)
jobs: dict[str, dict[str, Any]] = {}
idempotency: dict[str, str] = {}
queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
slot_lock = asyncio.Lock()
admission_lock = asyncio.Lock()
admission_reserved = False
worker_task: asyncio.Task | None = None
reaper_task: asyncio.Task | None = None


@app.middleware("http")
async def token_gate(request, call_next):
    # Reject unauthenticated requests before multipart parsing allocates temp files.
    if request.url.path.startswith("/v1/"):
        try:
            _authorized(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    return await call_next(request)


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job.get(key) for key in ("id", "status", "stage", "progress", "error", "steps", "mode", "resolution", "effective_width", "effective_height", "lora_name", "lora_strength", "lora_stack")}


def _available_lora_names() -> list[str]:
    return available_lora_names(COMFY_ROOT / "models" / "loras")


def _resolve_lora_name(mode: str, requested: str | None, available: list[str]) -> str | None:
    return resolve_lora_name(mode, requested, available, TURBO_LORA)


def _authorized(authorization: str | None):
    if not TOKEN:
        raise HTTPException(503, "Worker token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, TOKEN):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})


async def _bytes_to_file(upload: UploadFile, target: Path) -> tuple[int, str, str, int, int]:
    total, digest = 0, hashlib.sha256()
    with target.open("xb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Photo exceeds the upload limit")
            digest.update(chunk)
            output.write(chunk)
    if not total:
        raise HTTPException(422, "Photo is empty")
    try:
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Upload a JPEG, PNG, or WebP photo")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Photo dimensions are too large")
            image_format = image.format
            oriented = ImageOps.exif_transpose(image)
            image_width, image_height = oriented.size
            normalized = target.with_suffix(".normalized")
            if image_format == "JPEG" and oriented.mode not in {"RGB", "L"}:
                oriented = oriented.convert("RGB")
            oriented.save(normalized, format=image_format)
        normalized.replace(target)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Uploaded file is not a supported image") from None
    return total, digest.hexdigest(), image_format, image_width, image_height


def _safe_lora_filename(filename: str | None) -> str:
    if (not filename or len(filename) > 255 or filename.startswith(".")
            or Path(filename).name != filename or "/" in filename or "\\" in filename
            or "\x00" in filename or not filename.endswith(".safetensors")):
        raise HTTPException(422, "Upload a LoRA with a .safetensors basename")
    return filename


async def _store_lora(upload: UploadFile, filename: str) -> int:
    """Stream a LoRA to a same-directory temporary file, then link it without clobbering."""
    root = COMFY_ROOT / "models" / "loras"
    if not root.is_dir():
        raise HTTPException(503, "LoRA storage is not available")
    target = root / filename
    temp_path: Path | None = None
    total = 0
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".imagine-upload-", suffix=".part", dir=root)
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_LORA_UPLOAD_BYTES:
                    raise HTTPException(413, "LoRA exceeds the upload limit")
                output.write(chunk)
            if not total:
                raise HTTPException(422, "LoRA file is empty")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temp_path, target)
        except FileExistsError:
            raise HTTPException(409, "A LoRA with that filename already exists") from None
        return total
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _remove_owned_file(root: Path | None, relative: str):
    if not root:
        return
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and candidate.is_relative_to(root):
        candidate.unlink(missing_ok=True)


def _remove_owned_tree(root: Path | None, relative: str):
    if not root:
        return
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and candidate.is_relative_to(root):
        shutil.rmtree(candidate, ignore_errors=True)


def _purge_comfy_files(job: dict[str, Any]):
    if job.get("comfy_name"):
        _remove_owned_file(COMFY_INPUT_DIR, job["comfy_name"])
    for info in job.get("remote_outputs", []):
        root = COMFY_TEMP_DIR if info.get("type") == "temp" else COMFY_OUTPUT_DIR
        relative = "/".join(part for part in (info.get("subfolder", ""), info.get("filename", "")) if part)
        _remove_owned_file(root, relative)
    if job.get("id"):
        relative = str(Path("imagine_h3") / job["id"])
        _remove_owned_tree(COMFY_OUTPUT_DIR, relative)
        _remove_owned_tree(COMFY_TEMP_DIR, relative)


def _cleanup(job: dict[str, Any], keep_result: bool = False):
    folder: Path = job["folder"]
    result = job.get("result_path") if keep_result else None
    for child in folder.iterdir() if folder.exists() else ():
        if result and child == result:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    if not keep_result:
        shutil.rmtree(folder, ignore_errors=True)


async def _run_job(job_id: str):
    global _generation_count
    job = jobs.get(job_id)
    if not job or job["status"] != "queued":
        queue.task_done()
        return
    async with slot_lock:
        resource_stop = asyncio.Event()
        resource_peaks: dict[str, int | None] = {"vram_mb": None, "container_ram_mb": None}
        resource_task = asyncio.create_task(_sample_gpu_peak(resource_stop, resource_peaks)) if DEBUG_METRICS else None
        _generation_count += 1
        generation_number = _generation_count
        generation_started = time.perf_counter()
        phase_started = generation_started
        active_node: str | None = None
        node_started = generation_started
        previous_step: float | None = None
        node_seconds: dict[str, float] = {}
        node_labels = dict(COMFY_NODE_LABELS)
        step_seconds: list[float] = []
        _metric(
            "generation_started", generation=generation_number, warm=generation_number > 1,
            generation_state="initial" if generation_number == 1 else "subsequent",
            mode=job["mode"], steps=job["steps"], resolution=job["resolution"],
            duration_seconds=job["duration_seconds"], fps=VIDEO_FPS,
            frame_count=job["duration_seconds"] * VIDEO_FPS, audio=AUDIO,
            vram_profile=VRAM_PROFILE, vae_output=VAE_OUTPUT,
            effective_width=job["effective_width"], effective_height=job["effective_height"],
            lora_name=job.get("lora_name"), lora_strength=job["lora_strength"], lora_stack=job["lora_stack"],
            **_first_block_cache_diagnostics(),
        )
        job["status"], job["stage"], job["progress"] = "running", "uploading_photo", None
        try:
            content_type = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[job["input_path"].suffix]
            upload_name = f"imagine_{job_id}{job['input_path'].suffix}"
            job["comfy_name"] = upload_name
            (job["folder"] / "remote.json").write_text(json.dumps({"comfy_name": upload_name}))
            comfy_name = await comfy.upload(job["input_path"], upload_name, content_type)
            _metric("phase", name="image_upload", seconds=round(time.perf_counter() - phase_started, 3))
            job["comfy_name"] = comfy_name
            (job["folder"] / "remote.json").write_text(json.dumps({"comfy_name": comfy_name}))
            job["input_path"].unlink(missing_ok=True)
            if job["cancel_event"].is_set():
                return
            job["stage"], job["progress"] = "generating_video", None
            phase_started = time.perf_counter()
            if any(not (COMFY_ROOT / "models" / "loras" / item["name"]).is_file() for item in job["lora_stack"]):
                raise RuntimeError("A selected LoRA is no longer available")
            workflow = build_workflow(
                job["prompt"], comfy_name, job["duration_seconds"], job["seed"],
                job["image_width"], job["image_height"], job["id"],
                steps=job["steps"], audio=AUDIO, resolution=job["resolution"],
                lora_name=job.get("lora_name"), lora_strength=job["lora_strength"], lora_stack=job["lora_stack"],
                first_block_cache=FIRST_BLOCK_CACHE, vae_output=VAE_OUTPUT,
            )
            for index in range(len(job["lora_stack"])):
                node_labels[str(16 + index)] = f"lora_loader_{index + 1}"
            if FIRST_BLOCK_CACHE.mode != "off":
                node_labels[str(16 + len(job["lora_stack"]))] = "first_block_cache"
            job.pop("prompt", None)
            client_id = new_client_id()
            prompt_id = await comfy.queue(workflow, client_id)
            del workflow
            job["prompt_id"] = prompt_id
            (job["folder"] / "remote.json").write_text(json.dumps({"comfy_name": comfy_name, "prompt_id": prompt_id}))
            if job["cancel_event"].is_set():
                await comfy.cancel(prompt_id)
                return

            def report(data):
                nonlocal active_node, node_started, previous_step
                if DEBUG_METRICS:
                    now = time.perf_counter()
                    if "node_started" in data:
                        if active_node is not None:
                            node_seconds[active_node] = node_seconds.get(active_node, 0.0) + now - node_started
                        active_node = data["node_started"]
                        node_started = now
                        previous_step = now if active_node == "11" else None
                    elif str(data.get("node")) == "11" and data.get("value") is not None:
                        if previous_step is not None:
                            step_seconds.append(round(now - previous_step, 3))
                        previous_step = now
                if "node_started" in data:
                    return
                if data.get("stage"):
                    job["stage"] = data["stage"]
                    return
                if data.get("node") is not None and str(data["node"]) != "11":
                    return
                maximum = max(1, int(data.get("max", 1)))
                value = max(0, min(maximum, int(data.get("value", 0))))
                job["progress"] = min(99, int(100 * value / maximum))
                job["stage"] = "generating_video"

            history = await comfy.wait(prompt_id, client_id, report, job["cancel_event"].is_set, JOB_TIMEOUT_SECONDS)
            _metric(
                "phase", name="comfy_generation", generation=generation_number, seconds=round(time.perf_counter() - phase_started, 3),
                nodes={node_labels[node]: round(seconds, 3) for node, seconds in node_seconds.items() if node in node_labels},
                diffusion_step_seconds=step_seconds,
            )
            loader_nodes = ("1", "2", "3", "4", *(str(16 + index) for index in range(len(job["lora_stack"]))))
            for node in loader_nodes:
                if node in node_seconds:
                    _metric(
                        "model_loader_node", loader=node_labels[node],
                        seconds=round(node_seconds[node], 3), generation=generation_number,
                        load_state="initial" if generation_number == 1 else "subsequent",
                    )
            if job["cancel_event"].is_set() or history is None:
                return
            job["stage"] = "collecting_result"
            phase_started = time.perf_counter()
            if DEBUG_METRICS and isinstance(runtime_metrics := history.get(HISTORY_KEY), dict):
                _metric("h3_inference", generation=generation_number, **runtime_metrics)
            outputs = history.get("outputs", {})
            _report_export_metrics(outputs.get("15", {}), generation_number)
            job["remote_outputs"] = [item for node in outputs.values() for item in (node.get("images", []) + node.get("gifs", []) + node.get("videos", []) + node.get("files", [])) if item.get("filename")]
            (job["folder"] / "remote.json").write_text(json.dumps({"comfy_name": comfy_name, "remote_outputs": job["remote_outputs"], "prompt_id": prompt_id}))
            info = next((item for item in job["remote_outputs"] if item.get("filename", "").lower().endswith(".mp4")), None)
            if not info:
                raise RuntimeError("ComfyUI finished without a video output")
            result_path = job["folder"] / "result.mp4"
            collection = await _collect_video(job, info, result_path)
            _metric("phase", name="result_download", seconds=round(time.perf_counter() - phase_started, 3), method=collection)
            if job["cancel_event"].is_set():
                return
            phase_started = time.perf_counter()
            duration = job["duration_seconds"]
            await _validate_video(result_path, duration * VIDEO_FPS, duration, AUDIO,
                                  job["effective_width"], job["effective_height"])
            _metric("phase", name="mp4_finalize", seconds=round(time.perf_counter() - phase_started, 3))
            if job["cancel_event"].is_set():
                return
            job["result_path"] = result_path
            # Keep the cleanup marker until Comfy's copies and history are gone.
            _purge_comfy_files(job)
            if job.get("prompt_id"):
                await comfy.forget_prompt(job["prompt_id"])
            if job["cancel_event"].is_set():
                return
            # The source image and all temporary inputs/intermediates are removed before download is exposed.
            _cleanup(job, keep_result=True)
            job.update(status="completed", stage="completed", progress=100, completed_at=time.time(), error=None)
            _metric("generation_finished", status="completed", seconds=round(time.perf_counter() - generation_started, 3), generation=generation_number)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not job["cancel_event"].is_set():
                stopped = True
                if job.get("prompt_id"):
                    try:
                        stopped = await comfy.cancel(job["prompt_id"])
                    except Exception:
                        stopped = False
                if stopped:
                    if job.get("prompt_id"):
                        try:
                            await comfy.forget_prompt(job["prompt_id"])
                        except Exception:
                            pass
                    _purge_comfy_files(job)
                    _cleanup(job)
                else:
                    job["remote_cleanup_pending"] = True
                public_error = "Video generation failed. Please try again."
                if isinstance(exc, ModelLoaderError):
                    public_error = "Model loading failed. Check model storage, permissions, file completeness, and mmap compatibility."
                    _metric("model_loader_failure", generation=generation_number, loader=exc.loader, kind=exc.kind)
                job.update(status="failed", stage="failed", progress=None, error=public_error, completed_at=time.time())
                _metric("generation_finished", status="failed", seconds=round(time.perf_counter() - generation_started, 3), generation=generation_number)
        finally:
            if resource_task:
                resource_stop.set()
                await asyncio.gather(resource_task, return_exceptions=True)
                _metric("resource_peaks", generation=generation_number, **resource_peaks)
            if job["cancel_event"].is_set():
                job.pop("prompt", None)
                _metric("generation_finished", status="cancelled", seconds=round(time.perf_counter() - generation_started, 3), generation=generation_number)
                stopped = True
                if job.get("prompt_id"):
                    try:
                        stopped = await comfy.cancel(job["prompt_id"])
                    except Exception:
                        stopped = False
                if stopped:
                    if job.get("prompt_id"):
                        try:
                            await comfy.forget_prompt(job["prompt_id"])
                        except Exception:
                            pass
                    _purge_comfy_files(job)
                    _cleanup(job)
            job.pop("prompt", None)
            queue.task_done()


async def _worker():
    while True:
        job_id = await queue.get()
        await _run_job(job_id)


async def _reaper():
    while True:
        # Bounded sweep keeps terminal media cleanup within one second of expiry.
        await asyncio.sleep(1)
        now = time.time()
        for job in list(jobs.values()):
            if job["status"] not in {"completed", "failed", "cancelled", "expired"}:
                continue
            if job.get("remote_cleanup_pending") and job.get("prompt_id"):
                try:
                    if not await comfy.cancel(job["prompt_id"]):
                        continue
                    await comfy.forget_prompt(job["prompt_id"])
                except Exception:
                    continue
                _purge_comfy_files(job)
                _cleanup(job)
                job["remote_cleanup_pending"] = False
            if now - job.get("completed_at", job["created_at"]) >= RETENTION_SECONDS:
                if job["status"] == "cancelled" and job.get("prompt_id"):
                    try:
                        if not await comfy.cancel(job["prompt_id"]):
                            continue
                    except Exception:
                        continue
                _purge_comfy_files(job)
                if job.get("prompt_id"):
                    try:
                        await comfy.forget_prompt(job["prompt_id"])
                    except Exception:
                        pass
                _cleanup(job)
                job.update(status="expired", stage="expired", progress=None, result_path=None, error=None)


@app.on_event("startup")
async def startup():
    startup_started = time.perf_counter()
    if not DATA_ROOT.resolve().is_relative_to(Path("/tmp").resolve()):
        raise RuntimeError("Job storage must stay under /tmp")
    DATA_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    DATA_ROOT.chmod(0o700)
    # No job survives restart. Remove old ephemeral files and begin with a clean queue.
    for child in DATA_ROOT.iterdir():
        if child.is_dir():
            marker = child / "remote.json"
            if marker.is_file():
                try:
                    prior = json.loads(marker.read_text())
                    if prior.get("prompt_id"):
                        if not await comfy.cancel(prior["prompt_id"]):
                            raise RuntimeError("ComfyUI job is still active")
                        await comfy.forget_prompt(prior["prompt_id"])
                    _purge_comfy_files({"id": child.name, "comfy_name": prior.get("comfy_name"), "remote_outputs": prior.get("remote_outputs", [])})
                except Exception:
                    raise RuntimeError("Could not safely clean a previous job") from None
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    global worker_task, reaper_task
    if DEBUG_METRICS:
        try:
            import torch
            _metric(
                "runtime", torch=torch.__version__, cuda=torch.version.cuda,
                attention_backend_selected=ATTENTION_BACKEND_SELECTED,
                attention_selection_source="startup probe; ComfyUI startup log is authoritative",
            )
        except Exception:
            _metric(
                "runtime", attention_backend_selected=ATTENTION_BACKEND_SELECTED,
                attention_selection_source="startup probe; ComfyUI startup log is authoritative",
            )
    worker_task = asyncio.create_task(_worker())
    reaper_task = asyncio.create_task(_reaper())
    _metric(
        "worker_startup", seconds=round(time.perf_counter() - startup_started, 3),
        vram_profile=VRAM_PROFILE, audio=AUDIO,
        attention_backend_selected=ATTENTION_BACKEND_SELECTED,
        **_first_block_cache_diagnostics(),
    )


@app.on_event("shutdown")
async def shutdown():
    for task in (worker_task, reaper_task):
        if task:
            task.cancel()
    if worker_task and worker_task.done() is False:
        await asyncio.gather(worker_task, return_exceptions=True)
    if reaper_task and reaper_task.done() is False:
        await asyncio.gather(reaper_task, return_exceptions=True)
    for job in jobs.values():
        stopped = True
        if job.get("prompt_id"):
            try:
                stopped = await comfy.cancel(job["prompt_id"])
            except Exception:
                stopped = False
        if stopped:
            if job.get("prompt_id"):
                try:
                    await comfy.forget_prompt(job["prompt_id"])
                except Exception:
                    pass
            _purge_comfy_files(job)
            _cleanup(job)
    await comfy.close()


@app.get("/v1/health")
async def health(authorization: str | None = Header(None)):
    _authorized(authorization)
    models = (
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    )
    if AUDIO:
        models += ("vae/minimax_h3_audio_vae_fp32.safetensors",)
    if any(not (COMFY_ROOT / "models" / relative).is_file() for relative in models):
        raise HTTPException(503, "Worker is not ready")
    try:
        available_nodes = await comfy.health(audio=AUDIO)
    except Exception:
        raise HTTPException(503, "Worker is not ready") from None
    if FIRST_BLOCK_CACHE.mode != "off" and FIRST_BLOCK_CACHE_NODE not in available_nodes:
        raise HTTPException(503, "ComfyUI FirstBlockCache node is missing; install the pinned custom node or set IMAGINE_VIDEO_FIRST_BLOCK_CACHE=off")
    if VAE_OUTPUT == "gpu-buffer" and BUFFERED_VAE_NODE not in available_nodes:
        raise HTTPException(503, "ComfyUI H3 buffered VAE decode node is missing")
    lora_names = _available_lora_names() if "LoraLoaderModelOnly" in available_nodes else []
    turbo_available = TURBO_LORA in lora_names
    any_lora_available = bool(lora_names)
    return {
        "status": "ok",
        "model": "MiniMax H3 FL2VA INT8 ConvRot",
        "attention_backend_selection": ATTENTION_BACKEND_SELECTED,
        "attention_policy": "Selected at worker startup; ComfyUI may choose different kernels per operation",
        "first_block_cache": {
            "mode": FIRST_BLOCK_CACHE.mode,
            "preset": FIRST_BLOCK_CACHE.preset,
        },
        "capabilities": {
            "image_to_video": True,
            "audio_output": AUDIO,
            "audio_latents_generated": True,
            "loras": bool(lora_names),
            "default_lora": TURBO_LORA if turbo_available else None,
            "durations_seconds": SUPPORTED_DURATIONS,
        },
        "supported_durations_seconds": SUPPORTED_DURATIONS,
        "supported_steps": list(range(1, 51)),
        "supported_resolutions": list(SUPPORTED_RESOLUTIONS),
        "default_parameters": {"mode": "turbo", "steps": DEFAULT_STEPS, "resolution": DEFAULT_RESOLUTION, "lora_name": TURBO_LORA if turbo_available else None, "lora_strength": 1.0},
        "supported_modes": ["base", "turbo"] if any_lora_available else ["base"],
    }


@app.get("/v1/loras")
async def list_loras(authorization: str | None = Header(None)):
    _authorized(authorization)
    try:
        available_nodes = await comfy.health(audio=AUDIO)
    except Exception:
        raise HTTPException(503, "ComfyUI is not ready") from None
    names = _available_lora_names() if "LoraLoaderModelOnly" in available_nodes else []
    return {
        "loras": [{"name": name, "available": True, "default_strength": 1.0} for name in names],
        "default_lora": TURBO_LORA if TURBO_LORA in names else None,
    }


@app.post("/v1/loras", status_code=201)
async def upload_lora(file: UploadFile = File(...), authorization: str | None = Header(None)):
    _authorized(authorization)
    filename = _safe_lora_filename(file.filename)
    try:
        size = await _store_lora(file, filename)
    finally:
        await file.close()
    return {"name": filename, "size": size}


@app.post("/v1/jobs")
async def create_job(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    duration_seconds: int = Form(...),
    steps: int | None = Form(None),
    mode: str = Form("turbo"),
    resolution: int = Form(DEFAULT_RESOLUTION),
    lora_name: str | None = Form(None),
    lora_strength: float = Form(1.0),
    lora_stack: str | None = Form(None),
    authorization: str | None = Header(None),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
):
    _authorized(authorization)
    if duration_seconds not in SUPPORTED_DURATIONS:
        raise HTTPException(422, "duration_seconds must be between 3 and 30")
    if mode not in {"base", "turbo"}:
        raise HTTPException(422, "mode must be base or turbo")
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise HTTPException(422, "resolution must be 480, 576, 672, or 768")
    if lora_stack is None and (not math.isfinite(lora_strength) or not 0.0 <= lora_strength <= 2.0):
        raise HTTPException(422, "lora_strength must be between 0 and 2")
    available_loras = _available_lora_names()
    if mode == "turbo" and lora_stack is None and not lora_name and TURBO_LORA not in available_loras:
        raise HTTPException(503, "Turbo mode is unavailable because its default LoRA is not installed")
    try:
        if lora_stack is not None:
            if lora_name is not None:
                raise ValueError("Use either lora_stack or legacy lora_name/lora_strength, not both")
            selected_stack = resolve_lora_stack(mode, json.loads(lora_stack), available_loras, TURBO_LORA)
        elif lora_name:
            selected_stack = resolve_lora_stack(mode, [{"name": lora_name, "strength": lora_strength}], available_loras, TURBO_LORA)
        else:
            selected_stack = resolve_lora_stack(mode, None, available_loras, TURBO_LORA)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(422, "lora_stack must be a JSON array") from None
    if FIRST_BLOCK_CACHE.mode != "off" or selected_stack:
        try:
            available_nodes = await comfy.health(audio=AUDIO)
        except Exception:
            raise HTTPException(503, "ComfyUI is not ready") from None
        if FIRST_BLOCK_CACHE.mode != "off" and FIRST_BLOCK_CACHE_NODE not in available_nodes:
            raise HTTPException(503, "ComfyUI FirstBlockCache node is missing; install the pinned custom node or set IMAGINE_VIDEO_FIRST_BLOCK_CACHE=off")
        if selected_stack and "LoraLoaderModelOnly" not in available_nodes:
            raise HTTPException(503, "ComfyUI cannot load LoRAs")
    try:
        selected_steps = sampling_steps(steps)
    except ValueError as exc:
        raise HTTPException(422, "steps must be an integer from 1 to 50") from exc
    if not prompt.strip() or len(prompt) > 10000:
        raise HTTPException(422, "Prompt must contain 1 to 10000 characters")
    if idempotency_key:
        if len(idempotency_key) > 200:
            raise HTTPException(422, "Idempotency-Key is too long")
        prior = idempotency.get(idempotency_key)
        if prior and prior in jobs:
            return JSONResponse(_snapshot(jobs[prior]), status_code=200)
    global admission_reserved
    async with admission_lock:
        if admission_reserved or queue.full() or slot_lock.locked() or any(j["status"] in {"queued", "running"} for j in jobs.values()):
            raise HTTPException(409, "The worker is processing another job")
        admission_reserved = True
    job_id = uuid.uuid4().hex
    folder = DATA_ROOT / job_id
    try:
        folder.mkdir(mode=0o700, parents=True)
        input_path = folder / "input.upload"
        _, digest, image_format, image_width, image_height = await _bytes_to_file(file, input_path)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        async with admission_lock:
            admission_reserved = False
    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    if input_path.suffix != suffix:
        renamed = folder / ("input" + suffix)
        input_path.rename(renamed)
        input_path = renamed
    effective_width, effective_height = canvas_for_image(image_width, image_height, resolution)
    legacy_name = selected_stack[0]["name"] if len(selected_stack) == 1 else None
    legacy_strength = selected_stack[0]["strength"] if len(selected_stack) == 1 else float(lora_strength)
    job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0, "error": None, "prompt": prompt, "duration_seconds": duration_seconds, "steps": selected_steps, "mode": mode, "resolution": resolution, "effective_width": effective_width, "effective_height": effective_height, "lora_name": legacy_name, "lora_strength": legacy_strength, "lora_stack": selected_stack, "folder": folder, "input_path": input_path, "created_at": time.time(), "input_sha256": digest, "seed": int.from_bytes(os.urandom(8), "big"), "image_width": image_width, "image_height": image_height, "cancel_event": asyncio.Event()}
    jobs[job_id] = job
    if idempotency_key:
        idempotency[idempotency_key] = job_id
    queue.put_nowait(job_id)
    return JSONResponse(_snapshot(job), status_code=202)


def _get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def _expire_if_due(job: dict[str, Any]):
    completed_at = job.get("completed_at")
    if job["status"] == "completed" and completed_at is not None and time.time() >= completed_at + RETENTION_SECONDS:
        _cleanup(job)
        job.update(status="expired", stage="expired", progress=None, result_path=None, error=None)


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    _expire_if_due(job)
    return _snapshot(job)


@app.get("/v1/jobs/{job_id}/result")
async def get_result(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    _expire_if_due(job)
    if job["status"] != "completed" or not job.get("result_path") or not job["result_path"].is_file():
        raise HTTPException(410 if job["status"] == "expired" else 409, "Job result is not available")
    return FileResponse(job["result_path"], media_type="video/mp4", filename=f"imagine-{job_id}.mp4", headers={"Cache-Control": "no-store"})


@app.post("/v1/jobs/{job_id}/ack", status_code=204)
async def ack_job(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    _expire_if_due(job)
    if job["status"] in {"queued", "running"}:
        raise HTTPException(409, "Cannot acknowledge a job while it is running")
    if job.get("remote_cleanup_pending") and job.get("prompt_id"):
        try:
            if not await comfy.cancel(job["prompt_id"]):
                raise HTTPException(409, "Job cleanup is still in progress")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(409, "Job cleanup is still in progress") from None
        job["remote_cleanup_pending"] = False
        await comfy.forget_prompt(job["prompt_id"])
        _purge_comfy_files(job)
    _cleanup(job)
    job.update(status="expired", stage="expired", progress=None, result_path=None, error=None)
    return Response(status_code=204)


@app.delete("/v1/jobs/{job_id}")
async def delete_job(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    if job["status"] in {"queued", "running"}:
        was_running = job["status"] == "running"
        job["cancel_event"].set()
        job.update(status="cancelled", stage="cancelled", progress=None, completed_at=time.time())
        if was_running and job.get("prompt_id"):
            try:
                await comfy.cancel(job["prompt_id"])
            except Exception:
                pass
        if was_running:
            return _snapshot(job)
        job.pop("prompt", None)
        _purge_comfy_files(job)
        _cleanup(job)
        return _snapshot(job)
    if job.get("remote_cleanup_pending") and job.get("prompt_id"):
        try:
            if not await comfy.cancel(job["prompt_id"]):
                return _snapshot(job)
            await comfy.forget_prompt(job["prompt_id"])
        except Exception:
            return _snapshot(job)
        job["remote_cleanup_pending"] = False
    _purge_comfy_files(job)
    if job.get("prompt_id"):
        try:
            await comfy.forget_prompt(job["prompt_id"])
        except Exception:
            pass
    _cleanup(job)
    if job["status"] != "cancelled":
        job.update(status="expired", stage="expired", progress=None, result_path=None, error=None)
    return _snapshot(job)
