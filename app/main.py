"""Ephemeral, single-slot API for private image-to-video generation."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from .comfy import ComfyClient, ModelLoaderError, new_client_id
from .workflow import SUPPORTED_DURATIONS, VIDEO_FPS, TURBO_LORA, build_turbo_workflow, build_workflow, sampling_steps

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
JOB_TIMEOUT_SECONDS = max(60, int(os.environ.get("IMAGINE_VIDEO_JOB_TIMEOUT_SECONDS", str(90 * 60))))
MAX_PIXELS = 40_000_000
RETENTION_SECONDS = 15 * 60
DEBUG_METRICS = os.environ.get("IMAGINE_DEBUG_METRICS", "0") == "1"
AUDIO = os.environ.get("IMAGINE_H3_AUDIO", "1") == "1"
COMFY_NODE_LABELS = {
    "1": "diffusion_model_loader", "2": "text_encoder_loader",
    "3": "video_vae_loader", "4": "audio_vae_loader", "5": "image_load",
    "6": "prompt_and_image_encode", "11": "diffusion", "12": "video_vae_decode",
    "13": "audio_decode", "14": "video_mux", "15": "mp4_export", "16": "turbo_lora",
}
VRAM_PROFILE = os.environ.get("IMAGINE_H3_VRAM_PROFILE", "auto")
_generation_count = 0


def _metric(event: str, **values):
    """Emit technical-only diagnostics; never include request or credential data."""
    if DEBUG_METRICS:
        print("imagine_metric " + json.dumps({"event": event, **values}, sort_keys=True), flush=True)


async def _sample_gpu_peak(stop: asyncio.Event, peaks: dict[str, int | None]):
    """Sample NVIDIA device memory without touching prompts or media."""
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
    return {key: job.get(key) for key in ("id", "status", "stage", "progress", "error", "steps", "mode")}


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
        step_seconds: list[float] = []
        _metric(
            "generation_started", generation=generation_number, warm=generation_number > 1,
            generation_state="initial" if generation_number == 1 else "subsequent",
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
            if job["mode"] == "turbo":
                lora_file = COMFY_ROOT / "models" / "loras" / TURBO_LORA
                if not lora_file.is_file():
                    raise RuntimeError("Turbo mode requested but its 8-step LoRA is not installed")
                workflow = build_turbo_workflow(job["prompt"], comfy_name, job["duration_seconds"], job["seed"], job["image_width"], job["image_height"], job["id"], audio=AUDIO)
            else:
                workflow = build_workflow(job["prompt"], comfy_name, job["duration_seconds"], job["seed"], job["image_width"], job["image_height"], job["id"], steps=job["steps"], audio=AUDIO)
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
                "phase", name="comfy_generation", seconds=round(time.perf_counter() - phase_started, 3),
                nodes={COMFY_NODE_LABELS[node]: round(seconds, 3) for node, seconds in node_seconds.items() if node in COMFY_NODE_LABELS},
                diffusion_step_seconds=step_seconds,
            )
            for node in ("1", "2", "3", "4", "16"):
                if node in node_seconds:
                    _metric(
                        "model_loader_node", loader=COMFY_NODE_LABELS[node],
                        seconds=round(node_seconds[node], 3), generation=generation_number,
                        load_state="initial" if generation_number == 1 else "subsequent",
                    )
            if job["cancel_event"].is_set() or history is None:
                return
            job["stage"] = "collecting_result"
            phase_started = time.perf_counter()
            outputs = history.get("outputs", {})
            job["remote_outputs"] = [item for node in outputs.values() for item in (node.get("images", []) + node.get("gifs", []) + node.get("videos", []) + node.get("files", [])) if item.get("filename")]
            (job["folder"] / "remote.json").write_text(json.dumps({"comfy_name": comfy_name, "remote_outputs": job["remote_outputs"], "prompt_id": prompt_id}))
            info = next((item for item in job["remote_outputs"] if item.get("filename", "").lower().endswith(".mp4")), None)
            if not info:
                raise RuntimeError("ComfyUI finished without a video output")
            payload = await comfy.download(info)
            _metric("phase", name="result_download", seconds=round(time.perf_counter() - phase_started, 3))
            if job["cancel_event"].is_set():
                return
            if not payload:
                raise RuntimeError("ComfyUI returned an empty video")
            raw_path = job["folder"] / "comfy-output.mp4"
            result_path = job["folder"] / "result.mp4"
            raw_path.write_bytes(payload)
            frame_count = job["duration_seconds"] * VIDEO_FPS
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file,pipe", "-i", str(raw_path), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1", "-frames:v", str(frame_count), "-t", str(job["duration_seconds"]), "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", f"atrim=duration={job['duration_seconds']}", "-movflags", "+faststart", str(result_path)]
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            phase_started = time.perf_counter()
            communicate = asyncio.create_task(process.communicate())
            cancelled = asyncio.create_task(job["cancel_event"].wait())
            try:
                done, _ = await asyncio.wait({communicate, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                if cancelled in done:
                    return
                communicate.result()
            finally:
                cancelled.cancel()
                await asyncio.gather(cancelled, return_exceptions=True)
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(asyncio.shield(communicate), timeout=5)
                    except asyncio.TimeoutError:
                        if process.returncode is None:
                            process.kill()
                        await communicate
            if process.returncode:
                raise RuntimeError("Video finalization failed")
            probe = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=nb_frames,avg_frame_rate", "-of", "json", str(result_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                output, _ = await asyncio.wait_for(probe.communicate(), timeout=10)
            finally:
                if probe.returncode is None:
                    probe.kill()
                    await probe.wait()
            streams = json.loads(output)["streams"] if probe.returncode == 0 else []
            if (len(streams) != 1 or streams[0].get("nb_frames") != str(frame_count)
                    or streams[0].get("avg_frame_rate") != f"{VIDEO_FPS}/1"):
                raise RuntimeError("Video did not match the requested native frame count and FPS")
            _metric("phase", name="mp4_finalize", seconds=round(time.perf_counter() - phase_started, 3))
            if job["cancel_event"].is_set():
                return
            raw_path.unlink(missing_ok=True)
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
            _metric("runtime", torch=torch.__version__, cuda=torch.version.cuda, sdpa="PyTorch automatic dispatch (ComfyUI default)", optional_attention_backend="none pinned by this image")
        except Exception:
            _metric("runtime", sdpa="PyTorch automatic dispatch (ComfyUI default)", optional_attention_backend="none pinned by this image")
    worker_task = asyncio.create_task(_worker())
    reaper_task = asyncio.create_task(_reaper())
    _metric("worker_startup", seconds=round(time.perf_counter() - startup_started, 3), vram_profile=VRAM_PROFILE, audio=AUDIO)


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
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    )
    if any(not (COMFY_ROOT / "models" / relative).is_file() for relative in models):
        raise HTTPException(503, "Worker is not ready")
    try:
        available_nodes = await comfy.health()
    except Exception:
        raise HTTPException(503, "Worker is not ready") from None
    turbo_file = COMFY_ROOT / "models" / "loras" / TURBO_LORA
    turbo_available = turbo_file.is_file() and "LoraLoaderModelOnly" in available_nodes
    return {"status": "ok", "model": "MiniMax H3 FL2VA INT8 ConvRot", "attention_policy": "PyTorch automatic dispatch via ComfyUI; selected kernel may vary by operation", "capabilities": {"image_to_video": True, "audio_output": AUDIO, "audio_latents_generated": True, "turbo_lora_8step": turbo_available, "durations_seconds": SUPPORTED_DURATIONS}, "supported_durations_seconds": SUPPORTED_DURATIONS, "supported_steps": [8, 12, 20], "supported_modes": ["base", "turbo"] if turbo_available else ["base"]}


@app.post("/v1/jobs")
async def create_job(file: UploadFile = File(...), prompt: str = Form(...), duration_seconds: int = Form(...), steps: int | None = Form(None), mode: str = Form("base"), authorization: str | None = Header(None), idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    _authorized(authorization)
    if duration_seconds not in SUPPORTED_DURATIONS:
        raise HTTPException(422, "duration_seconds must be between 3 and 30")
    if mode not in {"base", "turbo"}:
        raise HTTPException(422, "mode must be base or turbo")
    if mode == "turbo":
        if steps not in (None, 8):
            raise HTTPException(422, "Turbo mode requires 8 steps")
        if not (COMFY_ROOT / "models" / "loras" / TURBO_LORA).is_file():
            raise HTTPException(503, "Turbo mode is unavailable because its 8-step LoRA is not installed")
        try:
            available_nodes = await comfy.health()
        except Exception:
            raise HTTPException(503, "ComfyUI is not ready") from None
        if "LoraLoaderModelOnly" not in available_nodes:
            raise HTTPException(503, "Turbo mode is unavailable because ComfyUI lacks LoraLoaderModelOnly")
        selected_steps = 8
    else:
        try:
            selected_steps = sampling_steps(steps)
        except ValueError as exc:
            raise HTTPException(422, "steps must be 8, 12, or 20") from exc
    prompt = prompt.strip()
    if not prompt or len(prompt) > 2000:
        raise HTTPException(422, "Prompt must contain 1 to 2000 characters")
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
    job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0, "error": None, "prompt": prompt, "duration_seconds": duration_seconds, "steps": selected_steps, "mode": mode, "folder": folder, "input_path": input_path, "created_at": time.time(), "input_sha256": digest, "seed": int.from_bytes(os.urandom(8), "big"), "image_width": image_width, "image_height": image_height, "cancel_event": asyncio.Event()}
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
