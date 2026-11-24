"""Private, single-slot Imagine API backed by LightWan through LightX2V."""
from __future__ import annotations

import asyncio
import hmac
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from .lightwan import LightWanRuntime

TOKEN = os.environ.get("IMAGINE_VIDEO_TOKEN", "")
DATA_ROOT = Path(os.environ.get("IMAGINE_VIDEO_DATA", "/tmp/imagine-video-worker"))
MAX_UPLOAD_BYTES = int(os.environ.get("IMAGINE_VIDEO_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
JOB_TIMEOUT_SECONDS = max(60, int(os.environ.get("IMAGINE_VIDEO_JOB_TIMEOUT_SECONDS", str(90 * 60))))
RETENTION_SECONDS = 15 * 60
MAX_PIXELS = 40_000_000
LONG_VIDEO_ENABLED = os.environ.get("IMAGINE_LIGHTWAN_ENABLE_LONG_VIDEO", "0") == "1"
FPS = 16

app = FastAPI(title="Imagine Video Worker", docs_url=None, redoc_url=None)
runtime = LightWanRuntime()
jobs: dict[str, dict[str, Any]] = {}
idempotency: dict[str, str] = {}
queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
slot_lock = asyncio.Lock()
admission_lock = asyncio.Lock()
admission_reserved = False
worker_task: asyncio.Task | None = None
reaper_task: asyncio.Task | None = None


def _authorized(authorization: str | None) -> None:
    if not TOKEN:
        raise HTTPException(503, "Worker token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, TOKEN):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})


@app.middleware("http")
async def token_gate(request, call_next):
    if request.url.path.startswith("/v1/"):
        try:
            _authorized(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    return await call_next(request)


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job.get(key) for key in ("id", "status", "stage", "progress", "error", "steps", "mode", "model_steps")}


def _cleanup(job: dict[str, Any]) -> None:
    folder = job.get("folder")
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
    job["result_path"] = None
    job.pop("prompt", None)


def _frame_counts(duration_seconds: int) -> tuple[int, int]:
    """Wan alignment is 4n+1; encode only the first exact-duration frames."""
    if duration_seconds not in (10, 15):
        raise ValueError("duration must be 10 or 15 seconds")
    output_frames = duration_seconds * FPS
    model_frames = output_frames + 1
    if (model_frames - 1) % 4:
        raise RuntimeError("Computed frame count violates Wan temporal alignment")
    return model_frames, output_frames


def _video_size(width: int, height: int) -> tuple[int, int]:
    """Keep the input orientation and request an approximately 720p canvas."""
    ratio = width / height
    short_edge = 720
    long_edge = min(1280, max(720, round(max(ratio, 1 / ratio) * short_edge / 32) * 32))
    if ratio >= 1:
        return short_edge, long_edge
    return long_edge, short_edge


async def _store_upload(upload: UploadFile, path: Path) -> tuple[int, int]:
    total = 0
    with path.open("xb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Photo exceeds the upload limit")
            output.write(chunk)
    if total == 0:
        raise HTTPException(422, "Photo is empty")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Upload a JPEG, PNG, or WebP photo")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Photo dimensions are too large")
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if oriented.mode not in {"RGB", "L"}:
                oriented = oriented.convert("RGB")
            normalized = path.with_name("input-normalized" + path.suffix)
            oriented.save(normalized, format=image.format)
        normalized.replace(path)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Upload is not a supported image") from None
    return width, height


async def _wait_or_cancel(awaitable, event, timeout):
    operation = asyncio.create_task(awaitable)
    cancelled = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait({operation, cancelled}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if cancelled in done:
            return False
        if operation not in done:
            raise TimeoutError
        operation.result()
        return True
    finally:
        cancelled.cancel()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, cancelled, return_exceptions=True)


async def _finalize(raw_path, result_path, output_frames):
    # Trim one generated frame at the native frame rate.
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-protocol_whitelist", "file,pipe", "-i", str(raw_path),
        "-map", "0:v:0", "-an", "-map_metadata", "-1", "-frames:v", str(output_frames),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(result_path),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        if await process.wait():
            raise RuntimeError("Video finalization failed")
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def _run_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        queue.task_done()
        return
    async with slot_lock:
        cancel_event = job["cancel_event"]
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        try:
            if cancel_event.is_set():
                return
            job.update(status="running", stage="loading_image", progress=5)
            model_frames, output_frames = _frame_counts(job["duration_seconds"])
            height, width = _video_size(job["image_width"], job["image_height"])
            raw_path = job["folder"] / "lightwan-output.mp4"
            result_path = job["folder"] / "result.mp4"
            job.update(stage="generating_video", progress=10, backend_active=True)
            if not await _wait_or_cancel(
                runtime.generate(
                    prompt=job.pop("prompt"), image_path=job["input_path"], output_path=raw_path,
                    seed=job["seed"], height=height, width=width, model_frames=model_frames,
                ),
                cancel_event, JOB_TIMEOUT_SECONDS,
            ):
                return
            job["input_path"].unlink(missing_ok=True)
            if not raw_path.is_file() or raw_path.stat().st_size == 0:
                raise RuntimeError("LightX2V finished without a video file")
            job.update(stage="finalizing_video", progress=90)
            if not await _wait_or_cancel(
                _finalize(raw_path, result_path, output_frames), cancel_event, max(0, deadline - time.monotonic()),
            ):
                return
            raw_path.unlink(missing_ok=True)
            job.update(
                status="completed", stage="completed", progress=100, result_path=result_path,
                completed_at=time.time(), error=None,
            )
        except TimeoutError:
            job.update(status="failed", stage="failed", progress=None, error="Video generation timed out.", completed_at=time.time())
        except Exception:
            if not cancel_event.is_set():
                job.update(status="failed", stage="failed", progress=None, error="Video generation failed. Please try again.", completed_at=time.time())
        finally:
            job["backend_active"] = False
            if job["status"] != "completed":
                _cleanup(job)
            queue.task_done()


async def _worker() -> None:
    while True:
        await _run_job(await queue.get())
        while not runtime.ready:
            try:
                await runtime.load()
            except Exception:
                print("LightWan runtime recovery failed; retrying", flush=True)
                await asyncio.sleep(10)


async def _reaper() -> None:
    while True:
        await asyncio.sleep(1)
        now = time.time()
        for job in list(jobs.values()):
            if not job.get("backend_active") and job.get("status") in {"completed", "failed", "cancelled", "expired"} and now - job.get("completed_at", job["created_at"]) >= RETENTION_SECONDS:
                _cleanup(job)
                job.update(status="expired", stage="expired", progress=None, error=None)


@app.on_event("startup")
async def startup() -> None:
    if not DATA_ROOT.resolve().is_relative_to(Path("/tmp").resolve()):
        raise RuntimeError("Job storage must stay under /tmp")
    DATA_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    DATA_ROOT.chmod(0o700)
    for child in DATA_ROOT.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    if not TOKEN:
        raise RuntimeError("Configure IMAGINE_VIDEO_TOKEN")
    await runtime.load()
    print("LightWan runtime ready", flush=True)
    global worker_task, reaper_task
    worker_task = asyncio.create_task(_worker())
    reaper_task = asyncio.create_task(_reaper())


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in (worker_task, reaper_task):
        if task:
            task.cancel()
    await asyncio.gather(*(task for task in (worker_task, reaper_task) if task), return_exceptions=True)
    await runtime.close()
    for job in jobs.values():
        _cleanup(job)


@app.get("/v1/health")
async def health(authorization: str | None = Header(None)):
    _authorized(authorization)
    if not runtime.ready:
        raise HTTPException(503, "Worker is not ready")
    return {
        "status": "ok",
        "model": "LightWan2.2-A14B I2V",
        "runtime": "LightX2V 0.6.0",
        "precision": "NVFP4",
        "actual_steps": 4,
        "supported_steps": [8],
        "supported_modes": ["turbo"],
        "wire_compatibility": {"mode": "turbo", "steps": 8, "actual_model_steps": 4},
        "attention": {"self": "dynamic_sparse_attn (sage2, 90% sparsity)", "cross": "sage_attn2"},
        "resolution": "approximately 720p, aspect-preserving",
        "fps": FPS,
        "supported_durations_seconds": [10, 15] if LONG_VIDEO_ENABLED else [],
        "experimental_long_video": LONG_VIDEO_ENABLED,
        "gpu": runtime.device_name,
    }


@app.post("/v1/jobs")
async def create_job(
    file: UploadFile = File(...), prompt: str = Form(...), duration_seconds: int = Form(...),
    steps: int | None = Form(None), mode: str = Form("turbo"), authorization: str | None = Header(None),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
):
    _authorized(authorization)
    if not LONG_VIDEO_ENABLED:
        raise HTTPException(422, "LightWan 10/15-second output is experimental; enable IMAGINE_LIGHTWAN_ENABLE_LONG_VIDEO=1")
    if duration_seconds not in (10, 15):
        raise HTTPException(422, "duration_seconds must be 10 or 15")
    if mode != "turbo":
        raise HTTPException(422, "LightWan uses the turbo wire compatibility mode")
    if steps not in (None, 8):
        raise HTTPException(422, "LightWan uses the 8-step turbo compatibility value")
    if not runtime.ready:
        raise HTTPException(503, "LightWan runtime is not ready")
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
        suffix = Path(file.filename or "input.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".upload"
        input_path = folder / ("input" + suffix)
        image_width, image_height = await _store_upload(file, input_path)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        async with admission_lock:
            admission_reserved = False
    job = {
        "id": job_id, "status": "queued", "stage": "queued", "progress": 0,
        "error": None, "prompt": prompt, "duration_seconds": duration_seconds,
        "steps": 8, "model_steps": 4, "mode": "turbo", "folder": folder,
        "input_path": input_path, "created_at": time.time(), "seed": int.from_bytes(os.urandom(8), "big"),
        "image_width": image_width, "image_height": image_height, "cancel_event": asyncio.Event(),
    }
    jobs[job_id] = job
    if idempotency_key:
        idempotency[idempotency_key] = job_id
    queue.put_nowait(job_id)
    return JSONResponse(_snapshot(job), status_code=202)


def _get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def _expire_if_due(job: dict[str, Any]) -> None:
    if job["status"] == "completed" and time.time() >= job.get("completed_at", time.time()) + RETENTION_SECONDS:
        _cleanup(job)
        job.update(status="expired", stage="expired", progress=None, error=None)


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
    path = job.get("result_path")
    if job["status"] != "completed" or not path or not path.is_file():
        raise HTTPException(410 if job["status"] == "expired" else 409, "Job result is not available")
    return FileResponse(path, media_type="video/mp4", filename=f"imagine-{job_id}.mp4", headers={"Cache-Control": "no-store"})


@app.post("/v1/jobs/{job_id}/ack", status_code=204)
async def ack_job(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    if job["status"] in {"queued", "running"}:
        raise HTTPException(409, "Cannot acknowledge a job while it is running")
    if job.get("backend_active"):
        raise HTTPException(409, "Job cleanup is still in progress")
    _cleanup(job)
    job.update(status="expired", stage="expired", progress=None, error=None)
    return Response(status_code=204)


@app.delete("/v1/jobs/{job_id}")
async def delete_job(job_id: str, authorization: str | None = Header(None)):
    _authorized(authorization)
    job = _get_job(job_id)
    if job["status"] in {"queued", "running"}:
        was_queued = job["status"] == "queued"
        job["cancel_event"].set()
        job.update(status="cancelled", stage="cancelled", progress=None, completed_at=time.time())
        if was_queued:
            _cleanup(job)
        return _snapshot(job)
    if job.get("backend_active"):
        return _snapshot(job)
    _cleanup(job)
    if job["status"] != "cancelled":
        job.update(status="expired", stage="expired", progress=None, error=None)
    return _snapshot(job)
