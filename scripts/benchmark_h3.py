"""Run fixed-seed silent H3 comparisons on an idle worker's loopback ComfyUI."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
from statistics import median
import sys
from time import perf_counter
import uuid

from PIL import Image

from app.comfy import new_client_id
from app.lora_catalog import available_lora_names, resolve_lora_stack
from app.workflow import (
    BUFFERED_VAE_NODE, FIRST_BLOCK_CACHE_MODES, FIRST_BLOCK_CACHE_NODE,
    TURBO_LORA, VIDEO_FPS, build_workflow, canvas_for_image, model_frame_count, sampling_steps,
)
from h3_diagnostics import HISTORY_KEY


class BenchmarkError(Exception):
    """A static, privacy-safe error message for the benchmark CLI."""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--lora-stack", type=Path, required=True, help="JSON file in worker lora_stack format")
    parser.add_argument("--out", type=Path, required=True, help="New case directory for private MP4s and JSON")
    parser.add_argument("--case", required=True, help="Short technical label for the run")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--cache", choices=FIRST_BLOCK_CACHE_MODES, default="safe")
    parser.add_argument("--vae-output", choices=("cpu", "gpu-buffer"), default="cpu")
    parser.add_argument("--expect-vram", choices=("auto", "high"), required=True)
    parser.add_argument("--warmups", type=int, choices=range(4), default=1, metavar="0..3")
    parser.add_argument("--runs", type=int, choices=range(1, 6), default=1, metavar="1..5")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.case):
        parser.error("--case must be a lowercase technical label of 1 to 40 characters")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")
    try:
        model_frame_count(args.seconds)
        sampling_steps(args.steps)
    except ValueError as error:
        parser.error(str(error))
    return args


def fixture(args, lora_root):
    try:
        with Image.open(args.image) as image:
            source_size = image.size
            suffix, content_type = {
                "JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"),
                "WEBP": (".webp", "image/webp"),
            }[image.format]
            image.verify()
        prompt = args.prompt_file.read_text(encoding="utf-8")
        loras = resolve_lora_stack("base", json.loads(args.lora_stack.read_text()),
                                   available_lora_names(lora_root), TURBO_LORA)
    except (OSError, ValueError, KeyError, SyntaxError):
        raise BenchmarkError("Check the image, prompt file and ordered LoRA stack on disk") from None
    if not 0 < len(prompt.strip()) <= 10_000:
        raise BenchmarkError("Prompt file must contain 1 to 10000 characters")
    if canvas_for_image(*source_size, 672) != (672, 1184):
        raise BenchmarkError("Use an input image that maps to the baseline 672x1184 canvas")
    return source_size, suffix, content_type, prompt, loras


def require_metrics(history, args):
    metrics = history.get(HISTORY_KEY, {})
    if not metrics:
        raise BenchmarkError("Start ComfyUI with IMAGINE_DEBUG_METRICS=1")
    runtime = metrics["runtime"]
    state = runtime["vram_state"]
    if (args.expect_vram == "high") != (state == "HIGH_VRAM"):
        raise BenchmarkError("The server's effective VRAM state does not match --expect-vram")
    if metrics["diffusion_step_observed"] != args.steps or "video_vae_decode" not in metrics["node_seconds"]:
        raise BenchmarkError("ComfyUI did not report the expected diffusion steps and actual VAE execution")
    cache = metrics["first_block_cache"]
    if args.cache != "off" and (not cache["hook_ready"] or not cache["samples"]):
        raise BenchmarkError("Pinned FirstBlockCache metrics are missing")
    export, = history["outputs"]["15"]["video_export_metrics"]
    if export["encoder"] != "h264_nvenc":
        raise BenchmarkError("Expected NVENC on this benchmark; check the preflight and runtime logs")
    return metrics, export


async def run(args):
    # Loopback only, on a maintenance Pod. The public API deliberately chooses its own seeds.
    from app.main import _sample_gpu_peak, _validate_video
    from app.comfy import ComfyClient

    base = Path(os.environ.get("IMAGINE_VIDEO_RUN_ROOT", "/tmp/imagine-video-worker")) / "comfy"
    comfy_root = Path(os.environ.get("COMFY_ROOT", "ComfyUI"))
    source_size, suffix, mime, prompt, loras = fixture(args, comfy_root / "models" / "loras")
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    client = ComfyClient("http://127.0.0.1:8188", "ws://127.0.0.1:8188/ws")
    try:
        nodes = await client.health(audio=False)
        if args.cache != "off" and FIRST_BLOCK_CACHE_NODE not in nodes:
            raise BenchmarkError("The pinned FirstBlockCache node is missing")
        if args.vae_output == "gpu-buffer" and BUFFERED_VAE_NODE not in nodes:
            raise BenchmarkError("The H3 buffered VAE node is missing")
        measured = []
        for index in range(args.warmups + args.runs):
            warmup = index < args.warmups
            sequence = index + 1 if warmup else index - args.warmups + 1
            seed = (args.seed + sequence) % 2**64 if warmup else args.seed
            report = await generation(client, args, base, prompt, loras, source_size,
                                      suffix, mime, seed, sequence, warmup,
                                      _sample_gpu_peak, _validate_video)
            print(json.dumps(report, sort_keys=True), flush=True)
            if not warmup and report["inference"]["cuda_trace"] == "off":
                measured.append(report)
        if measured:
            summary = {"event": "summary", "case": args.case, "runs": len(measured)}
            summary["median_seconds"] = {
                "total": median(row["total_seconds"] for row in measured),
                "diffusion": median(row["inference"]["node_seconds"]["diffusion"] for row in measured),
                "vae": median(row["inference"]["node_seconds"]["video_vae_decode"] for row in measured),
                "mp4_export": median(row["export"]["seconds"] for row in measured),
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        await client.close()


async def generation(client, args, base, prompt, loras, source_size, suffix, mime,
                     seed, sequence, warmup, sample_peaks, validate_video):
    queue = (await client.http.get("/queue")).raise_for_status().json()
    if queue["queue_running"] or queue["queue_pending"]:
        raise BenchmarkError("ComfyUI is busy; drain the Pod before benchmarking")
    job_id = uuid.uuid4().hex
    name = f"h3_bench_{job_id}{suffix}"
    prefix = Path("imagine_h3", job_id)
    prompt_id = None
    done = False
    started = perf_counter()
    stop = asyncio.Event()
    peaks = {"vram_mb": None, "container_ram_mb": None}
    resource_task = asyncio.create_task(sample_peaks(stop, peaks))
    try:
        uploaded = await client.upload(args.image, name, mime)
        if uploaded != name:
            raise BenchmarkError("Unexpected path returned by the benchmark input upload")
        workflow = build_workflow(
            prompt, name, args.seconds, seed, *source_size, job_id, steps=args.steps,
            audio=False, lora_stack=loras, first_block_cache=args.cache, vae_output=args.vae_output,
        )
        client_id = new_client_id()
        prompt_id = await client.queue(workflow, client_id)
        history = await client.wait(prompt_id, client_id, lambda _: None, lambda: False)
        done = True
        if history["status"]["status_str"] != "success":
            raise BenchmarkError("ComfyUI did not complete the benchmark generation")
        metrics, export = require_metrics(history, args)
        output, = history["outputs"]["15"]["images"]
        filename = output["filename"]
        if (output.get("subfolder") != str(prefix) or output.get("type") != "output"
                or Path(filename).name != filename or not filename.endswith(".mp4")):
            raise BenchmarkError("Unexpected benchmark output location")
        source = base / "output" / prefix / filename
        if not source.is_file():
            raise BenchmarkError("Benchmark must run inside the worker container with its local output directory")
        finalize = perf_counter()
        await validate_video(source, args.seconds * VIDEO_FPS, args.seconds, False, 672, 1184)
        finalize_seconds = round(perf_counter() - finalize, 6)
        total = round(perf_counter() - started, 6)
        if not warmup:
            shutil.copyfile(source, args.out / f"run-{sequence}.mp4")
    finally:
        stop.set()
        await asyncio.gather(resource_task, return_exceptions=True)
        if prompt_id is not None and not done:
            done = await client.cancel(prompt_id)
        if done or prompt_id is None:
            (base / "input" / name).unlink(missing_ok=True)
            shutil.rmtree(base / "output" / prefix, ignore_errors=True)
            if prompt_id is not None:
                await client.forget_prompt(prompt_id)

    report = {
        "event": "warmup" if warmup else "measured", "case": args.case, "run": sequence,
        "seed": seed, "steps": args.steps, "sampler": "res_multistep", "scheduler": "simple",
        "width": 672, "height": 1184, "duration_seconds": args.seconds, "fps": VIDEO_FPS,
        "model_frames": model_frame_count(args.seconds), "frames": args.seconds * VIDEO_FPS,
        "lora_count": len(loras), "lora_strengths": [item["strength"] for item in loras],
        "audio": False, "cache_mode": args.cache, "vae_output": args.vae_output,
        "vram_profile": args.expect_vram, "total_seconds": total, "finalize_seconds": finalize_seconds,
        "peaks": peaks, "inference": metrics, "export": export,
    }
    path = args.out / f'{report["event"]}-{sequence}.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    args = arguments()
    try:
        asyncio.run(run(args))
    except BenchmarkError as error:
        print(f"H3 benchmark: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("H3 benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # Upstream exception strings can contain prompt payloads or local filenames.
        print(f"H3 benchmark failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
