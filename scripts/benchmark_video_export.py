#!/usr/bin/env python3
"""Compare the old two-pass export, the worker exporter, and an FFmpeg RGB pipe."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter

import av
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_export import ExportProfile, FrameBatches, add_video_stream, export_video, set_sdr_color


FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
MIB = 1024**2


def cpu_environment():
    def read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return None

    root = Path("/sys/fs/cgroup")
    cgroups = (read(Path("/proc/self/cgroup")) or "").splitlines()
    cgroup = next((line[3:] for line in cgroups if line.startswith("0::")), "/")
    quota_paths = {root, root / cgroup.lstrip("/")}
    while cgroup != "/":
        cgroup = str(Path(cgroup).parent)
        quota_paths.add(root / cgroup.lstrip("/"))
    return {
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_quota_v2": [line for path in sorted(quota_paths) if (line := read(path / "cpu.max")) is not None],
        "cpu_quota_v1_us": read(root / "cpu/cpu.cfs_quota_us"),
        "cpu_period_v1_us": read(root / "cpu/cpu.cfs_period_us"),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


class Resources:
    def __init__(self, device):
        self.device = device
        self.proc = psutil.Process()
        self.child = None
        self.include_gpu = shutil.which("nvidia-smi") is not None
        self.stop = Event()
        self.cpu_thread = Thread(target=self._rss_loop)
        self.gpu_thread = Thread(target=self._gpu_loop)
        self.gpu_peak = None

    def _rss(self):
        rss = self.proc.memory_info().rss
        child = self.child
        if child is not None:
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        self.rss_peak = max(self.rss_peak, rss)
        return rss

    def track(self, process):
        self.child = psutil.Process(process.pid)

    def _rss_loop(self):
        while not self.stop.wait(0.025):
            self._rss()

    def _gpu(self):
        try:
            query = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            )
            used = sum(int(line) for line in query.splitlines())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        self.gpu_peak = max(self.gpu_peak or 0, used)
        return used

    def _gpu_loop(self):
        while not self.stop.is_set():
            self._gpu()
            self.stop.wait(0.2)

    @staticmethod
    def _cpu_seconds():
        return sum(
            usage.ru_utime + usage.ru_stime
            for usage in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
        )

    def __enter__(self):
        self.rss_peak = 0
        self.rss_initial = self._rss()
        self.gpu_initial = self._gpu() if self.include_gpu else None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.cpu_thread.start()
        if self.include_gpu:
            self.gpu_thread.start()
        self.cpu_started = self._cpu_seconds()
        return self

    def __exit__(self, *_):
        self.cpu_used = self._cpu_seconds() - self.cpu_started
        self._rss()
        self.stop.set()
        self.cpu_thread.join()
        if self.include_gpu:
            self.gpu_thread.join()

    def report(self):
        return {
            "process_tree_rss_start_mib": round(self.rss_initial / MIB, 2),
            "process_tree_rss_peak_mib": round(self.rss_peak / MIB, 2),
            "gpu_device_memory_start_mib": self.gpu_initial, "gpu_device_memory_peak_mib": self.gpu_peak,
            "torch_cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / MIB, 2) if self.device.type == "cuda" else None,
            "cpu_seconds": round(self.cpu_used, 3),
        }


def input_frames(args, count):
    """Create deterministic moving gradients and fine texture in H3's planar layout."""
    height, width = args.height, args.width
    frames = np.empty((3, count, height, width), dtype=np.float32)
    if args.input_video:
        with av.open(str(args.input_video)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if (frame.width, frame.height) != (width, height):
                    frame = frame.reformat(width=width, height=height)
                frames[:, i] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1) / np.float32(255)
                if i + 1 == count:
                    break
            else:
                raise ValueError(f"The input clip must contain at least {count} decoded frames")
    else:
        rng = np.random.default_rng(1408)
        y, x = np.ogrid[:height, :width]
        x, y = (x / width).astype("float32"), (y / height).astype("float32")
        grain = rng.standard_normal((height, width), dtype=np.float32) * np.float32(0.035)
        texture = 0.10 * np.cos(54 * x + 17 * y) + 0.12 * np.sin(28 * y + 16 * x) + grain
        for i in range(count):
            t = i / args.fps
            background = np.roll(texture, (i // 3, i // 2), axis=(0, 1))
            subject = np.exp(-22 * ((x - 0.5 - 0.35 * math.sin(t)) ** 2 + (y - 0.48 - 0.2 * math.cos(t)) ** 2))
            frames[0, i] = 0.13 + 0.25 * x + background + 0.45 * subject
            frames[1, i] = 0.27 + 0.18 * y + background + 0.25 * subject
            frames[2, i] = 0.36 + 0.17 * (1 - x) + background - 0.17 * subject
        np.clip(frames, 0, 1, out=frames)
    images = torch.from_numpy(frames).to(device=args.device, dtype=getattr(torch, args.dtype)).movedim(0, -1)
    return images.contiguous() if args.layout == "packed" else images


def legacy(path, images, fps, count, encoder, resources):
    """Reproduce the old patched 8-bit SDR SaveVideo and worker finalization, without audio."""
    raw = path.with_name("legacy-first-pass.mp4")
    timings = ExportProfile(images, fps, len(images))
    measure = timings.measure
    output = av.open(str(raw), mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"})
    try:
        stream = add_video_stream(output, encoder, fps, images.shape[2], images.shape[1])
        cuda_stream = torch.cuda.current_stream(images.device) if images.is_cuda else None
        if cuda_stream is not None:
            before, prepared = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for frame in images:
            if cuda_stream is None:
                pixels = measure("frame_preparation", lambda: (frame * 255).clamp(0, 255).byte())
            else:
                before.record(cuda_stream)
                pixels = (frame * 255).clamp(0, 255).byte()
                prepared.record(cuda_stream)
                copy_started = perf_counter()
                pixels = pixels.cpu()
                copy_wall = perf_counter() - copy_started
                prep = before.elapsed_time(prepared) / 1000
                timings.seconds["frame_preparation"] += prep
                timings.seconds["device_to_cpu"] += max(0, copy_wall - prep)
            frame = measure("frame_wrapping", av.VideoFrame.from_ndarray, pixels.numpy(), format="rgb24")
            frame = measure("pixel_conversion", frame.reformat, format="yuv420p", dst_colorspace=1)
            set_sdr_color(frame)
            measure("muxing", output.mux, measure("encoding", stream.encode, frame))
        measure("muxing", output.mux, measure("encoding", stream.encode, None))
    finally:
        measure("finalization", output.close)
    report = timings.report(encoder)
    finalize_started = perf_counter()
    command = FFMPEG + [
        "-i", str(raw), "-map", "0:v:0", "-map_metadata", "-1",
        "-frames:v", str(count), "-t", str(count / fps), "-fps_mode", "passthrough",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    with subprocess.Popen(command) as process:
        resources.track(process)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    report["worker_reencode_seconds"] = round(perf_counter() - finalize_started, 6)
    raw.unlink()
    return report


def ffmpeg_pipe(path, images, fps, count, encoder, resources):
    timings = ExportProfile(images, fps, count)
    batches = FrameBatches(images, count, timings)
    options = ["-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"] if encoder == "h264_nvenc" else ["-crf", "18"]
    command = FFMPEG + [
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{images.shape[2]}x{images.shape[1]}",
        "-framerate", str(fps), "-i", "pipe:0", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
        "-c:v", encoder, *options, "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        resources.track(process)
        for start in range(0, count, batches.size):
            batch = batches.read(start, min(count, start + batches.size))
            timings.measure("pipe_write", process.stdin.write, memoryview(batch).cast("B"))
        process.stdin.close()
        timings.measure("pipe_exit", process.wait)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    report = timings.report(encoder)
    for stage in ("frame_wrapping", "pixel_conversion", "encoding", "muxing", "finalization"):
        report["stages_seconds"][stage] = None  # These stages run concurrently inside FFmpeg.
    return report


def reference_file(path, images, fps, count):
    """Encode the original 8-bit BT.709 conversion losslessly for PSNR and SSIM."""
    batches = FrameBatches(images, count)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("ffv1", rate=fps)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        for start in range(0, count, batches.size):
            for img in batches.read(start, min(count, start + batches.size)):
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame = frame.reformat(format="yuv420p", dst_colorspace=1)
                output.mux(stream.encode(frame))
        output.mux(stream.encode(None))


def quality(path, reference, count, fps):
    with av.open(str(path)) as output:
        video = output.streams.video[0]
        if (video.frames != count or video.average_rate != fps or video.start_time != 0
                or video.duration * video.time_base != Fraction(count, fps) or video.pix_fmt != "yuv420p"):
            raise RuntimeError("Benchmark output did not match the requested video format and timeline")
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-i", str(reference),
        "-filter_complex", "[0:v]settb=AVTB,setpts=N/FRAME_RATE/TB,split[a][b];"
        "[1:v]settb=AVTB,setpts=N/FRAME_RATE/TB,split[c][d];[a][c]psnr[p];[b][d]ssim[s]",
        "-map", "[p]", "-map", "[s]", "-f", "null", "-",
    ], capture_output=True, text=True, check=True)
    psnr = re.search(r"average:(inf|[\d.]+)", result.stderr)[1]
    return {"psnr_db": float(psnr) if psnr != "inf" else "inf", "ssim": float(re.search(r"All:([\d.]+)", result.stderr)[1])}


def has_encoder(encoder):
    try:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.mp4"
            export_video(path, torch.zeros(2, 256, 256, 3), 24, 2, encoder=encoder)
    except (RuntimeError, ValueError, av.FFmpegError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=1184)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--layout", choices=("planar", "packed"), default="planar")
    parser.add_argument("--input-video", type=Path, help="Optional local reference clip, decoded in frame order and resized")
    parser.add_argument("--encoder", choices=("auto", "both", "h264_nvenc", "libx264"), default="auto")
    parser.add_argument("--strategies", nargs="+", choices=("legacy", "pyav", "ffmpeg"), default=("legacy", "pyav", "ffmpeg"))
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--keep-output", type=Path, help="Copy encoded results here for side-by-side viewing")
    args = parser.parse_args()
    if (min(args.width, args.height, args.seconds, args.fps, args.repeat) < 1 or args.width % 2 or args.height % 2
            or (args.torch_threads is not None and args.torch_threads < 1)):
        parser.error("Use positive arguments and even frame dimensions")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not shutil.which("ffmpeg"):
        parser.error("FFmpeg is required for the comparison and quality measurements")
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    nvenc = has_encoder("h264_nvenc") if args.encoder != "libx264" else False
    if args.encoder == "h264_nvenc" and not nvenc:
        parser.error("PyAV could not open and encode with h264_nvenc")
    encoders = (["h264_nvenc"] if nvenc else []) + (["libx264"] if args.encoder in ("both", "libx264") or not nvenc else [])
    count = args.seconds * args.fps
    source_count = 17 * ((count - 5 + 16) // 17) + 5
    images = input_frames(args, source_count)
    if images.is_cuda:
        torch.cuda.synchronize(images.device)
    print(json.dumps({"environment": cpu_environment(), "torch": torch.__version__, "pyav": av.__version__,
                      "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                      "source_frames": source_count, "output_frames": count,
                      "pyav_nvenc_test": nvenc if args.encoder != "libx264" else None}), flush=True)
    if args.keep_output:
        args.keep_output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="imagine-export-bench-") as directory:
        root = Path(directory)
        reference = root / "reference.mkv"
        reference_file(reference, images, args.fps, count)
        for trial in range(1, args.repeat + 1):
            for encoder in encoders:
                for strategy in args.strategies:
                    path = root / f"{strategy}-{encoder}-{trial}.mp4"
                    try:
                        with Resources(images.device) as resources:
                            started = perf_counter()
                            if strategy == "pyav":
                                report = export_video(path, images, args.fps, count, encoder=encoder, profile=True)
                            else:
                                run = legacy if strategy == "legacy" else ffmpeg_pipe
                                report = run(path, images, args.fps, count, encoder, resources)
                            elapsed = perf_counter() - started
                    except (BrokenPipeError, subprocess.CalledProcessError):
                        if strategy != "ffmpeg" or encoder != "h264_nvenc":
                            raise
                        print(json.dumps({"strategy": strategy, "encoder": encoder, "trial": trial, "error": "ffmpeg_nvenc_failed"}), flush=True)
                        continue
                    print(json.dumps({"strategy": strategy, "trial": trial, "total_seconds": round(elapsed, 6),
                                      "total_throughput_fps": round(count / elapsed, 3), "output_bytes": path.stat().st_size,
                                      **report, **resources.report(), **quality(path, reference, count, args.fps)}), flush=True)
                    if args.keep_output:
                        shutil.copyfile(path, args.keep_output / path.name)
                    path.unlink()


if __name__ == "__main__":
    main()
