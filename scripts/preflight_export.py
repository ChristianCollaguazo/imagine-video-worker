#!/usr/bin/env python3
"""Register the pinned ComfyUI exporter and validate a short synthetic MP4."""
import argparse
import asyncio
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import av
import torch


ROOT = Path(__file__).resolve().parents[1]
NODE = "ImagineH3ExportMP4"
WIDTH, HEIGHT, FPS = 672, 896, 24


async def discover_exporter(comfy_root, output_dir):
    manifest = json.loads((ROOT / "config/model-manifest.json").read_text())
    path_config = ROOT / "config/comfy-paths.yaml"
    # Match the server's import order and use its scanner, with only the worker node enabled.
    sys.path[:0] = [str(comfy_root), str(ROOT)]
    sys.argv[1:] = ["--cpu", "--base-directory", str(comfy_root), "--disable-metadata",
                    "--extra-model-paths-config", str(path_config),
                    "--disable-all-custom-nodes", "--whitelist-custom-nodes", "imagine_h3_export.py"]
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfyui_version import __version__
    if f"v{__version__}" != manifest["comfyui"]["ref"]:
        raise RuntimeError(f"Expected ComfyUI {manifest['comfyui']['ref']}, found v{__version__}")
    import folder_paths
    import nodes
    from utils.extra_config import load_extra_path_config

    load_extra_path_config(str(path_config))
    folder_paths.set_output_directory(str(output_dir))
    await nodes.init_external_custom_nodes()
    exporter = nodes.NODE_CLASS_MAPPINGS.get(NODE)
    if exporter is None or not exporter.OUTPUT_NODE:
        raise RuntimeError("ComfyUI did not register the worker export node")
    return exporter, __version__


def validate_mp4(path):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,start_pts,duration_ts,time_base,"
        "color_range,color_space,color_transfer,color_primaries",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=10)
    streams = json.loads(probe.stdout)["streams"]
    expected = {
        "codec_type": "video", "codec_name": "h264", "width": WIDTH, "height": HEIGHT,
        "pix_fmt": "yuv420p", "avg_frame_rate": f"{FPS}/1", "nb_frames": str(FPS), "start_pts": 0,
        "color_range": "tv", "color_space": "bt709", "color_transfer": "iec61966-2-1", "color_primaries": "bt709",
    }
    if (len(streams) != 1 or any(streams[0].get(key) != value for key, value in expected.items())
            or streams[0]["duration_ts"] * Fraction(streams[0]["time_base"]) != 1):
        raise RuntimeError(f"Preflight expected a one-second, video-only SDR H.264 MP4, found {streams}")
    with av.open(str(path)) as container:
        timestamps = [frame.pts * frame.time_base for frame in container.decode(video=0)]
    if timestamps != [Fraction(index, FPS) for index in range(FPS)]:
        raise RuntimeError("Preflight failed to decode the expected video frames and timestamps")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=ROOT / "ComfyUI")
    parser.add_argument("--cpu-smoke", action="store_true", help="Allow x264 for build or local CPU validation")
    args = parser.parse_args()
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise SystemExit("NVENC preflight requires an NVIDIA GPU visible to CUDA")
    os.environ["IMAGINE_DEBUG_METRICS"] = "0"
    with TemporaryDirectory(prefix="imagine-export-preflight-") as directory:
        output_dir = Path(directory)
        exporter, comfy_version = await discover_exporter(args.comfy_root.resolve(), output_dir)
        # Synthetic CPU IMAGE input with a surplus frame exercises the production path.
        images = torch.linspace(0.05, 0.7, FPS + 1).view(-1, 1, 1, 1).expand(-1, HEIGHT, WIDTH, 3)
        ui = exporter().save(images, "video", FPS, FPS)["ui"]
        report, = ui["video_export_metrics"]
        if not args.cpu_smoke and (report["encoder"] != "h264_nvenc" or report["hardware_fallback"]):
            raise SystemExit("NVENC preflight selected x264. Check PyAV h264_nvenc support and the host NVIDIA video driver mount.")
        info, = ui["images"]
        validate_mp4(output_dir / info["subfolder"] / info["filename"])
        print(json.dumps({"event": "video_export_preflight", "comfyui": comfy_version, "pyav": av.__version__,
                          "torch": torch.__version__, "encoder": report["encoder"],
                          "hardware_fallback": report["hardware_fallback"], "frames": FPS, "fps": FPS,
                          "seconds": 1, "width": WIDTH, "height": HEIGHT, "audio": False}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
