"""Keep the LightX2V CUDA process warm and isolate each active job from the API."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

MODEL_ROOT = Path(os.environ.get("LIGHTWAN_MODEL_DIR", "/opt/imagine/models/Wan2.2-I2V-A14B"))
CONFIG = Path(__file__).resolve().parents[1] / "config" / "lightwan-i2v-nvfp4.json"
STARTUP_TIMEOUT_SECONDS = 15 * 60


class LightWanRuntime:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.device_name = "unknown"
        self._ready = False

    @property
    def ready(self):
        return self._ready and self.process is not None and self.process.returncode is None

    async def load(self):
        await self.close()
        # The CUDA child needs models and scratch. The API bearer belongs to the API.
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"IMAGINE_VIDEO_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "app.lightwan",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True, env=environment,
        )
        try:
            message = await asyncio.wait_for(self._receive(), timeout=STARTUP_TIMEOUT_SECONDS)
            if message.get("status") != "ready":
                print("LightX2V initialization failed (" + str(message.get("error_type", "unknown")) + ")", flush=True)
                raise RuntimeError("LightX2V initialization failed")
            self.device_name = str(message["gpu"])
            self._ready = True
        except BaseException:
            await self.close()
            raise

    async def _receive(self):
        assert self.process is not None and self.process.stdout is not None
        raw = await self.process.stdout.readline()
        if not raw:
            raise RuntimeError("LightX2V process exited")
        return json.loads(raw)

    async def generate(self, *, prompt, image_path, output_path, seed, height, width, model_frames, report=None):
        if not self.ready or self.process is None or self.process.stdin is None:
            raise RuntimeError("LightX2V is unavailable")
        request = {
            "prompt": prompt, "image_path": str(image_path), "save_result_path": str(output_path),
            "seed": seed, "size": [height, width], "num_frames": model_frames,
        }
        try:
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            await self.process.stdin.drain()
            response = await self._receive()
            while response.get("status") == "progress":
                if report is not None:
                    report(int(response["percent"]))
                response = await self._receive()
        except BaseException:
            await self.close()
            raise
        if response.get("status") != "completed":
            # A failed CUDA call can leave process-local state partially initialized.
            await self.close()
            raise RuntimeError("LightX2V inference failed")

    async def close(self):
        process, self.process = self.process, None
        self._ready = False
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def _load_pipeline():
    import torch
    import spas_sage_attn._fused
    import spas_sage_attn._qattn
    from sageattention import sageattn
    from lightx2v import LightX2VPipeline
    from lightx2v.common.ops.mm.mm_weight import cutlass_scaled_nvfp4_mm, scaled_nvfp4_quant

    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("The pinned LightWan runtime targets Blackwell SM120")
    if not callable(cutlass_scaled_nvfp4_mm) or not callable(scaled_nvfp4_quant):
        raise RuntimeError("LightWan NVFP4 CUTLASS operators are unavailable")
    settings = json.loads(CONFIG.read_text())
    for key, filename in (
        ("high_noise_quantized_ckpt", "high_nvfp4.safetensors"),
        ("low_noise_quantized_ckpt", "low_nvfp4.safetensors"),
    ):
        settings[key] = str(MODEL_ROOT / filename)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
        json.dump(settings, config)
        config.flush()
        pipeline = LightX2VPipeline(model_path=str(MODEL_ROOT), model_cls="wan2.2_moe", task="i2v")
        pipeline.create_generator(config_json=config.name)
    return pipeline, torch.cuda.get_device_name(0)


def _worker():
    # Only structured status messages leave this process. Upstream output can
    # include request fields, so redirect standard descriptors before importing it.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol, open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), sys.stdout.fileno())
        os.dup2(sink.fileno(), sys.stderr.fileno())
        try:
            pipeline, gpu = _load_pipeline()
        except Exception as error:
            print(json.dumps({"status": "startup_failed", "error_type": type(error).__name__}), file=protocol)
            return
        def report(value, maximum):
            percent = round(100 * value / maximum)
            print(json.dumps({"status": "progress", "percent": percent}), file=protocol)

        pipeline.runner.set_progress_callback(report)
        print(json.dumps({"status": "ready", "gpu": gpu}), file=protocol)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                pipeline.generate(**request)
                pipeline.runner.gen_video = None
                pipeline.runner.gen_video_final = None
            except Exception:
                print('{"status":"failed"}', file=protocol)
                return
            finally:
                # Drop this adapter's own request references, even after failures.
                request = None
                line = None
            print('{"status":"completed"}', file=protocol)


if __name__ == "__main__":
    _worker()
