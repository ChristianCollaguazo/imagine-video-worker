import asyncio
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

import av
import httpx
import numpy as np
import torch

from app import main
from app.comfy import ComfyClient
from app.workflow import build_workflow, model_frame_count, VIDEO_EXPORT_NODE, VIDEO_FPS
from video_export import FrameBatches, add_video_stream, export_video


class VideoExportTests(unittest.TestCase):
    def test_workflow_requests_exact_frames_without_changing_the_model_grid(self):
        for audio in (True, False):
            graph = build_workflow("fixture", "input.png", 3, 0, audio=audio)
            self.assertEqual(graph["6"]["inputs"]["length"], model_frame_count(3))
            self.assertGreater(model_frame_count(3), 3 * VIDEO_FPS)
            self.assertEqual(graph["15"]["class_type"], VIDEO_EXPORT_NODE)
            inputs = graph["15"]["inputs"]
            self.assertEqual(inputs["frame_count"], 3 * VIDEO_FPS)
            self.assertEqual(inputs["images"], ["12", 0])
            self.assertEqual(inputs.get("audio"), ["13", 0] if audio else None)
            self.assertEqual("4" in graph, audio)
            self.assertEqual("13" in graph, audio)
            self.assertEqual("audio" in inputs, audio)

    def test_audio_decoder_is_optional_for_comfy_readiness(self):
        graph = build_workflow("fixture", "input.png", 3, 0, audio=False)
        available = {node["class_type"]: {} for node in graph.values()}

        def respond(request):
            data = {"devices": [{"type": "cuda"}]} if request.url.path == "/system_stats" else available
            return httpx.Response(200, json=data)

        async def check():
            client = ComfyClient("http://comfy", "ws://comfy/ws")
            await client.close()
            async with httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(respond)) as http:
                client.http = http
                await client.health(audio=False)
                with self.assertRaisesRegex(RuntimeError, "Required ComfyUI nodes"):
                    await client.health(audio=True)

        asyncio.run(check())

    def test_batched_rgb_quantization_matches_patched_exporter(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            planar = (torch.rand((4, 5, 30, 34), generator=generator) * 1.6 - 0.3).to(dtype).movedim(0, -1)
            for images in (planar, planar.contiguous()):
                frames = FrameBatches(images, 4)
                expected = (images[:4, ..., :3] * 255).clamp(0, 255).byte().numpy()
                np.testing.assert_array_equal(frames.read(0, 4), expected)

    def test_trims_before_encoding_and_keeps_exact_timeline_and_color(self):
        # Extra bright frames would produce a sharp luminance change if the wrong end were trimmed.
        images = torch.zeros((model_frame_count(3), 32, 48, 3))
        images[:72, ..., 0] = 0.25
        images[72:] = 1
        sample_rate = 44100
        time = torch.arange(sample_rate * 4) / sample_rate
        waveform = torch.zeros((1, 2, len(time)))
        waveform[:, :, sample_rate // 2:sample_rate] = 0.3 * torch.sin(2 * torch.pi * 440 * time[:sample_rate // 2])
        audio = {"sample_rate": sample_rate, "waveform": waveform}

        with TemporaryDirectory() as directory:
            for track in (None, audio):
                path = Path(directory) / "video.mp4"
                with patch("video_export.MAX_PREP_PIXELS", 3 * 32 * 48):
                    metrics = export_video(path, images, 24, 72, audio=track, encoder="libx264", profile=True)
                asyncio.run(main._validate_video(path, 72, 3, track is not None, 48, 32))
                with self.assertRaisesRegex(RuntimeError, "resolution"):
                    asyncio.run(main._validate_video(path, 72, 3, track is not None, 64, 32))
                with self.assertRaisesRegex(RuntimeError, "stream layout"):
                    asyncio.run(main._validate_video(path, 72, 3, track is None, 48, 32))
                self.assertEqual(metrics["tensor_shape"], list(images.shape))
                self.assertEqual(metrics["frame_count"], 72)
                self.assertEqual(metrics["encoder"], "libx264")
                self.assertEqual(metrics["stages_seconds"]["device_to_cpu"], 0)
                header = path.read_bytes()
                self.assertLess(header.index(b"moov"), header.index(b"mdat"))
                with av.open(str(path)) as output:
                    frames = list(output.decode(video=0))
                    self.assertEqual(len(frames), 72)
                    self.assertEqual([frame.pts * frame.time_base for frame in frames], [Fraction(i, 24) for i in range(72)])
                    last = frames[-1]
                    self.assertLess(last.to_ndarray(format="rgb24").max(), 85)
                    self.assertEqual((last.colorspace, last.color_primaries, last.color_trc), (1, 1, 13))
                if track:
                    with av.open(str(path)) as output:
                        decoded_audio = np.concatenate([frame.to_ndarray() for frame in output.decode(audio=0)], axis=-1)
                    self.assertLess(np.max(np.abs(decoded_audio[:, :sample_rate // 3])), 0.01)
                    self.assertGreater(np.max(np.abs(decoded_audio[:, sample_rate * 2 // 3:sample_rate])), 0.2)
                    self.assertLess(np.max(np.abs(decoded_audio[:, sample_rate:])), 0.05)

    def test_hardware_failure_retries_in_software_and_cancellation_cleans_partial_output(self):
        def codec(output, encoder, *args):
            if encoder == "h264_nvenc":
                raise av.codec.codec.UnknownCodecError(encoder)
            return add_video_stream(output, encoder, *args)

        checks = 0

        def cancel():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise InterruptedError

        images = torch.zeros((24, 32, 48, 3))
        with TemporaryDirectory() as directory, patch("video_export.add_video_stream", side_effect=codec):
            path = Path(directory) / "video.mp4"
            with patch("video_export.MAX_PREP_PIXELS", 5 * 32 * 48), self.assertRaises(InterruptedError):
                export_video(path, images, 24, 24, encoder="libx264", check_cancel=cancel)
            self.assertFalse(list(path.parent.iterdir()))
            result = export_video(path, images, 24, 24)
            self.assertTrue(result["hardware_fallback"])
            self.assertEqual(result["encoder"], "libx264")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_worker_moves_shared_output_and_streams_remote_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            job_dir = output / "imagine_h3" / "job"
            job_dir.mkdir(parents=True)
            source = job_dir / "video.mp4"
            source.write_bytes(b"fixture")
            target = root / "result.mp4"
            info = {"subfolder": "imagine_h3/job", "filename": "video.mp4", "type": "output"}

            async def collect():
                job = {"id": "job", "cancel_event": asyncio.Event()}
                self.assertEqual(await main._collect_video(job, info, target), "local_move")
                self.assertFalse(source.exists())
                self.assertEqual(target.read_bytes(), b"fixture")
                self.assertEqual(await main._collect_video(job, info, target), "http_stream")
                main.comfy.download.assert_awaited_once_with(info, target, job["cancel_event"].is_set)

            with patch.object(main, "COMFY_OUTPUT_DIR", output), patch.object(main.comfy, "download", new_callable=AsyncMock):
                asyncio.run(collect())

    def test_http_collection_stops_after_cancellation(self):
        size = 256 * 1024
        cancelled = False

        class Blocks(httpx.AsyncByteStream):
            async def __aiter__(self):
                nonlocal cancelled
                yield b"a" * size
                cancelled = True
                yield b"b" * size

        async def download(path):
            transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Blocks()))
            client = ComfyClient("http://comfy", "ws://comfy/ws")
            await client.close()
            async with httpx.AsyncClient(transport=transport, base_url=client.base_url) as http:
                client.http = http
                await client.download({"filename": "video.mp4"}, path, lambda: cancelled)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            asyncio.run(download(path))
            self.assertEqual(path.stat().st_size, size)

    def test_metrics_only_forward_approved_technical_fields(self):
        with patch.object(main, "DEBUG_METRICS", True), patch.object(main, "_metric") as metric:
            main._report_export_metrics({"video_export_metrics": [{
                "encoder": "h264_nvenc", "tensor_device": "cuda:0", "duration_seconds": 3,
                "tensor_dtype": "arbitrary text", "prompt": "private data",
                "stages_seconds": {"encoding": 0.1, "muxing": "arbitrary text", "private key": 12},
            }]}, 4)
        metric.assert_called_once_with("video_export", generation=4, encoder="h264_nvenc", tensor_device="cuda",
                                       tensor_device_index=0, duration_seconds=3, stages_seconds={"encoding": 0.1})

    def test_hardware_fallback_is_reported_with_debug_metrics_disabled(self):
        output = StringIO()
        with patch.object(main, "DEBUG_METRICS", False), redirect_stdout(output):
            main._report_export_metrics({"video_export_metrics": [{
                "encoder": "libx264", "hardware_fallback": True, "prompt": "private data",
            }]}, 2)
        prefix, record = output.getvalue().split(" ", 1)
        self.assertEqual(prefix, "imagine_metric")
        self.assertEqual(json.loads(record), {"event": "video_encoder_fallback", "generation": 2, "encoder": "libx264"})
