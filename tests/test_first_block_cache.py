import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from app import main
from app.workflow import build_workflow, first_block_cache_settings


MODE = "IMAGINE_VIDEO_FIRST_BLOCK_CACHE"
THRESHOLD = "IMAGINE_VIDEO_FIRST_BLOCK_CACHE_THRESHOLD"
MAX_HITS = "IMAGINE_VIDEO_FIRST_BLOCK_CACHE_MAX_HITS"
CUSTOM_ENV = {MODE: "custom", THRESHOLD: "0.15", MAX_HITS: "2"}


class FirstBlockCacheTests(unittest.TestCase):
    def test_custom_values_map_to_pinned_node_after_loras(self):
        for threshold, hits in (("0.15", 2), ("0.18", 1), ("0.20", 4), ("0", 20), ("1", 2)):
            with self.subTest(threshold=threshold, hits=hits), patch.dict(
                os.environ, {MODE: " Custom ", THRESHOLD: threshold, MAX_HITS: str(hits)}, clear=True,
            ):
                workflow = build_workflow(
                    "walk", "input.png", 5, 4,
                    lora_stack=[{"name": "one.safetensors", "strength": 0.5},
                                {"name": "two.safetensors", "strength": 1.0}],
                )
                self.assertEqual(workflow["18"]["class_type"], "ApplyMiniMaxH3FirstBlockCache")
                self.assertEqual(workflow["18"]["inputs"], {
                    "model": ["17", 0], "mode": "Custom — manual values",
                    "threshold": float(threshold), "max_consecutive_hits": hits,
                    "start_percent": 0.10, "end_percent": 0.95, "temporal_guard": False,
                })
                self.assertEqual(workflow["7"]["inputs"]["model"], ["18", 0])
                self.assertEqual(workflow["10"]["inputs"]["model"], ["18", 0])

    def test_invalid_custom_settings_name_the_environment_variable(self):
        cases = {
            MODE: ("", "unknown"),
            THRESHOLD: (None, "", "invalid", "nan", "inf", "-inf", "-0.01", "1.01"),
            MAX_HITS: (None, "", "invalid", "2.0", "1.5", "0", "-1", "21"),
        }
        for name, values in cases.items():
            for value in values:
                env = CUSTOM_ENV.copy()
                if value is None:
                    env.pop(name)
                else:
                    env[name] = value
                with self.subTest(name=name, value=value), patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, rf"^{name} must be"):
                        first_block_cache_settings()

    def test_startup_validation_reports_error_on_stderr(self):
        result = subprocess.run(
            [sys.executable, "-m", "app.workflow"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, **CUSTOM_ENV, MAX_HITS: "0"},
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, f"{MAX_HITS} must be an integer from 1 to 20 in custom mode\n")

    def test_existing_modes_ignore_custom_environment_values(self):
        with patch.dict(os.environ, {THRESHOLD: "invalid", MAX_HITS: "invalid"}, clear=True):
            for mode, preset, threshold in (
                (None, "H3 Safe — 0.08 / max 2", 0.08),
                ("safe", "H3 Safe — 0.08 / max 2", 0.08),
                ("fast", "H3 Fast — 0.10 / max 2", 0.10),
            ):
                with self.subTest(mode=mode):
                    workflow = build_workflow("walk", "input.png", 5, 4, first_block_cache=mode)
                    inputs = workflow["16"]["inputs"]
                    self.assertEqual((inputs["mode"], inputs["threshold"], inputs["max_consecutive_hits"]),
                                     (preset, threshold, 2))
            workflow = build_workflow("walk", "input.png", 5, 4, first_block_cache="off")
            self.assertNotIn("16", workflow)
            self.assertEqual(workflow["10"]["inputs"]["model"], ["1", 0])


class FirstBlockCacheDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_and_generation_report_the_settings_used_in_the_graph(self):
        with patch.dict(os.environ, CUSTOM_ENV):
            settings = first_block_cache_settings()

        client = AsyncMock()
        client.upload.return_value = "input.png"
        client.queue.return_value = "prompt-id"
        client.wait.return_value = None
        log = StringIO()
        with TemporaryDirectory(dir="/tmp") as directory, redirect_stdout(log), patch.multiple(
            main, DATA_ROOT=Path(directory), FIRST_BLOCK_CACHE=settings, DEBUG_METRICS=True,
            comfy=client, jobs={}, queue=asyncio.Queue(), slot_lock=asyncio.Lock(),
            worker_task=None, reaper_task=None, _generation_count=0, _sample_gpu_peak=AsyncMock(),
        ):
            await main.startup()
            try:
                folder = Path(directory) / "test"
                folder.mkdir()
                source = folder / "source.png"
                source.touch()
                main.jobs["test"] = {
                    "id": "test", "status": "queued", "mode": "base", "steps": 8, "resolution": 672,
                    "duration_seconds": 5, "effective_width": 1184, "effective_height": 672,
                    "lora_strength": 1.0, "lora_stack": [], "input_path": source, "folder": folder,
                    "cancel_event": asyncio.Event(), "prompt": "walk", "seed": 4,
                    "image_width": 16, "image_height": 9,
                }
                main.queue.put_nowait("test")
                await main._run_job(main.queue.get_nowait())
            finally:
                main.jobs.clear()
                await main.shutdown()

        inputs = client.queue.call_args.args[0]["16"]["inputs"]
        self.assertEqual((inputs["threshold"], inputs["max_consecutive_hits"]), (0.15, 2))
        events = [json.loads(line.removeprefix("imagine_metric ")) for line in log.getvalue().splitlines()]
        for name in ("worker_startup", "generation_started"):
            event, = (event for event in events if event["event"] == name)
            self.assertEqual(event["first_block_cache_mode"], "custom")
            self.assertIsNone(event["first_block_cache_preset"])
            self.assertEqual(event["first_block_cache_threshold"], inputs["threshold"])
            self.assertEqual(event["first_block_cache_max_consecutive_hits"], inputs["max_consecutive_hits"])
