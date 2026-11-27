import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from weakref import ref

import torch

from app.workflow import build_workflow
from h3_diagnostics import (
    HISTORY_KEY, _active, _memory_cgroup, _memory_snapshot, _process_ram, hook_cache_summary, install,
    install_frame_cache_cleanup,
)
from h3_runtime import _cpu_cgroups, counter_delta, cpu_limits, cpu_throttling, runtime_settings
from scripts.benchmark_h3 import BenchmarkError, require_metrics


class RuntimeSettingsTests(unittest.TestCase):
    def test_v2_parent_quota_controls_auto_threads_and_throttling(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = root / "parent" / "child"
            leaf.mkdir(parents=True)
            membership = root / "membership"
            membership.write_text("0::/parent/child\n")
            (root / "cpu.stat").touch()
            (leaf / "cpu.max").write_text("max 100000")
            (leaf.parent / "cpu.max").write_text("2720000 100000")
            stat = leaf.parent / "cpu.stat"
            stat.write_text("nr_periods 100\nnr_throttled 20\nthrottled_usec 15000\n")
            paths = _cpu_cgroups(root, membership)
            with patch("os.sched_getaffinity", return_value=set(range(256)), create=True):
                limits = cpu_limits(paths)
            self.assertEqual(limits, {"affinity_cpus": 256, "quota_cpus": 27.2, "effective_cpus": 27.2})
            before = cpu_throttling(paths)
            stat.write_text("nr_periods 110\nnr_throttled 23\nthrottled_usec 18000\n")
            self.assertEqual(counter_delta(before, cpu_throttling(paths)), {
                "ancestor_1": {"nr_periods": 10, "nr_throttled": 3, "throttled_usec": 3000},
            })
            with patch("h3_runtime.cpu_limits", return_value=limits), patch.dict(os.environ, {
                "IMAGINE_H3_VRAM_PROFILE": "high",
                "IMAGINE_H3_INTRAOP_THREADS": "auto", "IMAGINE_H3_INTEROP_THREADS": "auto",
            }, clear=True):
                self.assertEqual(runtime_settings(), (27, 4, 0, "12", "cpu"))

    def test_v1_cpu_and_default_runtime(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "cpu,cpuacct"
            base.mkdir()
            membership = root / "membership"
            membership.write_text("2:cpu,cpuacct:/\n")
            (base / "cpu.stat").write_text("nr_throttled 2\nthrottled_time 2001\n")
            (base / "cpu.cfs_quota_us").write_text("200000")
            (base / "cpu.cfs_period_us").write_text("100000")
            paths = _cpu_cgroups(root, membership)
            self.assertEqual(cpu_limits(paths)["quota_cpus"], 2)
            self.assertEqual(cpu_throttling(paths), {"leaf": {"nr_throttled": 2, "throttled_usec": 2}})
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_settings(), (0, 0, 0, "-", "cpu"))
        with patch.dict(os.environ, {"IMAGINE_H3_CUDA_TRACE_PROMPT": "2"}, clear=True):
            with self.assertRaises(ValueError):
                runtime_settings()
        with patch.dict(os.environ, {"IMAGINE_H3_RESERVE_VRAM_GIB": "nan"}, clear=True):
            with self.assertRaises(ValueError):
                runtime_settings()

    def test_buffered_graph_keeps_the_silent_export_and_lora_chain(self):
        config = dict(audio=False, steps=8, lora_stack=[
            {"name": "first.safetensors", "strength": 0.6},
            {"name": "second.safetensors", "strength": 0.9},
            {"name": "third.safetensors", "strength": 1.0},
        ], first_block_cache="safe")
        base = build_workflow("fixture", "asset.png", 10, 1, 672, 1184, vae_output="cpu", **config)
        buffered = build_workflow("fixture", "asset.png", 10, 1, 672, 1184, vae_output="gpu-buffer", **config)
        self.assertEqual(base["15"]["inputs"]["frame_count"], 240)
        self.assertNotIn("audio", base["15"]["inputs"])
        self.assertEqual(buffered["12"]["class_type"], "ImagineH3BufferedVAEDecode")
        buffered["12"]["class_type"] = "VAEDecode"
        self.assertEqual(base, buffered)


class CacheAndBenchmarkTests(unittest.TestCase):
    def test_cache_hook_counts_finished_evaluations_before_reset(self):
        module = ModuleType("test_pinned_cache")

        class Cache:
            full_steps, cached_steps, block_count = 5, 3, 50
            diff_values = [0.07, 0.12, 0.05, float("inf")]
            cached_step_numbers = [2, 4, 6]

            def summary(self):
                return "upstream summary"

        module.MiniMaxH3FirstBlockCache = Cache
        node = type("CacheNode", (), {"__module__": module.__name__})
        with patch.dict(sys.modules, {module.__name__: module}):
            for _ in range(2):
                self.assertTrue(hook_cache_summary({"ApplyMiniMaxH3FirstBlockCache": node}))
            samples = []
            token = _active.set({"first_block_cache": {"samples": samples}})
            try:
                self.assertEqual(Cache().summary(), "upstream summary")
            finally:
                _active.reset(token)
        self.assertEqual(samples, [{
            "hits": 3, "misses": 5, "compared": 4, "hit_rate": 0.375, "blocks_executed": 253,
            "blocks_skipped": 147, "hit_model_evaluations": [2, 4, 6],
            "residual_diff_min_median_max": [0.05, 0.07, 0.12],
        }])

    def test_prompt_hook_tracks_executed_steps_and_keeps_payload_out_of_metrics(self):
        cpu = SimpleNamespace(type="cpu")

        class Progress:
            total = 2

            def update_absolute(self, value, total=None, preview=None):
                pass

        class Model:
            blocks = [1, 2, 3]

            def _forward(self):
                return "result"

        class VAE:
            device = output_device = cpu

            def decode(self, samples):
                return SimpleNamespace(device=cpu)

            def _decode_tiled_owned(self, samples):
                return self.decode(samples)

        class LoadedModel:
            def model_load(self):
                pass

            def model_unload(self):
                pass

        attention = SimpleNamespace(attention_pytorch=lambda: None)
        attention.optimized_attention = attention.attention_pytorch
        h3_model = SimpleNamespace(MiniMaxH3Model=Model, optimized_attention=attention.optimized_attention)
        h3_vae = SimpleNamespace(optimized_attention=attention.optimized_attention)
        mm = SimpleNamespace(
            LoadedModel=LoadedModel, get_torch_device=lambda: cpu, loaded_models=lambda: [],
            load_models_gpu=lambda models: None, extra_reserved_memory=lambda: 400,
            vram_state=SimpleNamespace(name="NORMAL_VRAM"),
        )

        async def get_output_data(prompt_id, node_id, payload):
            if node_id == "11":
                progress = Progress()
                mm.load_models_gpu([])
                for step in (1, 2):
                    Model()._forward()
                    h3_model.optimized_attention()
                    progress.update_absolute(step)
            elif node_id == "12":
                VAE().decode(SimpleNamespace(device=cpu))
            return payload

        class Executor:
            def execute(self, prompt, prompt_id):
                for node, payload in prompt.items():
                    asyncio.run(execution.get_output_data(prompt_id, node, payload))
                self.history_result = {"outputs": {}}

        execution = SimpleNamespace(PromptExecutor=Executor, get_output_data=get_output_data)
        comfy = ModuleType("comfy")
        comfy.memory_management = SimpleNamespace(aimdo_enabled=False)
        comfy.model_management = mm
        comfy.sd = SimpleNamespace(VAE=VAE)
        comfy.utils = SimpleNamespace(ProgressBar=Progress)
        modules = {
            "execution": execution, "nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={}),
            "torch": SimpleNamespace(get_num_threads=lambda: 27, get_num_interop_threads=lambda: 4,
                                     __version__="fixture", version=SimpleNamespace(cuda=None)),
            "comfy": comfy, "comfy.ldm.minimax": SimpleNamespace(model=h3_model, vae=h3_vae),
            "comfy.ldm.modules": SimpleNamespace(attention=attention),
        }
        with patch.dict(sys.modules, modules):
            install(0)
            executor = Executor()
            executor.execute({"7": {"inputs": {"steps": 2}}, "11": {}, "12": {},
                              "15": {"class_type": "ImagineH3ExportMP4", "private": "confidential"}},
                             "private prompt ID")
        metrics = executor.history_result[HISTORY_KEY]
        self.assertEqual(metrics["diffusion_step_observed"], 2)
        self.assertEqual(len(metrics["diffusion_step_host_seconds"]), 2)
        self.assertEqual(metrics["attention_dispatch"], {"diffusion": {"pytorch_sdpa": 2}})
        self.assertEqual(metrics["first_block_cache"]["samples"][0]["blocks_executed"], 6)
        self.assertEqual(metrics["model_load_calls"][0]["stage"], "diffusion")
        self.assertEqual(metrics["vae"][0]["decoded_device"], "cpu")
        self.assertIsNone(_active.get())
        self.assertNotIn("confidential", json.dumps(metrics))
        self.assertNotIn("private prompt ID", json.dumps(metrics))

    def test_benchmark_rejects_missing_cache_telemetry_and_cpu_fallback(self):
        args = SimpleNamespace(expect_vram="auto", cache="safe", steps=8)
        metrics = {
            "runtime": {"vram_state": "NORMAL_VRAM"},
            "diffusion_step_observed": 8, "node_seconds": {"video_vae_decode": 18},
            "first_block_cache": {"hook_ready": True, "samples": []},
        }
        export = {"encoder": "libx264"}
        history = {"imagine_h3_metrics": metrics, "outputs": {"15": {"video_export_metrics": [export]}}}
        with self.assertRaisesRegex(BenchmarkError, "FirstBlockCache"):
            require_metrics(history, args)
        metrics["first_block_cache"]["samples"].append({"hits": 0, "misses": 8})
        with self.assertRaisesRegex(BenchmarkError, "NVENC"):
            require_metrics(history, args)
        export["encoder"] = "h264_nvenc"
        self.assertEqual(require_metrics(history, args), (metrics, export))


class FrameCacheCleanupTests(unittest.TestCase):
    def setUp(self):
        class Cache:
            initialized = False

            def __init__(self):
                self.cache = {"model": object(), "latent": object(), "mp4": object()}
                for field in ("used_generation", "timestamps", "children"):
                    setattr(self, field, dict.fromkeys(self.cache))

        class Executor:
            def __init__(self):
                self.caches = SimpleNamespace(outputs=Cache(), objects={"decoder": object()})

            def execute(self, prompt, prompt_id):
                if prompt_id == "before-cache":
                    raise InterruptedError()
                cache = self.caches.outputs
                cache.initialized = True
                cache.dynprompt = SimpleNamespace(original_prompt=prompt)
                cache.cache_key_set = SimpleNamespace(get_data_key=lambda _: prompt_id)
                frames = torch.zeros(3, 8, 4 if prompt_id in {"6", "10"} else 6, 3)
                cache.cache[prompt_id] = SimpleNamespace(outputs=[[frames]])
                for field in ("used_generation", "timestamps", "children"):
                    getattr(cache, field)[prompt_id] = None
                self.latest_frames = ref(frames)
                if prompt_id == "during-export":
                    raise InterruptedError()
                self.history_result = {"outputs": {"15": {"mp4": "fixture"}}, HISTORY_KEY: {}}

        self.modules = {"execution": SimpleNamespace(PromptExecutor=Executor),
                        "comfy_execution.caching": SimpleNamespace(BasicCache=Cache)}
        self.executor = Executor()
        self.prompt = build_workflow("fixture", "asset.png", 10, 1, audio=False)
        self.original_entries = dict(self.executor.caches.outputs.cache)
        self.original_objects = dict(self.executor.caches.objects)

    def install_cleanup(self, *, report):
        with patch.dict(sys.modules, self.modules):
            install_frame_cache_cleanup(report=report)

    def test_sixteen_unique_decodes_release_frames_and_keep_other_caches(self):
        self.install_cleanup(report=True)
        cache = self.executor.caches.outputs
        for index in range(1, 17):
            with self.subTest(index=index):
                self.prompt["12"]["class_type"] = ("VAEDecode" if index % 2
                                                   else "ImagineH3BufferedVAEDecode")
                with patch("h3_diagnostics._memory_snapshot", side_effect=[{"before": 1}, {"peak": 2}, {"after": 3}]):
                    self.executor.execute(self.prompt, str(index))
                self.assertIsNone(self.executor.latest_frames())
                self.assertEqual(self.executor.history_result["outputs"], {"15": {"mp4": "fixture"}})
                memory = self.executor.history_result[HISTORY_KEY]["host_memory"]
                expected_width = 4 if index in {6, 10} else 6
                self.assertEqual(memory["decoded_frame_cache_evicted_bytes"], 3 * 8 * expected_width * 3 * 4)
                self.assertTrue(memory["decoded_tensor_released"])
                self.assertEqual([memory["start"], memory["before_frame_eviction"], memory["end"]],
                                 [{"before": 1}, {"peak": 2}, {"after": 3}])
                self.assertEqual(cache.cache, self.original_entries)
                self.assertEqual(self.executor.caches.objects, self.original_objects)
                for field in ("used_generation", "timestamps", "children"):
                    self.assertEqual(getattr(cache, field).keys(), self.original_entries.keys())

    def test_debug_off_still_releases_and_cancellation_preserves_other_prompt(self):
        self.install_cleanup(report=False)
        cache = self.executor.caches.outputs
        unrelated = {"12": {"class_type": "VAEDecode"}}
        self.executor.execute(unrelated, "unrelated")
        self.assertIn("unrelated", cache.cache)
        with self.assertRaises(InterruptedError):
            self.executor.execute(self.prompt, "before-cache")
        self.assertIn("unrelated", cache.cache)
        with self.assertRaises(InterruptedError):
            self.executor.execute(self.prompt, "during-export")
        self.assertNotIn("during-export", cache.cache)
        self.executor.execute(self.prompt, "success")
        self.assertNotIn("success", cache.cache)
        self.assertNotIn("host_memory", self.executor.history_result[HISTORY_KEY])

    def test_memory_snapshot_distinguishes_current_rss_and_lifetime_peak(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            membership = root / "membership"
            membership.write_text("0::/pod/worker\n")
            leaf = root / "pod" / "worker"
            leaf.mkdir(parents=True)
            (root / "memory.current").write_text("9999\n")
            (leaf / "memory.current").write_text("2100\n")
            (leaf / "memory.stat").write_text("file 900\nanon 1200\nshmem 100\ninactive_file 600\n")
            status = root / "status"
            status.write_text("VmHWM:\t64 kB\nVmRSS:\t32 kB\nRssAnon:\t24 kB\nRssFile:\t8 kB\n")
            self.assertEqual(_memory_cgroup(root, membership), leaf)
            process = _process_ram(status)
            self.assertEqual(process, {"process_lifetime_peak_rss_bytes": 65536, "process_rss_bytes": 32768,
                                       "process_rss_anon_bytes": 24576, "process_rss_file_bytes": 8192})
            with patch("h3_diagnostics._process_ram", return_value=process):
                self.assertEqual(_memory_snapshot(leaf), {
                    **process, "cgroup_current_bytes": 2100, "cgroup_file_bytes": 900,
                    "cgroup_anon_bytes": 1200, "cgroup_shmem_bytes": 100, "cgroup_inactive_file_bytes": 600,
                })


class BufferedVAETests(unittest.TestCase):
    def setUp(self):
        self.cpu, self.cuda = SimpleNamespace(type="cpu"), SimpleNamespace(type="cuda")
        self.soft_empties = []
        self.calls = []

        class FakeOOM(RuntimeError):
            pass

        def raise_non_oom(error):
            if not isinstance(error, FakeOOM):
                raise error

        class H3:
            def decode_output_shape(self, shape):
                return (1, 3, 243, 1184, 672)

        calls = self.calls

        class NativeDecode:
            def decode(self, vae, samples):
                calls.append((vae.output_device.type, vae.memory_used_decode((1,), vae.vae_dtype)))
                if vae.output_device.type == "cuda" and vae.error is not None:
                    raise vae.error
                return (SimpleNamespace(to=lambda device: ("same pixels", device.type)),)

        mm = SimpleNamespace(dtype_size=lambda _: 4, minimum_inference_memory=lambda: 400,
                             raise_non_oom=raise_non_oom,
                             soft_empty_cache=lambda: self.soft_empties.append(True))
        comfy = ModuleType("comfy")
        comfy.model_management = mm
        modules = {"comfy": comfy, "nodes": SimpleNamespace(VAEDecode=NativeDecode),
                   "torch": ModuleType("torch"),
                   "comfy.ldm.minimax.vae": SimpleNamespace(MiniMaxH3VideoVAE=H3)}
        path = Path(__file__).resolve().parents[1] / "custom_nodes" / "imagine_h3_runtime.py"
        spec = importlib.util.spec_from_file_location("test_runtime_custom_node", path)
        module = importlib.util.module_from_spec(spec)
        with (patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True),
              patch("h3_diagnostics.install_frame_cache_cleanup")):
            spec.loader.exec_module(module)
        self.decode = module.ImagineH3BufferedVAEDecode().decode
        self.vae = SimpleNamespace(
            first_stage_model=H3(), device=self.cuda, output_device=self.cpu, vae_dtype="float16",
            vae_output_dtype=lambda: "float32", memory_used_decode=lambda *_: 1000,
            patcher=SimpleNamespace(model_size=lambda: 500, get_free_memory=lambda _: 96 * 1024**3),
            error=None,
        )
        self.samples = {"samples": SimpleNamespace(is_nested=False, shape=(1, 24, 72, 74, 42))}
        self.oom = FakeOOM

    def test_buffers_without_mutating_the_shared_vae_and_returns_cpu_pixels(self):
        self.assertEqual(self.decode(self.vae, self.samples), (("same pixels", "cpu"),))
        self.assertEqual(self.calls, [("cuda", 1000 + 3 * 243 * 1184 * 672 * 4)])
        self.assertIs(self.vae.output_device, self.cpu)

    def test_low_memory_and_oom_use_native_decode(self):
        with patch.object(self.vae.patcher, "get_free_memory", return_value=1):
            self.decode(self.vae, self.samples)
        self.vae.error = self.oom()
        self.decode(self.vae, self.samples)
        self.assertEqual([device for device, _ in self.calls], ["cpu", "cuda", "cpu"])
        self.assertEqual(self.soft_empties, [True])

    def test_non_oom_propagates_without_retry(self):
        self.vae.error = RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.decode(self.vae, self.samples)
        self.assertEqual([device for device, _ in self.calls], ["cuda"])
        self.assertEqual(self.soft_empties, [])


if __name__ == "__main__":
    unittest.main()
