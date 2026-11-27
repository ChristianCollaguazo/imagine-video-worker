"""Frame-cache lifetime and opt-in observability for the pinned ComfyUI H3 graph."""
from collections import Counter
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import math
import os
from pathlib import Path
from statistics import median
import sys
import tempfile
from time import perf_counter
from weakref import ref

from h3_runtime import _cpu_cgroups, _read, counter_delta, cpu_limits, cpu_throttling

HISTORY_KEY = "imagine_h3_metrics"
CACHE_NODE = "ApplyMiniMaxH3FirstBlockCache"
STAGES = {
    "1": "diffusion_model_loader", "2": "text_encoder_loader",
    "3": "video_vae_loader", "4": "audio_vae_loader", "5": "image_load",
    "6": "prompt_and_image_encode", "11": "diffusion", "12": "video_vae_decode",
    "13": "audio_decode", "15": "mp4_export",
}
_active = ContextVar("h3_diagnostics", default=None)


def seconds(start):
    return round(perf_counter() - start, 6)


def cache_counts(cache):
    hits, misses = cache.cached_steps, cache.full_steps
    diffs = [value for value in cache.diff_values if math.isfinite(value)]
    return {
        "hits": hits, "misses": misses, "compared": len(cache.diff_values),
        "hit_rate": round(hits / (hits + misses), 6) if hits + misses else None,
        "blocks_executed": misses * cache.block_count + hits,
        "blocks_skipped": hits * (cache.block_count - 1),
        "hit_model_evaluations": list(cache.cached_step_numbers),
        "residual_diff_min_median_max": [min(diffs), median(diffs), max(diffs)] if diffs else [],
    }


def hook_cache_summary(node_mappings):
    node = node_mappings.get(CACHE_NODE)
    if node is None:
        return False
    cache_class = getattr(sys.modules.get(node.__module__), "MiniMaxH3FirstBlockCache", None)
    if cache_class is None:
        return False
    original = cache_class.summary
    if getattr(original, "_imagine_h3_metrics", False):
        return True

    @wraps(original)
    def summary(cache):
        run = _active.get()
        if run is not None:
            run["first_block_cache"]["samples"].append(cache_counts(cache))
        return original(cache)

    summary._imagine_h3_metrics = True
    cache_class.summary = summary
    return True


def _model_kind(patcher):
    name = type(patcher.model).__name__
    if name in {"MiniMaxH3", "MiniMaxH3Model", "MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"}:
        return name
    # BaseModel wraps the FL2VA denoiser and CLIP wraps the text model.
    inner = getattr(patcher.model, "diffusion_model", None)
    if type(inner).__name__ == "MiniMaxH3Model":
        return "MiniMaxH3Model"
    if hasattr(patcher.model, "clip_l"):
        return "text_encoder"
    return "other"


def _residency(patcher):
    return {
        "kind": _model_kind(patcher), "dynamic": patcher.is_dynamic(),
        "load_device": patcher.load_device.type, "offload_device": patcher.offload_device.type,
        "loaded_device": patcher.current_loaded_device().type,
        "loaded_bytes": int(patcher.loaded_size()), "weight_bytes": int(patcher.model_size()),
    }


def _record_model_action(fn, action):
    @wraps(fn)
    def wrapped(loaded, *args, **kwargs):
        run = _active.get()
        patcher = loaded.model
        if run is None or patcher is None:
            return fn(loaded, *args, **kwargs)
        before = _residency(patcher)
        start = perf_counter()
        try:
            return fn(loaded, *args, **kwargs)
        finally:
            elapsed = seconds(start)
            run["model_actions"].append({
                "stage": run["stage"], "action": action, "host_seconds": elapsed,
                "before": before, "after": _residency(patcher),
            })

    return wrapped


def _start_trace(torch, run_root):
    directory = Path(run_root) / "h3-traces"
    directory.mkdir(mode=0o700, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="trace-", suffix=".json", dir=directory)
    os.close(fd)
    try:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        )
        profiler.start()
    except (RuntimeError, OSError):
        Path(path).unlink(missing_ok=True)
        raise
    return profiler, Path(path)


def _release_decoded_frames(cache, cache_type, prompt, is_tensor, report):
    # The flat worker graph stores this decode output under its input signature.
    # Check prompt identity in case execution failed before initializing its cache.
    if (not isinstance(cache, cache_type) or not cache.initialized
            or cache.dynprompt.original_prompt is not prompt):
        return 0, None
    key = cache.cache_key_set.get_data_key("12")
    if key is None:
        return 0, None
    entry = cache.cache.pop(key, None)
    for attr in ("used_generation", "timestamps", "children"):
        bookkeeping = getattr(cache, attr, None)
        if bookkeeping is not None:
            bookkeeping.pop(key, None)
    if report and entry is not None and entry.outputs and entry.outputs[0]:
        frames = entry.outputs[0][0]
        if is_tensor(frames):
            return frames.numel() * frames.element_size(), ref(frames)
    return 0, None


def install_frame_cache_cleanup(*, report):
    import execution
    import torch
    from comfy_execution.caching import BasicCache

    original = execution.PromptExecutor.execute

    @wraps(original)
    def execute(executor, prompt, prompt_id, *args, **kwargs):
        export = prompt.get("15", {})
        if (export.get("class_type") != "ImagineH3ExportMP4"
                or export.get("inputs", {}).get("images") != ["12", 0]
                or prompt.get("12", {}).get("class_type") not in {"VAEDecode", "ImagineH3BufferedVAEDecode"}):
            return original(executor, prompt, prompt_id, *args, **kwargs)

        cgroup = _memory_cgroup() if report else None
        start = _memory_snapshot(cgroup) if report else None
        try:
            result = original(executor, prompt, prompt_id, *args, **kwargs)
        finally:
            # The prompt has finished consuming IMAGE, including any other
            # consumers. Keep history, the MP4 and every other cache entry.
            before_eviction = _memory_snapshot(cgroup) if report else None
            cache = executor.caches.outputs
            frame_bytes, frames_ref = _release_decoded_frames(cache, BasicCache, prompt, torch.is_tensor, report)
        if report and HISTORY_KEY in executor.history_result:
            executor.history_result[HISTORY_KEY]["host_memory"] = {
                "cache_type": type(cache).__name__,
                "decoded_frame_cache_evicted_bytes": frame_bytes,
                "decoded_tensor_released": frames_ref() is None if frames_ref is not None else None,
                "start": start,
                "before_frame_eviction": before_eviction,
                "end": _memory_snapshot(cgroup),
            }
        return result

    execution.PromptExecutor.execute = execute


def install(trace_prompt):
    import execution
    import nodes
    import torch
    from comfy import memory_management, model_management as mm, sd, utils
    from comfy.ldm.minimax import model as h3_model, vae as h3_vae
    from comfy.ldm.modules import attention

    original_prompt = execution.PromptExecutor.execute
    original_output = execution.get_output_data
    original_progress = utils.ProgressBar.update_absolute
    original_forward = h3_model.MiniMaxH3Model._forward
    original_vae_decode = sd.VAE.decode
    original_owned_tiled = sd.VAE._decode_tiled_owned
    original_load = mm.load_models_gpu
    cgroups = _cpu_cgroups()
    prompt_number = 0

    def patch_attention(name, label):
        original = getattr(attention, name, None)
        if original is None:
            return

        @wraps(original)
        def wrapped(*args, **kwargs):
            run = _active.get()
            if run is None:
                return original(*args, **kwargs)
            stage = run["stage"]
            try:
                result = original(*args, **kwargs)
            except Exception:
                run["attention"][stage][label + "_errors"] += 1
                raise
            run["attention"][stage][label] += 1
            return result

        setattr(attention, name, wrapped)
        if attention.optimized_attention is original:
            attention.optimized_attention = wrapped
        if h3_model.optimized_attention is original:
            h3_model.optimized_attention = wrapped
        # VAE has its own imported alias and uses the same Comfy backend.
        if h3_vae.optimized_attention is original:
            h3_vae.optimized_attention = wrapped

    for name, label in (("sageattn", "sage_dispatch"), ("attention_pytorch", "pytorch_sdpa"),
                        ("attention_flash", "flash"), ("attention_xformers", "xformers"),
                        ("attention_sub_quad", "sub_quad"), ("attention_split", "split"),
                        ("attention_comfy_kitchen", "comfy_kitchen"),
                        ("attention_comfy_kitchen_int8", "comfy_kitchen_int8")):
        patch_attention(name, label)

    @wraps(original_prompt)
    def run_prompt(executor, prompt, prompt_id, *args, **kwargs):
        nonlocal prompt_number
        if prompt.get("15", {}).get("class_type") != "ImagineH3ExportMP4":
            return original_prompt(executor, prompt, prompt_id, *args, **kwargs)
        prompt_number += 1
        has_cache = any(node.get("class_type") == CACHE_NODE for node in prompt.values())
        stage_names = {**STAGES}
        lora_index = 0
        for node_id, node in prompt.items():
            if node.get("class_type") == "LoraLoaderModelOnly":
                lora_index += 1
                stage_names[node_id] = f"lora_loader_{lora_index}"
            elif node.get("class_type") == CACHE_NODE:
                stage_names[node_id] = "first_block_cache"
        run = {
            "stage": "graph_overhead", "stage_names": stage_names, "node_seconds": {},
            "step_times": [], "step_max": prompt.get("7", {}).get("inputs", {}).get("steps"),
            "attention": {stage: Counter() for stage in {"graph_overhead", *stage_names.values()}},
            "model_actions": [], "model_load_calls": [], "vae": [],
            "forward_evaluations": 0, "dit_blocks": 0,
            "first_block_cache": {"enabled": has_cache, "hook_ready": hook_cache_summary(nodes.NODE_CLASS_MAPPINGS)
                                  if has_cache else False, "samples": []},
        }
        device = mm.get_torch_device()
        cuda = device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        run_root = os.environ.get("IMAGINE_VIDEO_RUN_ROOT", "/tmp/imagine-video-worker")
        trace = None
        trace_status = "off"
        if trace_prompt and prompt_number == trace_prompt and cuda:
            try:
                trace = _start_trace(torch, run_root)
                trace_status = "recording"
            except (RuntimeError, OSError):
                trace_status = "unavailable"
        run["trace"] = trace is not None

        start = perf_counter()
        before = cpu_throttling(cgroups)
        token = _active.set(run)
        try:
            result = original_prompt(executor, prompt, prompt_id, *args, **kwargs)
        finally:
            _active.reset(token)
            duration = seconds(start)
            after = cpu_throttling(cgroups)
            if trace is not None:
                profiler, path = trace
                try:
                    profiler.stop()
                    profiler.export_chrome_trace(str(path))
                    trace_status = "saved"
                except (RuntimeError, OSError):
                    path.unlink(missing_ok=True)
                    trace_status = "failed"

        progress = run["step_times"]
        start_step = run.get("step_start")
        step_intervals = [round(end - last, 6) for last, end in
                          zip([start_step, *progress], progress)]
        fbc = run["first_block_cache"]
        if not has_cache and run["forward_evaluations"]:
            fbc["samples"] = [{"hits": 0, "misses": 0, "compared": 0, "hit_rate": None,
                               "blocks_executed": run["forward_evaluations"] * run["dit_blocks"],
                               "blocks_skipped": 0, "hit_model_evaluations": []}]
        executor.history_result[HISTORY_KEY] = {
            "prompt_seconds": duration, "node_seconds": run["node_seconds"],
            "diffusion_step_host_seconds": step_intervals,
            "diffusion_step_expected": run["step_max"], "diffusion_step_observed": len(progress),
            "diffusion_step_tail_host_seconds": run.get("step_tail_seconds"),
            "diffusion_forward_evaluations": run["forward_evaluations"], "dit_blocks": run["dit_blocks"],
            "first_block_cache": fbc,
            "attention_dispatch": {stage: dict(counts) for stage, counts in run["attention"].items() if counts},
            "model_actions": run["model_actions"], "model_load_calls": run["model_load_calls"],
            "vae": run["vae"], "cpu_throttling": counter_delta(before, after),
            "runtime": {
                **cpu_limits(cgroups), "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "omp_threads": _thread_env("OMP_NUM_THREADS"), "mkl_threads": _thread_env("MKL_NUM_THREADS"),
                "numa_nodes_visible": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
                "numa_nodes_allowed": _numa_allowed(), **_process_ram(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "vram_state": mm.vram_state.name, "dynamic_vram": bool(memory_management.aimdo_enabled),
                "reserve_vram_bytes": int(mm.extra_reserved_memory()),
                "allocator": torch.cuda.get_allocator_backend() if cuda else None,
                "allocator_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                "allocator_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
                "compute_capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
            },
            "cuda_trace": trace_status,
        }
        return result

    @wraps(original_output)
    async def get_output(prompt_id, unique_id, *args, **kwargs):
        run = _active.get()
        stage = run["stage_names"].get(str(unique_id)) if run is not None else None
        if stage is None:
            return await original_output(prompt_id, unique_id, *args, **kwargs)
        start = perf_counter()
        previous_stage = run["stage"]
        run["stage"] = stage
        if stage == "diffusion":
            run["step_start"] = start
        try:
            scope = torch.profiler.record_function("imagine_h3/" + stage) if run["trace"] else nullcontext()
            with scope:
                return await original_output(prompt_id, unique_id, *args, **kwargs)
        finally:
            run["node_seconds"][stage] = seconds(start)
            if stage == "diffusion" and run["step_times"]:
                run["step_tail_seconds"] = seconds(run["step_times"][-1])
            run["stage"] = previous_stage

    @wraps(original_progress)
    def progress(bar, value, total=None, preview=None):
        run = _active.get()
        if (run is not None and run["stage"] == "diffusion"
                and (total if total is not None else bar.total) == run["step_max"]
                and value == len(run["step_times"]) + 1):
            run["step_times"].append(perf_counter())
        return original_progress(bar, value, total, preview)

    @wraps(original_forward)
    def forward(model, *args, **kwargs):
        run = _active.get()
        if run is not None:
            run["forward_evaluations"] += 1
            run["dit_blocks"] = len(model.blocks)
        return original_forward(model, *args, **kwargs)

    @wraps(original_vae_decode)
    def decode(vae, samples, *args, **kwargs):
        run = _active.get()
        if run is None or run["stage"] != "video_vae_decode":
            return original_vae_decode(vae, samples, *args, **kwargs)
        record = {"input_device": samples.device.type, "compute_device": vae.device.type,
                  "output_buffer_device": vae.output_device.type, "oom_tiling_fallback": False}
        run["vae"].append(record)
        result = original_vae_decode(vae, samples, *args, **kwargs)
        record["decoded_device"] = result.device.type
        return result

    @wraps(original_owned_tiled)
    def owned_tiled(vae, samples, *args, **kwargs):
        run = _active.get()
        if run is not None and run["stage"] == "video_vae_decode" and run["vae"]:
            run["vae"][-1]["oom_tiling_fallback"] = True
        return original_owned_tiled(vae, samples, *args, **kwargs)

    @wraps(original_load)
    def load_models(models, *args, **kwargs):
        run = _active.get()
        if run is None:
            return original_load(models, *args, **kwargs)
        before_models = {id(model): _model_kind(model) for model in mm.loaded_models() if model is not None}
        start = perf_counter()
        try:
            return original_load(models, *args, **kwargs)
        finally:
            host_seconds = seconds(start)
            after_models = {id(model): _residency(model) for model in mm.loaded_models() if model is not None}
            run["model_load_calls"].append({
                "stage": run["stage"], "host_seconds": host_seconds,
                "removed": [kind for identity, kind in before_models.items() if identity not in after_models],
                "resident_after": list(after_models.values()),
            })

    execution.PromptExecutor.execute = run_prompt
    execution.get_output_data = get_output
    utils.ProgressBar.update_absolute = progress
    h3_model.MiniMaxH3Model._forward = forward
    sd.VAE.decode = decode
    sd.VAE._decode_tiled_owned = owned_tiled
    mm.load_models_gpu = load_models
    mm.LoadedModel.model_load = _record_model_action(mm.LoadedModel.model_load, "load")
    mm.LoadedModel.model_unload = _record_model_action(mm.LoadedModel.model_unload, "unload")


def _thread_env(name):
    value = os.environ.get(name, "")
    return int(value) if value.isascii() and value.isdecimal() and len(value) <= 8 else None


def _numa_allowed():
    for line in _read(Path("/proc/self/status")).splitlines():
        if line.startswith("Mems_allowed_list:"):
            value = line.split(":", 1)[1].strip()
            if len(value) < 256 and all(char in "0123456789,-" for char in value):
                return value
    return None


def _memory_cgroup(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    for line in _read(membership).splitlines():
        if line.startswith("0::"):
            leaf = root / line[3:].lstrip("/")
            if (leaf / "memory.current").is_file():
                return leaf
            break
    return root if (root / "memory.current").is_file() else None


def _memory_snapshot(cgroup):
    stats = _process_ram()
    if cgroup is not None:
        current = _read(cgroup / "memory.current")
        if current.isdecimal():
            stats["cgroup_current_bytes"] = int(current)
        for line in _read(cgroup / "memory.stat").splitlines():
            key, _, value = line.partition(" ")
            if key in {"anon", "file", "shmem", "inactive_file", "kernel"} and value.isdecimal():
                stats[f"cgroup_{key}_bytes"] = int(value)
    return stats


def _process_ram(status=Path("/proc/self/status")):
    labels = {
        "VmRSS:": "process_rss_bytes", "VmHWM:": "process_lifetime_peak_rss_bytes",
        "RssAnon:": "process_rss_anon_bytes", "RssFile:": "process_rss_file_bytes",
        "RssShmem:": "process_rss_shmem_bytes",
    }
    stats = {}
    for line in _read(status).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in labels and parts[1].isdecimal() and parts[2] == "kB":
            stats[labels[parts[0]]] = int(parts[1]) * 1024
    return stats
