# H3 inference release v3

Baseline is the `release-v2` tag at `95dc1314abef74aadecdced9c94a3da622aade83`. The target image is `christiancollaguazo/imagine-video-worker:release-v3`. Keep the `release-v2` image and tag for rollback.

For the custom FirstBlockCache rollout, see the [release-v3.1 RunPod configuration](README-Docker.md#runpod-en-blackwell-de-96-gb-con-descarga-a-disco-local).

## Evidence and investigation priorities

The supplied Blackwell runs used a 96 GB RTX PRO 6000, 10 seconds, 672×1184, 24 FPS, eight steps, three LoRAs and silent output. The two warm jobs averaged 115.94 s. Diffusion averaged 93.77 s, VAE decode 18.42 s and export 1.88 s. The third run's first step took 9.672 s and its remaining seven took 11.603–11.704 s. Relative to the warm baseline, the targets of 60 s diffusion, 12 s VAE and 80 s total call for roughly 36%, 35% and 31% reductions, respectively. Hardware results for v3 are pending.

| Priority | Finding | Test on Blackwell |
|---|---|---|
| VRAM residency | The pin's NVIDIA `auto` can initialize DynamicVRAM (AIMDO). `--highvram` selects eager loading and disables DynamicVRAM. ComfyUI can still unload models under pressure. The release-v2 launcher sets 12 GiB only in `high`. The default Linux reserve in the pin is 400 MiB. DynamicVRAM has a separate headroom parameter. A peak around 50 GiB alone says little about transfers. | Compare `auto` with the existing `high` bundle. Check DiT resident bytes, loads, removals and profiler H2D/D2H during diffusion. If the result warrants isolating the reserve, compare two high runs with different reserves on a dedicated GPU. |
| FirstBlockCache | The pinned node patches native H3's `double_block` and wraps the actual diffusion and outer sampler calls. The workflow applies the wrapper after all LoRAs. A hit runs block zero and reuses the tail residual. Upstream still enters the dispatcher and can prefetch blocks before running the patch. It reads sigma and eligible residual comparisons via `.item()`, and misses can clone residuals. Its presets compare within 10–95% of the sampling schedule and cap consecutive hits at two. | Inspect hit, miss, comparison and skipped-block counts. Use `off` to quantify net overhead. Use `fast` as a quality experiment. The timing pattern alone cannot establish hit rate, and a model evaluation can differ from a sampler step. |
| CPU configuration | The supplied process's 128 intra-op and 128 inter-op pools each exceed the 27.2 CPU quota. Cgroup counters and GPU idle gaps can establish the impact of runnable demand. On this quota with at least 28 CPUs in affinity, `auto/auto` resolves to 27 intra-op and 4 inter-op threads. | Compare wall times, per-stage GPU idle gaps and cgroup throttling deltas across the two configurations. Check CPU affinity and allowed NUMA nodes. If warranted, try 16/4 and compare locality with `nvidia-smi topo -m` and `numactl --hardware` if installed. |
| VAE output buffering | The pinned H3 video VAE uses native 256 px spatial tiles and overlapping temporal chunks. Standard `VAEDecode` computes on the GPU and typically copies finalized chunks to a CPU tensor. FL2VA here decodes a single video sample, so increasing the outer VAE batch size provides no gain. The new option keeps the output buffer on GPU during the same chunk computation, then copies it to CPU once. A 243-frame float32 buffer at this canvas is about 2.16 GiB. | Use a trace to establish the cost of chunk transfers and synchronization. Compare `gpu-buffer` with `cpu` only if that cost is meaningful. Confirm the GPU buffer actually ran and compare total time, peak VRAM, output hashes and color properties. |

Pinned sources for the matching [VRAM CLI policy](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/cli_args.py), [model manager](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/model_management.py), [H3 transformer](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ldm/minimax/model.py), [cache implementation](https://raw.githubusercontent.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache/d474755360cc1784df1b7c933ca28195e55e3445/nodes.py) and [video VAE](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ldm/minimax/vae.py).

### Blackwell kernels and deferred changes

- H3 projects packed joint video and audio tokens, fuses RMSNorm/RoPE on Q/K and calls ComfyUI attention with `skip_reshape=True` in HND layout. The transformer clones V and transposes views, and the selected attention path may prepare different output layouts. ComfyUI can fall back to SDPA at runtime, so startup selection alone cannot establish dispatch. The SageAttention 2.2 upstream SM120 dispatcher selects CUDA QK INT8 and PV FP8. Confirm the actual wheel's kernel names and memory layout materializations in the trace. Silence still leaves the audio latent stream in the model. [Pinned H3 transformer](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ldm/minimax/model.py), [pinned ComfyUI attention](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ldm/modules/attention.py), [SageAttention dispatcher](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py).
- ComfyUI uses metadata-driven quantized layouts through comfy-kitchen. H3 already uses fused operations where supported, including SwiGLU routing. Weight functions and nonresident low-VRAM patches can incur dequantization and requantization. Time in these branches depends on actual LoRA targets, residency and kernel dispatch. Use the trace to separate INT8/FP dispatch, rotation, dequantization, GEMM, copies and launches. Use Nsight Compute on selected kernels, if available, to determine Tensor Core utilization. Kernel names and VRAM totals alone are insufficient. PyTorch allocator peaks may omit AIMDO, direct allocations and NVENC, and some counters can be unavailable under `cudaMallocAsync`. Compare with sampled device memory. [Pinned quantized ops](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/quant_ops.py), [pinned linear and patch paths](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ops.py), [PyTorch allocator statistics](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html).
- Whole-sampler CUDA graphs and `torch.compile` have unqualified interactions with the pinned dynamic allocator and prefetch paths, host cache decisions, cancellation and variable packed layouts. NVIDIA's Sol Engine reports results for a different model/runtime, 124 frames and 50 steps. Its graph work is platform-dependent. ComfyUI-Sol-H3's exact runtime changes and sparse attention backend are useful profiling leads, but have no measured result for this pinned CUDA 13, SM120, eight-step, three-LoRA job. If a trace finds material host overhead, isolate a fixed-shape kernel or block and measure compilation cost, warm replications and cache hits across resolution, duration and LoRA changes before integrating. [NVIDIA H3 Sol Engine](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/), [ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3).
- ComfyUI v0.36.0 ships additional H3 VAE kernels and matching comfy-kitchen changes. The pin remains v0.34.0. An upstream upgrade or backport needs separate compatibility and image-quality qualification. The buffering option reuses the pinned model, dtype, tiling, compositing and color path, accounts for extra VRAM, and retries the native decode after a CUDA OOM. [Upstream VAE change](https://github.com/Comfy-Org/ComfyUI/pull/16187), [v0.36.0 notes](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.36.0).

## Runtime settings

| Environment variable | Default | Effect |
|---|---|---|
| `IMAGINE_H3_INTRAOP_THREADS` | `0` | ComfyUI only. `0` leaves PyTorch, OpenMP and MKL defaults and inherited settings alone. `auto` uses the floored minimum of CPU affinity and the smallest visible cgroup quota, at least one. A positive integer sets PyTorch intra-op, OMP and MKL to that value. |
| `IMAGINE_H3_INTEROP_THREADS` | `0` | ComfyUI only. `0` leaves the inherited PyTorch behavior. `auto` uses at most four threads within the effective CPU count. A positive integer sets the PyTorch inter-op limit. |
| `IMAGINE_H3_RESERVE_VRAM_GIB` | unset | Unset uses the ComfyUI default in `auto` and 12 GiB in `high`. A number passes `--reserve-vram` explicitly. Keep the existing 90 GiB device guard for `high`. |
| `IMAGINE_H3_VAE_OUTPUT` | `cpu` | `gpu-buffer` opts into the H3-only CUDA output buffer. The exported tensor still goes through the existing CPU input path of the release-v2 exporter. |
| `IMAGINE_H3_CUDA_TRACE_PROMPT` | `0` | With debug metrics enabled, a positive integer profiles that H3 prompt ordinal in the ComfyUI process. The trace stores CPU operations, CUDA kernels and copy activity in a private `/tmp/imagine-video-worker/h3-traces/` file. Trace runs carry profiling overhead. |
| `IMAGINE_H3_BENCHMARK_MODE` | `0` | Docker entrypoint `1` stages models once, then waits for manual loopback-only ComfyUI runs. Only the loopback service runs in this mode. |
| `IMAGINE_DEBUG_METRICS` | `0` | Existing flag. `1` enables trusted in-process diagnostics, resource sampling and the `h3_inference` log event. |

With timings enabled, `node_seconds` and `diffusion_step_host_seconds` come from inside ComfyUI. Step one includes model preparation inside the sampler and callbacks record host intervals without adding device synchronization. Use the optional CPU/CUDA trace to time asynchronous kernel work. Model action times are host timings and resident byte snapshots are reported by ComfyUI. Check the trace for transfers on the critical path. `attention_dispatch` counts successful backend calls and errors. Identify specific GPU kernels in the trace. `first_block_cache.samples` counts completed model evaluations and omitted original DiT blocks. Measure prefetch overhead and wall-time gains separately. Container RAM and device VRAM are sampled every two seconds. The process RSS and lifetime RSS high-water mark provide a separate view. Throttling deltas can include other processes sharing a cgroup. Existing API telemetry remains available and debug metrics carry no prompt, token, image or frame payloads.

## Decoded-frame RAM retention

Release-v2's `container_ram_mb` is the largest sampled value of `/sys/fs/cgroup/memory.current` during each job, truncated to MiB. It is a cgroup total that includes anonymous memory, filesystem cache and kernel charges. The process lifetime RSS peak is a separate metric. Between generations 1 and 16 the sampled total rose by 33,541 MiB. The 13 full-size and two smaller tensors from generations 2–16 account for approximately 31,954 MiB of logical frame storage. [Linux cgroup v2 memory accounting](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).

ComfyUI v0.34.0 defaults to a RAM-pressure output cache. The VAE's completed `IMAGE` stays under its input-signature key until pressure triggers eviction. Each worker job uses a new input and seed. Successful job history holds UI metadata such as the MP4 location, and the worker forgets it after collecting the file. The exporter holds frames during encoding and returns only metadata. Release-v3 removes the worker graph's decoded-image entry and its cache bookkeeping after prompt execution. Other cached outputs and loaded models remain available. [Pinned execution cache](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy_execution/caching.py), [pinned executor and history](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/execution.py).

For the rollout, enable `IMAGINE_DEBUG_METRICS=1` and leave CUDA tracing off. Check `h3_inference.host_memory` after consecutive successful jobs on one Pod. The `start`, `before_frame_eviction`, and `end` checkpoints report bytes for cgroup total, anonymous and file memory, plus current ComfyUI RSS and its anonymous and file components. Compare the warm `start` and `end` anonymous-memory trends. An eviction of about 2.32 GB and `decoded_tensor_released: true` are expected for a fresh 672×1184 decode that reaches the cache. A true value means the tensor's Python object was released. Allocator retention, other tensor references and cgroup accounting can affect when the OS totals fall. File memory and the lifetime RSS peak can stay elevated even when the warm anonymous-memory baseline levels off.

## RunPod benchmark

Use an existing dedicated SM120 Pod with the release-v3 image, the pinned weights already available to the deployment, and the same three installed LoRA files. Start it with the following environment and leave the startup command empty. Use the Pod's shell over your existing private access. Only loopback port 8188 is used for the benchmark. Existing model staging and the NVENC preflight run once when the container starts.

```dotenv
IMAGINE_H3_BENCHMARK_MODE=1
IMAGINE_H3_AUDIO=0
IMAGINE_REQUIRE_NVENC=1
IMAGINE_DOWNLOAD_MODELS=0
IMAGINE_MODEL_STORAGE_MODE=local
IMAGINE_SAGE_ATTENTION=on
```

Prepare private fixture files `input.png`, `prompt.txt` and `loras.json` in `/workspace/h3-fixture`. The original portrait fixture must map to 672×1184. Place the three LoRAs in `/opt/imagine-video-worker/ComfyUI/models/loras` using the deployment's existing LoRA copies. Model staging covers the pinned manifest entries. Use the same image bytes and prompt file across cases, and record the Pod/GPU, image revision, driver and actual installed LoRA versions privately. `loras.json` uses the API's ordered `[{"name":"a.safetensors","strength":0.8}, ...]` format. Include the actual three names and strengths. The benchmark reads the prompt and image from files, pins seed 20261126 for measurements and writes private result files. Each run uses a fresh Comfy image path to force real inference, matching worker jobs. Warmups use a separate seed. Total includes upload, queue and inference wall time, and output verification. It excludes copying the result to the retained benchmark folder and differs slightly from API wall time.

Run the following on the Pod. The three ComfyUI processes reuse staged weights. The cache variants share the baseline process and may incur a one-time re-patch, which the model metrics show. This screen uses **eight** full generations, including three warmups.

```bash
set -euo pipefail
umask 077
cd /opt/imagine-video-worker
export IMAGINE_DEBUG_METRICS=1 IMAGINE_H3_AUDIO=0 IMAGINE_SAGE_ATTENTION=on
export IMAGINE_H3_INTRAOP_THREADS=0 IMAGINE_H3_INTEROP_THREADS=0
export IMAGINE_H3_VAE_OUTPUT=cpu IMAGINE_H3_CUDA_TRACE_PROMPT=0
unset IMAGINE_H3_RESERVE_VRAM_GIB
runtime="$PWD/.venv-comfy/bin/python"
fixture=/workspace/h3-fixture
results="$(mktemp -d /workspace/h3-results.XXXXXX)"
bench() {
  local name="$1"
  shift
  "$runtime" -m scripts.benchmark_h3 \
    --image "$fixture/input.png" --prompt-file "$fixture/prompt.txt" \
    --lora-stack "$fixture/loras.json" --out "$results/$name" --case "$name" \
    --seed 20261126 --seconds 10 --steps 8 "$@"
}

export IMAGINE_H3_VRAM_PROFILE=auto
bash scripts/start.sh --comfy-only
bench auto-safe --expect-vram auto
bench auto-off --expect-vram auto --cache off --warmups 0
bench auto-fast --expect-vram auto --cache fast --warmups 0
bash scripts/stop.sh

export IMAGINE_H3_VRAM_PROFILE=high
bash scripts/start.sh --comfy-only
bench high-safe --expect-vram high
bash scripts/stop.sh

export IMAGINE_H3_VRAM_PROFILE=auto
export IMAGINE_H3_INTRAOP_THREADS=auto IMAGINE_H3_INTEROP_THREADS=auto
bash scripts/start.sh --comfy-only
bench auto-cpu-threads --expect-vram auto
bash scripts/stop.sh
```

Compare each candidate to `auto-safe` with the same fixture and check that the observed attention dispatch agrees across runs. The `high` comparison tests the deployed bundle. It changes DynamicVRAM and the reserve together. Off isolates cache overhead, and fast isolates the threshold relative to safe. If a case looks faster, repeat only the baseline and that candidate with `--runs 2`, using unique case names. Try 12 or 20 steps only if eight-step cache comparisons show few eligible opportunities and the longer configuration is relevant to production. Keep the sampler, fixture and LoRAs fixed, and pair off with the candidate threshold at the selected step count.

If model movement, allocator activity or D2H copies remain unclear, trace one representative warm job in a fresh Comfy process. Review `imagine_h3/diffusion` and `imagine_h3/video_vae_decode` in a local Chrome trace viewer. Check H2D/D2H direction, bytes and overlap, attention CUDA kernels, quantized GEMM/cast kernels, launch gaps and synchronization. Keep the trace private and exclude its timing from the benchmark. If D2H copies during VAE decode are material, compare the two output modes on the same warmed process. The optional commands below use original CPU thread defaults and one extra decode candidate.

```bash
export IMAGINE_H3_INTRAOP_THREADS=0 IMAGINE_H3_INTEROP_THREADS=0
export IMAGINE_H3_VRAM_PROFILE=auto IMAGINE_H3_CUDA_TRACE_PROMPT=2
bash scripts/start.sh --comfy-only
bench auto-trace --expect-vram auto
# Run only when the trace or prior evidence warrants the VAE comparison.
bench auto-vae-cpu --expect-vram auto --warmups 0
bench auto-vae-gpu --expect-vram auto --vae-output gpu-buffer --warmups 0
bash scripts/stop.sh
```

A useful VAE trial has `output_buffer_device: "cuda"` for its successful decode and no retry. Compare end-to-end time so that moving time from decode to another stage cannot count as a win. Confirm NVENC, 672×1184, 240 frames, 24 FPS and a silent ten-second stream. Check video color metadata and decoded per-frame equivalence for residency, threads and buffering experiments. Example for the two VAE modes:

```bash
for name in auto-vae-cpu auto-vae-gpu; do
  clip="$results/$name/run-1.mp4"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=color_range,color_space,color_transfer,color_primaries \
    -of json "$clip" > "$results/$name/color.json"
  ffmpeg -v error -i "$clip" -map 0:v:0 -c:v rawvideo \
    -f framehash -hash sha256 "$results/$name/frames.sha256"
done
cmp "$results/auto-vae-cpu/color.json" "$results/auto-vae-gpu/color.json"
cmp -s "$results/auto-vae-cpu/frames.sha256" "$results/auto-vae-gpu/frames.sha256"
```

Identical decoded frames support equivalence after encoding. When hashes differ, repeat the baseline for normal variance and quantify the difference before accepting the candidate. For cache modes, compare each retained clip against the off and safe references at normal speed and frame by frame. Check facial identity against the input, motion, temporal texture stability, tile seams, fine detail and all three LoRA effects. Repeat promising cache settings on representative clips before promotion. The fixed seed is a fixture control.

## Local release checks

The Apple Silicon regression suite passed 46 tests, including the default silent and multi-LoRA graph, cache counters, privacy of the new metrics, VAE buffer selection and fallback, and existing API and export coverage. A synthetic CPU x264 export passed. Startup and benchmark shell syntax, Docker context entries and NVENC preflight ordering were checked. The exporter, exporter node and preflight match `release-v2`. The pinned Linux ComfyUI stack, NVENC and actual H3 quality and performance require the RunPod run above.

## Production and rollback

For silent production keep `IMAGINE_H3_AUDIO=0`, `IMAGINE_SAGE_ATTENTION=on`, `IMAGINE_H3_VRAM_PROFILE=auto`, `IMAGINE_VIDEO_FIRST_BLOCK_CACHE=safe`, `IMAGINE_H3_VAE_OUTPUT=cpu`, default thread pools, and `IMAGINE_REQUIRE_NVENC=1`. Keep debug metrics, CUDA trace and benchmark mode off in routine production. Preserve the deployment's selected inference steps. Promote measured settings individually after output and stability checks. Test a long job and a different LoRA stack before promoting changes to VRAM residency or VAE buffering. Longer durations grow the output buffer, and changing LoRA stacks can change weight residency and patch cost.

To roll back, redeploy `christiancollaguazo/imagine-video-worker:release-v2` with the same staged weights, secrets, audio configuration and NVENC preflight. The source reference for rollback is `release-v2` at `95dc1314abef74aadecdced9c94a3da622aade83`. The `release-v3` branch contains the corresponding Docker build instructions.
