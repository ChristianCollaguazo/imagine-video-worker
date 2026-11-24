# Imagine Video Worker

This branch contains H3 only. The MVP 1 base is commit `1a0eb25`, image tag
`turbo-v1`. Develop LightWan independently in `feature/lightwan-video-worker`.

- `app/main.py` owns the single-slot Bearer API and job lifecycle. `app/comfy.py`
  communicates with localhost ComfyUI. `app/workflow.py` builds the native H3 graph.
- `docker/entrypoint.sh` prepares models and supervises services from
  `scripts/start.sh`. `docker/stage_models.py` reads the five manifest-pinned
  weights from `/workspace-global/imagine-h3-turbo/models`.
- `IMAGINE_MODEL_STORAGE_MODE=local|direct|download` defaults to `local`. Direct mode
  uses per-file symlinks into Global Volume and fails closed; no silent local
  copy fallback. Download mode requires `IMAGINE_DOWNLOAD_MODELS=1` and
  verifies the five pinned weights on container disk without touching a volume. `IMAGINE_MODEL_VERIFY_MODE=full|cached|size` defaults to
  `full`; only `full` is cryptographic verification on each run. Cached receipts
  are metadata-based and cannot detect same-size changes with unchanged metadata.
- Global Volume is object-backed and does not promise full POSIX or high
  throughput semantics. Treat mmap, remote random reads, `stat` freshness, and
  page-cache benefits as unverified until measured in RunPod. See the storage
  section and A/B plan in `README-Docker.md`.
- Keep the default INT8/NVFP4 weights, eight-step Turbo LoRA, 24 fps and native
  frame geometry. Benchmark optional VRAM/audio settings before promoting them.
- Keep media, prompts, credentials and diagnostics out of Global Volume and of
  the build context. Publish only the branch-specific tag. See README-Docker.md.
- Lightweight checks: `python3 -m compileall -q app scripts docker`,
  `bash -n docker/entrypoint.sh`, `bash -n scripts/start.sh`, `git diff --check`.
