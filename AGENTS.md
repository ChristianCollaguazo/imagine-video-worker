# Imagine Video Worker

This branch contains LightWan only. It starts from MVP 1 commit `1a0eb25`.
Develop H3 independently in `feature/h3-performance-optimization`.

- `app/main.py` owns the single-slot Bearer API, retention and job lifecycle.
  `app/lightwan.py` runs the pinned LightX2V runtime in an isolated child.
  Cancel/timeout terminate that child, then the API reloads it for the next job.
- `config/model-manifest.json` pins model artifacts. Stage only its entries
  from `/workspace-global/imagine-lightwan/models` onto local disk.
- The HTTP `turbo`/8 pair is for Imagine compatibility. LightWan uses four
  actual denoising steps and emits video at 16 FPS without audio.
- Keep media, prompts, credentials and diagnostics out of Global Volume and
  the build context. Publish only `lightwan-v1`.
- Validate with `python3 -m compileall -q app docker`, `bash -n docker/entrypoint.sh`
  and `git diff --check`. See README-Docker.md for deployment and benchmarks.
