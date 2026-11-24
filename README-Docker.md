# Imagine Video Worker — RunPod Docker

Add these files to the root of the existing imagine-video-worker checkout. The
image uses the repository's manifest, app and start/stop scripts, not a new API.
Requires the quantized INT8/NVFP4 manifest with the 8-step Turbo LoRA.

## RunPod sin Global Volume

Usa `christiancollaguazo/imagine-video-worker:turbo-v1-download`, Container Disk de 100 GB,
puerto HTTP 8000 y conserva el secreto del worker. Para esta modalidad, deja
Global Volume sin adjuntar y configura:

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_MODEL_STORAGE_MODE=download
IMAGINE_DOWNLOAD_MODELS=1
IMAGINE_DEBUG_METRICS=1
```

El arranque descarga los cinco archivos fijados (42.03 GB), comprueba tamaño y
SHA-256 y solo entonces inicia ComfyUI. Los archivos válidos se reutilizan al
reiniciar el mismo contenedor. Un Pod nuevo descarga de nuevo. El evento
`model_staging_finished` registra `download_seconds`, `verification_seconds`
y `bytes_downloaded`. La API admite duraciones enteras de 3 a 30 segundos.
El rendimiento y la calidad de las duraciones extremas requieren pruebas en GPU.

## Mac: build and publish once

Install and start Docker Desktop. Unzip this archive into the repository root.
Replace `TU_USUARIO_DOCKERHUB` with your real Docker Hub username, and create a
Docker Hub repository if needed. Only publish source you intend to share (or use
a private repository and configure registry credentials in RunPod).

```bash
cd ~/Documents/projects/Imagine/imagine-video-worker
unzip -o ~/Downloads/imagine-runpod-docker.zip -d .
docker login
docker buildx build --platform linux/amd64 \
  -t TU_USUARIO_DOCKERHUB/imagine-video-worker:turbo-v1-download --push .
```

Docker Desktop on Apple Silicon builds amd64 via emulation. No NVIDIA GPU is
required for build; inference must run on Linux with a compatible NVIDIA driver.
The image contains no model weights or credentials. Build/push may be slow and
requires local disk space; no GPU Pod is needed for the build.

## RunPod: deploy

- Custom image: `TU_USUARIO_DOCKERHUB/imagine-video-worker:turbo-v1`
- 1 RTX PRO 6000 96 GB, Secure Cloud, in a region allowed by H3's license.
- Container disk: 100 GB (local model staging and runtime/output headroom).
- Global Volume: `imagine-turbo`, mount `/workspace-global`. If the UI uses
  `/workspace`, set `IMAGINE_GLOBAL_MOUNT=/workspace` instead. Do not place the
  runtime on this mount.
- HTTP port 8000 only. No public ComfyUI or Jupyter port needed.
- Leave Docker start command empty to use the image's entrypoint. Do not select
  template options which override the startup command.
- Environment: `IMAGINE_VIDEO_TOKEN` from a RunPod Secret (use the same token as
  the Mac), `IMAGINE_GLOBAL_MOUNT=/workspace-global`,
  `IMAGINE_DOWNLOAD_MODELS=1` for the first population only.
- If the model repo requires authorization, provide an authorized HF_TOKEN via
  a secret, and accept the applicable license before starting.

The first boot downloads five manifest-pinned files (~42 GB), validates SHA-256
and stores them in the Global Volume. Subsequent boots copy the weights onto
local disk and verify them there before starting the existing services. Set
`IMAGINE_DOWNLOAD_MODELS=0` for subsequent Pods to fail instead of fetching from
Hugging Face if any weight is missing. Only one Pod may populate this volume
at a time. If a write is interrupted and leaves a corrupt object, the next
local staging detects it; remove only the identified corrupt object before
retrying with downloads enabled. No prompts, images, logs or tokens are written
into Global Volume by this integration. Do not delete the volume when terminating
compute if you wish to keep the models.

## Mac: connect Imagine

Use the real Pod ID and same secret as RunPod in the existing Imagine `.env`:

```dotenv
IMAGINE_VIDEO_URL=https://NUEVO-POD-ID-8000.proxy.runpod.net
IMAGINE_VIDEO_TOKEN=TU_TOKEN_EXISTENTE
```

Then run:

```bash
cd ~/Documents/projects/Imagine && ./start_wardrobe_app.command
```

Select Turbo mode in Imagine. Test a 10-second video and monitor VRAM before
trying longer clips. The public root may return 404; `/v1/health` is the correct
endpoint with Bearer authentication.

## Limitations and validation

The image uses local staging instead of direct mmap on object-backed storage.
Each fresh container must copy approximately 42 GB from Global Volume: startup
is not instantaneous and GPU/IOPS charges apply. First download also takes time.
Do not deploy multiple initial writers simultaneously. Global Volume billing
and region support must be checked in RunPod. A zero balance can put stored
data at risk; see current RunPod retention policies.

Shell syntax, Python compilation and small local staging fixtures were checked.
A real image build, RunPod mount, CUDA inference, and end-to-end generation have
not been executed here. Inspect the first boot logs before sending a job.
