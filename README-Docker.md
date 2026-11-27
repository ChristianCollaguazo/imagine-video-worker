# Imagine Video Worker — Docker release v3

Imagen de release para desplegar el worker privado en una Pod Linux con NVIDIA.
La etiqueta de esta versión es `christiancollaguazo/imagine-video-worker:release-v3`.
Consulta la [guía de inferencia H3](README-H3-release-v3.md) para métricas, pruebas en RunPod y rollback a `release-v2`.
El `Dockerfile` parte de la imagen validada `turbo-v1`, conserva los pesos H3
fijados por manifiesto fuera de la imagen y añade el worker, FirstBlockCache y
SageAttention. La imagen requiere GPU SM120 (Blackwell) con la rueda SageAttention
incluida; consulta `IMAGINE_SAGE_ATTENTION` para elegir explícitamente otro backend.

## Construir y publicar

Desde la raíz del repositorio, con los cambios de release confirmados en Git, Docker Desktop iniciado y sesión en Docker Hub:

```bash
docker login
docker buildx build --platform linux/amd64 \
  --build-arg IMAGINE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t christiancollaguazo/imagine-video-worker:release-v3 --push .
```

Para construir y cargar localmente en vez de publicar, sustituye `--push` por
`--load`. Los argumentos `SAGEATTENTION_WHEEL_URL` y
`SAGEATTENTION_WHEEL_SHA256` permiten fijar otra rueda `.whl` y su SHA-256
conjuntamente. El Dockerfile verifica el hash antes de instalarla. La rueda
predeterminada es community-built para Linux x86_64, Python 3.12,
PyTorch 2.11.0+cu130 y SM120; no es una distribución oficial de SageAttention.
Las compilaciones nativas ejecutan pruebas de exportación con CPU. Las
compilaciones cruzadas validan la versión de ComfyUI y la disponibilidad de
los codificadores de PyAV. La prueba NVENC se ejecuta con una GPU NVIDIA.

## Desplegar en RunPod

- Imagen: `christiancollaguazo/imagine-video-worker:release-v3`.
- GPU NVIDIA Blackwell SM120 y disco local de contenedor de al menos 100 GB.
- Puerto HTTP `8000` únicamente. ComfyUI en `8188` no debe exponerse.
- Deja vacío el comando de inicio para usar el entrypoint de la imagen.
- Inyecta `IMAGINE_VIDEO_TOKEN` desde RunPod Secrets.
- Para videos sin pista de audio, configura `IMAGINE_H3_AUDIO=0`. Configura
  `IMAGINE_REQUIRE_NVENC=1` para comprobar NVENC antes de preparar los pesos.
- Para conservar modelos entre Pods, adjunta un Global Volume en
  `/workspace-global`. Para una descarga local al disco del contenedor, usa el
  modo `download` documentado en la tabla de variables.

El primer arranque de un volumen nuevo requiere `IMAGINE_DOWNLOAD_MODELS=1` para
poblar el volumen, lo que descarga aproximadamente 42 GB y verifica SHA-256.
Solo una Pod debe poblarlo a la vez. En los arranques posteriores, el worker
copia los pesos del volumen al disco local y los verifica. Sin el indicador de
descarga, los archivos faltantes causan un error y no se recurre a Hugging Face.
Configura `HF_TOKEN` como secreto si el repositorio de modelos exige acceso.

### Verificación de NVENC antes de iniciar el worker

Con `IMAGINE_REQUIRE_NVENC=1`, el entrypoint registra el nodo de exportación de
ComfyUI, comprueba que codificó con `h264_nvenc` y valida un MP4 de 24 cuadros
a 672×896 sin audio. Si falla, el contenedor sale antes de preparar los pesos.
El evento `video_export_preflight` indica que la comprobación terminó.

Para ejecutar únicamente la comprobación en un host Docker con GPU, sin
arrancar el worker ni preparar los pesos:

```bash
docker run --rm --gpus all \
  --entrypoint /opt/imagine-video-worker/.venv-comfy/bin/python \
  christiancollaguazo/imagine-video-worker:release-v3 \
  /opt/imagine-video-worker/scripts/preflight_export.py
```

## Variables de entorno

Los valores de la columna “Predeterminado” corresponden al Dockerfile o al
código actual. `IMAGINE_VIDEO_TOKEN` es obligatorio y no tiene default válido.
Los valores booleanos se activan solo con el texto exacto `1`.

| Variable | Predeterminado | Uso |
|---|---|---|
| `IMAGINE_VIDEO_TOKEN` | requerido | Token Bearer para todos los endpoints `/v1/`. Inyectarlo como secreto. |
| `IMAGINE_MODEL_STORAGE_MODE` | `local` | `local` lee el Global Volume; `download` descarga al disco local y requiere también `IMAGINE_DOWNLOAD_MODELS=1`. |
| `IMAGINE_DOWNLOAD_MODELS` | `0` | Con `1`, permite descargar los pesos fijados. En modo `local` también popula el volumen persistente. |
| `IMAGINE_GLOBAL_MOUNT` | `/workspace-global` | Ruta del Global Volume en modo `local`. |
| `HF_TOKEN` | vacío | Credencial opcional de Hugging Face para repositorios con acceso restringido. |
| `IMAGINE_VIDEO_STEPS` | `16` | Pasos por defecto; entero entre 1 y 50. Una petición puede sobrescribirlo. |
| `IMAGINE_VIDEO_FIRST_BLOCK_CACHE` | `safe` | `safe`, `fast` u `off`. `safe` usa `H3 Safe — 0.08 / max 2`; `fast` usa `H3 Fast — 0.10 / max 2`. |
| `IMAGINE_SAGE_ATTENTION` | `on` (Dockerfile) | `on` exige SM120 y una prueba Sage correcta; `auto` permite fallback a PyTorch SDPA; `off` selecciona PyTorch SDPA. |
| `IMAGINE_H3_AUDIO` | `1` (Dockerfile) | `1` genera una pista de audio y `0` omite su decodificación y exportación. |
| `IMAGINE_REQUIRE_NVENC` | `0` | Con `1`, exige una prueba de codificación NVENC antes de preparar los pesos. |
| `IMAGINE_H3_VRAM_PROFILE` | `auto` (Dockerfile) | Perfil de VRAM de ComfyUI. `high` requiere 90 GiB y reserva 12 GiB de forma predeterminada. |
| `IMAGINE_H3_RESERVE_VRAM_GIB` | vacío | Sobrescribe la reserva de VRAM de ComfyUI en GiB. |
| `IMAGINE_H3_INTRAOP_THREADS` | `0` | Hilos intra-op de PyTorch, OpenMP y MKL en ComfyUI. Acepta `auto` y enteros positivos. |
| `IMAGINE_H3_INTEROP_THREADS` | `0` | Hilos inter-op de PyTorch en ComfyUI. Acepta `auto` y enteros positivos. |
| `IMAGINE_H3_VAE_OUTPUT` | `cpu` | `gpu-buffer` prueba un buffer CUDA para la decodificación de video H3. |
| `IMAGINE_H3_CUDA_TRACE_PROMPT` | `0` | Con métricas activadas, captura CUDA para el trabajo con este ordinal. |
| `IMAGINE_H3_BENCHMARK_MODE` | `0` | `1` prepara los pesos y permite iniciar ComfyUI localmente para medir sin exponer la API. |
| `IMAGINE_DEBUG_METRICS` | `0` (Dockerfile) | Con `1`, emite métricas técnicas de arranque y generación a stdout. |
| `IMAGINE_VIDEO_JOB_TIMEOUT_SECONDS` | `5400` | Timeout de generación en segundos; valores menores se ajustan a 60. |
| `IMAGINE_VIDEO_MAX_UPLOAD_BYTES` | `52428800` | Tamaño máximo de imagen subida: 50 MiB. |
| `IMAGINE_VIDEO_MAX_LORA_UPLOAD_BYTES` | `4294967296` | Máximo de subida LoRA: 4 GiB. |
| `COMFY_URL` | `http://127.0.0.1:8188` | URL HTTP interna de ComfyUI. |
| `COMFY_WS` | derivada de `COMFY_URL` + `/ws` | URL WebSocket interna de ComfyUI. |
| `IMAGINE_VIDEO_DATA` | `/tmp/imagine-video-worker` | Directorio transitorio para trabajos, imágenes y resultados. |

El proceso de inicio fija `COMFY_ROOT`, `COMFY_INPUT_DIR`, `COMFY_OUTPUT_DIR`,
`COMFY_TEMP_DIR` e `IMAGINE_VIDEO_DATA` en rutas de trabajo bajo `/tmp` o en
`/opt/imagine-video-worker/ComfyUI`; no configures esos directorios para un
volumen persistente. La API exige `duration_seconds` entre 3 y 30, sin valor
predeterminado; la resolución por defecto es 672 (también 480, 576 y 768). Las opciones
de petición disponibles están en el README principal.

### Ejemplo: poblar Global Volume la primera vez

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_MODEL_STORAGE_MODE=local
IMAGINE_DOWNLOAD_MODELS=1
IMAGINE_GLOBAL_MOUNT=/workspace-global
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Después de verificar que los archivos quedaron en el volumen, cambia
`IMAGINE_DOWNLOAD_MODELS=0` en siguientes Pods. `HF_TOKEN` se puede retirar si
ya no se necesitan descargas.

### Ejemplo: descarga a disco local

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_MODEL_STORAGE_MODE=download
IMAGINE_DOWNLOAD_MODELS=1
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
IMAGINE_SAGE_ATTENTION=auto
```

En este modo los aproximadamente 42 GB se vuelven a descargar para cada
contenedor nuevo. En hardware distinto de SM120, establece
`IMAGINE_SAGE_ATTENTION=auto` para permitir fallback, o `off` para usar SDPA.

## Benchmark aislado de exportación

En una Pod existente con el runtime instalado, ejecuta estos comandos desde el
checkout actualizado del worker. En una imagen de release, usa
`cd /opt/imagine-video-worker`. El script solo carga PyTorch, PyAV y los
fotogramas de prueba. El script puede ejecutarse con los servicios detenidos y
no descarga pesos ni requiere el token de la API.

```bash
runtime=/opt/imagine-video-worker/.venv-comfy/bin/python
"$runtime" scripts/benchmark_video_export.py --width 672 --height 896 --seconds 3 --encoder both --repeat 3
"$runtime" scripts/benchmark_video_export.py --width 672 --height 1184 --seconds 15 --encoder both --repeat 3
"$runtime" scripts/benchmark_video_export.py --width 672 --height 1184 --seconds 15 --device cuda --encoder h264_nvenc --repeat 3
"$runtime" scripts/benchmark_video_export.py --width 672 --height 1184 --seconds 15 --encoder h264_nvenc --strategies legacy pyav --torch-threads 1 --repeat 3
```

- `legacy` reproduce el primer pase por fotograma y el segundo pase de FFmpeg
  con x264. `pyav` usa el exportador del worker. `ffmpeg` envía RGB por pipe
  a un único proceso FFmpeg. Las tres variantes reciben el mismo tensor.
- `--encoder h264_nvenc` exige una codificación de prueba en PyAV. `--encoder
  both` compara NVENC con x264 cuando NVENC está disponible. Para NVENC, el
  contenedor necesita acceso a la capacidad `video` del driver NVIDIA.
- La salida JSONL incluye tiempos, FPS, afinidad, cuotas de CPU, hilos de
  PyTorch, picos de RSS y memoria de GPU, y tamaño del MP4. PSNR y SSIM se
  calculan contra una referencia sin pérdida fuera del tiempo de exportación.
  La memoria de GPU de `nvidia-smi` se mide por dispositivo y puede incluir
  otros procesos. La memoria PyTorch CUDA se mide por proceso.
- Para evaluar contenido real, añade `--input-video /tmp/referencia.mp4` con
  al menos `source_frames` fotogramas decodificables. Añade
  `--keep-output /tmp/imagine-export-review` para comparar visualmente los MP4.
  Por defecto se usan fotogramas sintéticos y se eliminan los archivos
  temporales. El benchmark compara video sin audio.

Con `IMAGINE_DEBUG_METRICS=1`, `imagine_metric` con `event=video_export`
incluye el codificador efectivo, el fallback, la duración, los FPS, las
dimensiones y strides del tensor, la preparación, la transferencia al CPU,
la conversión de formato, la codificación, el mux y el cierre del MP4.
El evento `video_encoder_fallback` se emite también con métricas desactivadas.
`mp4_finalize` mide la validación final con `ffprobe`. El worker solo publica
el resultado tras validar el formato, el número exacto de cuadros y la duración.

## Conectar Imagine

En el `.env` local de Imagine, guarda la URL HTTPS del proxy RunPod y el mismo
token secreto:

```dotenv
IMAGINE_VIDEO_URL=https://<pod-id>-8000.proxy.runpod.net
IMAGINE_VIDEO_TOKEN=<token-inyectado-en-RunPod>
```

La salud se comprueba en `/v1/health` con Bearer token. El root puede devolver
404. Nunca publiques el puerto 8188 ni guardes tokens dentro del repositorio.
