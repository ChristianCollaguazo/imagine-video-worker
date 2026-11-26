# Imagine Video Worker — Docker release v1

Imagen de release para desplegar el worker privado en una Pod Linux con NVIDIA.
La etiqueta corregida es `christiancollaguazo/imagine-video-worker:release-v1`.
El `Dockerfile` parte de la imagen validada `turbo-v1`, conserva los pesos H3
fijados por manifiesto fuera de la imagen y añade el worker, FirstBlockCache y
SageAttention. La imagen requiere GPU SM120 (Blackwell) con la rueda SageAttention
incluida; consulta `IMAGINE_SAGE_ATTENTION` para elegir explícitamente otro backend.

## Construir y publicar

Desde la raíz del repositorio, con Docker Desktop iniciado y sesión en Docker Hub:

```bash
docker login
docker buildx build --platform linux/amd64 \
  --build-arg IMAGINE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t christiancollaguazo/imagine-video-worker:release-v1 --push .
```

Para construir y cargar localmente en vez de publicar, sustituye `--push` por
`--load`. Los argumentos `SAGEATTENTION_WHEEL_URL` y
`SAGEATTENTION_WHEEL_SHA256` permiten fijar otra rueda `.whl` y su SHA-256
conjuntamente. El Dockerfile verifica el hash antes de instalarla. La rueda
predeterminada es community-built para Linux x86_64, Python 3.12,
PyTorch 2.11.0+cu130 y SM120; no es una distribución oficial de SageAttention.
No se ha ejecutado aquí un build ni inferencia CUDA.

## Desplegar en RunPod

- Imagen: `christiancollaguazo/imagine-video-worker:release-v1`.
- GPU NVIDIA Blackwell SM120 y disco local de contenedor de al menos 100 GB.
- Puerto HTTP `8000` únicamente. ComfyUI en `8188` no debe exponerse.
- Deja vacío el comando de inicio para usar el entrypoint de la imagen.
- Inyecta `IMAGINE_VIDEO_TOKEN` desde RunPod Secrets.
- Para conservar modelos entre Pods, adjunta un Global Volume en
  `/workspace-global`. Para una descarga local al disco del contenedor, usa el
  modo `download` documentado en la tabla de variables.

El primer arranque de un volumen nuevo requiere `IMAGINE_DOWNLOAD_MODELS=1` para
poblar el volumen, lo que descarga aproximadamente 42 GB y verifica SHA-256.
Solo una Pod debe poblarlo a la vez. En los arranques posteriores, el worker
copia los pesos del volumen al disco local y los verifica. Sin el indicador de
descarga, los archivos faltantes causan un error y no se recurre a Hugging Face.
Configura `HF_TOKEN` como secreto si el repositorio de modelos exige acceso.

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
| `IMAGINE_H3_AUDIO` | `1` (Dockerfile) | `1` genera audio; cualquier otro valor lo desactiva. |
| `IMAGINE_H3_VRAM_PROFILE` | `auto` (Dockerfile) | Perfil de VRAM pasado al workflow H3. |
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

## Conectar Imagine

En el `.env` local de Imagine, guarda la URL HTTPS del proxy RunPod y el mismo
token secreto:

```dotenv
IMAGINE_VIDEO_URL=https://<pod-id>-8000.proxy.runpod.net
IMAGINE_VIDEO_TOKEN=<token-inyectado-en-RunPod>
```

La salud se comprueba en `/v1/health` con Bearer token. El root puede devolver
404. Nunca publiques el puerto 8188 ni guardes tokens dentro del repositorio.
