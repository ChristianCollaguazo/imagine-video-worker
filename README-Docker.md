# H3 Turbo optimizado

Esta imagen usa MiniMax H3 FL2VA INT8 ConvRot, el encoder Qwen3VL NVFP4 y la
LoRA Turbo de ocho pasos del MVP 1. El flujo predeterminado mantiene 24 FPS,
el lienzo con lado corto de 768 px y el audio. ComfyUI genera 243 o 362
cuadros y FFmpeg exporta 240 o 360 para entregar 10 o 15 segundos.

## Mac

Desde el worktree `feature/h3-performance-optimization`, con Docker Desktop
activo y la sesión de Docker Hub iniciada:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg IMAGINE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t christiancollaguazo/imagine-video-worker:h3-optimized-v1 --push .
docker buildx imagetools inspect christiancollaguazo/imagine-video-worker:h3-optimized-v1
```

En el `.env` local de Imagine, configura la URL del Pod y conserva el token
existente. Reinicia Imagine con `./start_wardrobe_app.command` y selecciona
Turbo, ocho pasos.

```dotenv
IMAGINE_VIDEO_URL=https://POD_ID-8000.proxy.runpod.net
```

## RunPod

Selecciona una RTX PRO 6000 Blackwell de 96 GB en Secure Cloud con driver
CUDA 13.2, Container Disk de 100 GB y la imagen
`christiancollaguazo/imagine-video-worker:h3-optimized-v1`. Asocia el Global
Volume existente en `/workspace-global`. Expón únicamente HTTP 8000 y deja
vacío el comando de inicio.

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_GLOBAL_MOUNT=/workspace-global
IMAGINE_DOWNLOAD_MODELS=0
IMAGINE_DEBUG_METRICS=1
IMAGINE_H3_VRAM_PROFILE=auto
IMAGINE_H3_AUDIO=1
```

El staging mueve unos 42.1 GB (39.2 GiB) al disco local. Calcula SHA-256 en
la misma pasada de copia y verifica cualquier copia local reutilizada. Si
falta un archivo o falla su verificación, el worker se detiene durante el
arranque. Los cinco archivos ya deben existir bajo `imagine-h3-turbo/models/`.

En una terminal del Pod, consulta salud y uso de recursos:

```bash
printf 'header = "Authorization: Bearer %s"\n' "$IMAGINE_VIDEO_TOKEN" |
  curl -fsS -K - http://127.0.0.1:8000/v1/health
nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -l 2
cat /sys/fs/cgroup/memory.current
```

En **Logs** de RunPod, busca `imagine_metric`. Los campos incluyen arranque,
staging, carga de nodos, codificación conjunta de imagen y prompt, intervalos
de difusión, VAE de video y audio, exportación y tiempo total. La primera
ejecución del proceso se marca como fría. ComfyUI inicializa algunos pesos
dentro de la primera operación que los usa. Los eventos no aíslan esas cargas
de las operaciones ni miden transferencias CPU/GPU por separado. El muestreo
de memoria puede omitir picos entre muestras. La duración de cada paso es una
aproximación de reloj de pared a partir de los eventos de ComfyUI.

- `IMAGINE_H3_VRAM_PROFILE=high` prueba `--highvram` con una reserva de
  12 GiB en una GPU con al menos 90 GiB visibles. El perfil cambia la política
  de descarga del runtime. Vuelve a `auto` si sube la VRAM o falla la
  decodificación de 15 s.
- `IMAGINE_H3_AUDIO=0` omite la carga y decodificación del VAE de audio y
  produce MP4 sin pista de audio. FL2VA sigue difundiendo latentes de audio
  junto con los de video. La opción ahorra el trabajo posterior a difusión.
- El backend predeterminado usa SDPA de PyTorch con despacho automático por
  operación. La instrumentación registra esa política. Para identificar cada
  kernel en ejecución hacen falta trazas del perfilador CUDA. Se conserva la
  ruta de cuantización y operadores Comfy Kitchen del runtime original.

La evaluación conserva PyTorch 2.11 cu130 y ComfyUI v0.34.0. GPU-only reduce el
margen del VAE. SageAttention, FlashAttention externo y la atención Comfy
Kitchen forzada requieren medir compatibilidad y calidad en el grafo cuantizado.
Torch Compile y CUDA Graphs requieren medir compilación, formas variables y
reutilización real en Pods de corta vida. Se mantiene la decodificación VAE
original y libx264. Tiling o NVENC requerirían medir calidad, picos de memoria y
soporte de las bibliotecas de video en el contenedor.
[Opciones de ComfyUI v0.34.0](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/cli_args.py)
[Atención de MiniMax en ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/ldm/minimax/model.py)

### Retención y benchmark

Los trabajos son temporales y el historial de ComfyUI reside en memoria. La
API elimina la entrada y archivos intermedios antes de habilitar la descarga.
El MP4 se elimina con ACK, DELETE o a los 15 minutos. La cancelación y los
fallos intentan detener el trabajo y vaciar archivos e historial. El reinicio
reconcilia los marcadores de limpieza. Las cachés volátiles de ComfyUI pueden
contener resultados de nodos hasta su reemplazo o hasta que termine ese
proceso. El apagado del contenedor libera estas cachés. La eliminación de
archivos es lógica y depende también de la gestión del almacenamiento del
proveedor.

Prueba `turbo-v1`, `h3-optimized-v1` y `lightwan-v1` en Pods de la
misma clase y precio. Usa imágenes y prompts idénticos. Mide el tiempo desde
inicio hasta salud, desde POST hasta completado y hasta descarga. Después de
una ejecución fría, repite tres ejecuciones por duración y perfil en el mismo
Pod. Prueba un solo cambio H3 por vez. Registra mediana, rango, memoria máxima,
tiempos de etapas disponibles y costo de cómputo como
`tarifa por hora × segundos facturados / 3600`. Registra por separado la
facturación de almacenamiento. El tag original permite medir tiempos externos
y recursos con `nvidia-smi`. La API original asigna semillas aleatorias.
Inspecciona el archivo con:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration -of json salida.mp4
```

Evalúa a ciegas rostro e identidad, anatomía, ropa, fondo, consistencia
temporal y cumplimiento del prompt. Usa una escala de 1 a 5 y los mismos
criterios en todos los clips. Comprueba Bearer inválido, cancelación, ACK y
expiración observando la disponibilidad del resultado y la limpieza local.
Los 3–4 minutos en H3 y 2–3 minutos para LightWan a 10 s son objetivos de
prueba. Los tiempos y la calidad deben medirse en RunPod.
