# H3 Turbo optimizado — almacenamiento directo

Esta imagen usa MiniMax H3 FL2VA INT8 ConvRot, el encoder Qwen3VL NVFP4 y la
LoRA Turbo de ocho pasos del MVP 1. El flujo predeterminado mantiene 24 FPS,
el lienzo con lado corto de 768 px y el audio. La API admite duraciones enteras
de 3 a 30 segundos. ComfyUI redondea la longitud al siguiente valor válido de
la cuadrícula H3 `17k+5`; FFmpeg recorta cuadros sobrantes y conserva 24 FPS,
sin interpolación. El nodo de ComfyUI 0.34.0 acepta hasta 3600 cuadros, pero su
descripción indica que el rango entrenado es aproximadamente 124–362 cuadros
(5,2–15,1 s), y que duraciones mayores no están probadas. Los clips de 3–4 s
y 16–30 s se aceptan técnicamente, pero su calidad y uso de VRAM siguen sin
validarse en GPU. El
Dockerfile se basa en el digest linux/amd64 de `turbo-v1` para conservar
las versiones exactas del runtime que ya se usó en RunPod.

## RunPod sin Global Volume

Usa `christiancollaguazo/imagine-video-worker:h3-optimized-v2-download`, Container Disk de 100 GB,
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

## Mac

Desde el worktree `feature/h3-performance-optimization`, con Docker Desktop
activo y la sesión de Docker Hub iniciada:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg IMAGINE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t christiancollaguazo/imagine-video-worker:h3-optimized-v2-download --push .
docker buildx imagetools inspect christiancollaguazo/imagine-video-worker:h3-optimized-v2-download
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
`christiancollaguazo/imagine-video-worker:h3-optimized-v2`. Asocia el Global
Volume existente en `/workspace-global`. Expón únicamente HTTP 8000 y deja
vacío el comando de inicio.

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_GLOBAL_MOUNT=/workspace-global
IMAGINE_DOWNLOAD_MODELS=0
IMAGINE_DEBUG_METRICS=1
IMAGINE_H3_VRAM_PROFILE=high
IMAGINE_H3_AUDIO=0
IMAGINE_MODEL_STORAGE_MODE=direct
IMAGINE_MODEL_VERIFY_MODE=cached
```

Los cinco archivos deben existir bajo `imagine-h3-turbo/models/`. El modo
predeterminado `local` mantiene el comportamiento anterior: copia los modelos
al disco local del contenedor y valida SHA-256. `direct` crea enlaces
simbólicos en las rutas estándar de ComfyUI que apuntan a los archivos del
Global Volume; no copia ni modifica los pesos. Si falta un modelo o falla una
comprobación, el arranque falla. No hay cambio silencioso de `direct` a `local`.
No se implementó `auto`: la accesibilidad de un archivo no permite inferir su
rendimiento ni la compatibilidad real de mmap. Si `direct` informa que una ruta
local ya contiene un archivo que no es un enlace válido, usa una Pod nueva con
disco local limpio o selecciona `local`; no se reemplaza automáticamente.

`IMAGINE_MODEL_VERIFY_MODE` controla la integridad. `local` siempre calcula
SHA-256 en la pasada de copia; para archivos locales existentes, `full` y
`cached` actualmente vuelven a calcular SHA-256, mientras `size` comprueba el
tamaño. La caché evita el hash completo solo en modo `direct`:

| Valor | Comprobación |
|---|---|
| `full` (predeterminado) | SHA-256 completo de cada peso; `direct` vuelve a leer los pesos persistentes. |
| `cached` | En modo `direct`, reutiliza solo una verificación SHA-256 anterior que coincide con el manifiesto y metadatos observados. Un fallo de caché realiza una verificación completa antes de guardar un recibo nuevo. |
| `size` | Comprueba existencia y tamaño exacto esperado; no es una comprobación criptográfica. |

En modo `direct`, la verificación cacheada evita releer los 40 GiB únicamente
si existe un recibo válido de una verificación SHA-256 completa previa. La
primera preparación `direct+cached` puede por tanto tardar lo mismo que `full`;
las siguientes usan el recibo si los metadatos siguen coincidiendo. Metadatos idénticos no
demuestran que el contenido no haya cambiado; usa `full` cuando necesites
validar el contenido actual. Los recibos guardan solo datos técnicos de
verificación en Global Volume; cada recibo válido se guarda en un archivo
inmutable y no se sobrescribe. Un recibo parcial o ilegible se ignora. Los
modelos deben haberse validado íntegramente al descargarse.

El modo `direct` hace una comprobación básica de que los archivos admiten un
mapa de solo lectura y lee bytes al inicio y al final; no prueba la carga completa de tensores
ni el rendimiento sostenido de lecturas aleatorias. ComfyUI v0.34.0 carga
`safetensors` con `safe_open` y obtiene sus tensores desde el archivo. Si
`comfy-aimdo` está activo, usa su propio cargador basado en mapas de memoria. El
argumento `--disable-mmap` de esa versión copia cada tensor después de
`get_tensor`; no garantiza que la apertura inicial evite mmap. No lo uses como
prueba de que el almacenamiento directo será compatible o más rápido.
[Código del cargador en v0.34.0](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/utils.py) y
[argumentos de CLI](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.0/comfy/cli_args.py).

RunPod describe Global Volumes como almacenamiento respaldado por object
storage, pensado para modelos y otros datos que se escriben poco y se leen en
despliegues sucesivos. RunPod advierte que no ofrece semántica POSIX completa
ni equivale a un volumen regional de alto rendimiento; tampoco ofrece
renombrado atómico ni hard links. Por eso, no se puede
deducir solo desde el contenedor si mmap, acceso aleatorio, `stat` y la caché de
páginas tendrán el comportamiento o la velocidad deseados. [Detalles de RunPod
Global Volumes](https://www.runpod.io/blog/global-volumes-beta) y [limitaciones
del almacenamiento object-backed](https://www.runpod.io/blog/global-volumes-serverless-beta).

El acceso directo puede reducir el tiempo antes de que la API esté lista y a
la vez aumentar el tiempo de carga inicial o la primera generación. Los datos
leídos pueden quedar en la caché de páginas del sistema operativo; eso puede
ayudar en accesos posteriores mientras el proceso y la Pod sigan activos,
pero no se promete que el contenido de una Global Volume permanezca cacheado
entre Pods. [Descripción del page cache de Linux](https://docs.kernel.org/admin-guide/mm/concepts.html#page-cache).
La carga completa de los tensores a CPU/RAM puede evitar lecturas repetidas
desde el almacenamiento remoto, pero también requiere memoria y podría elevar
el pico de RAM. No se sustituye mmap por una carga completa automáticamente.

En una terminal del Pod, consulta salud y uso de recursos:

```bash
printf 'header = "Authorization: Bearer %s"\n' "$IMAGINE_VIDEO_TOKEN" |
  curl -fsS -K - http://127.0.0.1:8000/v1/health
nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -l 2
cat /sys/fs/cgroup/memory.current
findmnt -T /workspace-global/imagine-h3-turbo/models
readlink -f /opt/imagine-video-worker/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
stat -c '%s %Y %i %n' /workspace-global/imagine-h3-turbo/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

`readlink -f` confirma que el enlace de ComfyUI resuelve al archivo
persistent; `findmnt` y `stat` registran el tipo de montaje y metadatos que
RunPod presenta a la Pod. Estos comandos no miden acceso aleatorio ni prueban
que todos los fallos de mmap estén cubiertos. No ejecutes una lectura completa
de 40 GiB solo para diagnosticar un arranque: usa el modo `full` cuando
realmente quieras comprobar cada byte.

En **Logs** de RunPod, busca `imagine_metric`. El evento
`model_staging_finished` identifica modos solicitados y efectivos, tiempos de
montaje, verificación, copia, prueba básica de mmap y configuración de rutas,
bytes copiados/hasheados, recuentos de archivos y el fingerprint del manifiesto.
`comfy_available`, `api_available` y `container_ready` marcan disponibilidad;
`model_loader_node`, `generation_started`, `generation_finished` y
`resource_peaks` describen cargas observadas y generaciones. ComfyUI inicializa
algunos pesos dentro de la primera operación que los usa. Los tiempos de nodo
son aproximaciones de reloj de pared y no aíslan transferencias CPU/GPU. El
muestreo de memoria puede omitir picos entre muestras. Las métricas contienen
datos técnicos; no incluyen prompts, imágenes, videos ni tokens.

- `IMAGINE_H3_VRAM_PROFILE=high` prueba `--highvram` con una reserva de
  12 GiB en una GPU con al menos 90 GiB visibles. El perfil cambia la política
  de descarga del runtime. Vuelve a `auto` si sube la VRAM o falla la
  decodificación de 15 s.
- `IMAGINE_H3_AUDIO=0` omite la carga y decodificación del VAE de audio y
  produce MP4 sin pista de audio. FL2VA sigue difundiendo latentes de audio
  junto con los de video. La opción ahorra el trabajo posterior a difusión.
- El backend predeterminado usa SDPA de PyTorch con despacho automático por
  operación. `comfy_attention_selected` registra el backend elegido por
  ComfyUI al arrancar. Para identificar cada
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

La métrica principal es el tiempo externo desde iniciar el despliegue hasta
obtener el primer MP4. Compara `h3-optimized-v1` con `h3-optimized-v2` en Pods
separados de la misma clase, manteniendo el mismo Global Volume, GPU, imagen
del modelo, prompt, resolución, duración, FPS y parámetros. Mantén
`IMAGINE_H3_VRAM_PROFILE=high` y `IMAGINE_H3_AUDIO=0` en ambas. Conserva
`turbo-v1` como referencia aparte; no aísla el cambio de almacenamiento.

Para separar el efecto del almacenamiento y el de la verificación, registra
estos casos:

| Medición | v1 local | v2 local + full | v2 direct + full | v2 direct + cached |
|---|---:|---:|---:|---:|
| Deploy externo → primer MP4 | Pendiente | Pendiente | Pendiente | Pendiente |
| API disponible (`container_ready`) | 650 s observado* | Pendiente | Pendiente | Pendiente |
| Preparación (`model_staging_finished`) | 634 s observado* | Pendiente | Pendiente | Pendiente |
| Primera generación y cargas iniciales | Pendiente | Pendiente | Pendiente | Pendiente |
| Generaciones posteriores | Pendiente | Pendiente | Pendiente | Pendiente |
| Pico de VRAM y RAM | Pendiente | Pendiente | Pendiente | Pendiente |
| Disco local ocupado | Pendiente | Pendiente | Pendiente | Pendiente |
| Costo de sesión y almacenamiento | Pendiente | Pendiente | Pendiente | Pendiente |

`*` Los 634 s de staging y 650 s hasta API son observaciones anteriores de v1;
`container_ready` cuenta desde dentro del contenedor y no incluye descarga o
extracción de la imagen. Cronometra además deploy→salud, POST→completado y
descarga del MP4. El primer `direct+cached` puede ser un fallo de caché y leer
y calcular hash de todos los modelos; anótalo como cache miss. Repite una
sesión fría en una Pod nueva y, después, al menos tres generaciones en la misma
Pod para medir la primera carga y las posteriores. Registra mediana y rango,
costo de cómputo (`tarifa por hora × segundos facturados / 3600`) y costo de
almacenamiento por separado.

El worker asigna semillas aleatorias y la API actual no permite fijar una.
No cambies el workflow ni la API para introducir una semilla en esta
comparación. Usa varias ejecuciones, evalúa calidad visual a ciegas y registra
esta limitación: las salidas no estarán emparejadas con la misma secuencia
aleatoria. Inspecciona el archivo con:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration -of json salida.mp4
```

Evalúa a ciegas rostro e identidad, anatomía, ropa, fondo, consistencia
temporal y cumplimiento del prompt. Usa una escala de 1 a 5 y los mismos
criterios en todos los clips. Comprueba Bearer inválido, cancelación, ACK y
expiración observando la disponibilidad del resultado y la limpieza local.
Los 3–4 minutos en H3 y 2–3 minutos para LightWan a 10 s son objetivos de
prueba. Los tiempos y la calidad deben medirse en RunPod.
