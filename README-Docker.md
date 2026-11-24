# LightWan 2.2 A14B NVFP4

LightX2V proporciona el camino oficial NVFP4 de cuatro pasos, dos por cada
experto, y atención dispersa Sage2. El preset publicado usa 81 cuadros a
480 × 832. Los tiempos de inferencia publicados para RTX 5090 son una
referencia del proyecto y corresponden a su propia prueba. La configuración
de 720p y las duraciones de 3 a 30 s requieren validación con este worker en
la RTX PRO 6000. El rango de duración es una política experimental de la API;
no representa un rango de calidad o rendimiento validado por el modelo. Cuando
la opción está activa, health informa los valores aceptados por la API y los
marca como experimentales.

La API acepta la pareja `turbo`/8 que envía Imagine y reporta los cuatro
pasos internos por separado. Se elige la duración en pasos de un segundo. A
16 FPS, LightX2V solicita un cuadro adicional por la restricción temporal
`4n+1` y el exportador recorta el último cuadro para conservar la duración
solicitada. No se cambia la velocidad ni se interpolan cuadros. Los MP4 no
incluyen audio. Esta ruta es experimental.
[LightWan y licencia Apache 2.0](https://huggingface.co/lightx2v/LightWan2.2-A14B)
[Runtime oficial LightX2V](https://github.com/ModelTC/LightX2V)
[Preset oficial NVFP4 con atención dispersa](https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json)

## Mac

Desde el worktree `feature/lightwan-video-worker`, con Docker Desktop
activo y la sesión de Docker Hub iniciada:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg IMAGINE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t christiancollaguazo/imagine-video-worker:lightwan-v1 --push .
docker buildx imagetools inspect christiancollaguazo/imagine-video-worker:lightwan-v1
```

En el `.env` local de Imagine, configura la URL del Pod y conserva el token
existente. Reinicia Imagine con `./start_wardrobe_app.command` y selecciona
Turbo, ocho pasos, 10 segundos en la primera prueba.

```dotenv
IMAGINE_VIDEO_URL=https://POD_ID-8000.proxy.runpod.net
```

## RunPod

Selecciona una RTX PRO 6000 Blackwell de 96 GB en Secure Cloud con driver
CUDA 13.2, Container Disk de 100 GB e imagen
`christiancollaguazo/imagine-video-worker:lightwan-v1`. Asocia el mismo
Global Volume en `/workspace-global`. Expón únicamente HTTP 8000 y deja
vacío el comando de inicio.

```dotenv
IMAGINE_VIDEO_TOKEN={{ RUNPOD_SECRET_imagine_video_token }}
IMAGINE_GLOBAL_MOUNT=/workspace-global
IMAGINE_DOWNLOAD_MODELS=1
IMAGINE_LIGHTWAN_ENABLE_LONG_VIDEO=1
IMAGINE_DEBUG_METRICS=1
```

Después de comprobar la primera carga, usa `IMAGINE_DOWNLOAD_MODELS=0` en
los siguientes Pods. El primer arranque puede consumir tiempo facturable
mientras descarga archivos de forma secuencial. Usa un solo Pod para poblar
el directorio. Revisa un fallo de verificación antes de volver a intentar
una descarga. Con los archivos presentes y el volumen montado, el arranque
los prepara en disco local.

El manifiesto selecciona únicamente los dos expertos I2V NVFP4, el text
encoder, el tokenizador y el VAE. Usa aproximadamente 28.7 GB de almacenamiento
persistente para LightWan (28,649,013,413 bytes). H3 usa aproximadamente 42.1 GB en su propio
directorio. Juntos son aproximadamente 70.7 GB (65.9 GiB), bajo 100 GB de
capacidad para pesos. Confirma la franquicia y tarifa en tu cuenta de RunPod.
[Artefactos de LightWan](https://huggingface.co/lightx2v/LightWan2.2-A14B/tree/a6f09bcd5ffbd079e7397cb7c185691a13445af7)
[Componentes Wan fijados](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/tree/206a9ee1b7bfaaf8f7e4d81335650533490646a3)

Busca `imagine_metric` en los logs para medir staging, carga, inferencia y
tiempo total. Desde una terminal del Pod:

```bash
printf 'header = "Authorization: Bearer %s"\n' "$IMAGINE_VIDEO_TOKEN" |
  curl -fsS -K - http://127.0.0.1:8000/v1/health
nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -l 2
cat /sys/fs/cgroup/memory.current
```

### Retención y benchmark

Las entradas y resultados viven bajo `/tmp` en el contenedor. La API
conserva el resultado descargable hasta ACK, DELETE o 15 minutos. Una
cancelación o timeout detiene el proceso CUDA y elimina los archivos del
trabajo. El worker recarga pesos para recuperar la salud antes de admitir
otro trabajo. El apagado cierra el proceso CUDA y el reinicio purga los
archivos del trabajo anterior. La eliminación
de archivos es lógica y depende también del almacenamiento del proveedor.

Prueba `turbo-v1`, `h3-optimized-v1` y `lightwan-v1` en la misma clase de
Pod con las mismas imágenes y prompts. Registra el arranque hasta salud,
tiempo del POST hasta completado y hasta descarga. Mide una ejecución fría
y tres ejecuciones consecutivas por variante y duración. Compara medianas y
rangos, tiempos de etapas disponibles, picos de RAM/VRAM y
`tarifa por hora × segundos facturados / 3600`. Registra el storage por
separado. La API del MVP 1 asigna semillas aleatorias.
Comprueba FPS, resolución, duración y pistas:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration -of json salida.mp4
```

Puntúa a ciegas rostro e identidad, anatomía, ropa, fondo, consistencia
temporal y cumplimiento del prompt con una escala fija de 1 a 5. Antes de
repetir la comparación con 15 s, comprueba duración y consumo de memoria
del primer clip de 10 s. Verifica autenticación, cancelación, ACK y expiración
observando el estado en la API y la limpieza local. El objetivo experimental
es 2–3 minutos o menos para un clip LightWan de 10 s. La latencia y calidad
en RunPod están pendientes.

El Dockerfile fija el digest de la base oficial Blackwell de LightX2V y el
commit `94f0f19459e66f61a723db7690f1707f308599ef` de la etiqueta Git `0.6.0`.
La base incluye compilaciones CUDA para CUTLASS, SageAttention y SpargeAttn,
así como herramientas y bibliotecas upstream adicionales. La imagen usa
SM120 y carga la variante NVFP4 de I2V. El diagnóstico de salud queda listo
cuando termina la carga de pesos.
