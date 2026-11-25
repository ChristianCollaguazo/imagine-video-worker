# Imagine Video Worker (standalone)

Servicio privado de una sola cola para convertir **una imagen inicial y un prompt** en un MP4 de 10 o 15 segundos. Base y Turbo reutilizan un único stack oficial H3 cuantizado: difusión FL2VA INT8 ConvRot, encoder Qwen3VL NVFP4-AWQ y VAE de video INT8. Base no usa LoRA; Turbo aplica la LoRA Lightning oficial a 8 pasos. Turbo es un experimento que se espera que genere más rápido, pero la calidad no está garantizada; la guía oficial advierte que puede suavizar el movimiento y el audio. La cuantización y Turbo tampoco cuentan con una evaluación de calidad en este worker. Este worker se ejecuta desde el repositorio privado desplegado en la Pod de servicio.

## Estado y alcance

- `app/` contiene el worker privado y el adaptador ComfyUI.
- `config/model-manifest.json` es la única fuente de verdad para revisiones, archivos y SHA-256 de pesos.
- `scripts/` trae bootstrap, descarga verificada, inicio y apagado. Bootstrap no instala ni descarga nada si no se pasa una opción explícita.
- No se incluyen secretos, fotos, resultados ni pesos en Git. El `.gitignore` local excluye esos archivos.

## Requisitos y tamaño

El manifiesto fija ComfyUI `v0.34.0`, que incluye los nodos MiniMax H3 en core. No hacen falta nodos externos. En una máquina NVIDIA se recomienda Python 3.12, driver NVIDIA compatible con CUDA 13 y almacenamiento persistente de **al menos 100 GB** para el código, unos 40 GB del stack cuantizado, archivos auxiliares, caché de descargas y margen. La LoRA Turbo opcional añade unos 2 GB.

Pesos fijados en el manifiesto:

| Archivo | Función | Tamaño publicado |
|---|---|---:|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Difusión FL2VA INT8 ConvRot | 21 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Encoder Qwen3VL NVFP4-AWQ | 15.7 GB |
| `minimax_h3_video_vae_int8_convrot.safetensors` | VAE de video INT8 ConvRot | 2.81 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | VAE de audio FP32 | 605 MB |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | LoRA Turbo opcional de 8 pasos | 1.96 GB |

El manifiesto fija Python 3.12, PyTorch 2.11.0, torchvision 0.26.0 y torchaudio 2.11.0 con la rueda oficial CUDA 13.0; `TORCH_INDEX_URL` permite seleccionar otra rueda oficial compatible. PyTorch documenta estas ruedas CUDA y amplía soporte CUDA 13 a Blackwell y Hopper; comprueba la combinación concreta de driver, wheel y GPU antes de instalar. El encoder NVFP4 tiene aceleración nativa en Blackwell; otras GPU pueden cargarlo con dequantización y sin esa aceleración. Este worker no tiene validaciones de rendimiento ni de memoria para 10 s o 15 s en RTX PRO 6000 ni H200. La cuantización reduce el tamaño de los archivos, pero no implica una garantía sobre calidad o VRAM. Haz una prueba controlada y mira el uso de VRAM antes de presupuestar cargas recurrentes.

El preflight muestra GPU, VRAM, driver, memoria del sistema y espacio libre; bloquea la descarga si no hay espacio para los archivos faltantes más margen de caché. Después de instalar, bootstrap verifica CUDA/BF16 en PyTorch y que `ffmpeg`/`ffprobe` estén disponibles. El preflight no mide el tiempo por video ni garantiza que una tarea quepa en la GPU.

## Duración, resolución y exportación

H3 acepta longitudes en la cuadrícula `17k + 5` cuadros a 24 fps. Para satisfacer el objetivo de video exacto, el workflow genera el legal más cercano hacia arriba y el worker recorta y vuelve a codificar a H.264 a 24 fps:

| Pedido | Cuadros generados por H3 | MP4 final | Duración final |
|---|---:|---:|---:|
| 10 s | 243 (10.125 s) | 240 | 10.000 s |
| 15 s | 362 (15.083 s) | 360 | 15.000 s |

La resolución usa lados múltiplos de 32 y un lado corto de 768, dentro del canvas nativo máximo de 1344 × 768 (apaisado) o 768 × 1344 (vertical). El nodo nativo estira el primer cuadro para llenar el canvas; fotos muy panorámicas, muy altas o de otra relación de aspecto pueden deformarse. Recorta la imagen de origen a la relación de aspecto de salida si quieres conservar proporciones.

El modo Base usa **12 pasos por defecto**. Para cambiar el valor predeterminado del worker, configura `IMAGINE_VIDEO_STEPS` como `8`, `12` o `20` antes de iniciarlo. Cada petición también puede incluir el campo multipart opcional `steps` con uno de esos valores para seleccionar los pasos del trabajo. La respuesta del trabajo incluye la selección. `GET /v1/health` anuncia `supported_steps: [8, 12, 20]` para que el cliente compruebe compatibilidad. Cualquier otro valor se rechaza.

El campo multipart opcional `mode` acepta `base` o `turbo` y usa `base` por defecto. Turbo siempre usa 8 pasos y la LoRA `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`; rechaza cualquier selección distinta de 8 y falla con HTTP 503 si falta el archivo. `GET /v1/health` incluye `turbo` en `supported_modes` solo cuando la LoRA está instalada. Base y Turbo usan el mismo stack cuantizado; cambiar de modo no descarga otro modelo base.

## Configuración inicial en una máquina CUDA

Las instrucciones son deliberadamente explícitas: ejecutar `--install-runtime` instala paquetes y clona ComfyUI; ejecutar `--download-models` descarga unos 40 GB del stack cuantizado compartido por Base y Turbo. Añadir `--download-turbo` descarga la LoRA de unos 2 GB. No las ejecutes antes de revisar el presupuesto, el almacenamiento, la licencia y la ubicación física.

```bash
cd imagine-video-worker
scripts/bootstrap.sh --install-runtime
scripts/bootstrap.sh --download-models --download-turbo
```

Si Hugging Face solicita acceso al modelo, acepta la licencia en la cuenta autorizada y facilita `HF_TOKEN` solo en el entorno del proceso de descarga. No lo pegues en el repositorio, comandos guardados, historial de terminal ni archivo `.env`.

La descarga valida los cuatro hashes SHA-256 del stack base y el hash de la LoRA cuando se solicita. Si un hash no coincide, descarga fallará con el nombre exacto del archivo; elimina solo ese archivo corrupto de `ComfyUI/models/<ruta>` y vuelve a ejecutar la descarga correspondiente. Los archivos parciales de caché de Hugging Face se pueden reanudar.

Para habilitar la prueba Turbo después de instalar los pesos base, ejecuta `scripts/bootstrap.sh --download-turbo`. Primero comprueba que el stack base esté instalado; después descarga únicamente la LoRA opcional de aproximadamente 1.96 GB, fijada por revisión y SHA-256 en el manifiesto. No vuelve a descargar los modelos base. Para un Pod nuevo, usa `scripts/bootstrap.sh --download-models --download-turbo` una sola vez para instalar el stack compartido y la LoRA juntos.

Antes de iniciar, inyecta `IMAGINE_VIDEO_TOKEN` desde el gestor de secretos del sistema donde corre el servicio. Debe ser un token aleatorio largo. Para una sesión local de shell puede exportarse temporalmente en el entorno; no se guarda en Git. En la Pod de servicio, usa el gestor de secretos del proveedor.

```bash
export IMAGINE_VIDEO_TOKEN='(inyectado desde el gestor de secretos)'
scripts/start.sh
```

ComfyUI escucha solo en `127.0.0.1:8188`. El worker escucha en el puerto `8000`; en Pod, expón únicamente su proxy **HTTPS**. La ruta `GET /v1/health` requiere Bearer token y comprueba CUDA y nodos de ComfyUI. Los archivos de fotos, videos, temporales y estado de cola van a `/tmp/imagine-video-worker`; los logs se descartan y ComfyUI no escribe metadatos de prompt en el MP4. El worker borra los medios ligados al trabajo al completar, fallar, cancelar, expirar o reiniciar. El límite de carga del worker es 50 MiB por imagen.

Para apagar:

```bash
scripts/stop.sh
```

El apagado pide terminación normal al worker y a ComfyUI. Al volver a iniciar, el worker vacía sus trabajos transitorios. Revisa `scripts/stop.sh` y el estado del proceso; no des por detenida una máquina facturable por cerrar la terminal.

## API privada

Cada petición de trabajo necesita `Authorization: Bearer <IMAGINE_VIDEO_TOKEN>`, una imagen JPEG/PNG/WebP, el prompt y `duration_seconds` igual a 10 o 15. Hay una sola tarea activa y no se aceptan imágenes por lote ni imágenes finales opcionales. La respuesta del worker incluye un ID; consulta ese trabajo hasta `completed`, descarga el MP4 y envía el ACK para purgarlo. Los endpoints y los límites de carga están en `app/main.py`.

| Ruta | Acción | Comportamiento |
|---|---|---|
| `GET /v1/health` | Salud autenticada | Comprueba CUDA y los nodos nativos requeridos |
| `POST /v1/jobs` | Subir una foto, prompt y duración | Máximo 50 MiB; soporta `Idempotency-Key`; una tarea activa |
| `GET /v1/jobs/{id}` | Consultar progreso/estado | Fases de carga, muestreo, decodificación y exportación |
| `GET /v1/jobs/{id}/result` | Descargar MP4 | Disponible mientras el resultado esté retenido |
| `POST /v1/jobs/{id}/ack` | Confirmar importación | Borra la copia remota del resultado y vence el trabajo |
| `DELETE /v1/jobs/{id}` | Cancelar o purgar | Interrumpe el trabajo y borra sus temporales |

`prompt` acepta de 1 a 10.000 caracteres y el worker lo pasa completo a ComfyUI sin recortarlo. El encoder Qwen3VL puede hacer que prompts largos consuman más tiempo y memoria; el límite interno de contexto efectivo no está fijado aquí. LoRAs disponibles se redescubren en cada consulta de `GET /v1/loras`. `POST /v1/loras` acepta `file` como multipart con un basename `.safetensors`, requiere Bearer token, carga en streaming hasta 4 GiB por defecto (`IMAGINE_VIDEO_MAX_LORA_UPLOAD_BYTES`) y responde `{"name":"...","size":123}`; responde 409 si el nombre ya existe y nunca lo reemplaza. `POST /v1/jobs` acepta el campo multipart opcional `lora_stack` como un array JSON ordenado, por ejemplo `[{"name":"style.safetensors","strength":0.7},{"name":"motion.safetensors","strength":1.1}]`; cada entrada se aplica secuencialmente al modelo. Los nombres deben ser basenames instalados, no se permiten duplicados y cada intensidad debe estar entre 0 y 2. `lora_stack=[]` desactiva LoRAs. Si se omite, se conserva el comportamiento de modo actual (Turbo usa su LoRA por defecto; Base no aplica una). Los campos anteriores `lora_name` y `lora_strength` siguen aceptándose para clientes con una sola LoRA; no se pueden combinar con `lora_stack`.

El resultado se conserva hasta que se confirme con ACK o caduque: 15 minutos de retención, revisados por un barrido cada 30 segundos. `IMAGINE_VIDEO_JOB_TIMEOUT_SECONDS` ajusta el límite de generación; por defecto es 5400 segundos (90 minutos; mínimo 60). El token también se exige en salud. No se registran prompts ni medios en logs persistentes.

Comprueba la salud:

```bash
printf 'url = "https://<HOST-HTTPS-DEL-WORKER>/v1/health"\nheader = "Authorization: Bearer %s"\nfail\nsilent\nshow-error\n' "$IMAGINE_VIDEO_TOKEN" | curl --config - --output /dev/null --max-time 10
```

La URL HTTP local del worker se puede usar en un entorno local de confianza. No publiques 8188 ni habilites la interfaz web de ComfyUI en Internet.

## Operación y recuperación: Pod privada con CUDA

Este procedimiento documenta la Pod privada en la que corre el worker desplegado. Los pasos para crear una Pod y clonar el repositorio sirven para reconstruir la instalación desde cero. El código se aloja en el repositorio privado `ChristianCollaguazo/imagine-video-worker`.

1. **Revisar licencia y región antes de crear la Pod o descargar pesos.** La licencia comunitaria de MiniMax H3 define como territorios excluidos Estados Unidos, Unión Europea, Reino Unido y República de Corea. El uso alojado también está sujeto a esa restricción; elige una región física fuera de esas zonas o consigue autorización previa de MiniMax. Confirma la licencia vigente y región concreta del proveedor, no solo la ubicación de quien opera la cuenta.
2. **Preflight del costo.** El objetivo de gasto de este ensayo es **US$7 total**. Antes de iniciar, verifica en el panel el precio por hora vigente de la GPU, almacenamiento persistente, cargo mientras la Pod está detenida, descarga inicial de unos 42 GB si usarás Turbo, transferencia y tiempo de arranque. Estima el total con margen para primer inicio y generación; si no cabe bajo US$7, no inicies. El precio cambia por región y disponibilidad; este README no cotiza ni reserva infraestructura.
3. **Recrear la Pod privada (Secure Cloud) solo si hace falta** con acceso SSH privado, entorno Linux CUDA, volumen persistente montado en `/workspace` de al menos 100 GB. La descarga completa del stack cuantizado con Turbo requiere alrededor de 42 GB más espacio para caché, runtime y resultados. Configura el proveedor para no admitir acceso público directo por SSH y usa clave SSH controlada por ti. RunPod usa `https://<pod-id>-<port>.proxy.runpod.net` para un puerto HTTP expuesto; selecciona únicamente `8000/http` y la opción de SSH privada necesaria para administración. No expongas `8188`.
4. **Clonar el repositorio privado en el volumen**, con autenticación Git de solo lectura configurada como clave deploy de solo lectura en el gestor de secretos del proveedor. No añadas credenciales al código ni guardes la clave en el volumen.

   ```bash
   cd /workspace
   git clone --depth 1 git@github.com:ChristianCollaguazo/imagine-video-worker.git imagine-video-worker
   cd imagine-video-worker
   ```

5. **Instalar una vez** en el volumen persistente con el mismo bootstrap anterior: ejecuta `scripts/bootstrap.sh --install-runtime` y luego `scripts/bootstrap.sh --download-models --download-turbo` para descargar el único stack cuantizado compartido y la LoRA opcional. ComfyUI y los pesos quedan bajo el árbol montado en `/workspace`; detén la Pod sin borrar el volumen para conservarlos. No coloques fotos de usuarios, resultados ni tokens en ese volumen.
6. **Configurar el entorno fuera del repositorio**: inyecta `IMAGINE_VIDEO_TOKEN` desde el gestor de secretos del proveedor. El muestreo base usa 12 pasos por defecto; configura `IMAGINE_VIDEO_STEPS=8`, `12` o `20` en el entorno de la Pod para cambiarlo. Si la LoRA opcional no se instaló en el paso anterior, `scripts/bootstrap.sh --download-turbo` comprueba el stack base y descarga solo la LoRA de aproximadamente 1.96 GB. En una Pod que ya tiene ese stack, no vuelvas a ejecutar `--download-models` para habilitar Turbo. No copies `.env`, claves privadas, contraseñas ni tokens al directorio de trabajo. Usa un token nuevo para la instancia remota.
7. **Limitar red**: deja el servidor ComfyUI `8188` accesible solo por loopback; expón el servicio del worker por el endpoint HTTPS del proveedor en `8000`; no abras otros puertos de aplicación. Mantén SSH privado para la administración. Revisa reglas de red y URL antes de enviar una foto.
8. **Arrancar y comprobar salud**, hacer validación aislada con la misma foto autorizada: genera primero 10 s en modo Base y verifica MP4 exacto de 240 cuadros, 24 fps y H.264; ACK/purga ese trabajo. Luego, si el presupuesto lo permite, compara un video Turbo de 10 s con el mismo prompt/foto y verifica el MP4; la calidad y la velocidad de la comparación son experimentales y no están garantizadas. Incluye la descarga fría y los trabajos en el límite total de US$7; mira el gasto entre trabajos y no inicies el siguiente si la proyección supera el límite. La URL HTTPS es `https://<pod-id>-8000.proxy.runpod.net`. Si hay error CUDA/OOM, detén el servicio y la Pod.
9. **Conectar Imagine local al Pod**: en el archivo raíz `Imagine/.env` (ignorado por Git), añade `IMAGINE_VIDEO_URL=https://<pod-id>-8000.proxy.runpod.net` y `IMAGINE_VIDEO_TOKEN=<el mismo secreto inyectado en la Pod>`. Luego detén la app local y arráncala de nuevo con `./start_wardrobe_app.command`; el backend lee `.env` durante el arranque. Comprueba que la salud HTTPS con token responde, inicia un único video de prueba desde Imagine y confirma que el MP4 se importa localmente. Mantén las claves solo en el gestor de secretos y `.env` local.
10. **Cerrar después de cada sesión**: ejecuta `scripts/stop.sh`, comprueba `GET /v1/health` deja de responder con una petición Bearer que tenga timeout, detén la Pod desde el panel y verifica que la GPU ya no factura. Revisa el costo acumulado cada vez que inicias o detienes; una terminal cerrada no detiene la facturación. La persistencia o costo del volumen puede continuar al parar la Pod; confirma los cargos en el panel.
11. **Limpieza final**: descarga solo los MP4 que quieras conservar; luego elimina la Pod y, si ya no necesitas el caché, borra el volumen persistente desde el panel. Comprueba que desaparecieron los recursos facturables. La limpieza normal de trabajos elimina imagen, video e intermedios temporales; borrar el volumen además borra los pesos que tendrías que volver a descargar.

### Inicio de sesiones posteriores

1. En el panel inicia la Pod ya creada y confirma que quedó `RUNNING`, en la región física autorizada y con el volumen `/workspace` adjunto.
2. Comprueba el gasto acumulado y la tarifa actual antes de abrir el servicio. El volumen puede conservar cargos aunque la GPU esté parada.
3. Conéctate por SSH privado, entra a `/workspace/imagine-video-worker` y ejecuta `scripts/start.sh`. Los pesos permanecen en el volumen; no repitas `--install-runtime` ni `--download-models`.
4. Desde Imagine, confirma la salud HTTPS con token y procesa solo los trabajos necesarios.
5. Al acabar, `scripts/stop.sh`, revisa la limpieza y detén la Pod desde el panel. Verifica de nuevo el gasto y el estado de la GPU.

## Fuentes oficiales fijadas

- [Guía oficial de MiniMax H3 en ComfyUI](https://github.com/Comfy-Org/docs/blob/main/tutorials/video/minimax/minimax-h3-native.mdx): 768 px en el lado corto, 1344 × 768 nativo, cuadro `17k + 5`, nodos nativos, Turbo opcional.
- [Plantilla oficial H3 I2V de ComfyUI](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json): LoRA Lightning opcional de 8 pasos para Turbo.
- [ComfyUI v0.34.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.34.0): release de core con nodos MiniMax H3.
- [Manifiesto de archivos Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main): stack FL2VA INT8 ConvRot, encoder NVFP4-AWQ y VAE INT8 usados por este flujo.
- [Licencia comunitaria oficial MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE): alcance territorial, restricciones y condiciones para servicios alojados.
- [PyTorch: instalación local](https://pytorch.org/get-started/locally/) y [versiones CUDA anteriores](https://pytorch.org/get-started/previous-versions/): elige la rueda adecuada para el driver y GPU concretos.
- [PyTorch: soporte CUDA 13 y Blackwell](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337).
- [RunPod: URLs de Pod, regiones, puertos y ciclo de vida](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods) y [persistencia de `/workspace`](https://docs.runpod.io/pods/troubleshooting/zero-gpus).
