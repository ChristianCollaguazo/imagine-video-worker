"""Native ComfyUI MiniMax H3 FL2VA base workflow and optional 8-step Turbo variant."""
from __future__ import annotations

import json
import os
from pathlib import Path

_manifest = json.loads((Path(__file__).resolve().parents[1] / "config/model-manifest.json").read_text())
_models = {item["id"]: item["name"] for item in _manifest["models"]}
_optional_models = {item["id"]: item["name"] for item in _manifest.get("optional_models", [])}
MODEL = _models["fl2va_model"]
CLIP = _models["text_encoder"]
VIDEO_VAE = _models["video_vae"]
AUDIO_VAE = _models["audio_vae"]
TURBO_LORA = _optional_models["fl2v_turbo_lora_8step"]
VIDEO_FPS = 24
SUPPORTED_DURATIONS = tuple(range(3, 31))
DEFAULT_STEPS = 12
ALLOWED_STEPS = tuple(range(1, 51))
SUPPORTED_RESOLUTIONS = (480, 576, 768)
DEFAULT_RESOLUTION = 768


def sampling_steps(configured: int | None = None, *, default: int = DEFAULT_STEPS) -> int:
    """Resolve an integer H3 sampler count in the ComfyUI-supported 1..50 range."""
    raw = os.environ.get("IMAGINE_VIDEO_STEPS", str(default)) if configured is None else configured
    try:
        steps = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be an integer from 1 to 50") from exc
    if steps not in ALLOWED_STEPS:
        raise ValueError("steps must be an integer from 1 to 50")
    return steps


def canvas_for_image(width: int, height: int, resolution: int = DEFAULT_RESOLUTION) -> tuple[int, int]:
    """Scale toward the selected short edge, keeping H3's native canvas bounds.

    Both dimensions stay on ComfyUI's 32-pixel grid. When the source aspect ratio
    would exceed H3's 1344x768 (or rotated) native canvas, both are reduced.
    """
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError("resolution must be 480, 576, or 768")
    ratio = width / height
    if resolution == 768 and 1.70 <= ratio <= 1.85:
        # Preserve ComfyUI's documented native 16:9-ish canvas at the default.
        return (1344, 768) if width >= height else (768, 1344)
    scale = resolution / min(width, height)
    max_long_edge = round((resolution * 1344 / 768) / 32) * 32
    if max(width, height) * scale > max_long_edge:
        scale = max_long_edge / max(width, height)
    ideal_w, ideal_h = width * scale, height * scale
    max_w, max_h = (max_long_edge, resolution) if width >= height else (resolution, max_long_edge)
    # Search nearby grid points for the closest aspect ratio within the canvas.
    base_w, base_h = max(32, round(ideal_w / 32) * 32), max(32, round(ideal_h / 32) * 32)
    candidates = []
    for out_w in range(max(32, base_w - 64), base_w + 65, 32):
        for out_h in range(max(32, base_h - 64), base_h + 65, 32):
            if out_w <= max_w and out_h <= max_h:
                aspect_error = abs((out_w / out_h) / ratio - 1)
                scale_error = abs((out_w / width) - (out_h / height)) / max(ideal_w / width, ideal_h / height)
                candidates.append((aspect_error, scale_error, -out_w * out_h, out_w, out_h))
    _, _, _, out_w, out_h = min(candidates)
    return out_w, out_h


def model_frame_count(duration_seconds: int) -> int:
    """Round up to the H3 17k+5 grid so the export can trim surplus frames."""
    if duration_seconds not in SUPPORTED_DURATIONS:
        raise ValueError("duration_seconds must be between 3 and 30")
    target = duration_seconds * VIDEO_FPS
    return 17 * ((target - 5 + 16) // 17) + 5


def build_workflow(
    prompt: str,
    image_filename: str,
    duration_seconds: int,
    seed: int,
    image_width: int = 16,
    image_height: int = 9,
    job_id: str | None = None,
    steps: int | None = None,
    *,
    audio: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    lora_stack: list[dict] | None = None,
) -> dict:
    """Build the ComfyUI API prompt graph for one image plus one prompt."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not image_filename:
        raise ValueError("image_filename must be the ComfyUI uploaded filename")
    width, height = canvas_for_image(image_width, image_height, resolution)
    length = model_frame_count(duration_seconds)
    steps = sampling_steps(steps)
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "width": width, "height": height, "length": length,
            "first_frame": ["5", 0],
        }},
        "7": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["6", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["10", 0], "sampler": ["8", 0],
            "sigmas": ["7", 0], "latent_image": ["6", 1],
        }},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": VIDEO_FPS}},
        "15": {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0], "filename_prefix": f"imagine_h3/{job_id or 'job'}/video",
            "format": "mp4", "format.codec": "h264",
            "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 18,
        }},
    }
    stack = lora_stack if lora_stack is not None else ([{"name": lora_name, "strength": lora_strength}] if lora_name else [])
    previous_model = ["1", 0]
    for index, entry in enumerate(stack):
        node_id = str(16 + index)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": previous_model, "lora_name": entry["name"], "strength_model": float(entry["strength"])},
        }
        previous_model = [node_id, 0]
    if stack:
        graph["7"]["inputs"]["model"] = previous_model
        graph["10"]["inputs"]["model"] = previous_model
    if not audio:
        # FL2VA still generates joint latents. Only skip decoding and muxing the audio.
        del graph["4"]
        del graph["13"]
        del graph["14"]["inputs"]["audio"]
    return graph


def build_turbo_workflow(
    prompt: str,
    image_filename: str,
    duration_seconds: int,
    seed: int,
    image_width: int = 16,
    image_height: int = 9,
    job_id: str | None = None,
    *,
    audio: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
    lora_name: str = TURBO_LORA,
    lora_strength: float = 1.0,
    steps: int = 8,
) -> dict:
    """Build the separate official 8-step LoRA I2V experiment graph."""
    graph = build_workflow(
        prompt, image_filename, duration_seconds, seed,
        image_width, image_height, job_id, steps=steps, audio=audio,
        resolution=resolution, lora_name=lora_name, lora_strength=lora_strength,
    )
    return graph
