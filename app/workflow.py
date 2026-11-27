"""Native ComfyUI MiniMax H3 FL2VA base workflow and optional 8-step Turbo variant."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from h3_runtime import vae_output_mode

_manifest = json.loads((Path(__file__).resolve().parents[1] / "config/model-manifest.json").read_text())
_models = {item["id"]: item["name"] for item in _manifest["models"]}
_optional_models = {item["id"]: item["name"] for item in _manifest.get("optional_models", [])}
MODEL = _models["fl2va_model"]
CLIP = _models["text_encoder"]
VIDEO_VAE = _models["video_vae"]
AUDIO_VAE = _models["audio_vae"]
TURBO_LORA = _optional_models["fl2v_turbo_lora_8step"]
VIDEO_FPS = 24
VIDEO_EXPORT_NODE = "ImagineH3ExportMP4"
BUFFERED_VAE_NODE = "ImagineH3BufferedVAEDecode"
SUPPORTED_DURATIONS = tuple(range(3, 31))
DEFAULT_STEPS = 16
ALLOWED_STEPS = tuple(range(1, 51))
SUPPORTED_RESOLUTIONS = (480, 576, 672, 768)
DEFAULT_RESOLUTION = 672
FIRST_BLOCK_CACHE_NODE = "ApplyMiniMaxH3FirstBlockCache"
FIRST_BLOCK_CACHE_SAFE_PRESET = "H3 Safe — 0.08 / max 2"
FIRST_BLOCK_CACHE_FAST_PRESET = "H3 Fast — 0.10 / max 2"
FIRST_BLOCK_CACHE_CUSTOM_MODE = "Custom — manual values"
FIRST_BLOCK_CACHE_PRESETS = {
    "safe": FIRST_BLOCK_CACHE_SAFE_PRESET,
    "fast": FIRST_BLOCK_CACHE_FAST_PRESET,
}
FIRST_BLOCK_CACHE_MODES = ("fast", "safe", "off", "custom")


@dataclass(frozen=True)
class FirstBlockCacheSettings:
    mode: str
    preset: str | None
    threshold: float | None
    max_consecutive_hits: int | None


def first_block_cache_mode(configured: str | None = None) -> str:
    """Return configured FirstBlockCache mode; Safe is the default."""
    raw = os.environ.get("IMAGINE_VIDEO_FIRST_BLOCK_CACHE", "safe") if configured is None else configured
    mode = str(raw).strip().lower()
    if mode not in FIRST_BLOCK_CACHE_MODES:
        raise ValueError("IMAGINE_VIDEO_FIRST_BLOCK_CACHE must be 'fast', 'safe', 'off', or 'custom'")
    return mode


def first_block_cache_settings(configured: str | None = None) -> FirstBlockCacheSettings:
    mode = first_block_cache_mode(configured)
    if mode == "off":
        return FirstBlockCacheSettings(mode, None, None, None)
    if mode != "custom":
        threshold = 0.10 if mode == "fast" else 0.08
        return FirstBlockCacheSettings(mode, FIRST_BLOCK_CACHE_PRESETS[mode], threshold, 2)

    threshold_error = "IMAGINE_VIDEO_FIRST_BLOCK_CACHE_THRESHOLD must be a finite number from 0 to 1 in custom mode"
    try:
        threshold = float(os.environ.get("IMAGINE_VIDEO_FIRST_BLOCK_CACHE_THRESHOLD", ""))
    except ValueError:
        raise ValueError(threshold_error) from None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(threshold_error)

    max_hits_error = "IMAGINE_VIDEO_FIRST_BLOCK_CACHE_MAX_HITS must be an integer from 1 to 20 in custom mode"
    try:
        max_hits = int(os.environ.get("IMAGINE_VIDEO_FIRST_BLOCK_CACHE_MAX_HITS", ""))
    except ValueError:
        raise ValueError(max_hits_error) from None
    if not 1 <= max_hits <= 20:
        raise ValueError(max_hits_error)
    return FirstBlockCacheSettings(mode, None, threshold, max_hits)


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
        raise ValueError("resolution must be 480, 576, 672, or 768")
    ratio = width / height
    if resolution == 768 and 1.70 <= ratio <= 1.85:
        # Preserve ComfyUI's documented native 16:9-ish canvas for this preset.
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
    first_block_cache: str | FirstBlockCacheSettings | None = None,
    vae_output: str | None = None,
) -> dict:
    """Build the ComfyUI API prompt graph for one image plus one prompt."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not image_filename:
        raise ValueError("image_filename must be the ComfyUI uploaded filename")
    width, height = canvas_for_image(image_width, image_height, resolution)
    length = model_frame_count(duration_seconds)
    steps = sampling_steps(steps)
    decode_node = BUFFERED_VAE_NODE if vae_output_mode(vae_output) == "gpu-buffer" else "VAEDecode"
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
        "12": {"class_type": decode_node, "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "15": {"class_type": VIDEO_EXPORT_NODE, "inputs": {
            "images": ["12", 0], "audio": ["13", 0],
            "filename_prefix": f"imagine_h3/{job_id or 'job'}/video",
            "fps": VIDEO_FPS, "frame_count": duration_seconds * VIDEO_FPS,
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
    cache = (
        first_block_cache if isinstance(first_block_cache, FirstBlockCacheSettings)
        else first_block_cache_settings(first_block_cache)
    )
    if cache.mode != "off":
        cache_node_id = str(16 + len(stack))
        graph[cache_node_id] = {
            "class_type": FIRST_BLOCK_CACHE_NODE,
            "inputs": {
                "model": previous_model,
                "mode": cache.preset or FIRST_BLOCK_CACHE_CUSTOM_MODE,
                "threshold": cache.threshold,
                "start_percent": 0.10,
                "end_percent": 0.95,
                "max_consecutive_hits": cache.max_consecutive_hits,
                "temporal_guard": False,
            },
        }
        graph["7"]["inputs"]["model"] = [cache_node_id, 0]
        graph["10"]["inputs"]["model"] = [cache_node_id, 0]
    if not audio:
        # FL2VA still generates joint latents. Only skip decoding and muxing the audio.
        del graph["4"]
        del graph["13"]
        del graph["15"]["inputs"]["audio"]
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
    steps: int | None = None,
    first_block_cache: str | None = None,
) -> dict:
    """Build the separate Turbo LoRA I2V experiment graph."""
    graph = build_workflow(
        prompt, image_filename, duration_seconds, seed,
        image_width, image_height, job_id, steps=steps, audio=audio,
        resolution=resolution, lora_name=lora_name, lora_strength=lora_strength,
        first_block_cache=first_block_cache,
    )
    return graph


if __name__ == "__main__":
    try:
        first_block_cache_settings()
    except ValueError as error:
        raise SystemExit(str(error)) from None
