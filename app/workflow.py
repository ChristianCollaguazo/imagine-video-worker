"""Native ComfyUI MiniMax H3 FL2VA workflow (full BF16, no turbo/LoRA)."""
from __future__ import annotations

import json
from pathlib import Path

_manifest = json.loads((Path(__file__).resolve().parents[1] / "config/model-manifest.json").read_text())
_models = {item["id"]: item["name"] for item in _manifest["models"]}
MODEL = _models["fl2va_model"]
CLIP = _models["text_encoder"]
VIDEO_VAE = _models["video_vae"]
AUDIO_VAE = _models["audio_vae"]


def canvas_for_image(width: int, height: int) -> tuple[int, int]:
    """Use 768 px on the short edge, stay within 1344x768 native 16:9 canvas."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    ratio = width / height
    if ratio >= 1:
        out_h = 768
        out_w = min(1344, max(768, round(width / height * 768 / 32) * 32))
    else:
        out_w = 768
        out_h = min(1344, max(768, round(height / width * 768 / 32) * 32))
    return out_w, out_h


def model_frame_count(duration_seconds: int) -> int:
    """Return nearest legal H3 17k+5 frame length (24 fps); final clip is trimmed."""
    if duration_seconds not in (10, 15):
        raise ValueError("duration_seconds must be 10 or 15")
    target = duration_seconds * 24
    k = round((target - 5) / 17)
    return 17 * k + 5  # 243 or 362 frames, the legal H3 grid


def build_workflow(
    prompt: str,
    image_filename: str,
    duration_seconds: int,
    seed: int,
    image_width: int = 16,
    image_height: int = 9,
    job_id: str | None = None,
) -> dict:
    """Build the ComfyUI API prompt graph for one image plus one prompt."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not image_filename:
        raise ValueError("image_filename must be the ComfyUI uploaded filename")
    width, height = canvas_for_image(image_width, image_height)
    length = model_frame_count(duration_seconds)
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
        "7": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["6", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["10", 0], "sampler": ["8", 0],
            "sigmas": ["7", 0], "latent_image": ["6", 1],
        }},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24}},
        "15": {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0], "filename_prefix": f"imagine_h3/{job_id or 'job'}/video",
            "format": "mp4", "format.codec": "h264",
            "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 18,
        }},
    }
    return graph
