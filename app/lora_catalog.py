"""Safe, dynamic catalog for locally available ComfyUI LoRA files."""
from pathlib import Path
import math


def available_lora_names(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    names = []
    for candidate in root.glob("*.safetensors"):
        name = candidate.name
        if name != Path(name).name or "/" in name or "\\" in name or name.startswith("."):
            continue
        try:
            if candidate.resolve(strict=True).is_file():
                names.append(name)
        except OSError:
            continue
    return sorted(set(names))


def resolve_lora_name(mode: str, requested: str | None, available: list[str], default_lora: str) -> str | None:
    name = (requested or "").strip()
    if not name and mode == "turbo":
        name = default_lora
    if not name:
        return None
    if Path(name).name != name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("lora_name must be an installed filename from /v1/loras")
    if name not in available:
        raise ValueError("lora_name must be an installed filename from /v1/loras")
    return name


def resolve_lora_stack(mode: str, requested: list[dict] | None, available: list[str], default_lora: str) -> list[dict[str, str | float]]:
    """Validate an ordered stack; None preserves legacy mode defaults, [] disables LoRAs."""
    if requested is None:
        name = resolve_lora_name(mode, None, available, default_lora)
        return [] if name is None else [{"name": name, "strength": 1.0}]
    if not isinstance(requested, list):
        raise ValueError("lora_stack must be a JSON array")
    result = []
    seen = set()
    for entry in requested:
        if not isinstance(entry, dict) or set(entry) != {"name", "strength"}:
            raise ValueError('each lora_stack entry must contain only "name" and "strength"')
        name = entry["name"]
        strength = entry["strength"]
        if not isinstance(name, str) or not name:
            raise ValueError("LoRA name must be an installed filename")
        name = resolve_lora_name("base", name, available, default_lora)
        if name in seen:
            raise ValueError("lora_stack cannot contain duplicate names")
        seen.add(name)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError("LoRA strength must be between 0 and 2")
        result.append({"name": name, "strength": float(strength)})
    return result
