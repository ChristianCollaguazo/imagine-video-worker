"""Safe, dynamic catalog for locally available ComfyUI LoRA files."""
from pathlib import Path


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
