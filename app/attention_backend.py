"""Select an optional ComfyUI attention backend before starting the server."""

from __future__ import annotations

import json
import os
import sys


def _sage_probe(torch) -> tuple[bool, str]:
    """Import SageAttention and execute a tiny CUDA call to catch ABI/SM gaps."""
    try:
        from sageattention import sageattn

        q = torch.zeros((1, 16, 1, 64), device="cuda", dtype=torch.bfloat16)
        sageattn(q, q, q)
    except Exception as exc:  # optional binary extensions vary by CUDA/GPU build
        return False, type(exc).__name__
    return True, "ready"


def select_attention(requested: str, capability: tuple[int, int] | None,
                     sage_probe: tuple[bool, str]) -> tuple[str, str]:
    """Return (backend, reason); Sage is enabled only on SM120 after a probe."""
    if requested not in {"auto", "on", "off"}:
        raise ValueError("IMAGINE_SAGE_ATTENTION must be auto, on, or off")
    if requested == "off":
        return "pytorch_sdpa", "disabled_by_configuration"
    if capability != (12, 0):
        reason = "requires_sm120"
    elif not sage_probe[0]:
        reason = f"sage_probe_failed:{sage_probe[1]}"
    else:
        return "sage", "sm120_probe_passed"
    if requested == "on":
        raise RuntimeError(f"SageAttention was requested but unavailable: {reason}")
    return "pytorch_sdpa", reason


def detect() -> dict[str, object]:
    import torch

    capability = (tuple(torch.cuda.get_device_capability(0))
                  if torch.cuda.is_available() else None)
    requested = os.environ.get("IMAGINE_SAGE_ATTENTION", "on").lower()
    probe = _sage_probe(torch) if capability == (12, 0) and requested != "off" else (False, "not_probed")
    backend, reason = select_attention(requested, capability, probe)
    return {
        "backend": backend,
        "reason": reason,
        "compute_capability": ".".join(map(str, capability)) if capability else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(detect(), sort_keys=True), flush=True)
    except Exception as exc:
        print(f"attention backend selection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
