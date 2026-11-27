"""Worker-only runtime setup and opt-in H3 VAE output buffering."""
from copy import copy
from math import prod
import os

import nodes
import torch
from comfy import model_management
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from h3_diagnostics import install, install_frame_cache_cleanup
from h3_runtime import runtime_settings


class ImagineH3BufferedVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        model = vae.first_stage_model
        if (not isinstance(model, MiniMaxH3VideoVAE)
                or vae.device.type != "cuda" or vae.output_device.type != "cpu"):
            return super().decode(vae, samples)

        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        dtype_bytes = model_management.dtype_size(vae.vae_output_dtype())

        def output_bytes(shape):
            return prod(model.decode_output_shape(shape)) * dtype_bytes

        buffer_bytes = output_bytes(latent.shape)
        workspace = vae.memory_used_decode(latent.shape, vae.vae_dtype)
        required = (buffer_bytes + workspace + vae.patcher.model_size()
                    + model_management.minimum_inference_memory())
        if vae.patcher.get_free_memory(vae.device) < required:
            return super().decode(vae, samples)

        gpu_vae = copy(vae)
        gpu_vae.output_device = vae.device
        gpu_vae.memory_used_decode = lambda shape, dtype: vae.memory_used_decode(shape, dtype) + output_bytes(shape)
        try:
            images, = super().decode(gpu_vae, samples)
            return (images.to(vae.output_device),)
        except Exception as error:
            model_management.raise_non_oom(error)
        # Retry after the failed allocation's exception frame is released.
        model_management.soft_empty_cache()
        return super().decode(vae, samples)


intra, inter, trace, _, _ = runtime_settings()
if inter:
    torch.set_num_interop_threads(inter)
if intra:
    torch.set_num_threads(intra)
debug_metrics = os.environ.get("IMAGINE_DEBUG_METRICS") == "1"
if debug_metrics:
    install(trace)
install_frame_cache_cleanup(report=debug_metrics)

NODE_CLASS_MAPPINGS = {"ImagineH3BufferedVAEDecode": ImagineH3BufferedVAEDecode}
