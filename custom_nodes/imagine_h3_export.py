"""ComfyUI output node for the worker's private H3 workflow."""
import os
from pathlib import Path

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted

from video_export import export_video


class ImagineH3ExportMP4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "frame_count": ("INT", {"default": 72, "min": 1}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "video"

    def save(self, images, filename_prefix, fps, frame_count, audio=None):
        folder, prefix, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1]
        )
        filename = f"{prefix}_{counter:05}_.mp4"
        metrics = export_video(
            Path(folder) / filename, images, fps, frame_count, audio=audio,
            check_cancel=throw_exception_if_processing_interrupted,
            profile=os.environ.get("IMAGINE_DEBUG_METRICS") == "1",
        )
        return {"ui": {
            "images": [{"filename": filename, "subfolder": subfolder, "type": "output"}],
            "animated": (True,),
            "video_export_metrics": [metrics],
        }}


NODE_CLASS_MAPPINGS = {"ImagineH3ExportMP4": ImagineH3ExportMP4}
