"""Run the Docker entrypoint's pinned model staging/download procedure."""
from __future__ import annotations

import os
from pathlib import Path
import runpy

if os.environ.get("IMAGINE_DOWNLOAD_MODELS") != "1":
    raise SystemExit("Set IMAGINE_DOWNLOAD_MODELS=1 for first-run model download")
runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker" / "stage_models.py"), run_name="__main__")
