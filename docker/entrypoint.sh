#!/usr/bin/env bash
set -euo pipefail
cd /opt/imagine-video-worker
: "${IMAGINE_VIDEO_TOKEN:?Configure IMAGINE_VIDEO_TOKEN in RunPod Secrets}"
export IMAGINE_VIDEO_TOKEN
export IMAGINE_VIDEO_DATA=/tmp/imagine-video-worker/jobs

# Runtime and GPU checks happen at deployment, never during the amd64 image build.
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
assert torch.cuda.is_bf16_supported(), "GPU does not report BF16 support"
capability = torch.cuda.get_device_capability(0)
assert capability == (12, 0), f"This NVFP4 LightWan build targets SM120, got compute capability {capability}"
print(f"LightWan GPU: {torch.cuda.get_device_name(0)}; compute capability {capability}; PyTorch {torch.__version__}", flush=True)
PY

python docker/stage_models.py
exec bash scripts/start.sh
