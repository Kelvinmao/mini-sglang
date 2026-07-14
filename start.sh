#!/usr/bin/env bash

set -Eeuo pipefail

################################################################################
# mini-sglang launcher
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Activate virtual environment
################################################################################

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found."
    exit 1
fi

source .venv/bin/activate

################################################################################
# Environment
################################################################################

export CUDA_VISIBLE_DEVICES=0

# Uncomment this line when debugging CUDA errors.
# export CUDA_LAUNCH_BLOCKING=1

# Let FlashInfer JIT compile kernels if needed.
unset FLASHINFER_DISABLE_JIT

################################################################################
# Check GPU
################################################################################

echo
echo "================ GPU ================"
nvidia-smi || {
    echo "ERROR: NVIDIA driver unavailable."
    exit 1
}

################################################################################
# Check Python
################################################################################

echo
echo "================ Python ================"

python - <<'PY'
import torch

print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)

assert torch.cuda.is_available(), "CUDA is unavailable"

print("GPU   :", torch.cuda.get_device_name(0))
PY

################################################################################
# Launch
################################################################################

echo
echo "================ Launch mini-sglang ================"
echo

exec python -m minisgl \
    --model-path "Qwen/Qwen3-0.6B" \
    --dtype float16 \
    --attention-backend fi \
    --max-running-requests 4 \
    --cuda-graph-max-bs 4 \
    --memory-ratio 0.7