#!/bin/bash
# Install all dependencies for KISS (auto-selects uv or plain venv)
# Usage: bash scripts/install.sh [cuda]
#   cuda: cu128 (default) | cu121
set -e

CUDA="${1:-cu128}"

case "$CUDA" in
    cu128) TORCH_VER="2.8.0" ;;
    cu121) TORCH_VER="2.3.1" ;;
    *)
        echo "Error: unsupported CUDA variant '$CUDA'. Use cu128 or cu121."
        exit 1
        ;;
esac

# --- resolve package manager ---
if command -v uv &>/dev/null; then
    USE_UV=1
else
    echo "uv not found, attempting to install..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$PATH"
        USE_UV=1
    else
        echo "uv install failed, falling back to plain venv + pip."
        USE_UV=0
    fi
fi

# --- create venv if not already active ---
if [ -z "$VIRTUAL_ENV" ]; then
    if [ "$USE_UV" = "1" ]; then
        uv venv .venv
    else
        python -m venv .venv
    fi
    source .venv/bin/activate
fi

PIP=$( [ "$USE_UV" = "1" ] && echo "uv pip" || echo "pip" )

echo "=== Using: $( [ "$USE_UV" = "1" ] && echo uv || echo pip ), PyTorch ${TORCH_VER} + ${CUDA} ==="

echo "=== Installing PyTorch ==="
$PIP install torch=="${TORCH_VER}" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "=== Installing torch-scatter ==="
$PIP install torch-scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA}.html"

echo "=== Installing other dependencies ==="
$PIP install numpy scipy tqdm pyyaml

echo "=== Done — activate with: source .venv/bin/activate ==="
