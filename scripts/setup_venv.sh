#!/usr/bin/env bash
# Creates .venv, installs the package (editable, from pyproject.toml), and wires repo-local cache env vars
# into the venv's activate script so they are exported automatically on
# `source .venv/bin/activate`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
CACHE_DIR="$REPO_ROOT/.cache"
ACTIVATE="$VENV_DIR/bin/activate"

mkdir -p "$CACHE_DIR"/{pip,torch,huggingface,rife,realesrgan,matplotlib}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Wire cache env vars into activate (idempotent: guarded by marker).
MARKER="# >>> ai-video-enhancer cache env >>>"
if ! grep -q "$MARKER" "$ACTIVATE"; then
  echo "[setup] injecting cache env vars into $ACTIVATE"
  cat >> "$ACTIVATE" <<EOF

$MARKER
# All caches/models stay inside the repo's .cache directory.
export REPO_ROOT="$REPO_ROOT"
export XDG_CACHE_HOME="\$REPO_ROOT/.cache"
export PIP_CACHE_DIR="\$REPO_ROOT/.cache/pip"
export TORCH_HOME="\$REPO_ROOT/.cache/torch"
export HF_HOME="\$REPO_ROOT/.cache/huggingface"
export MPLCONFIGDIR="\$REPO_ROOT/.cache/matplotlib"
export RIFE_MODEL_DIR="\$REPO_ROOT/.cache/rife"
export REALESRGAN_MODEL_DIR="\$REPO_ROOT/.cache/realesrgan"
# Allow MPS to fall back to CPU for ops not yet implemented on Apple GPU.
# This is primarily relevant on macOS and harmless on Linux/WSL2.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# <<< ai-video-enhancer cache env <<<
EOF
fi

# Older venvs may already have the main marker without this newer model cache.
if ! grep -q '^export REALESRGAN_MODEL_DIR=' "$ACTIVATE"; then
  cat >> "$ACTIVATE" <<EOF
export REALESRGAN_MODEL_DIR="\$REPO_ROOT/.cache/realesrgan"
EOF
fi

# shellcheck disable=SC1090
source "$ACTIVATE"

echo "[setup] upgrading pip + installing package (editable)"
pip install --upgrade pip
pip install -e "$REPO_ROOT"

echo "[setup] done. Activate with: source .venv/bin/activate"
