#!/usr/bin/env bash
# Creates .venv, installs requirements, and wires repo-local cache env vars
# into the venv's activate script so they are exported automatically on
# `source .venv/bin/activate`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
CACHE_DIR="$REPO_ROOT/.cache"
ACTIVATE="$VENV_DIR/bin/activate"

mkdir -p "$CACHE_DIR"/{pip,torch,huggingface,rife,matplotlib}

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
# Allow MPS to fall back to CPU for ops not yet implemented on Apple GPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# <<< ai-video-enhancer cache env <<<
EOF
fi

# shellcheck disable=SC1090
source "$ACTIVATE"

echo "[setup] upgrading pip + installing requirements"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "[setup] done. Activate with: source .venv/bin/activate"
