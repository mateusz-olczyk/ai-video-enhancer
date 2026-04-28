#!/usr/bin/env bash
# Downloads Practical-RIFE source + v4.6 model weights into $RIFE_MODEL_DIR
# (defaults to <repo>/.cache/rife). Idempotent: skips work that's already done.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIFE_MODEL_DIR="${RIFE_MODEL_DIR:-$REPO_ROOT/.cache/rife}"
SRC_DIR="$RIFE_MODEL_DIR/Practical-RIFE"
TRAIN_LOG="$SRC_DIR/train_log"

mkdir -p "$RIFE_MODEL_DIR"

# 1) Source: clone Practical-RIFE (model architecture code we import at runtime).
if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "[rife] cloning Practical-RIFE into $SRC_DIR"
  git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git "$SRC_DIR"
else
  echo "[rife] Practical-RIFE already cloned at $SRC_DIR"
fi

# 2) Weights: v4.6 train_log archive hosted on Google Drive (per upstream README).
#    https://github.com/hzwer/Practical-RIFE#model-list
GDRIVE_ID="1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE"
ZIP_PATH="$RIFE_MODEL_DIR/RIFE_trained_model_v4.6.zip"

if [[ -f "$TRAIN_LOG/flownet.pkl" ]]; then
  echo "[rife] weights already present at $TRAIN_LOG/flownet.pkl"
else
  if ! command -v gdown >/dev/null 2>&1; then
    echo "[rife] ERROR: gdown not installed. Activate the venv first: source .venv/bin/activate" >&2
    exit 1
  fi
  echo "[rife] downloading v4.6 weights via gdown"
  gdown "$GDRIVE_ID" -O "$ZIP_PATH"
  echo "[rife] extracting weights into $SRC_DIR"
  unzip -o -q "$ZIP_PATH" -d "$SRC_DIR"
  rm -f "$ZIP_PATH"
  if [[ ! -f "$TRAIN_LOG/flownet.pkl" ]]; then
    echo "[rife] ERROR: expected $TRAIN_LOG/flownet.pkl after extract." >&2
    echo "       The Google Drive layout may have changed; download manually" >&2
    echo "       from https://github.com/hzwer/Practical-RIFE and place" >&2
    echo "       train_log/* into $TRAIN_LOG/" >&2
    exit 1
  fi
fi

echo "[rife] ready: $TRAIN_LOG/flownet.pkl"
