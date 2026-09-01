#!/usr/bin/env bash
# Downloads the Practical-RIFE and Real-ESRGAN model weights into repo-local
# cache directories. Idempotent: skips work that is already done.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIFE_MODEL_DIR="${RIFE_MODEL_DIR:-$REPO_ROOT/.cache/rife}"
REALESRGAN_MODEL_DIR="${REALESRGAN_MODEL_DIR:-$REPO_ROOT/.cache/realesrgan}"
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

# 3) General photographic-video super-resolution weights. The official model
# is loaded locally through Spandrel; there are no remote inference calls.
REALESRGAN_WEIGHTS="$REALESRGAN_MODEL_DIR/RealESRGAN_x4plus.pth"
REALESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

mkdir -p "$REALESRGAN_MODEL_DIR"
if [[ -f "$REALESRGAN_WEIGHTS" ]]; then
  echo "[realesrgan] weights already present at $REALESRGAN_WEIGHTS"
else
  if ! command -v curl >/dev/null 2>&1; then
    echo "[realesrgan] ERROR: curl is required to download model weights." >&2
    exit 1
  fi
  echo "[realesrgan] downloading RealESRGAN_x4plus weights"
  TEMP_WEIGHTS="$REALESRGAN_WEIGHTS.part"
  rm -f "$TEMP_WEIGHTS"
  curl --fail --location --retry 3 --output "$TEMP_WEIGHTS" "$REALESRGAN_URL"
  mv "$TEMP_WEIGHTS" "$REALESRGAN_WEIGHTS"
fi

echo "[realesrgan] ready: $REALESRGAN_WEIGHTS"
