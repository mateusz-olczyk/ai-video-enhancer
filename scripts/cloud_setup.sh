#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap: install the one missing system package,
# build the project venv, and download model weights. Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The default Cloud Agent image ships python3/git/curl/unzip/ffmpeg/gcc but not
# the venv module (ensurepip). Install it non-interactively when absent.
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "[cloud-setup] installing python3-venv"
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

echo "[cloud-setup] building venv + installing package"
bash scripts/setup_venv.sh

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[cloud-setup] downloading model weights"
bash scripts/download_model.sh

echo "[cloud-setup] done"
