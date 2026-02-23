#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python executable not found. Set PYTHON_BIN or install Python." >&2
    exit 1
  fi
fi

LEVELS=(bottom middle top)

echo "Starting sequential Jukebox VQ-VAE training: ${LEVELS[*]}"
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"

for level in "${LEVELS[@]}"; do
  echo ""
  echo "============================================="
  echo "Training level: $level"
  echo "============================================="
  "$PYTHON_BIN" src/train_scripts/train_vqvae_jukebox.py --level "$level" "$@"
done

echo ""
echo "All levels trained successfully: ${LEVELS[*]}"
