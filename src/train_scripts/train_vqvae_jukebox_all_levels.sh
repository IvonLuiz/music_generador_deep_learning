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

DEFAULT_LEVELS=(bottom middle top)
LEVELS=("${DEFAULT_LEVELS[@]}")
FORWARD_ARGS=()

print_usage() {
  cat <<'EOF'
Usage: train_vqvae_jukebox_all_levels.sh [--levels LEVEL_LIST] [extra args...]

Options:
  --levels LIST   Comma-separated levels to train (subset/order allowed).
                  Valid values: bottom,middle,top
                  Example: --levels middle,top
  -h, --help      Show this help message.

Any extra args are forwarded to:
  src/train_scripts/train_vqvae_jukebox.py
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --levels)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --levels requires a value." >&2
        print_usage
        exit 1
      fi
      IFS=',' read -r -a LEVELS <<< "$1"
      ;;
    --levels=*)
      IFS=',' read -r -a LEVELS <<< "${1#*=}"
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      ;;
  esac
  shift
done

if [[ ${#LEVELS[@]} -eq 0 ]]; then
  LEVELS=("${DEFAULT_LEVELS[@]}")
fi

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  level="${level//[[:space:]]/}"
  LEVELS[$i]="$level"
  case "$level" in
    bottom|middle|top)
      ;;
    "")
      echo "Error: empty level detected in --levels value." >&2
      print_usage
      exit 1
      ;;
    *)
      echo "Error: invalid level '$level'. Valid values are: bottom,middle,top" >&2
      print_usage
      exit 1
      ;;
  esac
done

echo "Starting sequential Jukebox VQ-VAE training: ${LEVELS[*]}"
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"

for level in "${LEVELS[@]}"; do
  echo ""
  echo "============================================="
  echo "Training level: $level"
  echo "============================================="
  "$PYTHON_BIN" src/train_scripts/train_vqvae_jukebox.py --level "$level" "${FORWARD_ARGS[@]}"
done

echo ""
echo "All levels trained successfully: ${LEVELS[*]}"
