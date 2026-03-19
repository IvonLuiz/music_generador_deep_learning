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

DEFAULT_LEVELS=(top middle bottom)
LEVELS=("${DEFAULT_LEVELS[@]}")
FORWARD_ARGS=()
CONDITIONING_MODE="real"

print_usage() {
  cat <<'EOF'
Usage: train_pixelcnn_jukebox_all_levels.sh [--levels LEVEL_LIST] [--conditioning_mode MODE] [extra args...]

Options:
  --levels LIST            Comma-separated levels to train (subset/order allowed).
                           Valid values: top,middle,bottom
                           Default: top,middle,bottom
                           Example: --levels middle,bottom
  --conditioning_mode MODE Conditioning source for middle/bottom training.
                           Valid values: real,generated
                           Default: real
  -h, --help               Show this help message.

Any extra args are forwarded to:
  src/train_scripts/train_pixel_cnn_jukebox_hierarchical.py
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
    --conditioning_mode)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --conditioning_mode requires a value." >&2
        print_usage
        exit 1
      fi
      CONDITIONING_MODE="$1"
      ;;
    --conditioning_mode=*)
      CONDITIONING_MODE="${1#*=}"
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

case "$CONDITIONING_MODE" in
  real|generated)
    ;;
  *)
    echo "Error: invalid --conditioning_mode '$CONDITIONING_MODE'. Valid values are: real,generated" >&2
    exit 1
    ;;
esac

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  level="${level//[[:space:]]/}"
  LEVELS[$i]="$level"
  case "$level" in
    top|middle|bottom)
      ;;
    "")
      echo "Error: empty level detected in --levels value." >&2
      print_usage
      exit 1
      ;;
    *)
      echo "Error: invalid level '$level'. Valid values are: top,middle,bottom" >&2
      print_usage
      exit 1
      ;;
  esac
done

echo "Starting sequential Jukebox PixelCNN prior training: ${LEVELS[*]}"
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Conditioning mode: $CONDITIONING_MODE"

for level in "${LEVELS[@]}"; do
  echo ""
  echo "============================================="
  echo "Training prior level: $level"
  echo "============================================="
  "$PYTHON_BIN" src/train_scripts/train_pixel_cnn_jukebox_hierarchical.py \
    --level "$level" \
    --conditioning_mode "$CONDITIONING_MODE" \
    "${FORWARD_ARGS[@]}"
done

echo ""
echo "All requested prior levels trained successfully: ${LEVELS[*]}"
