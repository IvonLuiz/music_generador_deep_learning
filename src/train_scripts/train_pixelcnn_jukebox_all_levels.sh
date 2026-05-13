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
CONFIG_PATH="./config/config_pixelcnn_jukebox_hierarchical.yaml"

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
  --config PATH            Config path forwarded to trainer and used to infer save root/model name.
                           Default: ./config/config_pixelcnn_jukebox_hierarchical.yaml
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
    --config)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --config requires a value." >&2
        print_usage
        exit 1
      fi
      CONFIG_PATH="$1"
      FORWARD_ARGS+=("--config" "$CONFIG_PATH")
      ;;
    --config=*)
      CONFIG_PATH="${1#*=}"
      FORWARD_ARGS+=("--config=$CONFIG_PATH")
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

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found: $CONFIG_PATH" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SHARED_RUN_ROOT="$($PYTHON_BIN - "$CONFIG_PATH" "$TIMESTAMP" <<'PY'
import os
import sys
import yaml

cfg_path = sys.argv[1]
timestamp = sys.argv[2]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

save_root = str(cfg.get('training', {}).get('save_dir', './models/')).strip()
model_name = str(cfg.get('model', {}).get('name', 'pixelcnn_jukebox')).strip() or 'pixelcnn_jukebox'
print(os.path.join(save_root, f'{model_name}_prior', timestamp))
PY
)"

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
echo "Shared run root: $SHARED_RUN_ROOT"

for level in "${LEVELS[@]}"; do
  echo ""
  echo "============================================="
  echo "Training prior level: $level"
  echo "============================================="
  "$PYTHON_BIN" src/train_scripts/train_pixel_cnn_jukebox_hierarchical.py \
    --level "$level" \
    --conditioning_mode "$CONDITIONING_MODE" \
    --run_dir_root "$SHARED_RUN_ROOT" \
    "${FORWARD_ARGS[@]}"
done

echo ""
echo "All requested prior levels trained successfully: ${LEVELS[*]}"
