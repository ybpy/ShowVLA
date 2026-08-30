#!/usr/bin/env bash
# Convert LumiData COCO grounding → RobotGroundingDataset HDF5 + meta_lumi JSONs.
set -euo pipefail

cd "$(dirname "$0")"

LUMI_ROOT="${LUMI_ROOT:-/home/hyx/LumiData}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hyx/datasets/Lumi_grounding}"
META_DIR="${META_DIR:-./meta_lumi}"
MIN_FRAMES="${MIN_FRAMES:-4}"

EXTRA_ARGS=("$@")

python convert_lumi_grounding.py \
  --lumi_root "$LUMI_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --meta_dir "$META_DIR" \
  --min_frames "$MIN_FRAMES" \
  --write_preview \
  "${EXTRA_ARGS[@]}"
