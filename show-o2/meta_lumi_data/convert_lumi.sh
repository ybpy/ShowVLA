#!/usr/bin/env bash
# Convert Lumi raw episodes to ShowVLA HDF5 (LumiHandler format).
# head/tail lengths mirror openpi/examples/lumi/convert_lumi_data_subtask.py RAW_DATA_META.
set -euo pipefail

cd "$(dirname "$0")"

META_PREFIX="${META_PREFIX:-Lumi}"
DATASET_NAME="${DATASET_NAME:-Lumi-mobile}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hyx/datasets/Lumi}"
SPEED_UP="${SPEED_UP:-1}"
CROP_MAIN="${CROP_MAIN:-(300, 20, 1280-220, 720)}"

# Extra flags for convert_lumi.py (e.g. --overwrite). Capture before run_one shadows "$@".
EXTRA_ARGS=("$@")

run_one() {
    local data_dir="$1"
    local head_length="$2"
    local tail_length="$3"
    local name
    name="$(basename "$data_dir")"
    python convert_lumi.py \
        --data_dir "$data_dir" \
        --output_dir "${OUTPUT_ROOT}/${name}" \
        --meta_prefix "$META_PREFIX" \
        --dataset_name "$DATASET_NAME" \
        --speed_up "$SPEED_UP" \
        --head_length "$head_length" \
        --tail_length "$tail_length" \
        --crop_main "$CROP_MAIN" \
        "${EXTRA_ARGS[@]}"
}

# (data_dir, head_length, tail_length)
run_one /home/hyx/LumiData/get_orange_juice_new 20 5
run_one /home/hyx/LumiData/get_potato_chips_new 20 5
run_one /home/hyx/LumiData/get_mineral_water_new 20 5
run_one /home/hyx/LumiData/put_orange_juice_new 5 0
run_one /home/hyx/LumiData/put_potato_chips_new 5 0
run_one /home/hyx/LumiData/put_mineral_water_new 5 0
run_one /home/hyx/LumiData/get_coca_cola_bottle 20 5
run_one /home/hyx/LumiData/get_coca_cola_can 20 5
run_one /home/hyx/LumiData/get_coconut_drink 20 5
run_one /home/hyx/LumiData/get_sprite_can 20 5
run_one /home/hyx/LumiData/get_biscuits 20 5
run_one /home/hyx/LumiData/get_grapefruit_drink 20 5
run_one /home/hyx/LumiData/get_oreo_rolls 20 5
run_one /home/hyx/LumiData/put_coca_cola_bottle 5 0
run_one /home/hyx/LumiData/put_coca_cola_can 5 0
run_one /home/hyx/LumiData/put_coconut_drink 5 0
run_one /home/hyx/LumiData/put_sprite_can 5 0
run_one /home/hyx/LumiData/put_biscuits 5 0
run_one /home/hyx/LumiData/put_grapefruit_drink 5 0
run_one /home/hyx/LumiData/put_oreo_rolls 5 0
run_one /home/hyx/LumiData/get2_biscuits 20 5
run_one /home/hyx/LumiData/get2_coca_cola_can 20 5
run_one /home/hyx/LumiData/get2_coconut_drink 20 5
run_one /home/hyx/LumiData/get2_grapefruit_drink 20 5
run_one /home/hyx/LumiData/get2_mineral_water_new 20 5
run_one /home/hyx/LumiData/get2_orange_juice_new 20 5
run_one /home/hyx/LumiData/get2_oreo_rolls 20 5
run_one /home/hyx/LumiData/get2_potato_chips_new 20 5
run_one /home/hyx/LumiData/get2_sprite_can 20 5
run_one /home/hyx/LumiData/go_to_plate_on_table 25 10
