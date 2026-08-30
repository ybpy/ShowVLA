#!/bin/bash
# Convert RoboMIND benchmark1_1 h5_agilex_3rgb to ShowVLA HDF5 format.

cd "$(dirname "$0")"

python convert_robomind_agilex.py \
    --data_dir /home/hyx/RoboMIND/benchmark1_1/h5_agilex_3rgb \
    --output_dir /home/hyx/datasets/RoboMIND/benchmark1_1/h5_agilex_3rgb \
    --annotation_json /home/hyx/RoboMIND/static/language_description_annotation_json/h5_agilex_3rgb.json \
    --meta_prefix robomind-agilex \
    --dataset_name robomind-agilex \
    --merge_steps \
    --overwrite \
    --max_main_frame_diff 35 \
    --max_wrist_frame_diff 60
