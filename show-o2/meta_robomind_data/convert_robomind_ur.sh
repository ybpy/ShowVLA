#!/bin/bash
# Convert RoboMIND benchmark1_1 h5_ur_1rgb to ShowVLA HDF5 format.

cd "$(dirname "$0")"

python convert_robomind_ur.py \
    --data_dir /home/hyx/RoboMIND/benchmark1_1/h5_ur_1rgb \
    --output_dir /home/hyx/datasets/RoboMIND/benchmark1_1/h5_ur_1rgb \
    --annotation_json /home/hyx/RoboMIND/static/language_description_annotation_json/h5_ur_1rgb.json \
    --meta_prefix robomind-ur \
    --dataset_name robomind-ur \
    --overwrite
