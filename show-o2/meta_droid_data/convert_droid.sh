#!/bin/bash
# Convert DROID RLDS dataset to ShowVLA HDF5 format.
# Requires: pip install tfds-nightly 'tensorflow[and-cuda]' mediapy h5py pillow
# If protobuf version error occurs: pip install 'protobuf>=6.31.1,<7'

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=6 python convert_droid.py \
    --input_dir /datasets3/droid/1.0.1 \
    --keep_ranges_path /datasets3/droid/KarIP/droid/keep_ranges_1_0_1.json \
    --left_dir /home/hyx/datasets/Droid-Left \
    --right_dir /home/hyx/datasets/Droid-Right \
    --left_metainfo ./Droid-Left_metainfo.json \
    --right_metainfo ./Droid-Right_metainfo.json \
    --split train
