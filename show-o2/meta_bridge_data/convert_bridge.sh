#!/bin/bash
# Convert Bridge RLDS dataset to ShowVLA HDF5 format.
# Requires: pip install tfds-nightly 'tensorflow[and-cuda]'
# If protobuf version error occurs: pip install 'protobuf>=6.31.1'

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=2 python convert_bridge.py \
    --input_dir /datasets2/hyx_data/bridge_orig/1.0.0 \
    --train_dir /home/hyx/datasets/Bridge/Train \
    --val_dir /home/hyx/datasets/Bridge/Val \
    --dataset_name Bridge \
    --split val \
