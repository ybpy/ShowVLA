#!/bin/bash
# Convert AgiBot Manipulation-RealRobot subtask clips to ShowVLA HDF5.
# Defaults are RealRobot-tuned (see convert_agibot_real_hdf5.py).

cd "$(dirname "$0")"

python convert_agibot_real_hdf5.py \
    --data_root /datasets3/agibot_world_challenge_2025/Manipulation-RealRobot/ \
    --output_dir /home/hyx/datasets/AGIBOT-Real \
    --metainfo ./AGIBOT-HDF5-Real_metainfo.json \
    --dataset_name AGIBOT-HDF5-Real \
    --max_frames 400 \
    --max_downsample_rate 3.5 \
    --min_downsample_rate 1.5 \
    --main_max_adjacent_diff 15.5 \
    --wrist_max_adjacent_diff 36.0
