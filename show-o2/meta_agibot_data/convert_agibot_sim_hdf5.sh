#!/bin/bash
# Convert AgiBot Manipulation-SimData subtask clips to ShowVLA HDF5.

cd "$(dirname "$0")"

python convert_agibot_sim_hdf5.py \
    --task_dir /datasets3/agibot_world_challenge_2025/Manipulation-SimData/ \
    --output_dir /home/hyx/datasets/AGIBOT-Sim \
    --metainfo ./AGIBOT-HDF5-Sim_metainfo.json \
    --dataset_name AGIBOT-HDF5-Sim \
    --max_frames 300 \
    --max_downsample_rate 3.5 \
    --min_downsample_rate 2.0
