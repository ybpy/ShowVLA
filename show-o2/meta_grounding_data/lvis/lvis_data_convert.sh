#!/bin/bash
# Preprocess LVIS v1 dataset annotations into per-image JSON files.
#
# LVIS images are shared with COCO. LVIS V1.0 splits do not match COCO splits
# one-to-one, so image paths are resolved per-sample from coco_url.
# --coco_img_root should point to the COCO root containing train2017/ and val2017/.
#
# Usage:
#   bash lvis_data_convert.sh
#
# Adjust --data_root and --coco_img_root to match your local paths.

LVIS_DATA_ROOT=/home/hyx/datasets/LVIS
COCO_IMG_ROOT=/home/hyx/datasets/coco

python lvis_data_convert.py \
    --data_root ${LVIS_DATA_ROOT} \
    --coco_img_root ${COCO_IMG_ROOT} \
    --split_name train

# python lvis_data_convert.py \
#     --data_root ${LVIS_DATA_ROOT} \
#     --coco_img_root ${COCO_IMG_ROOT} \
#     --split_name val
