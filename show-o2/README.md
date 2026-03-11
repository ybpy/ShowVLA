20260109

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 89% | 87% | 59% | 69% |

20260114 fix stage1

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 92.6% | 96.0% | 98.2% | 48.4% |

20260122 weighted LIBERO data (Long: 0.1  -> 0.15)

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 91.2% | 98.8% | 97.8% | 88.2% |

20260128 +something-something-v2 on stage1, weighted LIBERO data (Long: 0.1  -> 0.15)

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 95.4% | 97.6% | 94.8% | 79.8% |

20260215 future image -> future 4 frames, weighted LIBERO data (Long: 0.1  -> 0.15)

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 91.2% | 99.4% | 97.6% | 77.2% |

20260305 weighted LIBERO data (Spatial: 0.1 -> 0.11, Long: 0.15  -> 0.18)

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 93.8% | 99.6% | 98.6% | 85% |

20260311 remove grounding, weighted LIBERO data (Spatial: 0.1 -> 0.11, Long: 0.15  -> 0.18)

| Spatial | Object | Goal | Long |
| :---: | :---: | :---: | :---: |
| 92.4% | 99.4% | 98% | 80% |

# Data
## COCO Data

Download the COCO dataset, which consists of 3 folders `annotations`, `train2017`, `val2017`.

Then run the following commands.
```bash
cd ShowVLA/show-o2
cd meta_grounding_data/coco

python coco_data_convert.py --data_root PATH/TO/COCO/DATA --split_name train
python coco_data_convert.py --data_root PATH/TO/COCO/DATA --split_name val
```
An example script is provided as `show-o2/meta_grounding_data/coco/coco_data_convert.sh`

## ADE20K Data

Download the ADEChallengeData2016 dataset.

Then run the following commands.
```bash
cd ShowVLA/show-o2
cd meta_grounding_data/ade20k

python ade20k_data_convert.py --data_root PATH/TO/ADE20K/DATA --split_name train
python ade20k_data_convert.py --data_root PATH/TO/ADE20K/DATA --split_name val
```
An example script is provided as `show-o2/meta_grounding_data/ade20k/ade20k_data_convert.sh`


## LIBERO Data

1. Regenerate data for the five tasksuites respectively using `meta_libero_data/libero_data_regen.py`. An example script for `libero_10` tasksuite is provided as `meta_libero_data/libero_data_regen.sh`.
2. Copy all the 5 output metainfo json files (e.g., `meta_libero_data/libero_xxx_metainfo.json`) to `meta_libero_data/split_all/`.
3. Copy the metainfo json files for `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` to `meta_libero_data/split_spatial_object_goal_10/`.


# Training
## Stage 1
```
cd ShowVLA/show-o2
bash train_vla_mix.sh
```

## Stage 2

Asume that the final model checkpoint of stage 1 is `showvla-mix_weighted_itf/checkpoint-16000` and the target save dir for stage 2 is `showvla-future_act_weighted_itf`.
```
cd ShowVLA/show-o2

mkdir showvla-future_act_weighted_itf
cd showvla-future_act_weighted_itf
ln -s ../showvla-mix_weighted_itf/checkpoint-16000 checkpoint-0

cd ../
bash train_vla_mix.sh
```

# Evaluation
## Serve the model
Refer to `ShowVLA/show-o2/deploy.sh`

## Testing
```
cd ShowVLA/show-o2/evaluation/libero
```
Then refer to `libero_eval.sh`.
