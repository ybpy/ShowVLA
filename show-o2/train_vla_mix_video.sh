set -euo pipefail

export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix_video.py \
    config=configs/showvla-moe_mix_video-336x320.yaml

mkdir showvla-video_act_Vis_jaka-calvin-bridge-robomind-droid_lvis
cd showvla-video_act_Vis_jaka-calvin-bridge-robomind-droid_lvis
ln -s ../showvla-mix_video_Vis_jaka-calvin-bridge-robomind-droid_lvis/checkpoint-20000 checkpoint-0
cd ..

# accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
#     train_vla.py \
#     config=configs/showvla-moe_video_action-336x320_warmup.yaml

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_video_action-336x320.yaml

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_video_action-336x320_cont.yaml
