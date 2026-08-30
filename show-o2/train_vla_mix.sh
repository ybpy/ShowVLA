set -euo pipefail
# export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix_video.py \
    config=configs/showvla-moe_mix-336x320.yaml

mkdir showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825
cd showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825
ln -s ../showvla-mix_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825/checkpoint-25000 checkpoint-0
cd ..

# accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
#     train_vla.py \
#     config=configs/showvla-moe_future_action-336x320_warmup.yaml

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_future_action-336x320.yaml

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_future_action-336x320_cont.yaml
