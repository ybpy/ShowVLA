export OMP_NUM_THREADS=8

# mkdir showvla-video_act_weighted_grounding_jaka0529
# cd showvla-video_act_weighted_grounding_jaka0529
# ln -s ../showvla-mix_weighted_grounding_jaka_video0529/checkpoint-25000 checkpoint-0
# cd ../

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_video_action-336x320.yaml

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_video_action-336x320_cont.yaml
