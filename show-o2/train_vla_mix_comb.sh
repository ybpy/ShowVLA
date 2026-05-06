export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix_video_vqa.py \
    config=configs/showvla-moe_mix_combine-336x320.yaml