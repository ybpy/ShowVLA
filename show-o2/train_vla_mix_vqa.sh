export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix_vqa.py \
    config=configs/showvla-moe_mix_vqa-336x320.yaml