export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix.py \
    config=configs/show-vla-moe_mix-336x320.yaml