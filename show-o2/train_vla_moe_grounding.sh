export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_grounding.py \
    config=configs/show-vla-moe_grounding-256x256.yaml