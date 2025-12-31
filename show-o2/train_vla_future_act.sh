export OMP_NUM_THREADS=8

accelerate launch --config_file ../accelerate_configs/gpus.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_future_action-336x320.yaml