CUDA_VISIBLE_DEVICES=3 python deploy.py config=configs/show-vla_future_action-256x256.yaml \
                        model_path=/home/hyx/ShowVLA/show-o2/show-vla_future_act-256x256/checkpoint-final/unwrapped_model/pytorch_model.bin \
                        output_dir=./logs \
                        device=cuda \
                        port=8000 \
                        host=0.0.0.0 \