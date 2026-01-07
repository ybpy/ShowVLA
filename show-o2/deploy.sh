CUDA_VISIBLE_DEVICES=4 python deploy.py config=configs/showvla-moe_future_action-336x320.yaml \
                        model_path=showvla-future_act_seed/checkpoint-12000/unwrapped_model/pytorch_model.bin \
                        output_dir=./logs \
                        device=cuda \
                        port=8989 \
                        host=0.0.0.0 \