CUDA_VISIBLE_DEVICES=0 python inference_vla_clear.py config=configs/showvla-moe_future_action-336x320.yaml \
                        model_path="showvla-future_act_seed/checkpoint-14000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=50