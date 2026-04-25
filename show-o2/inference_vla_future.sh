CUDA_VISIBLE_DEVICES=2 python inference_vla_clear.py config=configs/showvla-moe_future_action-336x320.yaml \
                        model_path="showvla-future_act_weighted_grounding0421/checkpoint-16000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=10