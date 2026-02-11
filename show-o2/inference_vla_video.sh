CUDA_VISIBLE_DEVICES=0 python inference_vla_clear_video.py config=configs/showvla-moe_future_action-336x320.yaml \
                        model_path="showvla-future_act_weighted/checkpoint-16000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=10