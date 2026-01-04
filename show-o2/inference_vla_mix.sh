CUDA_VISIBLE_DEVICES=5 python inference_vla_mix.py config=configs/showvla-moe_mix-336x320.yaml \
                        model_path="showvla-mix/checkpoint-16000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=50