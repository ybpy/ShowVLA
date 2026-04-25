CUDA_VISIBLE_DEVICES=2 python inference_vla_grounding.py config=configs/showvla-moe_mix-336x320.yaml \
                        model_path="showvla-mix_weighted_grounding0421/checkpoint-16000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=50