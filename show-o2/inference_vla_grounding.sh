CUDA_VISIBLE_DEVICES=5 python inference_vla_grounding.py config=configs/show-vla-moe_grounding-336x320.yaml \
                        model_path="showvla-moe_grounding_fix/checkpoint-10000/unwrapped_model/pytorch_model.bin" \
                        batch_size=1 \
                        guidance_scale=0.0 \
                        num_inference_steps=50