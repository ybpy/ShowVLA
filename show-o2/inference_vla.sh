CUDA_VISIBLE_DEVICES=3 python inference_vla.py config=configs/show-vla_future_only-256x256.yaml \
                        model_path=show-vla_future_only-256x256_lr/checkpoint-15000/unwrapped_model/pytorch_model.bin \
                        batch_size=1 \
                        guidance_scale=0.0 \
                        num_inference_steps=50