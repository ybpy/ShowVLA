CUDA_VISIBLE_DEVICES=1 python inference_vla_clear_video.py config=configs/showvla-moe_mix_video-336x320.yaml \
                        model_path="showvla-mix_Vis_jaka-calvin-bridge-robomind-droid_lvis/checkpoint-20000/unwrapped_model/pytorch_model.bin" \
                        guidance_scale=0.0 \
                        num_inference_steps=10

# CUDA_VISIBLE_DEVICES=0 python inference_vla_clear.py config=configs/showvla-moe_future_action-336x320.yaml \
#                         model_path="showvla-mix_weighted_grounding_jaka0430/checkpoint-25000/unwrapped_model/pytorch_model.bin" \
#                         guidance_scale=0.0 \
#                         num_inference_steps=10