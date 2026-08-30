CUDA_VISIBLE_DEVICES=3 python deploy.py config=configs/showvla-moe_video_action-336x320.yaml \
                        model_path=showvla-video_act_weighted_grounding_jaka0529/checkpoint-20000/unwrapped_model/pytorch_model.bin \
                        output_dir=./logs-20000 \
                        device=cuda \
                        port=8920 \
                        host=0.0.0.0 \