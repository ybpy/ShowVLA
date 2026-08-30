CUDA_VISIBLE_DEVICES=7 python deploy.py config=configs/showvla-moe_future_action-336x320.yaml \
                        model_path=showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0830/checkpoint-25000/unwrapped_model/pytorch_model.bin \
                        output_dir=./25000 \
                        device=cuda \
                        port=8925 \
                        host=0.0.0.0 \