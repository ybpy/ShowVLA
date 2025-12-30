conda create -n show python=3.10
conda activate show

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://mirror.nju.edu.cn/pytorch/whl/cu126

pip install transformers==4.51.3;
pip install diffusers==0.31.0;
pip install einops==0.8.1;
pip install decord==0.6.0;
pip install gpustat==1.1.1;
pip install sentencepiece==0.2.0;
pip install ipdb==0.13.13;
pip install ftfy==6.3.1 regex tqdm==4.67.1;
pip install git+https://github.com/openai/CLIP.git;
pip install onnx==1.17.0;
pip install onnxsim==0.4.36;
pip install omegaconf==2.3.0;
pip install torchdiffeq==0.2.5;
pip install segment_anything==1.0;
pip install wandb==0.19.7;

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation;

pip install deepspeed==0.17.1;
pip install accelerate==1.12.0;
pip install timm==1.0.15;
pip install huggingface-hub==0.36.0;
pip install onnxruntime==1.20.1;
pip install dill==0.3.8;
pip install pandas==2.2.3;
pip install pyarrow==20.0.0;
pip install av==15.0.0;
pip install moviepy==1.0.3;
pip install jsonlines==4.0.0;

pip install fastapi
pip install peft==0.17.1
pip install uvicorn==0.34.3
pip install json_numpy==2.1.0
pip install safetensors==0.5.3
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
pip install scipy==1.15.3
pip install mmengine==0.10.5
pip install h5py==3.13.0
pip install mediapy==1.2.4

pip install bitsandbytes
pip install lion_pytorch


pip install hydra-core==1.2.0
pip install easydict==1.13
pip install robomimic==0.3.0
pip install thop==0.1.1.post2209072238
pip install robosuite==1.4.1
pip install bddl==1.0.1
pip install future==0.18.2
pip install cloudpickle==3.1.1
pip install gym==0.26.2
