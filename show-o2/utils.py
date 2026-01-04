from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import random
from decord import VideoReader, cpu
from models import Qwen2MoeForCausalLM
import re
from tqdm import tqdm

##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              misc
##################################################

def _count_params(module):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in module.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def _freeze_params(model, frozen_params=None):
    if frozen_params is not None:
        for n, p in model.named_parameters():
            for name in frozen_params:
                if name in n:
                    if "showo.model.embed_tokens" or "showo.lm_head" in n:
                        print(f"Freezing {n}!!!")
                    p.requires_grad = False


path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": 'qwen2_5',
    "Qwen/Qwen2.5-1.5B-Instruct": 'qwen2_5',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama3'
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def denorm_vid(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    # B, C, T, H, W --> B, T, C, H, W
    images = images.permute(0, 2, 1, 3, 4).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=False, is_hq=False):
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    max_seq_len = config.dataset.preprocessing.max_seq_length
    num_video_tokens = config.dataset.preprocessing.num_video_tokens
    if is_video:
        max_text_len = max_seq_len - num_video_tokens - 4
        latent_width = config.dataset.preprocessing.video_latent_width
        latent_height = config.dataset.preprocessing.video_latent_height
        num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
        num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
    else:
        if is_hq:
            latent_width = config.dataset.preprocessing.hq_latent_width
            latent_height = config.dataset.preprocessing.hq_latent_height
            num_t2i_image_tokens = config.dataset.preprocessing.num_hq_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            max_seq_len = config.dataset.preprocessing.max_hq_seq_length
            max_text_len = max_seq_len - num_t2i_image_tokens - 4
        else:
            num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            latent_width = config.dataset.preprocessing.latent_width
            latent_height = config.dataset.preprocessing.latent_height
            max_text_len = max_seq_len - num_t2i_image_tokens - 4

    image_latent_dim = config.model.showo.image_latent_dim
    patch_size = config.model.showo.patch_size

    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    boi_id = showo_token_ids['boi_id']
    eoi_id = showo_token_ids['eoi_id']
    bov_id = showo_token_ids['bov_id']
    eov_id = showo_token_ids['eov_id']
    img_pad_id = showo_token_ids['img_pad_id']
    vid_pad_id = showo_token_ids['vid_pad_id']

    guidance_scale = config.transport.guidance_scale

    return num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, \
           latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, \
           vid_pad_id, guidance_scale


# these save and recover functions are based on our internal packages
# please modified them when necessary
def save_dataloader_state(rank, loader, ckpt_path="./"):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    saved_state = deepcopy(loader.__getstate__())
    torch.save(saved_state, ckpt_path)

def recover_dataloader_state(rank, loader, ckpt_path='./'):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            loader_state_dict = torch.load(f)
            loader.__setstate__(loader_state_dict)
        print(f"rank {rank} loader state dict loaded successfully!")


def save_images_as_grid(pil_images, fn, path, grid_size=(2, 2)):

    os.makedirs(path, exist_ok=True)

    rows, cols = grid_size

    num_images = len(pil_images)
    if num_images > rows * cols:
        raise ValueError(f"Number of images ({num_images}) exceeds grid capacity ({rows * cols}).")

    img_width, img_height = pil_images[0].size

    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))  # 白色背景

    for idx, image in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * img_width
        y_offset = row * img_height
        grid_image.paste(image, (x_offset, y_offset))

    grid_image.save(os.path.join(path, f"{fn}.png"))

    return grid_image


def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path)
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            state_dict.update(checkpoint)

    return state_dict

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def load_video(video_path, max_frames_num, fps, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 432, 432, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = [Image.fromarray(item) for item in vr.get_batch(frame_idx).asnumpy()]

    return spare_frames, frame_time, video_time

def load_xvla_modules(
    logger,
    model, 
    xvla_checkpoint_path, 
    module_names=['action_encoder', 'action_decoder', 'norm', 'pos_emb', 'soft_prompt_hub', 'blocks'],
    layer_prefix='blocks',
    layer_indices=[22, 23],
    source_prefix='transformer',
    target_prefix=None,
):
    """
    Load XVLA modules from local path or HuggingFace Hub.
    
    Args:
        model: Showo2Qwen2_5
        xvla_checkpoint_path: Local file path or HuggingFace model ID (e.g., "username/model-name")
        module_names: List of module names to load
        source_prefix: transformer(XVLA prefix)
        target_prefix: Showo2 prefix
    
    Returns:
        bool: success
    """
    logger.info(f"Loading XVLA modules from {xvla_checkpoint_path}")
    
    # Detect if it's a HuggingFace model ID (contains '/' and not a local path)
    is_hf_model = '/' in xvla_checkpoint_path and not os.path.exists(xvla_checkpoint_path)
    
    if is_hf_model:
        logger.info(f"Detected HuggingFace model ID: {xvla_checkpoint_path}")
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # Try to download the model files from HuggingFace Hub
        try:
            # First, try to find safetensors file
            try:
                checkpoint_file = hf_hub_download(
                    repo_id=xvla_checkpoint_path,
                    filename="model.safetensors",
                )
                logger.info(f"Downloaded model.safetensors from HuggingFace Hub")
            except:
                # Fall back to pytorch_model.bin
                try:
                    checkpoint_file = hf_hub_download(
                        repo_id=xvla_checkpoint_path,
                        filename="pytorch_model.bin",
                    )
                    logger.info(f"Downloaded pytorch_model.bin from HuggingFace Hub")
                except:
                    # Try to download entire repo and look for checkpoint files
                    cache_dir = snapshot_download(repo_id=xvla_checkpoint_path)
                    logger.info(f"Downloaded full model repo to {cache_dir}")
                    
                    # Look for checkpoint files in the downloaded directory
                    if os.path.exists(os.path.join(cache_dir, "model.safetensors")):
                        checkpoint_file = os.path.join(cache_dir, "model.safetensors")
                    elif os.path.exists(os.path.join(cache_dir, "pytorch_model.bin")):
                        checkpoint_file = os.path.join(cache_dir, "pytorch_model.bin")
                    else:
                        raise FileNotFoundError(
                            f"Could not find model.safetensors or pytorch_model.bin in {xvla_checkpoint_path}"
                        )
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace Hub: {e}")
            return False
    else:
        # Local file path
        checkpoint_file = xvla_checkpoint_path
        if not os.path.exists(checkpoint_file):
            logger.error(f"Local checkpoint file not found: {checkpoint_file}")
            return False
    
    # Load the checkpoint
    if checkpoint_file.endswith('.safetensors'):
        from safetensors.torch import load_file
        xvla_state_dict = load_file(checkpoint_file)
        logger.info(f"Loaded state_dict from safetensors file")
    else:
        xvla_state_dict = torch.load(checkpoint_file, map_location="cpu")
        logger.info(f"Loaded state_dict from pytorch file")

    src_layer_id_2_tgt_layer_id = dict()
    for i, src_layer_id in enumerate(layer_indices):
        src_layer_id_2_tgt_layer_id[src_layer_id] = i
    
    filtered_state_dict = {}
    for key, value in xvla_state_dict.items():
        if source_prefix and not key.startswith(f"{source_prefix}."):
            continue

        new_key = key[len(source_prefix) + 1:]  # +1 for the dot
        new_key_splits = new_key.split('.')

        should_load = any(module_name == new_key_splits[0] for module_name in module_names)
        if not should_load:
            continue

        if new_key_splits[0] == layer_prefix:
            layer_id = int(new_key_splits[1])
            if layer_id in layer_indices:
                tgt_layer_id = src_layer_id_2_tgt_layer_id[layer_id]
                new_key_splits[1] = str(tgt_layer_id)
                new_key = '.'.join(new_key_splits)
            else:
                continue
        
        # add target_prefix
        if target_prefix:
            new_key = f"{target_prefix}.{new_key}"
        
        filtered_state_dict[new_key] = value
        logger.info(f"Mapping: {key} -> {new_key} (shape: {list(value.shape)})")
    
    if not filtered_state_dict:
        logger.warning(f"No matching parameters found for modules: {module_names}")
        logger.warning(f"Available keys in XVLA checkpoint (first 10):")
        for i, k in enumerate(list(xvla_state_dict.keys())[:10]):
            logger.warning(f"{i+1}. {k}")
        return False
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
    logger.info(f"Successfully loaded {list(filtered_state_dict.keys())} from XVLA")
    
    return len(filtered_state_dict) > 0



def initialize_weights(logger, size, method, std=0.02, mean=0):
    if method == "torch_normal":
        return torch.normal(mean=mean, std=std, size=size)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
def shuffle_and_partially_initialize(
    logger, tensor, perm, target_size, is_down_proj, layer_idx, expert_idx, ffn_init_ratio
):
    if is_down_proj:
        original_size = tensor.size(1)
        # For down_proj (w2), shuffle columns
        shuffled = tensor.index_select(1, perm[:target_size])
    else:
        original_size = tensor.size(0)
        # For gate_proj (w1) and up_proj (w3), shuffle rows
        shuffled = tensor.index_select(0, perm[:target_size])
    init_size = int(target_size * ffn_init_ratio)
    init_indices = torch.randperm(target_size)[:init_size]
    if is_down_proj:
        init_part = shuffled[:, init_indices]
        init_mean = init_part.mean().item()
        init_std = init_part.std().item()
        init_tensor = initialize_weights(
            logger, 
            (tensor.size(0), init_size),
            "torch_normal",
            std=init_std,
            mean=init_mean,
        )
        shuffled[:, init_indices] = init_tensor
    else:
        init_part = shuffled[init_indices, :]
        init_mean = init_part.mean().item()
        init_std = init_part.std().item()
        init_tensor = initialize_weights(
            logger, 
            (init_size, tensor.size(1)),
            "torch_normal",
            std=init_std,
            mean=init_mean,
        )
        shuffled[init_indices, :] = init_tensor

    return shuffled

def initialize_gate_weights(size, method, std=0.02):
    if method == "torch_rand":
        return torch.rand(size)
    elif method == "torch_rand_mean0":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return weights - weights_mean
    elif method == "torch_normal_002":
        return torch.normal(mean=0, std=0.02, size=size)
    elif method == "torch_normal_028":
        return torch.normal(mean=0, std=0.2886751345948129, size=size)
    elif method == "torch_rand_002":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return (weights - weights_mean) * 0.02 * (12**0.5)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def replace_model_parameters(
    logger,
    source_model,
    target_config,
    num_experts,
    num_layers,
    seed,
    init_method,
    ffn_init_ratio,
):

    target_model = Qwen2MoeForCausalLM(target_config)
    ori_intermediate_size = target_config.intermediate_size
    target_intermediate_size = target_config.moe_intermediate_size
    logger.info(f"Original intermediate size: {ori_intermediate_size}")
    logger.info(f"Target intermediate size: {target_intermediate_size}")

    base_src = "model.layers.{}.mlp.experts.{}"
    base_tgt = "model.layers.{}.mlp"
    replace_mapping = {
        f"{base_src}.gate_proj.weight": f"{base_tgt}.gate_proj.weight",
        f"{base_src}.down_proj.weight": f"{base_tgt}.down_proj.weight",
        f"{base_src}.up_proj.weight": f"{base_tgt}.up_proj.weight",
    }

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    no_find_parameters = [] 
    for name, param in target_state_dict.items():
        if name not in source_state_dict:
            assert ".mlp.gate." in name or ".mlp.experts." in name
            no_find_parameters.append(name)
            continue
        target_state_dict[name] = source_state_dict[name]

    for layer_idx in range(num_layers):
        gate_weight_name = f"model.layers.{layer_idx}.mlp.gate.weight"
        if gate_weight_name in target_state_dict:
            target_state_dict[gate_weight_name] = initialize_gate_weights(
                target_state_dict[gate_weight_name].size(), init_method
            )

    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            perm = torch.randperm(ori_intermediate_size)
            for target_pattern, source_pattern in replace_mapping.items():
                target_name = target_pattern.format(layer_idx, expert_idx)
                source_name = source_pattern.format(layer_idx)
                assert target_name in target_state_dict and source_name in source_state_dict, f"{target_name} {source_name}"

                source_tensor = source_state_dict[source_name].float().cpu()

                # Determine if it's down_proj (w2) or not
                is_down_proj = "down_proj" in source_name
                # Shuffle the tensor along the intermediate dimension
                shuffled_and_init_tensor = shuffle_and_partially_initialize(
                    logger,
                    source_tensor,
                    perm,
                    target_intermediate_size,
                    is_down_proj,
                    layer_idx,
                    expert_idx,
                    ffn_init_ratio,
                )
                target_state_dict[target_name] = shuffled_and_init_tensor

    target_model.load_state_dict(target_state_dict)

    return target_model

def remove_trailing_digits(s):
    # 匹配字符串结尾的'.'+连续数字，替换为空字符串
    return re.sub(r'.\d+$', '', s)