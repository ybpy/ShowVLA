# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
from tqdm import tqdm
import logging
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, \
    path_to_llm_name, load_state_dict, load_xvla_modules, replace_model_parameters, remove_trailing_digits

from omegaconf import OmegaConf
from transformers import Qwen2MoeConfig
from peft import LoraConfig, get_peft_model

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import numpy as np
from einops import rearrange
from datasets_vla import create_dataloader

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport

import mediapy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name,
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weight_type = get_weight_type(config)

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    use_img_trans_field = config.model.showo.use_img_trans_field if 'use_img_trans_field' in config.model.showo else False
    pred_act = config.model.showo.pred_act if 'pred_act' in config.model.showo else False 
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                                                         add_return_act_token_ids=pred_act)
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    print(config.model.showo)
    model = Showo2Qwen2_5(**config.model.showo).to(device)
    # Drop-upcycling if needed
    if config.model.showo.drop_upcycling:
        logger.info("Dropping upcycling modules...")
        # Create MoE config from yaml settings
        config.model.showo.moe_config.vocab_size = config.model.showo.llm_vocab_size
        moe_config_dict = OmegaConf.to_container(config.model.showo.moe_config, resolve=True)
        target_config = Qwen2MoeConfig(**moe_config_dict)
        model.showo = replace_model_parameters(
            logger=logger,
            source_model=model.showo,
            target_config=target_config,
            num_experts=config.model.showo.moe_config.num_experts,
            num_layers=config.model.showo.moe_config.num_hidden_layers,
            seed=config.training.seed,
            init_method=config.model.showo.init_method,
            ffn_init_ratio=config.model.showo.ffn_init_ratio,
        ).to(device)
        logger.info("Drop-upcycling completed. Model converted to MoE architecture.")
    
    # Load XVLA action modules
    xvla_checkpoint = config.model.showo.get('xvla_ckpt_path', None)
    if xvla_checkpoint is not None and config.model.showo.xvla_hidden_size is not None:
        logger.info("Loading XVLA action modules...")
        xvla_layers_to_load = config.model.showo.get('xvla_layers_to_load', [22, 23])
        assert len(xvla_layers_to_load) == model.xvla_depth
        success = load_xvla_modules(
            logger,
            model, 
            xvla_checkpoint,
            module_names=config.model.showo.get('xvla_modules_to_load', 
                ['action_encoder', 'action_decoder', 'norm', 'pos_emb', 'soft_prompt_hub', 'blocks']),
            layer_prefix=config.model.showo.get('xvla_layer_prefix', 'blocks'),
            layer_indices=xvla_layers_to_load,
            source_prefix=config.model.showo.get('source_prefix', 'transformer'),
            target_prefix=config.model.showo.get('target_prefix', None),
        )
        if not success:
            logger.error("Failed to load XVLA modules! Please check:")
        else:
            logger.info("XVLA action modules loaded successfully!")

    use_lora = config.training.get('use_lora', False)
    lr_multipler = config.training.get('lr_multipler', 1.0)
    if use_lora:
        exclude_modules = ["time_embed"]
        suffix_of_modules_to_save = [
            "mlp.gate",
            # "mlp.experts",
            "lm_head",
            "image_embedder_und",
            "image_embedder_gen",
            "position_embedding",
            # "fusion_proj",
            # "time_embed",
            "diff_proj",
            "time_embed_proj",
            "diffusion_head_b",
        ]
        modules_to_save = ["norm"]
        if config.model.showo.xvla_hidden_size is not None:
            modules_to_save = [
                "project_xvla_encode",
                "project_xvla_decode",
                "pos_emb",
                "norm",
                "action_encoder",
                "action_decoder",
                "soft_prompt_hub",
            ]
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) or isinstance(module, torch.nn.Sequential):
                continue
            if any((name.endswith(x) or remove_trailing_digits(name).endswith(x)) for x in suffix_of_modules_to_save): 
                modules_to_save.append(name)
        for name in modules_to_save:
            logger.info(f"[modules_to_save] {name}")
        
        lora_config = LoraConfig(
            lora_alpha=48,
            r=24,
            bias="none",
            target_modules="all-linear",
            exclude_modules=exclude_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    use_compile = config.training.get('use_compile', True)
    compile_mode = config.training.get('compile_mode', "default")
    if use_compile:
        try:
            if hasattr(torch, "compile"):
                compile_kwargs = {"mode": compile_mode}
                model = torch.compile(model, **compile_kwargs)
                logger.info(f"Enabled torch.compile with mode={compile_mode}")
            else:
                logger.warning("torch.compile is unavailable in the installed torch version.")
        except Exception as exc:
            logger.warning(f"Failed to enable torch.compile: {exc}. Continuing without compilation.")
            use_compile = False
    

    """ Loading Model Checkpoint """
    if config.model_path:
        state_dict = torch.load(config.model_path, map_location="cpu")
        # Unwrap model manually to match the state_dict structure
        unwrapped_model = model
        while hasattr(unwrapped_model, "_orig_mod"):
            unwrapped_model = unwrapped_model._orig_mod
        if hasattr(unwrapped_model, "base_model"):
            unwrapped_model = unwrapped_model.base_model.model
        unwrapped_model.load_state_dict(state_dict, strict=True)
        del state_dict
    """ Merge Lora """
    if use_lora:
        model = model.merge_and_unload()


    model.to(weight_type)
    model.eval()

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    ori_num_vla_image_tokens = config.dataset.preprocessing.num_vla_image_tokens

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_vla_image_tokens += 1
    

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, image_pad_id, video_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    # load users passed arguments
    guidance_scale = config.guidance_scale
    config.transport.num_inference_steps = config.num_inference_steps
    assert guidance_scale == 0.0

    # Iterable dataloader
    random_query_duration = config.xvla.random_query_duration if 'random_query_duration' in config.xvla else False
    num_future_imgs = config.xvla.num_future_imgs if 'num_future_imgs' in config.xvla else 1
    assert num_future_imgs == 4
    mixed_loader = create_dataloader(
        num_workers=dataset_config.num_workers,
        batch_size=config.training.batch_size_vla,
        metas_path=config.training.train_metas_path,
        num_actions=config.xvla.num_actions,
        action_mode=config.xvla.action_mode,
        training=True,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=preproc_config.max_vla_seq_len,
        image_size=preproc_config.vla_image_size,
        num_image_tokens=preproc_config.num_vla_image_tokens,
        pred_act=pred_act,
        random_query_duration=random_query_duration,
        num_future_imgs=num_future_imgs,
    )

    dtype = weight_type

    @torch.no_grad()
    def prepare_video_latents(pixel_values, num_obs_img=1):
        # b, n, pixel_c, pixel_h, pixel_w = pixel_values.shape
        if config.model.vae_model.type == 'wan21':
            # (b, 5, 3, h, w)
            pixel_values = rearrange(pixel_values, "b n c h w -> b c n h w")    # (b, 3, 5, h, w)
            image_latents = vae_model.sample(pixel_values)                      # (b, 16, 2, h/8, w/8)
            image_latents = image_latents.transpose(1, 2)                       # (b, 2, 16, h/8, w/8)
            b, n, c, h, w = image_latents.shape
        else:
            raise NotImplementedError

        t = torch.ones(b, n, dtype=dtype, device=device)
        t[:, -1] = 0.0
        t = t.reshape(-1)

        xt_list = []
        for i in range(b):
            for j in range(n):
                is_obs_img = j < num_obs_img
                # x0: src or noise, x1: tgt
                xt = image_latents[i][0]
                if not use_img_trans_field and not is_obs_img:
                    xt = torch.randn_like(xt)
                
                xt_list.append(xt)

        xt = torch.cat(xt_list, dim=0)
        xt = xt.reshape(b * n, c, h, w)
        return xt, t
    
    batch_idx = 0
    sample_idx = 0
    for batch in mixed_loader:
        print(f"\nbatch_idx: {batch_idx}")

        texts = batch['language_instruction']
        text_tokens = batch['text_tokens'].to(device)
        # text_labels = batch['text_labels'].to(device)
        # b n c h w
        pixel_values = batch['images'].to(device).to(weight_type)

        text_masks = batch['text_masks'].to(device)
        image_masks = batch['image_masks'].to(device)
        modality_positions = batch['modality_positions'].to(device)

        for text, text_tokens, pixel_values, text_masks, image_masks, modality_positions in zip(
            texts, torch.split(text_tokens, 1), torch.split(pixel_values, 1), torch.split(text_masks, 1), torch.split(image_masks, 1), torch.split(modality_positions, 1),
        ):
            assert text_tokens.size(0) == 1
            print(f"\nsample_idx: {sample_idx}")
            
            image_latents, t_img = prepare_video_latents(pixel_values)
            
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                                text_tokens.size(1),
                                                modality_positions,
                                                device).to(weight_type)

            steps = config.num_inference_steps
            dt = 1.0 / steps
            with torch.no_grad():
                for _ in range(steps):
                    _, v_pred_ = model.forward(text_tokens=text_tokens,
                                            image_latents=image_latents,
                                            t=t_img,
                                            attention_mask=block_mask,
                                            modality_positions=modality_positions,
                                            max_seq_len=max_seq_len,
                                            device=device,
                                            )
                    # Update image_latents and t_img
                    image_latents[1::2] = image_latents[1::2] + v_pred_[1::2] * dt
                    t_img[1::2] = (t_img[1::2] + dt).clamp(0, 1)

            # 2, 16, h/8, w/8
            samples = image_latents

            if config.model.vae_model.type == 'wan21':
                samples = rearrange(samples, "n c h w -> 1 c n h w")    # 1, 16, 2, h/8, w/8
                images = vae_model.batch_decode(samples)
                images = rearrange(images, "1 c n h w -> n c h w")      # 5, 3, h, w
            else:
                raise NotImplementedError
            
            pred_images = images
            pred_images = denorm(pred_images)   # 5, h, w, 3

            gt_images = pixel_values[0]
            gt_images = denorm(gt_images)       # 5, h, w, 3

            mediapy.write_video(f"demo_{sample_idx}pred_{text}.mp4", pred_images, fps=1)
            mediapy.write_video(f"demo_{sample_idx}gt_{text}.mp4", gt_images, fps=1)
            
            sample_idx += 1
        
        batch_idx += 1