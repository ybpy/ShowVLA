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
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type, interpolate_pos_encoding
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, \
    path_to_llm_name, load_state_dict, load_xvla_modules, replace_model_parameters, remove_trailing_digits

from omegaconf import OmegaConf
from transformers import Qwen2MoeConfig
from peft import LoraConfig, get_peft_model

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, IterableDataset
from datasets_vla import MixedDataLoader, VQAGroundingDataset, VQARobotGroundingDataset
from datasets_vla import create_dataloader
import cv2

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport

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
    pred_mobile_act = config.model.showo.get('pred_mobile_act', False)
    if pred_mobile_act:
        assert pred_act, "pred_mobile_act=True requires pred_act=True"
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                                                         add_return_act_token_ids=True)
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
            if pred_mobile_act:
                modules_to_save.append("mobile_norm")
                modules_to_save.append("mobile_decoder")
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

    def create_grounding_dataloader(dataset, batch_size, collate_fn):
        if isinstance(dataset, IterableDataset):
            # IterableDataset does not support Sampler or shuffle
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    num_workers=dataset_config.num_workers,
                                    shuffle=False,
                                    drop_last=True,
                                    pin_memory=True,
                                    persistent_workers=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                            sampler=None, collate_fn=collate_fn,
                                            shuffle=False, num_workers=dataset_config.num_workers,
                                            drop_last=True,
                                            pin_memory=True,
                                            persistent_workers=True)
        return dataloader

    loader_list = []
    if config.training.grounding_metas_path:
        vqa_grounding_dataset = VQAGroundingDataset(
            metas_path=config.training.grounding_metas_path,
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            max_seq_len=preproc_config.max_vla_seq_len,
            image_size=preproc_config.vla_image_size,
            num_image_tokens=preproc_config.num_vla_image_tokens,
        )
        train_dataloader_vqa_grounding = create_grounding_dataloader(vqa_grounding_dataset,
                                                        config.training.batch_size_grounding,
                                                        vqa_grounding_dataset.collate_fn)
        loader_list.append(train_dataloader_vqa_grounding)
    
    if config.training.get('robot_grounding_metas_path', None):
        vqa_grounding_robot_dataset = VQARobotGroundingDataset(
            meta_paths=config.training.robot_grounding_metas_path,
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            max_seq_len=preproc_config.max_vla_seq_len,
            image_size=preproc_config.vla_image_size,
            num_image_tokens=preproc_config.num_vla_image_tokens,
            num_samples_per_video=config.training.get('robot_grounding_num_samples_per_video', 4),
        )
        # 为 IterableDataset 设置分布式信息
        vqa_grounding_robot_dataset.set_process_info()
        train_dataloader_vqa_robot_grounding = create_grounding_dataloader(vqa_grounding_robot_dataset,
                                                                config.training.batch_size_robot_grounding,
                                                                vqa_grounding_robot_dataset.collate_fn)
        loader_list.append(train_dataloader_vqa_robot_grounding)

    assert len(loader_list) > 0

    mixed_loader = MixedDataLoader(
        loader_list=loader_list,
        mode=config.dataset.mixed_loader_mode
    )


    dtype = weight_type
    p = model.config.patch_size

    @torch.no_grad()
    def prepare_image_latents(pixel_values, num_obs_img=1):
        b, n, pixel_c, pixel_h, pixel_w = pixel_values.shape
        if config.model.vae_model.type == 'wan21':
            # (b, n, 3, 256, 256)
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            pixel_values = pixel_values.unsqueeze(2)    # b*n c 1 h w
            image_latents = vae_model.sample(pixel_values)
            image_latents = image_latents.squeeze(2)    # (b*n latent_c latent_h latent_w) == (b*n, 16, 32, 32)
            _, c, h, w = image_latents.shape
            image_latents = image_latents.reshape(b, n, c, h, w)
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
        modality_positions_batch = batch['modality_positions']

        for text, text_tokens, pixel_values, text_masks, image_masks, mp in zip(
            texts, torch.split(text_tokens, 1), torch.split(pixel_values, 1), torch.split(text_masks, 1), torch.split(image_masks, 1), modality_positions_batch,
        ):
            assert text_tokens.size(0) == 1
            print(f"\nsample_idx: {sample_idx}")
            modality_positions = [mp]

            h, w = pixel_values.shape[-2:]
            h_, w_ = h // 8 // p, w // 8 // p
            image_latents = vae_model.sample(pixel_values[:, 0].unsqueeze(2)).squeeze(2).to(weight_type)
            image_embeds_und = model.image_embedder_und(image_latents)
            image_embeds_gen = model.image_embedder_gen(image_latents)
            pos_encoding = interpolate_pos_encoding(
                model.config.clip_latent_dim,
                model.position_embedding,
                h_,
                w_,
                1,
            )
            image_embeds_und = image_embeds_und + pos_encoding
            image_embeds_und = model.und_trans(image_embeds_und)['last_hidden_state']
            image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))


            prefix_tokens = torch.tensor([showo_token_ids['bos_id'], showo_token_ids['boi_id']]).to(device)[None, :]
            prefix_embeds = model.showo.model.embed_tokens(prefix_tokens)

            print(text)
            question, gt_answer = text.split('\n')
            input_ids = text_tokenizer(question+'\n', add_special_tokens=False).input_ids
            prompt_tokens = torch.tensor([showo_token_ids['eoi_id']] + input_ids).to(device)[None, :]
            prompt_embeds = model.showo.model.embed_tokens(prompt_tokens)

            if config.model.showo.add_time_embeds:
                time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), prompt_embeds.dtype)
                if hasattr(model, 'time_embed_proj'):
                    time_embeds = model.time_embed_proj(time_embeds)
                input_embeds = torch.cat([
                    prefix_embeds,
                    time_embeds,
                    image_embeds,
                    prompt_embeds,
                ], dim=1).to(weight_type)
                modality_positions = [[(2, preproc_config.num_vla_image_tokens)]]
            else:
                input_embeds = torch.cat([
                    prefix_embeds,
                    image_embeds,
                    prompt_embeds
                ], dim=1).to(weight_type)
                modality_positions = [[(2, preproc_config.num_vla_image_tokens)]]

            attention_mask = omni_attn_mask_naive(
                B=input_embeds.size(0),
                LEN=input_embeds.size(1),
                modalities=modality_positions,
                device=device, inverted=True
            ).to(input_embeds.dtype)
            
            output_tokens = model.mmu_generate(input_embeds=input_embeds,
                                                attention_mask=attention_mask,
                                                top_k=1,
                                                max_new_tokens=300,
                                                eos_token=text_tokenizer.eos_token_id)

            output_tokens = torch.stack(output_tokens).squeeze()[None]

            pred_answer = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

            obs_images = pixel_values[:, 0]
            obs_img = denorm(obs_images)[0]

            try:
                pred_bboxes = [eval(x) for x in pred_answer.split(' ')]
            except:
                print(f"Invalid pred_answer: {pred_answer}")
                continue
            future_img = obs_img.copy()
            for bbox in pred_bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(future_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            gt_bboxes = [eval(x) for x in gt_answer.split(' ')]
            gt_img = obs_img.copy()
            for bbox in gt_bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            combine_img = np.concatenate([obs_img, future_img, gt_img], axis=1)
            combine_img = Image.fromarray(combine_img)
            save_name = f"demo{sample_idx}_{question}{pred_answer}"[:200]
            combine_img.save(f"{save_name}.jpg")

            # obs_img = Image.fromarray(obs_img)
            # future_img = Image.fromarray(future_img)
            # gt_img = Image.fromarray(gt_img)
            # obs_img.save(f"demo{batch_idx}_{text}_obs.png")
            # future_img.save(f"demo{batch_idx}_{text}_future.png")
            # gt_img.save(f"demo{batch_idx}_{text}_gt.png")
            
            sample_idx += 1
        
        batch_idx += 1