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
from torch.utils.data import DataLoader
from datasets_vla import COCODataset, MixedDataLoader
from datasets_vla import create_dataloader

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
        success = load_xvla_modules(
            logger,
            model, 
            xvla_checkpoint,
            module_names=config.model.showo.get('xvla_modules_to_load', 
                ['action_encoder', 'action_decoder', 'norm', 'pos_emb', 'soft_prompt_hub']),
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
    
    def create_grounding_dataloader(dataset, batch_size, collate_fn):
        sampler = None
        shuffle = True
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                                  sampler=sampler, collate_fn=collate_fn,
                                                  shuffle=shuffle, num_workers=dataset_config.num_workers,
                                                  drop_last=True)
        return dataloader

    dataset = COCODataset(
        metas_path=config.training.coco_metas_path,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=preproc_config.max_vla_seq_len,
        image_size=preproc_config.vla_image_size,
        num_image_tokens=preproc_config.num_vla_image_tokens,
    )
    train_dataloader_grounding = create_grounding_dataloader(dataset,
                                                     config.training.batch_size_grounding,
                                                     dataset.collate_fn)
    
    # X-VLA dataloader
    xvla_loader = create_dataloader(
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
    )

    # Combine these dataloaders into a single iterable
    mixed_loader = MixedDataLoader(
        loader_list=[train_dataloader_grounding, xvla_loader],
        samp_probs=config.dataset.samp_probs,
        accumulation=config.dataset.accumulation,
        mode=config.dataset.mixed_loader_mode
    )

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=ori_num_vla_image_tokens,
    )  # default: velocity;

    sampler = Sampler(transport)

    @torch.no_grad()
    def prepare_latents_and_labels(
            pixel_values,
            image_masks,
            modality_positions,
            num_obs_img=1,
    ):
        b, n, pixel_c, pixel_h, pixel_w = pixel_values.shape
        if config.model.vae_model.type == 'wan21':
            # (b, n, 3, 256, 256)
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            pixel_values = pixel_values.unsqueeze(2)    # b*n c 1 h w
            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)
            image_latents = image_latents.squeeze(2)    # (b*n latent_c latent_h latent_w) == (b*n, 16, 32, 32)
            recons_images = recons_images.squeeze(2)    # (b*n, 3, 256, 256)
            _, c, h, w = image_latents.shape
            image_latents = image_latents.reshape(b, n, c, h, w)
            recons_images = recons_images.reshape(b, n, pixel_c, pixel_h, pixel_w)
        else:
            raise NotImplementedError

        t_list, xt_list, ut_list = [], [], []
        masks = image_masks # b l
        for i in range(b):
            for j in range(n):
                is_obs_img = j < num_obs_img
                # x0->noise x1->image
                t, x0, x1 = transport.sample(image_latents[i][j][None],
                                            config.training.und_max_t0 if is_obs_img else None)
                # timesteps, noised image, velocity
                t, xt, ut = transport.path_sampler.plan(t, x0, x1)
                t_list.append(t)
                xt_list.append(xt)
                ut_list.append(ut)
                if is_obs_img:
                    assert j == 0, f"Only the first image is observation"
                    assert t == 1.0, f"The observation image should not be noisy"
                    # Do not calcuate the generation loss for the observation image
                    img_sid, length = modality_positions[i, j]
                    masks[i, img_sid: img_sid + length] = 0

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)

        ut = ut.reshape(b * n, c, h, w)
        xt = xt.reshape(b * n, c, h, w)
        t = t.reshape(b * n)

        return xt, t, ut, recons_images, masks

    @torch.no_grad()
    def prepare_latents(pixel_values, num_obs_img=1):
        b, n, pixel_c, pixel_h, pixel_w = pixel_values.shape
        if config.model.vae_model.type == 'wan21':
            # (b, n, 3, 256, 256)
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            pixel_values = pixel_values.unsqueeze(2)    # b*n c 1 h w
            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)
            image_latents = image_latents.squeeze(2)    # (b*n latent_c latent_h latent_w) == (b*n, 16, 32, 32)
            recons_images = recons_images.squeeze(2)    # (b*n, 3, 256, 256)
            _, c, h, w = image_latents.shape
            image_latents = image_latents.reshape(b, n, c, h, w)
            recons_images = recons_images.reshape(b, n, pixel_c, pixel_h, pixel_w)
        else:
            raise NotImplementedError

        xt_list = []
        for i in range(b):
            for j in range(n):
                is_obs_img = j < num_obs_img
                # x0->noise x1->image
                x1 = image_latents[i][j]
                
                xt = x1 if is_obs_img else torch.randn_like(x1)
                
                xt_list.append(xt)

        xt = torch.cat(xt_list, dim=0)
        xt = xt.reshape(b * n, c, h, w)
        return xt

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
            
            # prepare image latents and labels
            # image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(pixel_values,
            #                                                                                         image_masks,
            #                                                                                         modality_positions)
            image_latents = prepare_latents(pixel_values)
            
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                                text_tokens.size(1),
                                                modality_positions,
                                                device).to(weight_type)

            # z = torch.randn((len(prompts),
            #                  image_latent_dim, latent_height * patch_size,
            #                  latent_width * patch_size)).to(torch.bfloat16).to(device)

            z = image_latents

            model_kwargs = dict(
                text_tokens=text_tokens,
                attention_mask=block_mask,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
                guidance_scale=guidance_scale,
                only_denoise_last_image=True
            )

            sample_fn = sampler.sample_ode(
                sampling_method=config.transport.sampling_method,
                num_steps=config.transport.num_inference_steps,
                atol=config.transport.atol,
                rtol=config.transport.rtol,
                reverse=config.transport.reverse,
                time_shifting_factor=config.transport.time_shifting_factor
            )
            samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]

            if config.model.vae_model.type == 'wan21':
                samples = samples.unsqueeze(2)
                images = vae_model.batch_decode(samples)
                images = images.squeeze(2)
            else:
                raise NotImplementedError
            
            future_images = images[1:]
            future_images = denorm(future_images)

            obs_images = pixel_values[:, 0]
            obs_images = denorm(obs_images)

            gt_images = pixel_values[:, 1]
            gt_images = denorm(gt_images)

            for i, (obs_img, future_img, gt_img) in enumerate(zip(obs_images, future_images, gt_images)):
                combine_img = np.concatenate([obs_img, future_img, gt_img], axis=1)
                combine_img = Image.fromarray(combine_img)
                combine_img.save(f"demo{sample_idx}_{text}.png")

                # obs_img = Image.fromarray(obs_img)
                # future_img = Image.fromarray(future_img)
                # gt_img = Image.fromarray(gt_img)
                # obs_img.save(f"demo{batch_idx}_{text}_obs.png")
                # future_img.save(f"demo{batch_idx}_{text}_future.png")
                # gt_img.save(f"demo{batch_idx}_{text}_gt.png")
            
            sample_idx += 1
        
        batch_idx += 1