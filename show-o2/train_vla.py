# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
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
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import random
import torch
from torch.optim import AdamW
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from torch.nn.attention.flex_attention import flex_attention
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

# from datasets import create_imagetext_dataloader, MixedDataLoader, VISTDataset
from datasets_vla import create_dataloader
from utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    path_to_llm_name, _freeze_params, load_xvla_modules, replace_model_parameters

from transport import Sampler, create_transport

from transformers import Qwen2MoeConfig

logger = get_logger(__name__, log_level="INFO")

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    bs_mixed_modal = config.training.batch_size_vla

    if "concat" in config.dataset.mixed_loader_mode:
        raise NotImplementedError
    else:
        total_batch_size_per_gpu = bs_mixed_modal * config.dataset.accumulation
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        print(config)
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    weight_type = get_weight_type(config)

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type,
                           device=accelerator.device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    pred_act = config.model.showo.pred_act if 'pred_act' in config.model.showo else False 
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                                                         add_return_act_token_ids=pred_act)
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        # Load pretrained model and override action-related parameters from config
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, 
            use_safetensors=False,
            low_cpu_mem_usage=False,
            device_map=None,
            xvla_hidden_size=config.model.showo.get('xvla_hidden_size', None),
            action_dim=config.model.showo.get('action_dim', 20),
            proprio_dim=config.model.showo.get('proprio_dim', 20),
            time_dim=config.model.showo.get('time_dim', 32),
            len_soft_prompts=config.model.showo.get('len_soft_prompts', 32),
            max_len_seq=config.model.showo.get('max_len_seq', 512),
            num_domains=config.model.showo.get('num_domains', 20),
        ).to(accelerator.device)
        if config.model.showo.llm_vocab_size != model.showo.vocab_size:
            logger.info(f"Resize LLM vocabulary from {model.showo.vocab_size} to {config.model.showo.llm_vocab_size}")
            model.showo.resize_token_embeddings(config.model.showo.llm_vocab_size)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

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
        ).to(accelerator.device)
        logger.info("Drop-upcycling completed. Model converted to MoE architecture.")

        if accelerator.is_main_process:
            model.save_pretrained(config.uncycled_moe_init_model_path)
            logger.info(f"Modified model saved to {config.uncycled_moe_init_model_path}")
    
    # Load XVLA action modules
    xvla_checkpoint = config.model.showo.get('xvla_ckpt_path', None)
    if xvla_checkpoint is not None:
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


    # Choose layers to freeze
    _freeze_params(model, config.model.showo.frozen_params)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    ori_num_vla_image_tokens = config.dataset.preprocessing.num_vla_image_tokens

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_vla_image_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    optimizer_type = config.optimizer.name

    if accelerator.is_main_process:
        print(model)
        # for n, p in model.named_parameters():
        #     print(n)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       (('und_trans' in n or 'image_embedder' in n or 'position_embedding' in n)
                        and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_ve,
        },
        {
            "params": [p for n, p in model.named_parameters() if ('fusion_proj' in n and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_proj
        },
        {
            "params": [p for n, p in model.named_parameters() if ((
                'showo' in n or 'diffusion' in n or 'diff_proj' in n or 'time_embed_proj' in n) and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_showo
        },
        {
            "params": [p for n, p in model.named_parameters() if ((
                'pos_emb' in n or 'norm' == n.split('.')[0] or 'action_encoder' in n or 'action_decoder' in n) and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_act
        },
        {
            "params": [p for n, p in model.named_parameters() if ((
                'soft_prompt_hub' in n) and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_soft_prompt
        },
        {
            "params": [p for n, p in model.named_parameters() if ((
                'project_xvla' in n) and p.requires_grad)],
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_project_xvla
        },
    ]

    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    
    # Iterable dataloader
    mixed_loader = create_dataloader(
        num_workers=dataset_config.num_workers,
        batch_size=config.training.batch_size_vla,
        metas_path=config.training.train_metas_path,
        num_actions=config.xvla.num_actions+config.model.showo.get('len_soft_prompts', 32),
        action_mode=config.xvla.action_mode,
        training=True,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=preproc_config.max_vla_seq_len,
        image_size=preproc_config.vla_image_size,
        num_image_tokens=preproc_config.num_vla_image_tokens,
        pred_act=pred_act,
    )
    
    num_train_epochs = 1


    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")

            # not load some parameters
            if config.model.showo.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.showo.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False if config.model.showo.params_not_load is not None else True)
            del state_dict

    config.lr_scheduler.params.warmup_steps = int(
        config.training.max_train_steps * config.lr_scheduler.params.warmup_ratio)
    config.lr_scheduler.params.warmup_steps = int(
        config.training.max_train_steps * config.lr_scheduler.params.warmup_ratio)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps - global_step,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    # default: 1000 steps, linear noise schedule
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
            pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            image_masks,
            modality_positions,
            num_obs_img=1,
    ):
        b, n, pixel_c, pixel_h, pixel_w = pixel_values.shape
        if config.model.vae_model.type == 'wan21':
            # (b, n, 3, 432, 432)
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            pixel_values = pixel_values.unsqueeze(2)    # b*n c 1 h w
            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)
            image_latents = image_latents.squeeze(2)    # (b*n latent_c latent_h latent_w) == (b*n, 16, 54, 54)
            recons_images = recons_images.squeeze(2)    # (b*n, 3, 432, 432)
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
    def prepare_action_tokens_and_labels(
            actions,
            proprio,
            time_dim=32,
    ):
        action_labels = actions.clone()
        B = actions.shape[0] 
        t = (torch.rand(1, device=actions.device) + torch.arange(B, device=actions.device) / B) % (1 - 1e-5)

        noisy_actions = torch.randn_like(actions) * t.view(-1, 1, 1) + actions * (1 - t).view(-1, 1, 1)

        def timestep_embedding(t: torch.Tensor, dim: int = time_dim, max_period: int = 100) -> torch.Tensor:
            """
            Create sinusoidal timestep embeddings.

            Parameters
            ----------
            t : torch.Tensor
                Shape [B]. Each element is a timestep index, may be fractional.
            dim : int
                Dimensionality of the output embedding.
            max_period : int, default=100
                Controls the minimum frequency of the sinusoids.

            Returns
            -------
            torch.Tensor
                Shape [B, dim]. Sinusoidal embeddings.
            """
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
                / half
            )
            args = t[:, None] * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2 == 1:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

        time_emb = timestep_embedding(t)
        time_tokens = time_emb.unsqueeze(1).expand(B, actions.shape[1], time_emb.shape[-1])
        proprio_tokens = proprio.unsqueeze(1).expand(B, actions.shape[1], proprio.shape[-1])
        action_tokens = torch.cat([noisy_actions, proprio_tokens, time_tokens], dim=-1)

        return action_tokens, action_labels

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(num_train_epochs):
        model.train()
        for batch in mixed_loader:
            # print(f"batch['language_instruction']: {batch['language_instruction']}")
            text_tokens = batch['text_tokens'].to(accelerator.device)
            # text_labels = batch['text_labels'].to(accelerator.device)
            # b n c h w
            pixel_values = batch['images'].to(accelerator.device).to(weight_type)

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)
            if pred_act:
                action = batch['action'].to(accelerator.device).to(weight_type)
                proprio = batch['proprio'].to(accelerator.device).to(weight_type)
                action_masks = batch['action_masks'].to(accelerator.device)
                action_positions = batch['action_positions'].to(accelerator.device)
                domain_id = batch['domain_id'].to(accelerator.device)
            # prepare image latents and labels
            image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(pixel_values,
                                                                                                    image_masks,
                                                                                                    modality_positions)
            if pred_act: 
                action_tokens, action_labels = prepare_action_tokens_and_labels(
                    action,
                    proprio,
                    config.model.showo.time_dim,
                )
            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=text_tokens.shape[0], H=None,
            #                                Q_LEN=preproc_config.max_seq_length,
            #                                KV_LEN=preproc_config.max_seq_length, device=accelerator.device)
            # or use naive omni attention mask, which is more stable
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              text_tokens.size(1),
                                              modality_positions,
                                              accelerator.device,
                                              actions=action_positions if pred_act else None,
                                              ).to(weight_type)

            logits, loss_ntp, loss_flow, action_loss_dict = model(text_tokens=text_tokens,
                                                image_latents=image_latents,
                                                action_tokens=action_tokens.to(weight_type) if pred_act else None,
                                                t=t.to(weight_type),
                                                attention_mask=block_mask,
                                                text_masks=text_masks,
                                                image_masks=image_masks,
                                                # action_masks=action_masks,
                                                # text_labels=text_labels,
                                                image_labels=image_labels,
                                                action_labels=action_labels if pred_act else None,
                                                modality_positions=modality_positions,
                                                action_positions=action_positions,
                                                domain_id=domain_id if pred_act else None,
                                                output_hidden_states=True,
                                                max_seq_len=text_tokens.size(1),
                                                device=accelerator.device,
                                                )

            # Gather the losses across all processes for logging (if we use distributed training).
            # avg_loss_ntp = accelerator.gather(loss_ntp.repeat(total_batch_size_per_gpu)).mean()
            avg_loss_flow = accelerator.gather(loss_flow.repeat(total_batch_size_per_gpu)).mean()
            # loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow
            if pred_act:
                loss_action = sum(action_loss_dict.values())
                avg_action_loss = accelerator.gather(loss_action.repeat(total_batch_size_per_gpu)).mean()
                loss = config.training.flow_coeff * loss_flow + config.training.action_coeff * loss_action
            else:
                loss = config.training.flow_coeff * loss_flow


            

            accelerator.backward(loss.to(weight_type) / config.training.gradient_accumulation_steps)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            # log gradient norm before zeroing it
            if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    lr = [group["lr"] for group in optimizer.param_groups]
                    if len(lr) == 6:
                        logs = {
                            # "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr_ve": lr[0],
                            "lr_proj": lr[1],
                            "lr_showo": lr[2],
                        }
                        if pred_act:
                            logs.update({
                                "step_loss_action": avg_action_loss.item(),
                                "lr_act": lr[3],
                                "lr_soft_prompt": lr[4],
                                "lr_project_xvla": lr[5],
                            })
                        accelerator.log(logs, step=global_step + 1)
                        logger.info(
                            f"Step: {global_step + 1} "
                            # f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"LR_ve: {lr[0]:0.6f} "
                            f"LR_proj: {lr[1]:0.6f} "
                            f"LR_showo: {lr[2]:0.6f}"
                        )
                        if pred_act:
                            logger.info(
                                f"Loss_ACTION: {avg_action_loss.item():0.4f} "
                                f"LR_act: {lr[3]:0.6f} "
                                f"LR_soft_prompt: {lr[4]:0.6f} "
                                f"LR_project_xvla: {lr[5]:0.6f} "
                            )
                    else:
                        logs = {
                            # "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr": lr[0],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)
                        logger.info(
                            f"Epoch: {epoch} "
                            f"Step: {global_step + 1} "
                            # f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR: {lr[0]:0.6f}"
                        )
                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, "final")

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()




def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
