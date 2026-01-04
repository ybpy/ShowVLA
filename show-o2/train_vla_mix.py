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
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType
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

from datasets_vla import COCODataset, MixedDataLoader
from datasets_vla import create_dataloader
from utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    path_to_llm_name, _freeze_params, load_xvla_modules, replace_model_parameters, remove_trailing_digits, set_seed


from transport import Sampler, create_transport

from transformers import Qwen2MoeConfig
from peft import LoraConfig, get_peft_model

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

    bs_grounding = config.training.batch_size_grounding
    bs_vla = config.training.batch_size_vla

    if "concat" in config.dataset.mixed_loader_mode:
        assert config.dataset.accumulation == 1, "No need to enable accumulation in mixed-dataloader!"
        total_batch_size_per_gpu = bs_grounding + bs_vla
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps
    else:
        raise NotImplementedError

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
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        print(config)

        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed + accelerator.process_index)

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
                                                         add_return_act_token_ids=True)
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        # Load pretrained model and override action-related parameters from config
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, 
            use_safetensors=False,
            low_cpu_mem_usage=False,
            device_map=None,
            xvla_hidden_size=config.model.showo.get('xvla_hidden_size', None),
            xvla_depth=config.model.showo.get('xvla_depth', 2),
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
    drop_upcycling = config.model.showo.get('drop_upcycling', False)
    if drop_upcycling:
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
    lr_multipler = config.training.get('lr_multipler', 1.0) if use_lora else 1.0
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

    xval_norm_name_index = 0
    if use_lora:
        xval_norm_name_index += 2
    if use_compile:
        xval_norm_name_index += 1

    # 定义参数组配置
    group_configs = [
        {
            "name": "VE (und_trans/image_embedder/position_embedding)",
            "filter": lambda n, p: (('und_trans' in n or 'image_embedder' in n or 'position_embedding' in n)
                                    and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_ve * lr_multipler,
        },
        {
            "name": "Fusion Projection",
            "filter": lambda n, p: ('fusion_proj' in n and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_proj * lr_multipler
        },
        {
            "name": "ShowO2 (without experts)",
            "filter": lambda n, p: ((
                (('showo' in n) and ('experts' not in n)) or 
                'diffusion' in n or 'diff_proj' in n or 'time_embed_proj' in n) and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_showo * lr_multipler
        },
        {
            "name": "ShowO2 Experts",
            "filter": lambda n, p: (('experts' in n) and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_showo_expert * lr_multipler
        },
        {
            "name": "Action (pos_emb/norm/action_encoder/action_decoder)",
            "filter": lambda n, p: ((
                'pos_emb' in n or 'norm' == n.split('.')[xval_norm_name_index] or 'action_encoder' in n or 'action_decoder' in n or 'blocks' in n) and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_act
        },
        {
            "name": "Soft Prompt Hub",
            "filter": lambda n, p: (('soft_prompt_hub' in n) and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_soft_prompt
        },
        {
            "name": "Project XVLA",
            "filter": lambda n, p: (('project_xvla' in n) and p.requires_grad),
            "weight_decay": optimizer_config.weight_decay,
            "lr": optimizer_config.learning_rate_project_xvla
        },
    ]
    
    # 构建参数组并收集参数名称
    optimizer_grouped_parameters = []
    group_names_list = []
    
    for group_config in group_configs:
        params = []
        param_names = []
        for n, p in model.named_parameters():
            if group_config["filter"](n, p):
                params.append(p)
                param_names.append(n)
        if len(param_names) == 0:
            continue
        optimizer_grouped_parameters.append({
            "params": params,
            "weight_decay": group_config["weight_decay"],
            "lr": group_config["lr"],
        })
        group_names_list.append({
            "name": group_config["name"],
            "param_names": param_names,
            "lr": group_config["lr"],
            "weight_decay": group_config["weight_decay"],
        })
    
    # 打印每个group的参数名称
    logger.info("=" * 80)
    logger.info("Optimizer Parameter Groups:")
    logger.info("=" * 80)
    for i, group_info in enumerate(group_names_list):
        logger.info(f"\nGroup {i+1}: {group_info['name']}")
        logger.info(f"  Learning Rate: {group_info['lr']:.2e}")
        logger.info(f"  Weight Decay: {group_info['weight_decay']}")
        logger.info(f"  Number of Parameters: {len(group_info['param_names'])}")
        logger.info(f"  Parameter Names:")
        for param_name in group_info['param_names']:
            logger.info(f"    - {param_name}")
    logger.info("=" * 80)

    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    elif optimizer_type == "adamw8bit":
        if not BITSANDBYTES_AVAILABLE:
            raise ValueError("8-bit AdamW requires bitsandbytes library. Install with: pip install bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
        logger.info("Using 8-bit AdamW optimizer (memory efficient, ~50% memory reduction)")
    elif optimizer_type == "lion":
        if not LION_AVAILABLE:
            raise ValueError("Lion optimizer requires lion_pytorch library. Install with: pip install lion-pytorch")
        # Lion supports per-parameter-group learning rates
        # Lion typically requires learning rate 3-10x smaller than AdamW for stability
        # and weight_decay 3-10x larger to maintain regularization strength
        lion_lr_scale = optimizer_config.get('lion_lr_scale', 0.3)
        lion_wd_scale = optimizer_config.get('lion_wd_scale', 3.0)
        
        logger.info(f"Lion: scaling learning rates by {lion_lr_scale}x (recommended: 0.1-0.33)")
        
        lion_params = []
        for group in optimizer_grouped_parameters:
            scaled_weight_decay = group.get("weight_decay", optimizer_config.weight_decay) * lion_wd_scale
            lion_params.append({
                "params": group["params"],
                "lr": group["lr"] * lion_lr_scale,
                "weight_decay": scaled_weight_decay,
            })
        optimizer = Lion(
            lion_params,
            betas=(0.9, 0.99),
            use_triton=True,
        )
        logger.info("Using Lion optimizer (memory efficient, faster convergence)")
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported. Available: adamw, adamw8bit, lion")

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    def create_grounding_dataloader(dataset, batch_size, collate_fn):
        if accelerator.num_processes > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,
                                         drop_last=True,
                                         )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=sampler, collate_fn=collate_fn,
                                        shuffle=shuffle, num_workers=dataset_config.num_workers,
                                        drop_last=True,
                                        pin_memory=True,
                                        persistent_workers=True)
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

            # Unwrap model manually to match the state_dict structure
            unwrapped_model = model
            while hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod

            if hasattr(unwrapped_model, "base_model"):
                unwrapped_model = unwrapped_model.base_model.model

            missing_keys, unexpected_keys = unwrapped_model.load_state_dict(state_dict, strict=False)
            del state_dict

            assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
            logger.info(f"missing_keys: {missing_keys}")

    # Calculate steps for the scheduler (based on optimization steps, not micro-steps)
    num_training_steps = config.training.max_train_steps
    num_warmup_steps = int(num_training_steps * config.lr_scheduler.params.warmup_ratio)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps - global_step,
        num_warmup_steps=num_warmup_steps,
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer = accelerator.prepare(model, optimizer)

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
            image_latents = image_latents.squeeze(2)    # (b*n latent_c latent_h latent_w) == (b*n, 16, 54, 54)
            _, c, h, w = image_latents.shape
            image_latents = image_latents.reshape(b, n, c, h, w)
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

        return xt, t, ut, masks

    # Initialize loss meters for logging
    loss_flow_m = AverageMeter()
    loss_action_m = AverageMeter()

    model.train()
    for batch in mixed_loader:
        with accelerator.accumulate(model):
            # print(f"batch['language_instruction']: {batch['language_instruction']}")
            text_tokens = batch['text_tokens'].to(accelerator.device)
            # text_labels = batch['text_labels'].to(accelerator.device)
            # b n c h w
            pixel_values = batch['images'].to(accelerator.device).to(weight_type)

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)
            if pred_act:
                actions = batch['action'].to(accelerator.device).to(weight_type)
                proprio = batch['proprio'].to(accelerator.device).to(weight_type)
                action_positions = batch['action_positions'].to(accelerator.device)
                domain_id = batch['domain_id'].to(accelerator.device)
                action_labels = actions.clone().to(accelerator.device)
                t_action = (torch.rand(1, device=actions.device) + torch.arange(text_tokens.shape[0], device=actions.device) / text_tokens.shape[0]) % (1 - 1e-5)
            # prepare image latents and labels
            image_latents, t, image_labels, image_masks = prepare_latents_and_labels(pixel_values,
                                                                                        image_masks,
                                                                                        modality_positions)
            
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                                text_tokens.size(1),
                                                modality_positions,
                                                accelerator.device,
                                                actions=action_positions if pred_act else None,
                                                ).to(weight_type)

            logits, loss_ntp, loss_flow, action_loss_dict = model(text_tokens=text_tokens,
                                                image_latents=image_latents,
                                                t=t.to(weight_type),
                                                attention_mask=block_mask,
                                                text_masks=text_masks,
                                                image_masks=image_masks,
                                                # action_masks=action_masks,
                                                # text_labels=text_labels,
                                                image_labels=image_labels,
                                                modality_positions=modality_positions,
                                                domain_id=domain_id if pred_act else None,
                                                output_hidden_states=True,
                                                max_seq_len=text_tokens.size(1),
                                                device=accelerator.device,
                                                actions=actions if pred_act else None,
                                                proprio=proprio if pred_act else None,
                                                action_labels=action_labels if pred_act else None,
                                                action_positions=action_positions if pred_act else None,
                                                t_action=t_action.to(weight_type) if pred_act else None,
                                                )

            loss_flow_m.update(loss_flow.item())

            # loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow
            if pred_act:
                loss_action = sum(action_loss_dict.values())
                loss_action_m.update(loss_action.item())

                loss = config.training.flow_coeff * loss_flow + config.training.action_coeff * loss_action
            else:
                loss = config.training.flow_coeff * loss_flow

            accelerator.backward(loss.to(weight_type))

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            if accelerator.sync_gradients:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:

            # Log metrics
            if (global_step + 1) % config.experiment.log_every == 0:
                # 跨 GPU 汇总并计算平均 Loss
                Loss_flow = accelerator.gather(torch.tensor(loss_flow_m.avg, device=accelerator.device).repeat(total_batch_size_per_gpu)).mean().item()
                if pred_act:
                    Loss_action = accelerator.gather(torch.tensor(loss_action_m.avg, device=accelerator.device).repeat(total_batch_size_per_gpu)).mean().item()
                
                lr = [group["lr"] for group in optimizer.param_groups]
                if len(lr) >= 6:
                    logs = {
                        "Loss_flow": Loss_flow,
                        "lr_ve": lr[0],
                        "lr_proj": lr[1],
                        "lr_showo": lr[2],
                    }
                    if pred_act:
                        logs.update({
                            "Loss_action": Loss_action,
                            "lr_act": lr[-3],
                            "lr_soft_prompt": lr[-2],
                            "lr_project_xvla": lr[-1],
                        })
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step:{global_step + 1} "
                        f"Loss_flow:{Loss_flow:0.4f} "
                        f"LR_ve:{lr[0]:.2e} "
                        f"LR_proj:{lr[1]:.2e} "
                        f"LR_showo:{lr[2]:.2e}"
                    )
                    if pred_act:
                        logger.info(
                            f"Loss_action: {Loss_action:0.4f} "
                            f"LR_act: {lr[-3]:.2e} "
                            f"LR_soft_prompt: {lr[-2]:.2e} "
                            f"LR_project_xvla: {lr[-1]:.2e} "
                        )
                else:
                    logs = {
                        "Loss_flow": Loss_flow,
                        "lr_ve": lr[0],
                        "lr_proj": lr[1],
                        "lr_showo": lr[2],
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step:{global_step + 1} "
                        f"Loss_flow:{Loss_flow:0.4f} "
                        f"LR_ve:{lr[0]:.2e} "
                        f"LR_proj:{lr[1]:.2e} "
                        f"LR_showo:{lr[2]:.2e}"
                    )
                loss_flow_m.reset()
                loss_action_m.reset()

            # Save model checkpoint
            if (global_step + 1) % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step + 1)

            global_step += 1


        # Stop training if max steps is reached
        if global_step >= config.training.max_train_steps:
            break
        # End for

    accelerator.wait_for_everyone()

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
    if accelerator.is_main_process:
        # Get state_dict first before unwrapping to avoid issues
        # manually unwrap and get state_dict
        temp_model = model
        if hasattr(model, 'module'):
            temp_model = model.module
        # Unwrap torch.compile if present
        while hasattr(temp_model, '_orig_mod'):
            temp_model = temp_model._orig_mod
        state_dict = temp_model.state_dict()
        
        # Unwrap model manually to avoid accelerator's unwrap_model issues with torch.compile
        unwrapped_model = model
        # Unwrap accelerator wrapper
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        # Unwrap torch.compile wrapper if present
        while hasattr(unwrapped_model, '_orig_mod'):
            unwrapped_model = unwrapped_model._orig_mod
        
        # For PEFT models, we need to unwrap the base model
        if hasattr(unwrapped_model, 'base_model'):
            unwrapped_model = unwrapped_model.base_model.model
        
        # Use regular torch.save instead of accelerator.save to avoid unwrapping issues
        def safe_save_function(obj, path):
            """Save function that avoids accelerator's unwrapping logic"""
            torch.save(obj, path)
        
        # Try to save using save_pretrained, but fall back to direct saving if it fails
        # due to torch.compile unwrapping issues
        try:
            unwrapped_model.save_pretrained(
                save_path / "unwrapped_model",
                save_function=safe_save_function,
                state_dict=state_dict,
                safe_serialization=False
            )
        except (KeyError, AttributeError) as e:
            # If save_pretrained fails due to unwrapping issues, save directly
            logger.warning(f"save_pretrained failed with {e}. Saving state_dict directly...")
            unwrapped_model_dir = save_path / "unwrapped_model"
            unwrapped_model_dir.mkdir(parents=True, exist_ok=True)
            # Save state_dict directly
            safe_save_function(state_dict, unwrapped_model_dir / "pytorch_model.bin")
            # Save config if the model has one
            if hasattr(unwrapped_model, 'config'):
                config_path = unwrapped_model_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(unwrapped_model.config.to_dict(), f, indent=2)
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")
    else:
        # On non-main processes, still need to get state_dict for synchronization
        try:
            _ = accelerator.get_state_dict(model)
        except (KeyError, AttributeError):
            pass  # Ignore errors on non-main processes



if __name__ == "__main__":
    main()
