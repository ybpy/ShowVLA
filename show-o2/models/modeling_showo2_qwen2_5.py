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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig
from torch.nn.attention.flex_attention import BlockMask
from .misc import velocity_prediction, next_token_prediction, interpolate_pos_encoding
from .modeling_siglip import SiglipModel
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig
from .modules import ModulatedAttentionBlock, RMSNorm, PatchEmbed, TimestepEmbedder, FinalLayer
from .qwen2 import Qwen2ForCausalLM

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
import traceback
from typing import Any, Dict
import json_numpy
import cv2
import numpy as np
from PIL import Image
import math
from .omni_attention import omni_attn_mask_naive


def timestep_embedding(t: torch.Tensor, dim: int = 32, max_period: int = 100) -> torch.Tensor:
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


class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y

class Showo2Qwen2_5(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            llm_vocab_size=None,
            llm_model_path='',
            load_from_showo=False,
            image_latent_dim=16,
            image_latent_height=16,
            image_latent_width=16,
            video_latent_height=16,
            video_latent_width=16,
            patch_size=2,
            hidden_size=2048,
            clip_latent_dim=1152,
            num_diffusion_layers=10,
            add_time_embeds=True,
            add_qk_norm=False,
            clip_pretrained_model_path="google/siglip-so400m-patch14-384",
            xvla_hidden_size=1024,
            action_dim=20,
            proprio_dim=20,
            time_dim=32,
            len_soft_prompts=32,
            max_len_seq=512,
            num_domains=20,
            **kwargs,
    ):
        super().__init__()

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(llm_vocab_size)

        # patch embedding layer for semantic layers
        self.image_embedder_und = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=clip_latent_dim,
        )

        # projector
        self.image_embedder_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=hidden_size,
        )

        # initialize semantic layers from siglip
        siglip_model = SiglipModel.from_pretrained(clip_pretrained_model_path)
        self.position_embedding = siglip_model.vision_model.embeddings.position_embedding
        self.und_trans = siglip_model.vision_model.encoder
        del self.und_trans.layers[-1]
        self.register_buffer("image_position_ids",
                             torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
                             persistent=False)

        self.fusion_proj = nn.Sequential(
            RMSNorm(clip_latent_dim + hidden_size),
            nn.Linear(clip_latent_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        if xvla_hidden_size:
            self.xvla_hidden_size = xvla_hidden_size
            self.project_xvla_encode = nn.Linear(xvla_hidden_size, hidden_size)
            self.project_xvla_decode = nn.Linear(hidden_size, xvla_hidden_size)

            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, xvla_hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            self.norm = nn.LayerNorm(xvla_hidden_size)
            self.action_encoder = DomainAwareLinear(
                action_dim + proprio_dim + time_dim, xvla_hidden_size, num_domains=num_domains
            )
            self.action_decoder = DomainAwareLinear(xvla_hidden_size, action_dim, num_domains=num_domains)

            self.len_soft_prompts = len_soft_prompts
            if len_soft_prompts > 0:
                self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * xvla_hidden_size)
                nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
        

        # adjust for diffusion head
        self.diffusion_head_config = DiffusionHeadConfig()
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size)
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in
             range(num_diffusion_layers)]
        )
        self.diffusion_head_b = FinalLayer(self.diffusion_head_config.hidden_size, patch_size, image_latent_dim)

        # Action loss components (EE6D action space)
        self.action_mse = nn.MSELoss()
        self.action_bce = nn.BCEWithLogitsLoss()
        self.GRIPPER_IDX = (9, 19)
        self.GRIPPER_SCALE = 1.0
        self.XYZ_SCALE = 500.0
        self.ROT_SCALE = 10.0
        self.POS_IDX_1 = (0, 1, 2)
        self.POS_IDX_2 = (10, 11, 12)
        self.ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
        self.ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

        self.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):

        # Initialize image embedders
        w1 = self.image_embedder_und.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.image_embedder_und.proj.bias, 0)

        w2 = self.image_embedder_gen.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.image_embedder_gen.proj.bias, 0)

        # Initialize transformer layers for understanding encoding and diffusion head
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.und_trans)
        _basic_init(self.fusion_proj)
        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def compute_action_loss(self, pred, target):
        """
        Compute action loss for EE6D action space.
        
        Parameters
        ----------
        pred : Tensor
            Predicted actions, shape [B, T, action_dim]
        target : Tensor
            Target actions, shape [B, T, action_dim]
        
        Returns
        -------
        dict
            Dictionary containing position_loss, rotate6D_loss, gripper_loss
        """
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape

        # Gripper BCE loss
        g_losses = [self.action_bce(pred[:, :, gi], target[:, :, gi]) for gi in self.GRIPPER_IDX]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX) * self.GRIPPER_SCALE

        # XYZ position loss
        pos_loss = (
            self.action_mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.action_mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D loss
        rot_loss = (
            self.action_mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.action_mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def unpatchify(self, x, h, w, T=0):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.image_embedder_gen.patch_size[0]
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs

    def forward_und_only(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            **kwargs,
    ):
        T = 0
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        if T == 0:
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
        else:
            # (B, C, T, H, W) --> (BT, C, H, W)
            image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
            image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

        # go through semantic layers
        p = self.config.patch_size
        h_, w_ = h // p, w // p
        # specific for fixed resolution of 432x432
        if self.position_embedding.weight.shape[0] == h_ * w_:
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        # interpolate position embeddings for dynamic resolution
        else:
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim,
                self.position_embedding,
                h_,
                w_,
                1,
            )
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        if T != 0:
            image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

        # spatial (-temporal) fusion
        image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                    # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = \
                        image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]

        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            domain_id=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            guidance_scale=0.0,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            actions=None,
            proprio=None,
            action_labels=None,
            action_positions=None,
            t_action=None,
            **kwargs,
    ):
        B, L = text_tokens.shape
        T = 0
        if image_latents is None:
            # text-only
            logits = self.showo(input_ids=text_tokens, attention_mask=attention_mask)
            return logits
        elif image_latents is not None:
            # multimoidal understanding and generatiopn
            input_embeds = self.showo.model.embed_tokens(text_tokens)
            dtype = input_embeds.dtype
            if len(image_latents.shape) != 4:
                b, c, T, h, w = image_latents.shape
            else:
                b, c, h, w = image_latents.shape

            # go through dual-path extraction
            if T == 0:
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            else:
                # (B, C, T, H, W) --> (BT, C, H, W)
                image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
                # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
                image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
                image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

            # go through semantic layers
            p = self.config.patch_size
            h_, w_ = h // p, w // p
            # specific for fixed resolution of 432x432
            if self.position_embedding.weight.shape[0] == h_ * w_:
                image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
            # interpolate position embeddings for dynamic resolution
            else:
                image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                    self.config.clip_latent_dim,
                    self.position_embedding,
                    h_,
                    w_,
                    1,
                )
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
            if T != 0:
                image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

            # spatial (-temporal) fusion
            image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

            if image_labels is not None:
                if T == 0:
                    image_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
                    image_labels = image_labels.reshape(shape=(b, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, h_ * w_, p * p * c))
                else:
                    # (B, C, T, H/p, W/p)
                    image_labels = rearrange(image_labels, 'b c t h w -> b (t h w) c')
                    image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))

            time_embeds = self.time_embed(t, dtype)
            if hasattr(self, 'time_embed_proj'):
                time_embeds_proj = self.time_embed_proj(time_embeds)
            else:
                time_embeds_proj = time_embeds

            # structure text and image embeddings into sequences
            if image_labels is not None:
                new_image_labels = torch.zeros([b, max_seq_len, p * p * c], device=device, dtype=dtype)
                image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if self.config.add_time_embeds:
                        input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                        # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                        # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                        input_embeds[i, offset + 1:offset + 1 + length - 1] = image_embeds[
                                                                              i * modality_positions.size(1) + j,
                                                                              :max(length - 1, 0)]
                        if image_labels is not None:
                            # mask the position of time embedding
                            image_masks[i, offset] = 0
                            # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                            new_image_labels[i, offset + 1:offset + 1 + length - 1] = image_labels[
                                                                                      i * modality_positions.size(
                                                                                          1) + j, :max(length - 1, 0)]
                    else:
                        input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j,
                                                                  :length]
                        if image_labels is not None:
                            new_image_labels[i, offset:offset + length] = image_labels[
                                                                          i * modality_positions.size(1) + j, :length]

            if actions is not None:
                assert proprio is not None, "proprioception input is required when actions are provided"
                noisy_actions = torch.randn_like(actions) * t_action.view(-1, 1, 1) + actions * (1 - t_action).view(-1, 1, 1)
                # zero-out gripper channels in actions/proprio
                noisy_actions[..., self.GRIPPER_IDX] = 0.0
                proprio[..., self.GRIPPER_IDX] = 0.0

                time_emb = timestep_embedding(t_action)
                time_tokens = time_emb.unsqueeze(1).expand(B, actions.shape[1], time_emb.shape[-1])
                proprio_tokens = proprio.unsqueeze(1).expand(B, actions.shape[1], proprio.shape[-1])
                action_tokens = torch.cat([noisy_actions, proprio_tokens, time_tokens], dim=-1)
            
            if action_tokens is not None:
                # encode action tokens
                action_embeds = self.action_encoder(action_tokens, domain_id=domain_id)

                # Add positional embeddings (truncate if needed)
                num_actions = action_embeds.shape[1]
                if num_actions > self.pos_emb.shape[1]:
                    raise ValueError(
                        f"Sequence length {num_actions} exceeds max_len_seq={self.pos_emb.shape[1]}."
                    )
                action_embeds = action_embeds + self.pos_emb[:, :num_actions, :]

                # Append soft prompts
                if self.len_soft_prompts > 0:
                    soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.xvla_hidden_size)
                    action_embeds = torch.cat([soft_prompts, action_embeds], dim=1)

                action_embeds = self.project_xvla_encode(action_embeds)

                for i, action_batch in enumerate(action_positions):
                    for j, (offset, length) in enumerate(action_batch): 
                        input_embeds[i, offset:offset + length] = action_embeds[i * action_positions.size(1) + j]


            outputs = self.showo(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                output_hidden_states=output_hidden_states
            )

            logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]


            if action_tokens is not None: 
                # Extract action embeddings from last_hidden_states
                action_embeds_list = []
                for i, action_batch in enumerate(action_positions):
                    for j, (offset, length) in enumerate(action_batch):
                        action_embeds_list.append(last_hidden_states[i, offset+self.len_soft_prompts:offset+self.len_soft_prompts+num_actions])
                action_embeds_from_output = torch.stack(action_embeds_list, dim=0)  # [B, num_action_tokens, hidden_size]
                action_embeds_from_output = self.project_xvla_decode(action_embeds_from_output)
                # action head to predict actions
                pred_actions = self.action_decoder(self.norm(action_embeds_from_output), domain_id=domain_id)
            

            # diffusion head to predict vector fields
            if hasattr(self, 'diff_proj'):
                last_hidden_states = self.diff_proj(last_hidden_states)
            position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
            for layer in self.diffusion_head_a:
                last_hidden_states = layer(hidden_states=last_hidden_states,
                                           adaln_input=time_embeds,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           modality_positions=modality_positions,
                                           )[0]
            v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)

        # [:v_pred.shape[0]] is the valid image labels (special case for interleaved data training)
        if text_labels is not None and image_labels is not None and action_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            action_loss_dict = self.compute_action_loss(pred_actions, action_labels)
            return logits, loss_ntp, loss_flow, action_loss_dict
        
        elif text_labels is not None and image_labels is not None and action_labels is None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            return logits, loss_ntp, loss_flow, None
        
        elif text_labels is not None and image_labels is None and action_labels is None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp, None, None

        elif text_labels is None and image_labels is not None and action_labels is not None:
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            action_loss_dict = self.compute_action_loss(pred_actions, action_labels)
            return logits, None, loss_flow, action_loss_dict
        
        elif text_labels is None and image_labels is not None and action_labels is None:
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            return logits, None, loss_flow, None

        else:
            v_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    else:
                        v_pred_.append(v_pred[i, offset:offset + length])
                        num_imgs += 1
            v_pred_ = torch.stack(v_pred_)

            # remove the time embedding
            if self.config.add_time_embeds:
                v_pred_ = v_pred_[:, 1:, :]

            # unpatchify
            v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

            if T == 0:
                v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
                v_pred_ = v_pred_.reshape(num_imgs, c, h, w)
            else:
                v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
                v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

            # specific for image-to-video generation
            if first_frame_as_cond:
                # zero the v-prediction for the first frame
                v_pred_ = torch.cat([
                    torch.zeros_like(v_pred_)[:, :, :1],
                    v_pred_[:, :, 1:]
                ], dim=2)

            # specific for mixed-modality generation
            if only_denoise_last_image:
                if guidance_scale > 0:
                    v_pred_cond, v_pred_uncond = torch.chunk(v_pred_, 2)

                    v_pred_cond = torch.cat([
                        torch.zeros_like(v_pred_cond)[:-1, :, :],
                        v_pred_cond[-1:, :, :]
                    ], dim=0)

                    v_pred_uncond = torch.cat([
                        torch.zeros_like(v_pred_uncond)[:-1, :, :],
                        v_pred_uncond[-1:, :, :]
                    ], dim=0)

                    v_pred_ = torch.cat([v_pred_cond, v_pred_uncond], dim=0)
                else:
                    v_pred_ = torch.cat([
                        torch.zeros_like(v_pred_)[:-1, :, :],
                        v_pred_[-1:, :, :]
                    ], dim=0)

            if action_tokens is not None:
                return logits, v_pred_, pred_actions
            else:
                return logits, v_pred_

    @torch.no_grad()
    def t2i_generate(
            self,
            image_latents=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

        else:
            if t.shape[-1] != text_tokens.shape[0]:
                t[:-1] = 1.0
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            return v

    @torch.no_grad()
    def mmu_generate(
            self,
            input_embeds=None,
            attention_mask=None,
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            eos_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        device = input_embeds.device
        dtype = input_embeds.dtype
        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self.showo(inputs_embeds=input_embeds, attention_mask=attention_mask.to(dtype))['logits']

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            idx_next_embeds = self.showo.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1).to(dtype)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result

    @torch.no_grad()
    def lm_generate(
            self,
            input_ids=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        for _ in range(max_new_tokens):
            # Generate the next token
            outputs = self.showo(
                input_ids=torch.tensor([generated_tokens]).to(device),
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text

    @torch.no_grad()
    def mm_generate(
            self,
            input_ids=None,
            image_latents=None,
            t=None,
            modality_positions=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        if attention_mask is not None and type(attention_mask) == BlockMask:
            raise NotImplementedError

        for _ in range(max_new_tokens):
            # Generate the next token
            logits = self.forward_und_only(
                text_tokens=torch.tensor([generated_tokens]).to(device),
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
            )

            next_token_logits = logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(next_token_logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L + 1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b.to(image_latents.dtype)

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text
    
    # =============================== FastAPI service =============================
    def _build_app(self, config, vae_model, text_tokenizer, showo_token_ids):
        """
        Minimal FastAPI app for ShowVLA inference.
        """
        if self.app is not None:
            return

        app = FastAPI()

        device = vae_model.device
        dtype = vae_model.dtype

        pad_id = text_tokenizer.pad_token_id
        bos_id = showo_token_ids['bos_id']
        eos_id = showo_token_ids['eos_id']
        boi_id = showo_token_ids['boi_id']
        eoi_id = showo_token_ids['eoi_id']
        img_pad_id = showo_token_ids['img_pad_id']
        max_seq_len = config.preproc_config.max_vla_seq_len
        image_size = config.preproc_config.vla_image_size
        num_image_tokens = config.preproc_config.num_vla_image_tokens

        boa_id = showo_token_ids['boa_id']
        eoa_id = showo_token_ids['eoa_id']
        act_pad_id = showo_token_ids['act_pad_id']
        num_actions = config.xvla.num_actions
        num_action_tokens = config.xvla.num_actions + config.model.showo.len_soft_prompts
        action_dim = config.model.showo.action_dim

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
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
                            # x0->noise x1->image
                            x1 = image_latents[i][j]
                            
                            xt = x1 if is_obs_img else torch.randn_like(x1)
                            
                            xt_list.append(xt)

                    xt = torch.cat(xt_list, dim=0)
                    xt = xt.reshape(b * n, c, h, w)
                    return xt, t
                
                # Load image
                image = json_numpy.loads(payload["image0"])
                if isinstance(image, np.ndarray):
                    if image.ndim == 1:
                        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = Image.fromarray(image)
                elif isinstance(image, (list, tuple)):
                    image = Image.fromarray(np.array(image))
                elif isinstance(image, str):
                    image = Image.open(image)
                if not image:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)
                
                # Convert PIL.Image to tensor format for prepare_image_latents
                image_curr = np.array(image)  # (256, 256, 3)
                image_curr = image_curr.astype(np.float32) / 255.0
                image_curr = torch.from_numpy(image_curr).permute(2, 0, 1)  # (3, 256, 256)
                image_future = torch.zeros_like(image_curr)  # (3, 256, 256)
                pixel_values = torch.stack([image_curr, image_future], dim=0)  # (2, 3, 256, 256)
                pixel_values = pixel_values.unsqueeze(0)  # (1, 2, 3, 256, 256)
                pixel_values = pixel_values.to(device)
                
                # Get image latents
                image_latents, t_img = prepare_image_latents(pixel_values)

                # Formulate text tokens
                text_tokens = []
                modality_positions = []
                action_positions = []

                cur_len = 1 # bos token

                # One observation image
                text_tokens.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
                # +1 for one <|img_start|> token
                modality_positions.append((cur_len + 1, num_image_tokens))
                cur_len = cur_len + 1 + num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

                # Language command
                text = payload["language_instruction"]
                if text.endswith("."):
                    text = text + " Future image:"
                elif text[-1].isalpha():
                    text = text + ". Future image:"
                else:
                    raise ValueError(f"Unsupported Language Instruction: {text}")
                
                lang_tokens = text_tokenizer.encode(text, add_special_tokens=False, truncation=False).input_ids
                text_tokens.extend(lang_tokens)
                cur_len = cur_len + len(lang_tokens)

                # One future image
                text_tokens.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
                # +1 for one <|img_start|> token
                modality_positions.append((cur_len + 1, num_image_tokens))
                cur_len = cur_len + 1 + num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

                # Future action sequence to predict
                text_tokens.extend([boa_id] + [act_pad_id] * num_action_tokens + [eoa_id])
                action_positions.append((cur_len + 1, num_action_tokens))
                cur_len = cur_len + 1 + num_action_tokens + 1  # +2 to include <|act_start|> and <|act_end|>

                text_tokens = [bos_id] + text_tokens + [eos_id]

                assert len(text_tokens) <= max_seq_len, f"len(text_tokens): {len(text_tokens)}, max_seq_len: {max_seq_len}"
                text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
                text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(device) # (1, seq_len)

                modality_positions = torch.tensor(modality_positions).unsqueeze(0).to(device) # (1, num_modalities, 2)
                action_positions = torch.tensor(action_positions).unsqueeze(0).to(device) # (1, num_action_segments, 2)

                # Construct attention mask
                block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                                  text_tokens.size(1),
                                                  modality_positions,
                                                  device,
                                                  actions=action_positions,
                                                  ).to(dtype)
                
                # Load steps, proprio, domain_id
                steps = int(payload.get("steps", 10))
                steps = max(1, steps)
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"]))).to(device=device, dtype=dtype)
                proprio = proprio.unsqueeze(0)
                domain_id = torch.tensor([int(payload["domain_id"])]).to(device)
                domain_id = domain_id.unsqueeze(0)
                actions = torch.zeros((text_tokens.size(0), num_actions, action_dim), device=device, dtype=dtype)

                # Inference
                # Denoising loop
                dt = 1.0 / steps
                for i in range(steps, 0, -1):
                    t_action = torch.full((text_tokens.size(0),), fill_value=i / steps, device=device, dtype=dtype)
                    logits, v_pred_, actions = self.forward(text_tokens=text_tokens,
                                                    image_latents=image_latents,
                                                    t=t_img,
                                                    attention_mask=block_mask,
                                                    modality_positions=modality_positions,
                                                    domain_id=domain_id,
                                                    max_seq_len=max_seq_len,
                                                    device=device,
                                                    actions=actions,
                                                    proprio=proprio,
                                                    action_positions=action_positions,
                                                    t_action=t_action,
                                                   )
                    # Update image_latents and t_img
                    image_latents[1::2] = image_latents[1::2] + v_pred_[1::2] * dt
                    t_img[:, 1:] = (t_img[:, 1:] + dt).clamp(0, 1)
                    
                actions = actions.squeeze(0).tolist()
                
                return JSONResponse({"action": actions})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app
    
    def run(self, config, vae_model, text_tokenizer, showo_token_ids, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.
        """
        self._build_app(config, vae_model, text_tokenizer, showo_token_ids)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)