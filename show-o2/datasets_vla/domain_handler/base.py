# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

import io
from math import e
import random
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, Sequence, Any

import numpy as np
import h5py
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

import copy

class DomainHandler(ABC):
    """
    Minimal domain handler interface.

    Subclasses provide dataset-specific decoding by implementing an iterator
    that yields per-sample dictionaries compatible with the training loop.
    """
    dataset_name: str

    def __init__(self, meta: dict, num_views: int,
            text_tokenizer,
            showo_token_ids,
            max_seq_len: int = 2048,
            image_size: int = 432,
            num_image_tokens: int = 729) -> None:
        
        self.meta = meta
        self.num_views = num_views

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens

    @abstractmethod
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Yield samples for a single episode."""
        ...


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class BaseHDF5Handler(DomainHandler):
    """
    Generic HDF5 handler with resource-safe iteration.

    Subclasses only implement:
      - build_left_right(f) -> (left, right, left_time, right_time, freq, qdur)
          left/right: abs_trajectory [T, C], left_time/right_time: optional time arrays [T],
          freq (Hz), qdur (seconds of future window)
      - index_candidates(T_left, training) -> Iterable[int]

    Optionally override:
      - get_image_datasets(f): sequence of image arrays/datasets
      - read_instruction(f): string instruction
    """

    # --- Optional overrides -------------------------------------------------
    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        return [f[k][()] for k in keys]

    def read_instruction(self, f: h5py.File) -> str:
        key: str = self.meta["language_instruction_key"]
        ds = f[key]
        v = ds[()]
        return v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()

    # --- Required hooks -----------------------------------------------------
    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        raise NotImplementedError

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        raise NotImplementedError
    # -----------------------------------------------------------------------

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        from ..utils import decode_image_from_bytes
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def format_obs_text_future_seq(self, text: str, suffix=" Future image:"):
        text_tokens = []
        text_labels = []
        modality_positions = []

        cur_len = 1 # bos token
        
        # One observation image
        text_tokens.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        # +1 for one <|img_start|> token
        modality_positions.append((cur_len + 1, self.num_image_tokens))
        cur_len = cur_len + 1 + self.num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>
        
        # Language commmand
        if text.endswith('.'):
            text = text + suffix
        elif text[-1].isalpha():
            text = text + '.' + suffix
        else:
            raise ValueError(f"Unsupported Language Instruction: {text}")
        
        lang_tokens = self.text_tokenizer(text, add_special_tokens=False, truncation=False).input_ids
        text_tokens.extend(lang_tokens)
        cur_len += len(lang_tokens)

        text_labels = [-100 for _ in range(len(text_tokens))]

        # One future image
        text_tokens.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        text_labels.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        # +1 for one <|img_start|> token
        modality_positions.append((cur_len + 1, self.num_image_tokens))
        cur_len = cur_len + 1 + self.num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

        text_labels = [-100] + text_labels + [self.eos_id]
        text_tokens = [self.bos_id] + text_tokens + [self.eos_id]

        assert len(text_tokens) == len(text_labels) <= self.max_seq_len, f"len(text_tokens): {len(text_tokens)}, len(text_labels): {len(text_labels)}, self.max_seq_len: {self.max_seq_len}"
        text_labels = text_labels + [-100] * (self.max_seq_len - len(text_labels))
        text_tokens = text_tokens + [self.pad_id] * (self.max_seq_len - len(text_tokens))
        text_tokens = torch.tensor(text_tokens)
        text_labels = torch.tensor(text_labels)

        modality_positions = torch.tensor(modality_positions)

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Open once, yield many samples; file is always closed on exit."""
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with _open_h5(datapath) as f:
            # Images and mask
            images = self.get_image_datasets(f)
            # Language
            ins = self.read_instruction(f)
            # Domain-specific kinematics and timing
            left, right, lt, rt, freq, qdur = self.build_left_right(f)
        
        
        # image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        # image_mask[:len(images)] = True
        if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # Candidate indices (optionally shuffled)
        idxs = list(self.index_candidates(left.shape[0], training))
        if training: random.shuffle(idxs)

        # Interpolators; clamp to endpoints
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        assert V == 1
        for idx in idxs:

            # Query future window
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            # Skip static segments
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
            # Language augmentation
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            future_idx = int(min(idx + qdur*freq, ref.max() * freq))
            
            # Augment (Concatenate first, then split)
            concatenated_imgs = []
            for v in range(V):
                img_curr = self._pil_from_arr(images[v][idx])
                img_future = self._pil_from_arr(images[v][future_idx])
                concatenated_arr = np.concatenate([np.array(img_curr), np.array(img_future)], axis=1)
                # print(f"src image.shape: {concatenated_arr.shape}", flush=True)
                concatenated_imgs.append(image_aug(Image.fromarray(concatenated_arr)))

            # Split into current and future images
            # Each tensor has shape [C, H, 2*W], split into two [C, H, W] tensors
            imgs = []
            future_imgs = []
            for concat_tensor in concatenated_imgs:
                split_width = concat_tensor.shape[2] // 2
                imgs.append(concat_tensor[:, :, :split_width])
                future_imgs.append(concat_tensor[:, :, split_width:])

            assert len(imgs) == len(future_imgs) == 1
            image = torch.stack(imgs + future_imgs, dim=0)  # [2, C, H, W]
            # print(f"tgt image.shape: {image.shape}", flush=True)

            text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_future_seq(ins)

            yield {
                "language_instruction": ins,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
            }