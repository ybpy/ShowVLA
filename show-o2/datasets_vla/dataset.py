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
from typing import Dict, Iterable, List
import io, json, random, numpy as np, torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 num_views: int = 1, 
                 training: bool = True,
                 action_mode: str = "ee6d",
                 lang_aug: str = None,
                 text_tokenizer = None,
                 showo_token_ids = None,
                 max_seq_len: int = 1600,
                 image_size: int = 432,
                 num_image_tokens: int = 729,
                 ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f: meta = json.load(f)
            print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
            self.metas[meta["dataset_name"]] = meta

        self.image_aug = [
            transforms.Resize((432, 432*2), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) \
                if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

        self.text_tokenizer = text_tokenizer
        self.showo_token_ids = showo_token_ids
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training: random.shuffle(traj_indices)
        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views,
            text_tokenizer=self.text_tokenizer, showo_token_ids=self.showo_token_ids,
            max_seq_len=self.max_seq_len, image_size=self.image_size, num_image_tokens=self.num_image_tokens)
        
        for traj_idx in traj_indices:
            # try:
                
            for sample in handler.iter_episode(
                traj_idx,
                num_actions=self.num_actions,
                training=self.training,
                image_aug=self.image_aug,
                lang_aug_map= meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                action_mode = self.action_mode
            ):
                sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(dataset_name, 0))
                idx_for_delta = meta.get("idx_for_delta", [])
                sample.update(action_slice(sample["abs_trajectory"], idx_for_delta))
                del sample["abs_trajectory"]
                yield sample
            # except Exception as e:
            #     with open("error_log.txt", "a") as f: f.write(f"skip broken traj {meta['datalist'][traj_idx]} with {e}\n")
            #     continue
        if self.training: yield from self._iter_one_dataset(dataset_name)


    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training: 
            for n in names: yield from self._iter_one_dataset(n)
        else:
            #names = names * 2 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])
