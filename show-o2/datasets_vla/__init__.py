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

import collections
import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)

def collate_fn(batch):
    """Collate function to batch data."""
    batched = collections.defaultdict(list)
    for data in batch:
        for key, value in data.items():
            batched[key].append(value)
    for key, value in batched.items():
        if key not in ('language_instruction',):
            batched[key] = torch.stack(value, dim=0)
    return batched

def create_dataloader(
    num_workers,
    batch_size: int, 
    metas_path: str, 
    num_actions: int,
    training: bool,
    action_mode: str,
    text_tokenizer,
    showo_token_ids,
    max_seq_len,
):

    return DataLoader(
        InfiniteDataReader(metas_path, num_actions=num_actions, training=training, action_mode=action_mode,
            text_tokenizer=text_tokenizer, showo_token_ids=showo_token_ids, max_seq_len=max_seq_len),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )