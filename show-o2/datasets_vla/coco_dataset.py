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

import collections
import json
import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from mmengine import fileio
from pycocotools.coco import COCO
from PIL import Image

import io
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from datasets_vla.utils import BBOX_COLORS, MASK_COLORS, try_get_img_with_bbox, get_img_with_segment_mask


class COCODataset(Dataset):
    """Dataset for Coco Object Detection based on Any-to-Any Generation."""

    def __init__(
            self,
            metas_path,
            text_tokenizer,
            showo_token_ids,
            max_seq_len,
            image_size,
            num_image_tokens,
            prob_bbox: float = 0.5,
    ) -> None:

        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""

        self.all_datalist = []
        self.dataset_name_2_coco_helper = dict()
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f: meta = json.load(f)
            dataset_name = meta['dataset_name']
            ann_json_path = meta['ann_json_path']
            datalist = meta['datalist']
            print(f"== [{file}] Dataset {dataset_name} with {len(datalist)} images")

            coco = COCO(ann_json_path)
            # Remove unused parts to save memory
            coco.dataset, coco.anns, coco.cats, coco.imgToAnns, coco.catToImgs = None, None, None, None, None
            self.dataset_name_2_coco_helper[dataset_name] = coco
            for json_path in datalist:
                self.all_datalist.append([dataset_name, json_path])

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        if isinstance(image_size, int):
            self.image_height, self.image_width = image_size, image_size
        else:
            assert len(image_size) == 2
            self.image_height, self.image_width = image_size[0], image_size[1]
        self.num_image_tokens = num_image_tokens

        self.image_aug = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.)
        ]
        self.image_aug = transforms.Compose(self.image_aug)
        
        self.image_transform = [
            transforms.Resize((self.image_height, self.image_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        self.image_transform = transforms.Compose(self.image_transform)

        self.prob_bbox = prob_bbox
        self.bbox_colors = BBOX_COLORS
        self.mask_colors = MASK_COLORS

    def format_img_text_tgt_img_seq(self, text: str):
        text_tokens = []
        text_labels = []
        modality_positions = []

        cur_len = 1 # bos token
        
        # One image
        text_tokens.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        # +1 for one <|img_start|> token
        modality_positions.append((cur_len + 1, self.num_image_tokens))
        cur_len = cur_len + 1 + self.num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>
        
        # Language commmand
        lang_tokens = self.text_tokenizer(text, add_special_tokens=False, truncation=False).input_ids
        text_tokens.extend(lang_tokens)
        cur_len += len(lang_tokens)

        text_labels = [-100 for _ in range(len(text_tokens))]

        # One target image
        text_tokens.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        text_labels.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        # +1 for one <|img_start|> token
        modality_positions.append((cur_len + 1, self.num_image_tokens))
        cur_len = cur_len + 1 + self.num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

        text_labels = [-100] + text_labels + [self.eos_id]
        text_tokens = [self.bos_id] + text_tokens + [self.eos_id]

        assert len(text_tokens) == len(text_labels) <= self.max_seq_len, f"text: {text}, len(text_tokens): {len(text_tokens)}, len(text_labels): {len(text_labels)}, self.max_seq_len: {self.max_seq_len}"
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

    def __len__(self) -> int:
        return len(self.all_datalist)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        dataset_name, json_path = self.all_datalist[idx]
        coco_helper = self.dataset_name_2_coco_helper[dataset_name]
        data_dict = json.load(open(json_path))

        img_path = data_dict["img_path"]
        img = Image.open(img_path).convert('RGB')
        img = self.image_aug(img)

        category_2_instances = data_dict["anns"]
        category = np.random.choice(list(category_2_instances.keys()))
        instances = category_2_instances[category]

        use_bbox = np.random.rand() < self.prob_bbox
        if use_bbox:
            bbox_color_name = np.random.choice(list(self.bbox_colors.keys()))
            bbox_color_rgb = self.bbox_colors[bbox_color_name]
            tgt_img = try_get_img_with_bbox(img, instances, bbox_color_rgb)
            if tgt_img is None:
                use_bbox = False
            else:
                text = f"Mark all {category} in the image with {bbox_color_name} bounding box. Image with marked {category}:"
        
        if not use_bbox:
            mask_color_name = np.random.choice(list(self.mask_colors.keys()))
            mask_color_rgb = self.mask_colors[mask_color_name]
            tgt_img = get_img_with_segment_mask(img, instances, coco_helper, mask_color_rgb)
            text = f"Segment all {category} in the image with {mask_color_name} mask. Image with segmented {category}:"

        # vis = tgt_img
        # text_clean = text.replace('(', '').replace(')', '').replace('\"', '')
        # print(text_clean)
        # vis.save(f"{idx}_{text_clean}.jpg")

        text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_img_text_tgt_img_seq(text)

        img = self.image_transform(img)
        tgt_img = self.image_transform(tgt_img)
        image = torch.stack([img, tgt_img], dim=0)  # [2, C, H, W]

        return {
            'language_instruction': text,
            'text_tokens': text_tokens,
            'text_labels': text_labels,
            'images': image,
            'modality_positions': modality_positions,
            'text_masks': text_mask,
            'image_masks': image_mask,
        }


    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key not in ('language_instruction',):
                batched[key] = torch.stack(value, dim=0)
        return batched


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        # llm_name="llama3"
        llm_name="qwen2_5"
    )

    dataset = COCODataset(
        metas_path="./meta_coco_data",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=560,
        image_size=(384, 320),
        num_image_tokens=256+1,
        prob_bbox=0.5,
    )
    train_dataloader_img_edit = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn,
                                      shuffle=False, num_workers=0)


    for i, data in enumerate(train_dataloader_img_edit):
        print(f"[BATCH {i}]")
        print("text_tokens", data['text_tokens'].shape)
        print("images", data['images'].shape)
        print(data['modality_positions'][0])
        print()
