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
import cv2
from typing import Any, Dict, List, Optional
from mmengine import fileio
from PIL import Image

import io
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from datasets_vla.utils import BBOX_COLORS, MASK_COLORS, get_img_with_segment_mask, get_img_with_segment_mask_ade20k


class GroundingDataset(Dataset):
    """Dataset for Object Detection/Segmentation based on Any-to-Any Generation."""

    def __init__(
            self,
            metas_path,
            text_tokenizer,
            showo_token_ids,
            max_seq_len,
            image_size,
            num_image_tokens,
            vis_mode: str = "rand", # "bbox", "segment_mask", "combine", "rand"
            mask_color_weight: float = 0.5,
    ) -> None:

        # if fileio.isdir(metas_path):
        #     meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
        #     root = metas_path
        # else: meta_files, root = metas_path, ""

        self.all_datalist = []
        for file in metas_path:
            with io.BytesIO(fileio.get(file)) as f: meta = json.load(f)
            dataset_name = meta['dataset_name']
            datalist = meta['datalist']
            print(f"== [{file}] Dataset {dataset_name} with {len(datalist)} images")

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
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        self.image_transform = transforms.Compose(self.image_transform)

        assert vis_mode in ("bbox", "segment_mask", "combine", "rand"), vis_mode
        self.vis_mode = vis_mode
        self.bbox_colors = BBOX_COLORS
        self.mask_colors = MASK_COLORS
        self.mask_color_weight = mask_color_weight

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

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __len__(self) -> int:
        return len(self.all_datalist)

    def try_get_resized_img_with_bbox(self, img, instances, color, ensure_no_seg=False, max_num_bboxes=None):
        img_w, img_h = img.size
        
        bboxes = []
        for ann in instances:
            if ann.get("iscrowd", 0):
                return None
            segm = ann["segmentation"]
            assert type(segm) == list
            if ensure_no_seg and len(segm) > 1:
                return None
            
            bbox = ann["bbox"]
            bboxes.append(bbox)
        
        if max_num_bboxes is not None and len(bboxes) > max_num_bboxes:
            return None

        # Resize and transfer to numpy array
        tgt_img = np.array(img.resize((self.image_width, self.image_height)))

        scale_x = self.image_width / img_w
        scale_y = self.image_height / img_h

        # draw all the bboxes on the image with the color
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
            if w < 6:
                x1 = max(0, x1-3)
                x2 = min(img_w-1, x2+3)
            if h < 6:
                y1 = max(0, y1-3)
                y2 = min(img_h-1, y2+3)

            x1_ = int(round(x1 * scale_x))
            y1_ = int(round(y1 * scale_y))
            x2_ = int(round(x2 * scale_x))
            y2_ = int(round(y2 * scale_y))
            x1_ = max(0, min(self.image_width - 1, x1_))
            y1_ = max(0, min(self.image_height - 1, y1_))
            x2_ = max(0, min(self.image_width - 1, x2_))
            y2_ = max(0, min(self.image_height - 1, y2_))
            cv2.rectangle(tgt_img, (x1_, y1_), (x2_, y2_), color, 2)
        
        return Image.fromarray(tgt_img)

    def _get_coco_lvis_visualized_image(self, img, img_h, img_w, instances, category, force_comb=False, ensure_no_seg=False, max_num_bboxes=None):
        if force_comb:
            mode = "combine"
        elif self.vis_mode == "rand":
            mode = np.random.choice(["bbox", "segment_mask"])
            # mode = np.random.choice(["bbox", "segment_mask", "combine"])
        else:
            mode = self.vis_mode
        color_name = np.random.choice(list(self.mask_colors.keys()))
        bbox_color_rgb = self.bbox_colors[color_name]
        mask_color_rgb = self.mask_colors[color_name]

        if mode == "bbox":
            tgt_img = self.try_get_resized_img_with_bbox(img, instances, bbox_color_rgb, ensure_no_seg=ensure_no_seg, max_num_bboxes=max_num_bboxes)
            if tgt_img is not None:
                text = f"Mark all {category}(s) in the image with {color_name} bounding box. Image with marked {category}(s):"
                return tgt_img, text

        tgt_img = get_img_with_segment_mask(img, img_h, img_w, instances, mask_color_rgb, self.mask_color_weight)

        if mode == "combine":
            resized_tgt_img_with_bbox = self.try_get_resized_img_with_bbox(tgt_img, instances, bbox_color_rgb, ensure_no_seg=ensure_no_seg, max_num_bboxes=max_num_bboxes)
            if resized_tgt_img_with_bbox is not None:
                text = f"Mark with {color_name} bounding box and segment mask for all {category}(s) in the image:"
                return resized_tgt_img_with_bbox, text

        text = f"Segment all {category}(s) in the image with {color_name} mask. Image with segmented {category}(s):"
        return tgt_img.resize((self.image_width, self.image_height)), text

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        dataset_name, json_path = self.all_datalist[idx]
        data_dict = json.load(open(json_path))

        img_path = data_dict["img_path"]
        img = Image.open(img_path).convert('RGB')
        img = self.image_aug(img)

        img_h, img_w = data_dict["height"], data_dict["width"]
        assert img.size == (img_w, img_h), f"img.size {img.size} != ({img_w}, {img_h})"

        dataset_name_lower = dataset_name.lower()
        if 'coco' in dataset_name_lower:
            category_2_instances = data_dict["anns"]
            category = np.random.choice(list(category_2_instances.keys()))
            instances = category_2_instances[category]
            tgt_img, text = self._get_coco_lvis_visualized_image(img, img_h, img_w, instances, category, ensure_no_seg=True, max_num_bboxes=8)
            img = img.resize((self.image_width, self.image_height))
        elif 'lvis' in dataset_name_lower:
            category_2_instances = data_dict["anns"]
            category = np.random.choice(list(category_2_instances.keys()))
            instances = category_2_instances[category]
            is_small = data_dict["is_small"][category]
            tgt_img, text = self._get_coco_lvis_visualized_image(img, img_h, img_w, instances, category, force_comb=False)
            img = img.resize((self.image_width, self.image_height))
        elif 'ade20k' in dataset_name_lower:
            segm_path = data_dict["segm_path"]
            segm = Image.open(segm_path)
            assert segm.size == (img_w, img_h), f"segm.size {segm.size} != ({img_w}, {img_h})"

            list_categories = data_dict["list_categories"]
            (cat_id, category) = list_categories[np.random.choice(len(list_categories))]

            mask_color_name = np.random.choice(list(self.mask_colors.keys()))
            color = self.mask_colors[mask_color_name]
            tgt_img = get_img_with_segment_mask_ade20k(img, segm, cat_id, color, self.mask_color_weight)
            img = img.resize((self.image_width, self.image_height))
            tgt_img = tgt_img.resize((self.image_width, self.image_height))
            text = f"Segment instance(s) of {category} in the image with {mask_color_name} mask. Image with segmented {category}:"
        else:
            raise NotImplementedError(f"Unsupported grounding dataset: {dataset_name}")

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
            if key not in ('language_instruction', 'modality_positions', 'action_positions'):
                batched[key] = torch.stack(value, dim=0)
        return batched


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    metas_path = [
        # "./meta_grounding_data/coco/coco_train2017_meta.json",
        # "./meta_grounding_data/coco/coco_val2017_meta.json",
        "./meta_grounding_data/lvis/lvis_train2017_meta.json",
        "./meta_grounding_data/lvis/lvis_val2017_meta.json",
        "./meta_grounding_data/ade20k/ade20k_training_meta.json",
        "./meta_grounding_data/ade20k/ade20k_validation_meta.json",
    ]

    dataset = GroundingDataset(
        metas_path=metas_path,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=880,
        image_size=(336, 320),
        num_image_tokens=420+1,
    )
    train_dataloader_img_edit = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn,
                                      shuffle=True, num_workers=0)

    save_dir = "./vis_grounding_dataset"
    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(train_dataloader_img_edit):
        print(f"[BATCH {i}]")
        print("text_tokens", data['text_tokens'].shape)
        print("images", data['images'].shape)
        print(data['modality_positions'][0])

        images = data['images']
        texts = data['language_instruction']
        for j in range(images.shape[0]):
            sample_prefix = os.path.join(save_dir, f"batch_{i:04d}_sample_{j:02d}_{texts[j]}")
            save_image(images[j], f"{sample_prefix}.jpg", nrow=2, normalize=True, value_range=(-1, 1))

        print(f"saved images to {save_dir}")
        print()
