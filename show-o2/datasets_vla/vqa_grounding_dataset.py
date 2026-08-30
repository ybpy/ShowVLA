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
from PIL import Image

import io
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from datasets_vla.utils import check_can_use_bbox, get_bboxes_text


class VQAGroundingDataset(Dataset):
    """Dataset for Object Detection/Segmentation based on VQA."""

    def __init__(
            self,
            metas_path,
            text_tokenizer,
            showo_token_ids,
            max_seq_len,
            image_size,
            num_image_tokens,
    ) -> None:

        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""

        self.all_datalist = []
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f: meta = json.load(f)
            dataset_name = meta['dataset_name']
            datalist = meta['datalist']
            if 'ade20k' in dataset_name.lower():
                continue

            valid_datalist = []
            for json_path in datalist:
                data_dict = json.load(open(json_path))
                category_2_instances = data_dict["anns"]
                is_valid = False
                for instances in category_2_instances.values():
                    if check_can_use_bbox(instances):
                        is_valid = True
                        break
                if is_valid:
                    valid_datalist.append([dataset_name, json_path])
            self.all_datalist.extend(valid_datalist)
            
            print(f"== [{file}] VQA Dataset {dataset_name} with {len(valid_datalist)} images")
            

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

    def format_obs_text_seq(self, prompt: str, response: str):
        text_tokens = []
        text_labels = []
        modality_positions = []

        cur_len = 1 # bos token
        
        # Image part
        text_tokens.extend([self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id])
        modality_positions.append((cur_len + 1, self.num_image_tokens))
        cur_len = cur_len + 1 + self.num_image_tokens + 1

        # Prompt part
        prompt_tokens = self.text_tokenizer(prompt, add_special_tokens=False).input_ids
        text_tokens.extend(prompt_tokens)
        cur_len += len(prompt_tokens)

        if cur_len >= self.max_seq_len:
            print(f"{cur_len} >= {self.max_seq_len}\n{prompt}{response}", flush=True)
            return None

        # Response part
        response_tokens = self.text_tokenizer(response, add_special_tokens=False).input_ids
        text_tokens.extend(response_tokens)
        
        # Labels: -100 for everything except the response
        text_labels = [-100] * (len(text_tokens) - len(response_tokens)) + response_tokens

        # BOS and EOS
        text_tokens = [self.bos_id] + text_tokens + [self.eos_id]
        text_labels = [-100] + text_labels + [self.eos_id]
        
        # Padding or Truncation
        # assert len(text_tokens) <= self.max_seq_len, f"{len(text_tokens)}\n{prompt}{response}"
        if len(text_tokens) > self.max_seq_len:
            text_tokens = text_tokens[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]
        else:
            padding_len = self.max_seq_len - len(text_tokens)
            text_tokens.extend([self.pad_id] * padding_len)
            text_labels.extend([-100] * padding_len)

        text_tokens = torch.tensor(text_tokens)
        text_labels = torch.tensor(text_labels)

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __len__(self) -> int:
        return len(self.all_datalist)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        dataset_name, json_path = self.all_datalist[idx]
        data_dict = json.load(open(json_path))

        img_path = data_dict["img_path"]
        img = Image.open(img_path).convert('RGB')
        img = self.image_aug(img)

        img_h, img_w = data_dict["height"], data_dict["width"]
        assert img.size == (img_w, img_h), f"img.size {img.size} != ({img_w}, {img_h})"

        if 'coco' in dataset_name.lower():
            category_2_instances = data_dict["anns"]
            valid_categories = [k for k, v in category_2_instances.items() if check_can_use_bbox(v)]
            category = np.random.choice(valid_categories)
            instances = category_2_instances[category]
            prompt = f"Detect all {category}(s) in the image and output the bounding box(s) in text format.\n"
            bbox_xywh = [ann['bbox'] for ann in instances]
            response = get_bboxes_text(img_w, img_h, bbox_xywh)
        else:
            raise NotImplementedError(f"Unsupported grounding dataset: {dataset_name}")

        text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_seq(prompt, response)

        img = self.image_transform(img)
        # [C H W] -> [1 C H W]
        image = img.unsqueeze(0)

        return {
            'language_instruction': f"{prompt}{response}",
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
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        # llm_name="llama3"
        llm_name="qwen2_5"
    )

    dataset = VQAGroundingDataset(
        metas_path="./meta_grounding_data/",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=872,
        image_size=(336, 320),
        num_image_tokens=420+1,
    )
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn,
                                      shuffle=False, num_workers=0)


    from torchvision.utils import save_image
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}...")

    sample_count = 0
    for i, data in enumerate(dataloader):
        if i >= 1000:
            break
        print(f"[BATCH {i}]")

        images = data['images']  # [B, 1, C, H, W]
        texts = data['language_instruction']

        for j in range(images.shape[0]):
            img = images[j, 0]
            # Denormalize
            img = img * 0.5 + 0.5

            # Clean up text for filename
            clean_text = texts[j].replace("/", "_").replace("(", "").replace(")", "").replace(".", "").replace("\n", " ")
            clean_text = clean_text[:200]  # Limit filename length
            filename = f"sample_{sample_count}_{clean_text}.jpg"
            save_path = os.path.join(output_dir, filename)

            save_image(img, save_path)
            print(f"{texts[j]}\n")
            sample_count += 1
        
        print(flush=True)
