# EO-Data1.5M (17 Subsets)
# ├── 🔄 Interleaved Manipulation Data (5 subsets)
# │   ├── interleave-free_chat      # Free-form reasoning + action
# │   ├── interleave-random_qa      # Random QA + action
# │   ├── interleave-temporal       # Temporal reasoning + action
# │   ├── interleave-trajectory     # Trajectory prediction + action
# │   └── interleave-video_caption  # Video captioning + action
# │
# └── 💬 Embodied Reasoning QA Data (12 subsets)
#     ├── Temporal Reasoning (7 subsets)
#     │   ├── qa-task_planning          # Task decomposition & subtask planning
#     │   ├── qa-episode_caption        # Robot action description
#     │   ├── qa-affordance_qa          # Action feasibility assessment
#     │   ├── qa-process_verification   # Completed action recognition
#     │   ├── qa-subtask_qa             # Subtask QA
#     │   ├── qa-failure_detection      # Unsuccessful execution identification
#     │   └── qa-physical_common_sense  # Physical world commonsense
#     │
#     └── Spatial Reasoning (5 subsets)
#         ├── qa-trajectory_qa          # Trajectory reasoning & prediction
#         ├── qa-points_qa              # Point localization
#         ├── qa-multiview_qa           # Cross-view spatial understanding
#         ├── qa-object_referring_qa    # Object grounding
#         └── qa-relation_reasoning     # Spatial relationship understanding

# coding=utf-8
import collections
import json
import os
import random
import glob
import re
import numpy as np
import torch
import av
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import load_dataset, concatenate_datasets


class EO_VQADataset(Dataset):
    
    def __init__(
            self,
            eo_subsets,
            text_tokenizer,
            showo_token_ids,
            max_seq_len,
            image_size,
            num_image_tokens,
            training=True,
    ) -> None:
        
        if isinstance(eo_subsets, str):
            eo_subsets = [eo_subsets]

        list_subsets = []
        for subset_name in eo_subsets:
            subset = load_dataset("IPEC-COMMUNITY/EO-Data1.5M", name=subset_name,
                        split='train', keep_in_memory=False)
            # for i in range(len(subset)):
            #     print(f"{i}/{len(subset)}", flush=True)
            #     assert len(subset[i]['image']) == 1
            list_subsets.append(subset)
            print(f"== [{subset_name}] Loaded {len(subset)} items")
        
        # Concat all subsets using datasets.concatenate_datasets for better performance
        if len(list_subsets) > 1:
            self.data = concatenate_datasets(list_subsets)
        else:
            self.data = list_subsets[0]


        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.training = training

        if isinstance(image_size, int):
            self.image_height, self.image_width = image_size, image_size
        else:
            assert len(image_size) == 2
            self.image_height, self.image_width = image_size[0], image_size[1]
        self.num_image_tokens = num_image_tokens

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

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

        # Response part
        response_tokens = self.text_tokenizer(response, add_special_tokens=False).input_ids
        text_tokens.extend(response_tokens)
        
        # Labels: -100 for everything except the response
        text_labels = [-100] * (len(text_tokens) - len(response_tokens)) + response_tokens

        # BOS and EOS
        text_tokens = [self.bos_id] + text_tokens + [self.eos_id]
        text_labels = [-100] + text_labels + [self.eos_id]

        assert len(text_tokens) == len(text_labels)
        
        # Padding or Truncation
        assert len(text_tokens) <= self.max_seq_len, f"{len(text_tokens)}\n{prompt}{response}"
        if len(text_tokens) > self.max_seq_len:
            text_tokens = text_tokens[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]
        else:
            padding_len = self.max_seq_len - len(text_tokens)
            text_tokens.extend([self.pad_id] * padding_len)
            text_labels.extend([-100] * padding_len)

        text_tokens = torch.tensor(text_tokens)
        text_labels = torch.tensor(text_labels)
        modality_positions = torch.tensor(modality_positions)

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        images = sample['image']
        assert len(images) == 1
        if len(images) > 1:
            return self.__getitem__(random.randint(0, len(self) - 1))
        image = images[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        prompt = sample['conversation'][0]['value']
        response = sample['conversation'][1]['value']
        if len(prompt) + len(response) > 666:
            return self.__getitem__(random.randint(0, len(self) - 1))
        assert prompt.startswith('<image>') and not prompt.startswith('<image><image>'), sample['conversation']
        prompt = prompt[len('<image>'):]
        prompt += '\n'

        image = self.image_transform(image)
        # [C H W] -> [1 C H W]
        image = image.unsqueeze(0)

        text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_seq(prompt, response)

        return {
            'language_instruction': f"{prompt}{response}",
            'text_tokens': text_tokens,
            'text_labels': text_labels,
            'images': image,
            'modality_positions': modality_positions,
            'text_masks': text_mask,
            'image_masks': image_mask,
        }

    def collate_fn(self, batch: list) -> dict:
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
    from torchvision.utils import save_image

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    dataset = EO_VQADataset(
        eo_subsets=[
            'qa-task_planning',
            'qa-episode_caption',
            # 'qa-affordance_qa',
            # 'qa-process_verification',
            # 'qa-physical_common_sense',
            'qa-relation_reasoning',
        ],
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=600,
        image_size=(336, 320),
        num_image_tokens=420+1,
        training=True,
    )
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, num_workers=1, shuffle=True)

    output_dir = "vis_vqa_eo-3"
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
