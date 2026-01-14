# coding=utf-8
import collections
import json
import os
import random
import glob
import numpy as np
import torch
import av
from torch.utils.data import IterableDataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets_vla.utils import get_image_prompt_response_aokvqa, get_image_prompt_response_RoboVQA

class VQADataset(IterableDataset):
    """
    Dataset for Single-image VQA.
    Directly uses the original label JSON files.
    """
    def __init__(
            self,
            metas_paths,  # List of paths to the original label JSON files
            text_tokenizer,
            showo_token_ids,
            max_seq_len,
            image_size,
            num_image_tokens,
            training=True,
    ) -> None:
        self.all_samples = []
        
        if isinstance(metas_paths, str):
            metas_paths = [metas_paths]

        for label_path in metas_paths:
            if 'aokvqa' in label_path:
                with open(label_path, 'r') as f:
                    labels = json.load(f)

                print(f"== [{label_path}] Loaded {len(labels)} items")
                for item in labels:
                    data_root = os.path.join(os.path.dirname(label_path), f"{item['split']}2017")
                    self.all_samples.append({
                        'data_root': data_root,
                        'ann': item,
                    })
            elif 'RoboVQA' in label_path:
                # label_path is something like /home/hyx/datasets/RoboVQA/json/val
                # Load all jsonline files within it
                json_files = sorted(glob.glob(os.path.join(label_path, "*.json")))
                count = 0
                for json_file in json_files:
                    with open(json_file, 'r') as f:
                        for line in f:
                            item = json.loads(line)
                            # The videos are in ../../videos relative to the json/ directory
                            base_dir = os.path.normpath(label_path)
                            data_root = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "videos")
                            self.all_samples.append({
                                'data_root': data_root,
                                'ann': item,
                            })
                            count += 1
                print(f"== [{label_path}] Loaded {count} items from {len(json_files)} files")
            else:
                raise NotImplementedError

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            samples = self.all_samples
        else:
            per_worker = int(np.ceil(len(self.all_samples) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.all_samples))
            samples = self.all_samples[iter_start:iter_end]

        if self.training:
            random.shuffle(samples)

        for item in samples:
            data_root = item['data_root']
            ann = item['ann']

            if 'aokvqa' in data_root:
                image, prompt, response = get_image_prompt_response_aokvqa(data_root, ann)
            elif 'RoboVQA' in data_root:
                image, prompt, response = get_image_prompt_response_RoboVQA(data_root, ann)
                if image is None:
                    continue
            else:
                raise NotImplementedError

            image = self.image_transform(image)
            # [C H W] -> [1 C H W]
            image = image.unsqueeze(0)

            text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_seq(prompt, response)

            yield {
                'language_instruction': f"{prompt}{response}",
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
            }

        if self.training:
            yield from self.__iter__()

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

    dataset = VQADataset(
        metas_paths="/home/hyx/datasets/RoboVQA/json/val",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=872,
        image_size=(336, 320),
        num_image_tokens=420+1,
        training=False,
    )
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, num_workers=4)

    output_dir = "vis_vqa"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}...")

    sample_count = 0
    for i, data in enumerate(dataloader):
        if i >= 100:
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
        print()
