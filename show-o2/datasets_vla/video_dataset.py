# coding=utf-8
import collections
import json
import os
import random
import numpy as np
import torch
import av
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class SSv2Dataset(IterableDataset):
    """
    Something-Something-V2 Dataset as an IterableDataset.
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
            qdur=1.0,
    ) -> None:
        self.all_samples = []
        
        if isinstance(metas_paths, str):
            metas_paths = [metas_paths]

        for label_file in metas_paths:
            # The video frames are expected to be in a directory at the same level as the 'labels' directory
            # e.g., /home/hyx/datasets/Something-Something-V2/labels/train.json
            # -> /home/hyx/datasets/Something-Something-V2/20bn-something-something-v2/
            data_root = os.path.join(os.path.dirname(os.path.dirname(label_file)), '20bn-something-something-v2')
            
            with open(label_file, 'r') as f:
                labels = json.load(f)

            print(f"== [{label_file}] Loaded {len(labels)} videos")
            for item in labels:
                self.all_samples.append({
                    'data_root': data_root,
                    'id': item['id'],
                    'label': item['label'],
                })

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.training = training
        self.qdur = qdur

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
        
        # Language command
        if text.endswith('.'):
            text = text + suffix
        else:
            text = text + '.' + suffix
        
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

        assert len(text_tokens) == len(text_labels) <= self.max_seq_len, f"text: {text}\nlen(text_tokens): {len(text_tokens)}, len(text_labels): {len(text_labels)}, self.max_seq_len: {self.max_seq_len}"
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

    def __iter__(self):
        samples = self.all_samples

        if self.training:
            random.shuffle(samples)

        for item in samples:
            video_path = os.path.join(item['data_root'], f"{item['id']}.webm")
            text = item['label']

            if len(text) > 100:
                continue

            if text.startswith("holding"):
                continue
            
            # Use PyAV to read frames from webm
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate)
            assert 0 < fps <= 60, f"Invalid fps ({fps}) of video {video_path}"
            frames = [frame.to_image() for frame in container.decode(video=0)]

            # Use qdur and fps to determine idx2 (future image)
            offset = int(self.qdur * fps)
            assert offset > 0
            if len(frames) <= offset:
                continue

            # Time-wise center-clip the video
            while len(frames) > fps * 2.5:
                frames = frames[int(fps/4):-int(fps/4)]
            if len(frames) > fps * 2:
                frames = frames[:-int(fps/4)]

            if text.startswith("pretending"):
                frames = frames[:-int(fps/2)]
            elif text.startswith("showing"):
                offset = int(len(frames) * 0.9)
                

            # Randomly pick idx1, then idx2 is idx1 + offset
            # If video is shorter than offset, clamp idx2 to the last frame
            idx1 = random.randint(0, max(0, len(frames) - offset - 1))
            idx2 = min(idx1 + offset, len(frames) - 1)
            
            assert idx1 < idx2

            img1 = self.image_transform(frames[idx1])
            img2 = self.image_transform(frames[idx2])
            image = torch.stack([img1, img2], dim=0)

            text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_future_seq(text)

            yield {
                'language_instruction': text,
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


def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)

def create_video_dataset_loader(
    num_workers,
    batch_size, 
    metas_paths, 
    text_tokenizer,
    showo_token_ids,
    max_seq_len,
    image_size,
    num_image_tokens,
    training,
):
    video_dataset = SSv2Dataset(metas_paths, text_tokenizer=text_tokenizer, showo_token_ids=showo_token_ids,
        max_seq_len=max_seq_len, image_size=image_size, num_image_tokens=num_image_tokens, training=training)
    return DataLoader(
        video_dataset, 
        batch_size=batch_size,
        collate_fn=video_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )


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

    dataloader = create_video_dataset_loader(
        num_workers=4,
        batch_size=2,
        metas_paths="/home/hyx/datasets/Something-Something-V2/labels/train.json",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=880,
        image_size=(336, 320),
        num_image_tokens=420+1,
        training=True,
    )

    output_dir = "vis_ssv2_"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}...")

    sample_count = 0
    for i, data in enumerate(dataloader):
        if i >= 10000:
            break
        print(f"[BATCH {i}]")
        
        images = data['images'] # [B, 2, C, H, W]
        texts = data['language_instruction']

        for j in range(images.shape[0]):
            img1 = images[j, 0]
            img2 = images[j, 1]
            # Concat horizontally: [C, H, W1+W2]
            combined = torch.cat([img1, img2], dim=2)
            # Denormalize
            combined = combined * 0.5 + 0.5
            
            # Clean up text for filename
            clean_text = texts[j].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            filename = f"sample_{sample_count}_{clean_text}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            save_image(combined, save_path)
            print(f"  - Saved: {filename}")
            sample_count += 1
