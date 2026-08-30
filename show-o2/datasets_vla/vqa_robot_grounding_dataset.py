import json
import os
import h5py
import io
from mmengine import fileio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
from PIL import Image
import cv2
from pycocotools import mask as mask_utils
from typing import Dict, List, Any, Optional, Tuple
from datasets_vla.utils import get_bboxes_text
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def _swap_task_instruction_left_right(text: str) -> str:
    """Swap the phrases ``left`` and ``right`` in a task instruction string.
    """
    placeholder = "__LIBERO_TMP_WAS_LEFT__"
    assert placeholder not in text, text
    return text.replace("left", placeholder).replace("right", "left").replace(placeholder, "right")

def decode_jpeg_object(obj):
    arr = np.asarray(obj, dtype=np.uint8).reshape(-1)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG bytes from HDF5 element.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

class VQARobotGroundingDataset(IterableDataset):
    """
    Iterable Dataset for VQA-based Robot Grounding task.
    Supports random sampling N frames from each video to form N samples.
    """
    def __init__(
        self,
        meta_paths: List[str],
        text_tokenizer: Any,
        showo_token_ids: Dict[str, int],
        max_seq_len: int,
        image_size: Tuple[int, int],
        num_image_tokens: int,
        num_samples_per_video: int = 4, # Number of random frames to sample per video
    ):
        if isinstance(meta_paths, str):
            meta_paths = [meta_paths]
            
        self.datalist = []
        for path in meta_paths:
            with io.BytesIO(fileio.get(path)) as f: meta = json.load(f)
            print(f"== [{path}] with {len(meta['datalist'])} trajs", flush=True)
            self.datalist.extend(meta['datalist'])
        
        self.num_samples_per_video = num_samples_per_video
        
        # Consistent with GroundingDataset
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_height, self.image_width = image_size
        self.num_image_tokens = num_image_tokens

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.epoch = 0
        self.num_processes = 1
        self.process_index = 0

    def set_process_info(self, num_processes: int = 1, process_index: int = 0):
        self.num_processes = num_processes
        self.process_index = process_index

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # 1. 计算全局唯一的总并行度和当前 worker 的全局唯一 ID
        total_parallel_size = self.num_processes * num_workers
        global_worker_id = self.process_index * num_workers + worker_id

        # 2. 使用步长取模进行分片，确保数据在所有进程和 worker 之间均匀分布且不重复
        indices = [i for i in range(len(self.datalist)) if i % total_parallel_size == global_worker_id]
        
        # 3. 使用 epoch 和 global_worker_id 设置随机种子，确保每个 epoch 的 shuffle 不同
        random.seed(self.epoch + global_worker_id)
        random.shuffle(indices)

        for idx in indices:
            h5_path = self.datalist[idx]
            with h5py.File(h5_path, 'r') as f:
                rgb_ds = f["rgb_comb"]
                g = f["grounding"]
                num_frames = len(rgb_ds)

                assert num_frames >= self.num_samples_per_video
                
                # 1. 随机采样并排序（排序能极大地加速 HDF5 的底层读取速度）
                sampled_indices = np.random.choice(num_frames, self.num_samples_per_video, replace=False)
                sampled_indices.sort()

                # 2. 预取元数据
                object_names = []
                for n in g["object_names"][()]:
                    object_name = n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    object_name = object_name.replace('_', ' ').strip()
                    object_name = object_name.replace('black bowl', 'gray bowl')
                    object_name = _swap_task_instruction_left_right(object_name)
                    object_name = object_name.replace('top drawer of the wooden cabinet', 'top drawer')
                    object_name = object_name.replace('、', '')
                    if object_name.startswith('the ') or object_name.startswith('The '):
                        object_name = object_name[4:]
                    if object_name.endswith('rameki'):
                        object_name = object_name + 'n'
                    if 'libero' in h5_path and '_turn_on_the_stove_demo_' in h5_path and object_name == 'stove':
                        object_name = 'stove knob'

                    object_names.append(object_name)

                if 'libero' in h5_path and not(len(object_names) <= 4 and len(object_names) % 2 == 0):
                    assert 'KITCHEN_SCENE5_close_the_top_drawer' in h5_path, f"{h5_path}: {object_names}"

                # 3. 核心优化：批量读取所有需要的帧到内存，然后立即关闭文件
                # 如果存储的是 JPEG 字节流，内存占用很小
                all_img_bytes = rgb_ds[sampled_indices]
                all_bboxes = g["bbox_xywh"][sampled_indices]
                all_rles = g["rle"][sampled_indices]

            # 4. 文件句柄已关闭，在内存中迭代并 yield，不会阻塞 IO
            # sampled_indices是原视频帧号
            sampled_local_indices = list(range(len(sampled_indices)))
            random.shuffle(sampled_local_indices)
            for i in sampled_local_indices:
                sample = self._process_sample(
                    all_img_bytes[i], 
                    all_bboxes[i], 
                    all_rles[i], 
                    object_names
                )
                if sample is not None:
                    yield sample

        # 每跑完一轮分片后自增，下一轮 __iter__ 使用新 seed（各 DataLoader worker 进程内各自维护）
        self.epoch = self.epoch + 1

    def _process_sample(self, img_jpeg_bytes, bbox_xywh, rle_data, object_names):
        # Data decoding and processing
        img_rgb = decode_jpeg_object(img_jpeg_bytes)
        img_h, img_w = img_rgb.shape[:2]
        
        filtered = self._filter_grounding_to_frame(bbox_xywh, rle_data, object_names)
        if filtered is None:
            return None
        bbox_xywh, rle_data, object_names = filtered

        object_names_unique = list(set(object_names))
        category = np.random.choice(object_names_unique)
        bbox_xywh_of_category = []
        for bbox, object_name in zip(bbox_xywh, object_names):
            if object_name == category:
                bbox_xywh_of_category.append(bbox)

        prompt = f"Detect all {category}(s) in the image and output the bounding box(s) in text format.\n"
        response = get_bboxes_text(img_w, img_h, bbox_xywh_of_category)

        text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_obs_text_seq(prompt, response)
        
        # Image transformation
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.image_transform(img_pil)
        images = img_tensor.unsqueeze(0)

        return {
            'language_instruction': f"{prompt}{response}",
            'text_tokens': text_tokens,
            'text_labels': text_labels,
            'images': images,
            'modality_positions': modality_positions,
            'text_masks': text_mask,
            'image_masks': image_mask,
        }

    def _filter_grounding_to_frame(
        self,
        bbox_xywh: np.ndarray,
        rle_data,
        object_names: List[str],
    ) -> Optional[Tuple[np.ndarray, List[Any], List[str]]]:
        """
        Keep only objects with a positive-size bbox on this frame (has_bbox only).
        """
        if bbox_xywh is None or len(bbox_xywh) == 0:
            return None
        assert len(object_names) == len(bbox_xywh) == len(rle_data)
        n = len(bbox_xywh)
        keep: List[int] = []
        for i in range(n):
            x, y, bw, bh = bbox_xywh[i].astype(np.int32)
            has_bbox = bw > 0 and bh > 0
            if has_bbox:
                keep.append(i)
        if not keep:
            return None
        keep_idx = np.array(keep, dtype=np.int64)
        bbox_out = np.asarray(bbox_xywh[keep_idx], dtype=bbox_xywh.dtype)
        rle_out = [rle_data[i] for i in keep]
        names_out = [object_names[i] for i in keep]
        return bbox_out, rle_out, names_out

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        import collections
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

    meta_paths = [
        "/home/hyx/ShowVLA/show-o2/grounding_data_ann/meta_libero/split/libero_90_grounding_metainfo_0412.json",
        "/home/hyx/ShowVLA/show-o2/grounding_data_ann/meta_libero/split/libero_spatial_grounding_metainfo_0421.json",
    ]
    
    dataset = VQARobotGroundingDataset(
        meta_paths=meta_paths,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=880,
        image_size=(336, 320),
        num_image_tokens=420+1,
        num_samples_per_video=4,
    )
    
    # IterableDataset does not support Sampler or shuffle
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    def to_numpy_img(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        return img.astype(np.uint8)

    def safe_filename(s, max_len=120):
        bad = '\\/:*?"<>|\n\r'
        out = str(s)
        for c in bad:
            out = out.replace(c, "_")
        return out[:max_len]

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