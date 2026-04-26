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
from datasets_vla.utils import BBOX_COLORS, MASK_COLORS
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def decode_jpeg_object(obj):
    arr = np.asarray(obj, dtype=np.uint8).reshape(-1)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG bytes from HDF5 element.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

class RobotGroundingDataset(IterableDataset):
    """
    Iterable Dataset for Robot Grounding task, supporting bbox, segment_mask, rand, and combine visualization modes.
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
        vis_mode: str = "combine", # "bbox", "segment_mask", "rand", "combine"
        mask_color_weight: float = 0.5, # Consistent with GroundingDataset
        prob_bbox: float = 0.5, # Used when vis_mode is "rand" to decide between bbox and mask
        num_samples_per_video: int = 4, # Number of random frames to sample per video
    ):
        if isinstance(meta_paths, str):
            meta_paths = [meta_paths]
            
        self.datalist = []
        for path in meta_paths:
            with io.BytesIO(fileio.get(path)) as f: meta = json.load(f)
            print(f"== [{path}] with {len(meta['datalist'])} trajs", flush=True)
            self.datalist.extend(meta['datalist'])
        
        self.vis_mode = vis_mode
        self.mask_color_weight = mask_color_weight
        self.prob_bbox = prob_bbox
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
        
        # 统一使用来自 utils.py 的 MASK_COLORS
        self.colors_rgb = list(MASK_COLORS.values())
        self.color_names = list(MASK_COLORS.keys())

        self.epoch = 0
        self.num_processes = 1
        self.process_index = 0

    def set_epoch(self, epoch: int, num_processes: int = 1, process_index: int = 0):
        self.epoch = epoch
        self.num_processes = num_processes
        self.process_index = process_index

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

    def _process_sample(self, img_jpeg_bytes, bbox_xywh, rle_data, object_names):
        # Data decoding and processing
        img_rgb = decode_jpeg_object(img_jpeg_bytes)
        
        # Randomly choose color
        color_idx = np.random.randint(len(self.colors_rgb))
        color_rgb = self.colors_rgb[color_idx]
        color_name = self.color_names[color_idx]
        
        # Handle modes
        current_vis_mode = self.vis_mode
        task_mode = self.vis_mode
        
        if self.vis_mode == "rand":
            if np.random.rand() < self.prob_bbox:
                current_vis_mode = "bbox"
                task_mode = "bbox"
            else:
                current_vis_mode = "segment_mask"
                task_mode = "segment_mask"
        elif self.vis_mode == "combine":
            current_vis_mode = "combine"
            task_mode = "combine"

        filtered = self._filter_grounding_to_frame(bbox_xywh, rle_data, object_names)
        if filtered is None:
            return None
        bbox_xywh, rle_data, object_names = filtered

        vis_img = self._get_visualized_image(img_rgb, bbox_xywh, rle_data, current_vis_mode, color_rgb, object_names)
        # Generate instruction
        object_names_unique = list(set(object_names))
        assert 1<= len(object_names_unique) <= 2, object_names_unique
        obj_str = ", ".join(object_names_unique)
        if task_mode == "bbox":
            text = f"Mark {obj_str} in the image with {color_name} bounding box:"
        elif task_mode == "segment_mask":
            text = f"Segment {obj_str} in the image with {color_name} mask:"
        else: # combine
            text = f"Mark with {color_name} bounding box and segment mask for {obj_str} in the image:"

        # Tokenization and sequence formatting
        text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_img_text_tgt_img_seq(text)

        # Image transformation
        img_pil = Image.fromarray(img_rgb)
        vis_img_pil = Image.fromarray(vis_img)
        
        img_tensor = self.image_transform(img_pil)
        tgt_img_tensor = self.image_transform(vis_img_pil)
        images = torch.stack([img_tensor, tgt_img_tensor], dim=0)  # [2, C, H, W]

        return {
            'language_instruction': text,
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

    def _get_visualized_image(self, img_rgb, bbox_xywh, rle_list, mode, color_rgb, object_names=None):
        img = img_rgb.copy()
        
        # Determine what to draw based on mode
        draw_bbox = mode in ["bbox", "combine"]
        draw_mask = mode in ["segment_mask", "combine"]

        if draw_bbox:
            for bbox in bbox_xywh:
                x, y, bw, bh = bbox.astype(np.int32)
                assert bw > 0 and bh > 0
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color_rgb, 2)

        if draw_mask:
            # Draw masks
            mask_overlay = img.copy()
            for i, rle_str in enumerate(rle_list):
                assert rle_str

                if isinstance(rle_str, bytes):
                    rle_str = rle_str.decode("utf-8")
                rle_dict = json.loads(rle_str)
                if isinstance(rle_dict["counts"], str):
                    rle_dict["counts"] = rle_dict["counts"].encode("utf-8")
                mask = mask_utils.decode(rle_dict)
                mask_overlay[mask > 0] = color_rgb
            
            img = cv2.addWeighted(img, (1.0 - self.mask_color_weight), mask_overlay, self.mask_color_weight, 0)
            
        return img

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        import collections
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
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    meta_paths = [
        "/home/hyx/ShowVLA/show-o2/grounding_data_ann/meta_libero/split/libero_90_grounding_metainfo_0412.json",
        "/home/hyx/ShowVLA/show-o2/grounding_data_ann/meta_libero/split/libero_spatial_grounding_metainfo_0421.json",
    ]
    
    dataset = RobotGroundingDataset(
        meta_paths=meta_paths,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=880,
        image_size=(336, 320),
        num_image_tokens=420+1,
        vis_mode="combine",
        num_samples_per_video=4,
    )
    
    # IterableDataset does not support Sampler or shuffle
    train_dataloader = DataLoader(
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

    for i, data in enumerate(train_dataloader):
        print(f"[BATCH {i}]")
        print("text_tokens:", data["text_tokens"].shape)
        print("images:", data["images"].shape)
        bsz = data["images"].shape[0]
        # data['images'] shape: [B, 2, C, H, W] — 对每个样本左右拼接 orig | vis 并保存
        for b in range(bsz):
            print(f"  sample[{b}] language_instruction:", data["language_instruction"][b])
            print(f"  sample[{b}] modality_positions:", data["modality_positions"][b])
            orig_img = to_numpy_img(data["images"][b, 0])
            vis_img = to_numpy_img(data["images"][b, 1])
            combined_img = np.hstack([orig_img, vis_img])
            stem = safe_filename(data["language_instruction"][b])
            out_path = f"debug_batch_{i}_sample_{b}_{stem}.png"
            cv2.imwrite(out_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
            print(f"  Saved {out_path}")

        if i >= 2000:
            break  # 只测试前几个 batch
