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

import io, numpy as np, pyarrow.parquet as pq, av, cv2, os, random, re
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict
import torch

from pycocotools import mask as maskUtils


BBOX_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (238, 230, 0),
    "purple": (128, 0, 128),
    "orange": (255, 140, 0),
    "cyan": (0, 230, 230),
    "magenta": (255, 0, 255),
    "lime": (0, 255, 0),
    "pink": (255, 20, 147),
    "indigo": (75, 0, 130),
    "gold": (245,191,35),
    "olive": (128, 128, 0),
    "violet": (148, 0, 211),
    "khaki": (189, 183, 107),
}

MASK_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (238, 230, 0),
    "purple": (128, 0, 128),
    "orange": (255, 140, 0),
    "cyan": (0, 230, 230),
    "magenta": (255, 0, 255),
    "lime": (0, 255, 0),
    "pink": (255, 20, 147),
    "indigo": (75, 0, 130),
    "gold": (245,191,35),
    "olive": (128, 128, 0),
    "violet": (148, 0, 211),
    "khaki": (189, 183, 107),
}

def try_get_img_with_bbox(img, instances, color):
    img = np.array(img)
    img_h, img_w, c = img.shape
    
    bboxes = []
    for ann in instances:
        if ann["iscrowd"]:
            return None
        segm = ann["segmentation"]
        assert type(segm) == list
        if len(segm) > 1:
            return None
        
        bbox = ann["bbox"]
        bboxes.append(bbox)
    
    if len(bboxes) > 8:
        return None

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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return Image.fromarray(img)

def get_img_with_segment_mask(img, h, w, instances, color, mask_color_weight=0.5):
    img = np.array(img)
    comb_mask = None
    for ann in instances:
        mask = annToMask(ann, h, w)
        if comb_mask is None:
            comb_mask = mask
        else:
            comb_mask = comb_mask | mask

    colored_mask = img.copy()
    colored_mask[comb_mask==1] = color

    img = cv2.addWeighted(img, (1.0 - mask_color_weight), colored_mask, mask_color_weight, 0)
    
    return Image.fromarray(img)

def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, h, w)
    m = maskUtils.decode(rle)
    return m


def get_img_with_segment_mask_ade20k(img, segm, cat_id, mask_color_rgb, mask_color_weight=0.5):
    img = np.array(img)
    segm = np.array(segm)

    colored_mask = img.copy()
    colored_mask[segm == cat_id] = mask_color_rgb
    img = cv2.addWeighted(img, (1.0 - mask_color_weight), colored_mask, mask_color_weight, 0)
    
    return Image.fromarray(img)


def get_image_prompt_response_aokvqa(data_root, ann):
    image_path = os.path.join(data_root, f"{ann['image_id']:012d}.jpg")
    image = Image.open(image_path).convert('RGB')
    question = ann['question']
    
    choices = ann['choices']
    options_text = "\nOptions:\n"
    
    # Randomize the style of serial mark
    marker_type = random.choice(['alpha', 'numeric'])
    decoration = random.choice(['bracket', 'dot'])
    
    marks = []
    for i, choice in enumerate(choices):
        choice = choice.strip()
        if marker_type == 'alpha':
            label = chr(65 + i) # A, B, C...
        else:
            label = str(i + 1) # 1, 2, 3...
            
        if decoration == 'bracket':
            mark = f"({label})"
        else: # dot
            mark = f"{label}."
            
        options_text += f"{mark} {choice}\n"
        marks.append(mark)
    
    correct_idx = ann['correct_choice_idx']
    answer = choices[correct_idx].strip()
    correct_mark = marks[correct_idx]
    longest_rationale = max(ann['rationales'], key=len).strip()
    if not longest_rationale.endswith('.'):
        longest_rationale += '.'

    prompt = f"Question: {question}{options_text}\nAnswer with reasoning:\n"
    response = f"{longest_rationale} So, the answer is {correct_mark} {answer}."

    return image, prompt, response

def extract_robovqa_response(task_text):
    """从单个任务文本中提取清洗后的 response，如果是 'done' 则返回 None"""
    if '<PRED>' not in task_text:
        return None
    _, response_part = task_text.split('<PRED>', 1)
    # Remove 'A: ' prefix
    response = response_part.replace('A: ', '', 1).strip()
    # Remove </PRED> and any trailing whitespace
    response = response.replace('</PRED>', '').strip()
    # Remove all nested tags like <PRED:ANSWER>, <PRED:BINARY>, etc.
    response = re.sub(r'<[^>]+>', '', response).strip()

    response = response.replace('\n', ' ')

    # Special processing for remaining5_planning_with_context20: extract only the first step
    if task_text.startswith("remaining5_planning_with_context20"):
        match = re.search(r'1-\s*([\s\S]*?)(?:\s*\d+-|$)', response)
        if match:
            response = match.group(1).strip()
    
    if not response or response.lower() == 'done':
        return None
    return response

def get_image_prompt_response_RoboVQA(
        data_root,
        ann, 
        primary_tasks=(
            "planning:freeform",
            "immediate_planning_with_context20",
            "remaining5_planning_with_context20",
        ),
        secondary_tasks=(
            "success",
        ),
        tertiary_tasks=(
            "affordance:discriminative",
        ),
    ):
    video_path = os.path.join(data_root, ann['video'])
    container = av.open(video_path)
    # Get the last frame
    try:
        last_frame = None
        for frame in container.decode(video=0):
            last_frame = frame
        if last_frame is not None:
            image = last_frame.to_image()
        else:
            raise ValueError(f"No frames found in video: {video_path}")
    finally:
        container.close()
    
    text = ann['text']
    # RoboVQA text can contain multiple tasks, each starting with <task:...>
    # We split by <task: and filter out empty strings
    tasks = [t for t in text.split('<task:') if t.strip()]
    random.shuffle(tasks)

    task_text = None
    for priority_tasks in (primary_tasks, secondary_tasks, tertiary_tasks):
        for t in tasks:
            if any(t.startswith(name) for name in priority_tasks) and extract_robovqa_response(t) is not None:
                task_text = t
                break
        if task_text:
            break
    
    if task_text is None:
        return None, None, None
    
    # Extract response using helper
    response = extract_robovqa_response(task_text)
    assert response is not None

    # A task segment looks like: name>\nPROMPT <PRED>A: RESPONSE\n</PRED>
    # We want to extract PROMPT
    prompt_part, _ = task_text.split('<PRED>', 1)
    
    # Clean prompt: remove the task name (everything before the first \n or first >)
    if '>' in prompt_part:
        prompt = prompt_part.split('>', 1)[1].strip()
    else:
        prompt = prompt_part.strip()
    assert prompt
    prompt = prompt[0].upper() + prompt[1:]

    if task_text.startswith("success") or task_text.startswith("affordance:discriminative"):
        assert response in ["yes", "no"]
        use_true_false = random.random() < 0.5
        if use_true_false:
            prompt += " Answer with True or False."
            response = response.replace("yes", "True")
            response = response.replace("no", "False")
        else:
            prompt += " Answer with Yes or No."
            response = response.replace("yes", "Yes")
            response = response.replace("no", "No")
    prompt += '\n'
    
    if task_text.startswith("immediate_planning_with_context20"):
        # Transfer immediate_planning_with_context20 to planning:freeform
        prompt = re.sub(r'\s*last 20 steps:[\s\S]*?(?=Q:)', ' ', prompt)
        prompt = prompt.replace("immediate next step?", "Next action to achieve the goal?")

    if task_text.startswith("remaining5_planning_with_context20"):
        # Transfer remaining5_planning_with_context20 to planning:freeform
        prompt = re.sub(r'\s*last 20 steps:[\s\S]*?(?=Q:)', ' ', prompt)
        prompt = prompt.replace("next 5 steps?", "To fulfill the goal, what to do next?")
    
    if prompt.startswith("Current goal is") and random.random() < 0.5:
        prompt = prompt.replace("Current goal is", "The objective is")

    assert response
    if response.lower() == 'done':
        return None, None, None
    response = response[0].upper() + response[1:]
    if not response.endswith('.'):
        response += '.'
    
    return image, prompt, response


def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf, options={'threads': '2'})
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    # if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    # rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    # if rgb is None:
    #     rgb = np.frombuffer(x, dtype=np.uint8)
    #     if rgb.size == 2764800: rgb = rgb.reshape(720, 1280, 3)
    #     elif rgb.size == 921600: rgb = rgb.reshape(480, 640, 3)
    # return Image.fromarray(rgb)
    if isinstance(x, np.ndarray):
        x = x.tobytes()
    return Image.open(io.BytesIO(x))

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)


def action_slice(abs_traj: torch.Tensor, idx_for_delta: Sequence[int] = ()) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]

    if idx_for_delta:
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    return {"proprio": proprio, "action": action}