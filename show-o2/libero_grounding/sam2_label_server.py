"""
SAM2 Video Labeling Tool — Gradio Interactive Version

Usage:
    python sam2_label_server.py \
        --input_dir  /path/to/hdf5_folder \
        --output_dir /path/to/output \
        --model_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
        --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
        --device cuda \
        [--host 127.0.0.1] [--port 7860] [--share]
"""

import os


def _init_runtime_temp_dirs():
    """确保 Gradio 与 tempfile 使用可写目录，避免 /tmp 权限问题。"""
    if os.environ.get("GRADIO_TEMP_DIR"):
        base = os.environ["GRADIO_TEMP_DIR"]
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "gradio_tmp")

    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        # 兜底到当前工作目录
        base = os.path.abspath("./.gradio_tmp")
        os.makedirs(base, exist_ok=True)

    # Gradio 6.x 会读取该变量作为 DEFAULT_TEMP_DIR
    os.environ["GRADIO_TEMP_DIR"] = base
    # tempfile 默认也走同一路径，避免其他临时文件落到 /tmp
    os.environ.setdefault("TMPDIR", base)


_init_runtime_temp_dirs()

import cv2
import h5py
import json
import shutil
import torch
import argparse
import tempfile
import traceback
import numpy as np
import mediapy as media
import gradio as gr
from pathlib import Path
from pycocotools import mask as mask_utils
from sam2.build_sam import build_sam2_video_predictor


# ═══════════════════════════ HDF5 helpers ═══════════════════════════

def list_h5_files(folder: Path):
    files = list(folder.glob("*.hdf5")) + list(folder.glob("*.h5"))

    def parse_name(p):
        stem = p.stem
        if "_demo_" in stem:
            instruction, demo = stem.rsplit("_demo_", 1)
            demo_idx = int(demo)
        else:
            instruction, demo_idx = stem, 0
        return (instruction, demo_idx)

    return sorted(files, key=parse_name)


def decode_jpeg_object(obj):
    arr = np.asarray(obj, dtype=np.uint8).reshape(-1)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG bytes from HDF5 element.")
    return img_bgr


def read_all_frames_from_h5(h5_path, frames_key="rgb_comb"):
    with h5py.File(h5_path, "r") as f:
        ds = f[frames_key]
        frames = [decode_jpeg_object(ds[i]) for i in range(len(ds))]

        language_instruction = None
        if "language_instruction" in f:
            try:
                val = f["language_instruction"][()]
                language_instruction = (
                    val.decode("utf-8", errors="ignore")
                    if isinstance(val, bytes) else str(val)
                )
            except Exception:
                pass

    return frames, language_instruction


def export_frames_to_jpg(frames_bgr, out_dir, quality=95):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_digits = max(6, len(str(len(frames_bgr))))
    for i, frame_bgr in enumerate(frames_bgr):
        out_path = out_dir / f"{i:0{n_digits}d}.jpg"
        ok = cv2.imwrite(
            str(out_path), frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError(f"Failed to write frame to {out_path}")


def mask_frames_by_camera_view(frames_bgr, camera_view):
    """根据物体所属视角，对另一视角区域置黑（仅用于 SAM2 输入）。"""
    if not frames_bgr:
        return frames_bgr

    h = int(frames_bgr[0].shape[0])
    split_h = 224 if h == 336 else int(round(h * (2.0 / 3.0)))
    split_h = max(1, min(split_h, h - 1))

    out = []
    for frm in frames_bgr:
        img = frm.copy()
        if camera_view == "main":
            # 主视角物体：腕部视角(下半部分)置黑
            img[split_h:, :, :] = 0
        elif camera_view == "wrist":
            # 腕部视角物体：主视角(上半部分)置黑
            img[:split_h, :, :] = 0
        out.append(img)
    return out


# ═══════════════════════════ RLE / mask utils ═══════════════════════════

def mask_to_rle_json(mask_uint8):
    rle = mask_utils.encode(np.asfortranarray(mask_uint8))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return json.dumps(rle, ensure_ascii=False)


# ═══════════════════════════ SAM2 inference ═══════════════════════════

def run_sam2_on_one_video(predictor, frames_dir, objects, device, prompt_frame_idx=None):
    """
        objects : list of
                            {
                                "name": str,
                                "prompts": {
                                        "0":   {"points": [[x, y], ...], "labels": [0/1, ...]},
                                        "123": {"points": [[x, y], ...], "labels": [0/1, ...]},
                                }
                            }
              object i → SAM2 obj_id = i+1
    Returns : frame_idx, bbox_xywh, area, rle, masks
              masks 仅用于渲染视频，不写入 HDF5.
    """
    num_objects = len(objects)
    if num_objects < 1:
        raise ValueError("Need at least one object.")

    use_autocast = str(device).startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_autocast
        else torch.autocast("cpu", enabled=False)
    )

    # 自动选择传播锚点：所有已标注帧的中位帧（若为空则回退到 0）
    if prompt_frame_idx is None:
        all_prompt_frames = []
        for obj in objects:
            prompts = obj.get("prompts") or {}
            for fidx_str, prompt in prompts.items():
                if prompt.get("points", []):
                    all_prompt_frames.append(int(fidx_str))
        if all_prompt_frames:
            all_prompt_frames = sorted(all_prompt_frames)
            prompt_frame_idx = all_prompt_frames[len(all_prompt_frames) // 2]
        else:
            prompt_frame_idx = 0

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(frames_dir))

        for obj_idx, obj in enumerate(objects, start=1):
            prompts = obj.get("prompts")

            # 兼容旧格式：只有单帧 points/labels 时，挂到锚点帧
            if not prompts:
                prompts = {
                    str(int(prompt_frame_idx)): {
                        "points": obj.get("points", []),
                        "labels": obj.get("labels", [1] * len(obj.get("points", []))),
                    }
                }

            for fidx_str, prompt in prompts.items():
                pts_list = prompt.get("points", [])
                if not pts_list:
                    continue
                pts = np.array(pts_list, dtype=np.float32)
                lbls = np.array(
                    prompt.get("labels", [1] * len(pts_list)),
                    dtype=np.int32,
                )
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=int(fidx_str),
                    obj_id=obj_idx,
                    points=pts,
                    labels=lbls,
                )

        frame_results = []
        masks_all = []

        def _collect_from_generator(gen):
            for frame_idx_val, obj_ids, mask_logits in gen:
                obj_ids = [int(x) for x in obj_ids]
                frame_masks, frame_bboxes, frame_areas, frame_rles = [], [], [], []

                for i, oid in enumerate(obj_ids):
                    mask = (mask_logits[i] > 0.0).detach().cpu().numpy().astype(np.uint8)
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    if mask.ndim != 2:
                        raise RuntimeError(f"Unexpected mask shape: {mask.shape}")

                    area_val = int(mask.sum())
                    if area_val > 0:
                        rle_json = mask_to_rle_json(mask)
                        rle_dict = json.loads(rle_json)
                        bbox = mask_utils.toBbox({
                            "size": rle_dict["size"],
                            "counts": rle_dict["counts"].encode("utf-8"),
                        }).astype(np.float32).tolist()
                    else:
                        bbox = [0.0, 0.0, 0.0, 0.0]
                        rle_json = ""

                    frame_masks.append(mask)
                    frame_bboxes.append(bbox)
                    frame_areas.append(area_val)
                    frame_rles.append(rle_json)

                h, w = frame_masks[0].shape
                ordered_masks  = np.zeros((num_objects, h, w), dtype=np.uint8)
                ordered_bboxes = np.zeros((num_objects, 4), dtype=np.float32)
                ordered_areas  = np.zeros((num_objects,), dtype=np.int32)
                ordered_rles   = np.array([""] * num_objects, dtype=object)

                for li, oid in enumerate(obj_ids):
                    slot = oid - 1
                    ordered_masks[slot]  = frame_masks[li]
                    ordered_bboxes[slot] = np.array(frame_bboxes[li], dtype=np.float32)
                    ordered_areas[slot]  = int(frame_areas[li])
                    ordered_rles[slot]   = frame_rles[li]

                frame_results.append({
                    "frame_idx": int(frame_idx_val),
                    "bbox_xywh": ordered_bboxes,
                    "area":      ordered_areas,
                    "rle":       ordered_rles,
                })
                masks_all.append(ordered_masks)

        # 以锚点帧双向传播，覆盖全视频
        _collect_from_generator(
            predictor.propagate_in_video(
                state,
                start_frame_idx=int(prompt_frame_idx),
                reverse=False,
            )
        )

        if int(prompt_frame_idx) > 0:
            _collect_from_generator(
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=int(prompt_frame_idx),
                    reverse=True,
                )
            )

    # 双向传播会在锚点帧重复，按帧号去重并排序
    merged = {}
    for fr, mk in zip(frame_results, masks_all):
        merged[int(fr["frame_idx"])] = (fr, mk)
    sorted_items = sorted(merged.items(), key=lambda t: t[0])
    frame_results = [item[1][0] for item in sorted_items]
    masks_all = [item[1][1] for item in sorted_items]

    fidx  = np.array([r["frame_idx"] for r in frame_results], dtype=np.int32)
    bbox  = np.stack([r["bbox_xywh"] for r in frame_results], axis=0).astype(np.float32)  # [T, N, 4]
    area  = np.stack([r["area"]      for r in frame_results], axis=0).astype(np.int32)    # [T, N]
    rle   = np.stack([r["rle"]       for r in frame_results], axis=0).astype(object)       # [T, N]
    masks = np.stack(masks_all, axis=0).astype(np.uint8)                                   # [T, N, H, W]

    return fidx, bbox, area, rle, masks


# ═══════════════════════════ Save grounding to HDF5 (NO masks) ═══════════════════════════

def append_grounding_to_h5(out_path, objects, frame_idx, bbox_xywh, area, rle):
    """
    写入 grounding 分组，**不存储 masks** 以节省空间。

    结构:
        grounding/
            object_ids      : [N]       int32
            object_names    : [N]       string
            frame_idx       : [T]       int32
            bbox_xywh       : [T, N, 4] float32
            area            : [T, N]    int32
            rle             : [T, N]    string (UTF-8 JSON)
            prompt/
                obj_0/
                    frame_000000/
                        points_xy : [K, 2]  float32
                        labels    : [K]     int32 (1=positive, 0=negative)
                    frame_000123/ ...
                obj_1/ ...
    """
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "r+") as f:
        if "grounding" in f:
            del f["grounding"]

        g = f.create_group("grounding")
        num_obj = len(objects)

        # object meta
        g.create_dataset("object_ids", data=np.arange(1, num_obj + 1, dtype=np.int32))
        name_ds = g.create_dataset("object_names", shape=(num_obj,), dtype=str_dtype)
        for i, o in enumerate(objects):
            name_ds[i] = o["name"]

        # per-object prompt points (variable length per object)
        prompt_grp = g.create_group("prompt")
        for i, obj in enumerate(objects):
            og = prompt_grp.create_group(f"obj_{i}")
            og.create_dataset("camera_view", data=obj.get("camera_view", "main"), dtype=str_dtype)
            prompts = obj.get("prompts")
            if prompts:
                for fidx_str, prompt in sorted(prompts.items(), key=lambda kv: int(kv[0])):
                    fg = og.create_group(f"frame_{int(fidx_str):06d}")
                    pts = np.array(prompt.get("points", []), dtype=np.float32)
                    lbs = np.array(
                        prompt.get("labels", [1] * len(prompt.get("points", []))),
                        dtype=np.int32,
                    )
                    fg.create_dataset("points_xy", data=pts)
                    fg.create_dataset("labels", data=lbs)
            else:
                # 兼容旧格式
                og.create_dataset("points_xy", data=np.array(obj.get("points", []), dtype=np.float32))
                og.create_dataset(
                    "labels",
                    data=np.array(obj.get("labels", [1] * len(obj.get("points", []))), dtype=np.int32),
                )

        # tracking results
        g.create_dataset("frame_idx", data=frame_idx.astype(np.int32))
        g.create_dataset("bbox_xywh", data=bbox_xywh.astype(np.float32))
        g.create_dataset("area", data=area.astype(np.int32))

        rle_ds = g.create_dataset("rle", shape=rle.shape, dtype=str_dtype)
        rle_ds[...] = rle


# ═══════════════════════════ Render video (mediapy) ═══════════════════════════

DISPLAY_MAX_SIDE = 720   # 显示图片最大边长（像素），确保 numpy 尺寸 = 浏览器显示尺寸

COLORS_BGR = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
    (255, 255,   0), (255,   0, 255), (  0, 255, 255),
    (128,   0, 255), (255, 128,   0), (  0, 128, 255),
]


def compute_display_scale(frame_bgr):
    """计算从原图到显示尺寸的缩放因子."""
    h, w = frame_bgr.shape[:2]
    return DISPLAY_MAX_SIDE / max(h, w)


def render_grounding_video(video_path, frames_bgr, bbox_xywh, masks,
                           object_names=None, fps=10):
    """
    用 mediapy 写入 mp4（RGB 格式），而非 cv2.VideoWriter。
    masks 仅在此处用于渲染叠加层，不持久化。
    """
    if len(frames_bgr) == 0:
        raise ValueError("No frames to render.")

    alpha = 0.35
    T, N = bbox_xywh.shape[:2]
    rendered_rgb = []

    for t in range(T):
        frame = frames_bgr[t].copy()
        overlay = frame.copy()

        for n in range(N):
            color = COLORS_BGR[n % len(COLORS_BGR)]
            mask = masks[t, n]
            x, y, bw, bh = bbox_xywh[t, n].astype(np.int32).tolist()

            if mask.sum() > 0:
                overlay[mask > 0] = color

            if bw > 0 and bh > 0:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                label = object_names[n] if object_names else f"obj{n + 1}"
                cv2.putText(
                    frame, label,
                    (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        rendered_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # mediapy 接受 RGB uint8 列表 / 数组
    media.write_video(str(video_path), rendered_rgb, fps=fps)


# ═══════════════════════════ Annotation visualisation ═══════════════════════════

def draw_annotations_on_frame(frame_bgr, completed_objects,
                              current_points, current_obj_name):
    """
    在首帧上绘制已标注 / 正在标注的点和标签，返回 RGB numpy。

    关键：先将原图缩放到 DISPLAY_MAX_SIDE 的显示尺寸，
    所有坐标（原图空间）乘以 scale 后再绘制。
    返回的 numpy 尺寸就是浏览器中的实际显示尺寸，
    从而保证 Gradio select 事件返回的坐标与 numpy 像素一一对应。
    """
    scale = compute_display_scale(frame_bgr)
    h, w = frame_bgr.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    vis = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 已确认的物体（坐标在原图空间，需乘 scale）
    for i, obj in enumerate(completed_objects):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        labels = obj.get("labels", [1] * len(obj["points"]))
        for (px, py), lb in zip(obj["points"], labels):
            dx, dy = int(px * scale), int(py * scale)
            if int(lb) == 1:
                # 正点：实心圆
                cv2.circle(vis, (dx, dy), 6, color, -1)
                cv2.circle(vis, (dx, dy), 8, color, 2)
            else:
                # 负点：叉号
                cv2.circle(vis, (dx, dy), 8, color, 2)
                cv2.line(vis, (dx - 6, dy - 6), (dx + 6, dy + 6), color, 2)
                cv2.line(vis, (dx - 6, dy + 6), (dx + 6, dy - 6), color, 2)
        if obj["points"]:
            fx = int(obj["points"][0][0] * scale)
            fy = int(obj["points"][0][1] * scale)
            cv2.putText(vis, obj["name"], (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # 正在标注的物体（坐标同样在原图空间）
    if current_points:
        cidx = len(completed_objects)
        color = COLORS_BGR[cidx % len(COLORS_BGR)]
        for p in current_points:
            px, py = p["xy"]
            lb = p.get("label", 1)
            dx, dy = int(px * scale), int(py * scale)
            if int(lb) == 1:
                cv2.circle(vis, (dx, dy), 6, color, -1)
                cv2.circle(vis, (dx, dy), 8, color, 2)
            else:
                cv2.circle(vis, (dx, dy), 8, color, 2)
                cv2.line(vis, (dx - 6, dy - 6), (dx + 6, dy + 6), color, 2)
                cv2.line(vis, (dx - 6, dy + 6), (dx + 6, dy - 6), color, 2)
        if current_obj_name:
            fx = int(current_points[0]["xy"][0] * scale)
            fy = int(current_points[0]["xy"][1] * scale)
            cv2.putText(vis, f"{current_obj_name} (labeling...)",
                        (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ═══════════════════════════ Status text builder ═══════════════════════════

def build_status_text(state):
    frame_keys = ["f0", "f1", "f2", "f3", "f4", "f5"]
    frame_labels = ["初始帧", "1/6帧", "1/3帧", "1/2帧", "2/3帧", "5/6帧"]
    frame_indices = [int(state.get(f"{k}_frame_idx", 0)) for k in frame_keys]
    lang = state.get("language_instruction") or "(无)"

    def _count_prompt(prompt_dict):
        labels = prompt_dict.get("labels", [])
        pos_n = int(np.sum(np.array(labels, dtype=np.int32) == 1)) if labels else 0
        neg_n = int(np.sum(np.array(labels, dtype=np.int32) == 0)) if labels else 0
        return len(labels), pos_n, neg_n

    lines = [f"📝 当前指令: {lang}", f"📌 已接受物体: {len(state['completed_objects'])}"]
    for i, o in enumerate(state["completed_objects"]):
        view_txt = "主视角" if o.get("camera_view", "main") == "main" else "腕部视角"
        prompts = o.get("prompts", {})
        segs = []
        for k, lb, fidx in zip(frame_keys, frame_labels, frame_indices):
            p = prompts.get(str(fidx), {"points": [], "labels": []})
            n, pp, nn = _count_prompt(p)
            if n > 0:
                segs.append(f"{lb}({fidx + 1}): {n}点(+{pp}/-{nn})")
        seg_txt = " | ".join(segs) if segs else "无提示"
        lines.append(f"  {i + 1}. {o['name']} [{view_txt}]  {seg_txt}")

    draft_obj = state.get("draft_object")
    if draft_obj is not None:
        draft_view = "主视角" if draft_obj.get("camera_view", "main") == "main" else "腕部视角"
        lines.append("\n🧪 待处理物体（可重标注）:")
        lines.append(f"  • {draft_obj.get('name', '(未命名)')} [{draft_view}]")

    if state["phase"] == "clicking":
        lines.append(f"\n🔵 正在标注: {state['current_obj_name']}")
        cur_view = "主视角" if state.get("current_camera_view", "main") == "main" else "腕部视角"
        lines.append(f"   当前物体视角: {cur_view}")
        mode_text = "正点 (+)" if int(state.get("current_click_label", 1)) == 1 else "负点 (-)"
        lines.append(f"   当前点击类型: {mode_text}")
        active_key = state.get("active_prompt_key", "f3")
        cur_points = state.get("current_points", {}).get(active_key, [])
        cur_labels = [p.get("label", 1) for p in cur_points]
        cur_pos = int(np.sum(np.array(cur_labels, dtype=np.int32) == 1)) if cur_labels else 0
        cur_neg = int(np.sum(np.array(cur_labels, dtype=np.int32) == 0)) if cur_labels else 0
        key_to_idx = {k: i for i, k in enumerate(frame_keys)}
        ai = key_to_idx.get(active_key, 3)
        active_name, active_idx = frame_labels[ai], frame_indices[ai]
        lines.append(f"   当前标注帧: {active_name} ({active_idx + 1})")
        lines.append(f"   当前帧已点击: {len(cur_points)} 个点 (+{cur_pos} / -{cur_neg})")
        total_clicks = sum(len(state.get("current_points", {}).get(k, [])) for k in frame_keys)
        lines.append(f"   当前物体总点数: {total_clicks}")
    elif state["phase"] == "naming":
        if state["completed_objects"]:
            lines.append("\n💡 输入下一个物体名称开始标注，或点击「完成文件并处理全部已接受物体」")
        else:
            lines.append("\n💡 请输入物体名称开始标注")
    elif state["phase"] == "reviewing":
        lines.append("\n🧪 已生成当前物体预览：可重新标注，或接受该物体继续下一个")
    elif state["phase"] == "all_done":
        lines.append("\n✅ 所有文件已处理完毕！")

    return "\n".join(lines)


# ═══════════════════════════ CLI args ═══════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SAM2 Video Labeling — Gradio")
    p.add_argument("--input_dir",   type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--model_cfg",   type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--frames_key",  type=str, default="rgb_comb")
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--render_fps",  type=float, default=10.0)
    p.add_argument("--host",        type=str, default="127.0.0.1",
                   help="Gradio bind host, use 0.0.0.0 for LAN access")
    p.add_argument("--port",        type=int, default=7860)
    p.add_argument("--share",       action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


def normalize_model_cfg(model_cfg):
    if model_cfg.endswith(".yaml") and os.path.isabs(model_cfg):
        if "/configs/" in model_cfg:
            return model_cfg.split("/configs/", 1)[1]
    return model_cfg


# ═══════════════════════════ Main / Gradio app ═══════════════════════════

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    model_cfg = normalize_model_cfg(args.model_cfg)

    print("Loading SAM2 predictor …")
    predictor = build_sam2_video_predictor(
        model_cfg, args.checkpoint, device=args.device,
    )

    h5_files   = list_h5_files(Path(args.input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(h5_files)} HDF5 files total.")

    # ---------- filter already-processed ----------
    pending = []
    for f in h5_files:
        out_path = output_dir / (f.stem + "_grounding.hdf5")
        if out_path.exists() and not args.overwrite:
            print(f"  skip existing: {out_path.name}")
        else:
            pending.append(f)

    print(f"{len(pending)} file(s) to annotate.\n")

    if not pending:
        print("Nothing to do — all files already processed.")
        return

    # ────────────────────── server-side frame cache ──────────────────────
    # frames_bgr 不存入 gr.State（太大会导致序列化失败 / "unexpected token"）
    # 改为服务端 dict，仅按 file_idx 缓存当前文件帧
    _frame_cache = {}   # {file_idx: list[np.ndarray(BGR)]}
    # 进程内记录“本次服务运行期间”跳过过的文件。
    # 仅在 server 存活期间有效；server 重启后会自动清空（符合“重启后重新开始”预期）。
    _skipped_files_in_runtime = set()  # {str(path)}

    def _cache_frames(file_idx, frames_bgr):
        _frame_cache.clear()          # 只保留一个文件，节省内存
        _frame_cache[file_idx] = frames_bgr

    def _get_frames(state):
        """从缓存获取当前文件的帧列表，若不存在则重新读取."""
        idx = state["file_idx"]
        if idx in _frame_cache:
            return _frame_cache[idx]
        if idx >= len(pending):
            return None
        frames, _ = read_all_frames_from_h5(pending[idx], args.frames_key)
        _cache_frames(idx, frames)
        return frames

    # ────────────────────── state helpers ──────────────────────

    def make_state():
        return {
            "file_idx":             0,
            "language_instruction": None,
            "completed_objects":   [],       # [{"name": str, "points": [[x,y], ...]}]
            "completed_results":   [],       # [{"frame_idx","bbox","area","rle","masks"}] 与 completed_objects 对齐
            "draft_object":        None,     # 当前待处理（可重标注）的物体
            "draft_result":        None,     # 当前待处理物体的 tracking 结果
            "preview_video_path":  None,     # 当前文件的临时预览视频
            "current_camera_view": "main",  # main | wrist
            "current_obj_name":    "",
            "current_points":      {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": []},
            "current_click_label": 1,        # 1=positive, 0=negative
            "f0_frame_idx":        0,        # 0
            "f1_frame_idx":        0,        # 1/6
            "f2_frame_idx":        0,        # 1/3
            "f3_frame_idx":        0,        # 1/2
            "f4_frame_idx":        0,        # 2/3
            "f5_frame_idx":        0,        # 5/6
            "active_prompt_key":   "f0",   # f0|f1|f2|f3|f4|f5
            "phase":               "idle",   # idle | naming | clicking | reviewing | processing | all_done
            "last_out_hdf5":       None,     # 上次处理输出的 hdf5 路径 (str)
            "last_out_video":      None,     # 上次处理输出的 mp4 路径 (str)
            "last_file_idx":       None,     # 上次处理的 pending 索引
            "last_obj_name":       "",      # 记忆上一次输入的物体名称
        }

    def _cleanup_preview_video(state):
        p = state.get("preview_video_path")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        state["preview_video_path"] = None

    def _aggregate_completed_results(results):
        if not results:
            raise ValueError("No completed object results to aggregate.")

        base_fidx = np.array(results[0]["frame_idx"], dtype=np.int32)
        T = int(base_fidx.shape[0])
        frame_to_row = {int(f): i for i, f in enumerate(base_fidx.tolist())}

        bboxes, areas, rles, masks = [], [], [], []
        for r in results:
            cur_fidx = np.array(r["frame_idx"], dtype=np.int32)
            cur_bbox = np.array(r["bbox"], dtype=np.float32)
            cur_area = np.array(r["area"], dtype=np.int32)
            cur_rle = np.array(r["rle"], dtype=object)
            cur_masks = np.array(r["masks"], dtype=np.uint8)

            if np.array_equal(cur_fidx, base_fidx):
                bboxes.append(cur_bbox)
                areas.append(cur_area)
                rles.append(cur_rle)
                masks.append(cur_masks)
                continue

            if cur_bbox.shape[1] != 1:
                raise ValueError("Expected single-object result for aggregation.")

            h, w = cur_masks.shape[-2], cur_masks.shape[-1]
            aligned_bbox = np.zeros((T, 1, 4), dtype=np.float32)
            aligned_area = np.zeros((T, 1), dtype=np.int32)
            aligned_rle = np.array([[""] for _ in range(T)], dtype=object)
            aligned_masks = np.zeros((T, 1, h, w), dtype=np.uint8)

            for i, f in enumerate(cur_fidx.tolist()):
                row = frame_to_row.get(int(f))
                if row is None:
                    continue
                aligned_bbox[row, 0] = cur_bbox[i, 0]
                aligned_area[row, 0] = cur_area[i, 0]
                aligned_rle[row, 0] = cur_rle[i, 0]
                aligned_masks[row, 0] = cur_masks[i, 0]

            bboxes.append(aligned_bbox)
            areas.append(aligned_area)
            rles.append(aligned_rle)
            masks.append(aligned_masks)

        agg_bbox = np.concatenate(bboxes, axis=1).astype(np.float32)  # [T, N, 4]
        agg_area = np.concatenate(areas, axis=1).astype(np.int32)     # [T, N]
        agg_rle = np.concatenate(rles, axis=1).astype(object)          # [T, N]
        agg_masks = np.concatenate(masks, axis=1).astype(np.uint8)     # [T, N, H, W]

        return base_fidx, agg_bbox, agg_area, agg_rle, agg_masks

    def _advance_to_next_unprocessed(state):
        """从当前 file_idx 开始，跳过已有输出的文件（处理刷新后的恢复）."""
        idx = state["file_idx"]
        while idx < len(pending):
            cur_file = pending[idx]
            out_path = output_dir / (cur_file.stem + "_grounding.hdf5")
            if str(cur_file) in _skipped_files_in_runtime:
                print(f"  skip runtime-skipped: {cur_file.name}")
                idx += 1
                continue
            if out_path.exists() and not args.overwrite:
                print(f"  skip already processed: {cur_file.name}")
                idx += 1
            else:
                break
        state["file_idx"] = idx

    def load_current_file(state):
        """加载当前 file_idx 对应的帧数据（帧存缓存，不存 state）."""
        _advance_to_next_unprocessed(state)
        idx = state["file_idx"]
        if idx >= len(pending):
            state["phase"] = "all_done"
            return state
        h5_path = pending[idx]
        print(f"Loading {h5_path.name} …")
        frames, lang = read_all_frames_from_h5(h5_path, args.frames_key)
        _cache_frames(idx, frames)
        n = len(frames)
        state.update({
            "language_instruction": lang,
            "completed_objects":   [],
            "completed_results":   [],
            "draft_object":        None,
            "draft_result":        None,
            "preview_video_path":  None,
            "current_camera_view": "main",
            "current_obj_name":    "",
            "current_points":      {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": []},
            "current_click_label": 1,
            "f0_frame_idx":        0,
            "f1_frame_idx":        (n // 6) if n else 0,
            "f2_frame_idx":        (n // 3) if n else 0,
            "f3_frame_idx":        (n // 2) if n else 0,
            "f4_frame_idx":        ((2 * n) // 3) if n else 0,
            "f5_frame_idx":        ((5 * n) // 6) if n else 0,
            "active_prompt_key":   "f0",
            "phase":               "naming" if frames else "idle",
        })
        return state

    def file_info_text(state):
        idx = state["file_idx"]
        if idx >= len(pending):
            return "✅ 所有文件已处理完毕！"
        p = pending[idx]
        lang = state.get("language_instruction") or "(无)"
        frames = _get_frames(state)
        n = len(frames) if frames else 0
        f0 = int(state.get("f0_frame_idx", 0))
        f1 = int(state.get("f1_frame_idx", 0))
        f2 = int(state.get("f2_frame_idx", 0))
        f3 = int(state.get("f3_frame_idx", 0))
        f4 = int(state.get("f4_frame_idx", 0))
        f5 = int(state.get("f5_frame_idx", 0))
        return (
            f"📁 文件: {p.name}\n"
            f"📝 指令: {lang}\n"
            f"🎞️ 帧数: {n}\n"
            f"🖼️ 标注帧: 0={f0 + 1}, 1/6={f1 + 1}, 1/3={f2 + 1}, 1/2={f3 + 1}, 2/3={f4 + 1}, 5/6={f5 + 1} / {max(n, 1)}\n"
            f"📊 进度: {idx + 1} / {len(pending)}"
        )

    def get_display_image(state):
        frames = _get_frames(state)
        if not frames:
            return None
        active_key = state.get("active_prompt_key", "f3")
        prompt_idx = int(state.get(f"{active_key}_frame_idx", 0))
        prompt_idx = max(0, min(prompt_idx, len(frames) - 1))

        # 仅显示当前选中帧上的点
        completed_on_frame = []
        for o in state["completed_objects"]:
            prompt = o.get("prompts", {}).get(str(prompt_idx), {"points": [], "labels": []})
            completed_on_frame.append({
                "name": o["name"],
                "points": prompt.get("points", []),
                "labels": prompt.get("labels", []),
            })

        current_on_frame = state["current_points"].get(active_key, [])
        return draw_annotations_on_frame(
            frames[prompt_idx],
            completed_on_frame,
            current_on_frame,
            state["current_obj_name"],
        )

    # ────────────────────── build Gradio UI ──────────────────────

    with gr.Blocks(
        title="SAM2 标注工具",
    ) as demo:

        gr.Markdown(
            "# 🎯 SAM2 视频物体标注工具\n"
            "**标注流程**: ① 输入物体名称并回车确认 → "
            "② 切换 6 个关键帧（0,1/6,1/3,1/2,2/3,5/6）并添加点（正/负） → "
            "③ 点击「完成标注并处理当前物体」预览，可重标注；接受后继续下一个"
        )

        app_state = gr.State(make_state())

        with gr.Row():
            # ---- left column: image + video ----
            with gr.Column(scale=3):
                image_out = gr.Image(
                    label="标注帧（可切换 0、1/6、1/3、1/2、2/3、5/6）— 点击添加标注点",
                    type="numpy",
                    interactive=False,
                    elem_id="sam2-annot-image",
                )
                video_out = gr.Video(label="追踪结果预览")

            # ---- right column: info + controls ----
            with gr.Column(scale=2):
                info_box = gr.Textbox(
                    label="📄 文件信息", interactive=False, lines=4,
                )
                status_box = gr.Textbox(
                    label="📋 标注状态", interactive=False, lines=8,
                    value="正在加载 …\n提示：先在名称框输入并按回车确认，再选点类型并左键点击添加",
                )

                obj_name_in = gr.Textbox(
                    label="物体名称",
                    placeholder="输入名称，如: red_cup  (回车确认)",
                    interactive=True,
                )

                click_mode = gr.Radio(
                    choices=["正点 (+)", "负点 (-)"],
                    value="正点 (+)",
                    label="当前点击类型",
                    interactive=True,
                )

                frame_mode = gr.Radio(
                    choices=["初始帧", "1/6帧", "1/3帧", "1/2帧", "2/3帧", "5/6帧"],
                    value="初始帧",
                    label="当前标注帧",
                    interactive=True,
                )

                camera_mode = gr.Radio(
                    choices=["主视角", "腕部视角"],
                    value="主视角",
                    label="当前物体所属视角（仅影响SAM2输入）",
                    interactive=True,
                )

                gr.Markdown("- 点击说明：先选择标注帧与点类型，再用左键点击图片添加点")
                gr.Markdown("- 名称确认：在“物体名称”输入后按回车，无需点击按钮")

                with gr.Row():
                    btn_undo_pt  = gr.Button("↩️ 撤销上一个点", size="sm")

                with gr.Row():
                    btn_process_obj = gr.Button("🚀 完成标注并处理当前物体", variant="stop")

                with gr.Row():
                    btn_accept_obj = gr.Button("✅ 接受该物体并继续下一个", variant="primary", size="sm")
                    btn_redo_obj   = gr.Button("🔄 重新标注该物体", size="sm")

                with gr.Row():
                    btn_done = gr.Button("🏁 完成文件并处理全部已接受物体", variant="stop")
                    btn_skip = gr.Button("⏭️ 跳过此文件", size="sm")
                    btn_redo = gr.Button("🔄 删除上次文件结果并回退", size="sm")

        # ════════════════════ event handlers ════════════════════

        def h_load(state):
            """页面加载时：读取第一个待处理文件."""
            state = load_current_file(state)
            return (
                state,
                get_display_image(state),
                file_info_text(state),
                build_status_text(state),
                None,   # video_out
                state.get("last_obj_name", ""),
                "正点 (+)",
                "初始帧",
                "主视角",
            )

        def h_confirm_name(state, name):
            """用户确认物体名称 → 进入点击标注阶段."""
            if state["phase"] == "all_done":
                return state, gr.update(), "✅ 所有文件已完成", state.get("last_obj_name", ""), gr.update()
            name = name.strip()
            if not name:
                return state, gr.update(), "⚠️ 请输入物体名称！", gr.update(), gr.update()
            state["current_obj_name"] = name
            state["last_obj_name"] = name
            state["current_points"]   = {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": []}
            state["current_click_label"] = 1
            state["draft_object"] = None
            state["draft_result"] = None
            state["phase"]            = "clicking"
            return (
                state,
                get_display_image(state),
                build_status_text(state),
                name,
                gr.update(),
            )

        def _build_object_from_current_clicks(state):
            frame_keys = ["f0", "f1", "f2", "f3", "f4", "f5"]
            obj = {
                "name": state["current_obj_name"],
                "camera_view": state.get("current_camera_view", "main"),
                "prompts": {},
            }
            for fk in frame_keys:
                pts = state["current_points"].get(fk, [])
                if not pts:
                    continue
                fidx = int(state.get(f"{fk}_frame_idx", 0))
                obj["prompts"][str(fidx)] = {
                    "points": [p["xy"] for p in pts],
                    "labels": [int(p.get("label", 1)) for p in pts],
                }
            return obj

        def h_set_click_mode(state, mode_text):
            state["current_click_label"] = 0 if mode_text == "负点 (-)" else 1
            return state, build_status_text(state)

        def h_set_frame_mode(state, mode_text):
            mapping = {
                "初始帧": "f0",
                "1/6帧": "f1",
                "1/3帧": "f2",
                "1/2帧": "f3",
                "2/3帧": "f4",
                "5/6帧": "f5",
            }
            state["active_prompt_key"] = mapping.get(mode_text, "f3")
            return state, get_display_image(state), build_status_text(state)

        def h_set_camera_mode(state, mode_text):
            state["current_camera_view"] = "wrist" if mode_text == "腕部视角" else "main"
            return state, build_status_text(state)

        def h_click(state, evt: gr.SelectData):
            """用户在图片上点击 → 添加一个标注点."""
            if state["phase"] != "clicking":
                return state, gr.update(), "⚠️ 请先输入并确认物体名称"
            # evt.index → [x, y] 是浏览器显示空间的坐标
            # 需要除以 scale 转换回原图像素坐标（用于 SAM2 推理）
            frames = _get_frames(state)
            active_key = state.get("active_prompt_key", "f3")
            prompt_idx = int(state.get(f"{active_key}_frame_idx", 0))
            prompt_idx = max(0, min(prompt_idx, len(frames) - 1))
            scale = compute_display_scale(frames[prompt_idx])
            x_orig = float(evt.index[0]) / scale
            y_orig = float(evt.index[1]) / scale
            lb = int(state.get("current_click_label", 1))
            state["current_points"].setdefault(active_key, []).append(
                {"xy": [x_orig, y_orig], "label": int(lb)}
            )
            return (
                state,
                get_display_image(state),
                build_status_text(state),
            )

        def h_undo_point(state):
            """撤销最后一个标注点."""
            if state["phase"] == "clicking":
                active_key = state.get("active_prompt_key", "f3")
                if state["current_points"].get(active_key):
                    state["current_points"][active_key].pop()
            return (
                state,
                get_display_image(state),
                build_status_text(state),
            )

        def h_process_obj(state):
            """处理当前物体并生成预览（可重标注/接受）。"""
            if state["phase"] == "clicking":
                if not state.get("current_obj_name", "").strip():
                    return (
                        state,
                        gr.update(),
                        "⚠️ 请先在“物体名称”输入名称并按回车确认",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                total_points = sum(len(v) for v in state["current_points"].values())
                if total_points == 0:
                    return (
                        state,
                        gr.update(),
                        "⚠️ 当前物体至少需要在任意一帧点击一个点",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                state["draft_object"] = _build_object_from_current_clicks(state)
                state["draft_result"] = None
                state["current_obj_name"] = ""
                state["current_points"] = {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": []}

            obj = state.get("draft_object")
            if obj is None:
                return (
                    state,
                    gr.update(),
                    "⚠️ 没有可处理的物体，请先输入名称回车并完成点击标注",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            idx = state["file_idx"]
            h5_path = pending[idx]
            frames = _get_frames(state)
            preview_video = output_dir / (h5_path.stem + "_preview_current_obj.mp4")
            camera_view = obj.get("camera_view", "main")
            masked_frames = mask_frames_by_camera_view(frames, camera_view)

            view_text = "主视角" if camera_view == "main" else "腕部视角"
            status_lines = [f"⏳ 正在处理当前物体: {obj.get('name', '(未命名)')} [{view_text}] …"]
            try:
                with tempfile.TemporaryDirectory(prefix="sam2_frames_") as td:
                    export_frames_to_jpg(masked_frames, Path(td), args.jpg_quality)
                    fidx, bbox, area, rle, masks = run_sam2_on_one_video(
                        predictor,
                        Path(td),
                        [obj],
                        args.device,
                        prompt_frame_idx=None,
                    )

                render_grounding_video(
                    str(preview_video),
                    frames,
                    bbox,
                    masks,
                    [obj.get("name", "obj1")],
                    args.render_fps,
                )
                state["draft_result"] = {
                    "frame_idx": fidx,
                    "bbox": bbox,
                    "area": area,
                    "rle": rle,
                    "masks": masks,
                }
                state["preview_video_path"] = str(preview_video)
                state["phase"] = "reviewing"
                status_lines.append("✅ 当前物体预览已生成。可重标注或接受该物体。")
            except Exception as e:
                status_lines.append(f"❌ 当前物体处理失败: {e}")
                traceback.print_exc()

            return (
                state,
                get_display_image(state),
                "\n".join(status_lines) + "\n\n" + build_status_text(state),
                file_info_text(state),
                str(preview_video) if preview_video.exists() else None,
                state.get("last_obj_name", ""),
            )

        def h_accept_obj(state):
            """接受当前 draft 物体，加入已接受列表。"""
            obj = state.get("draft_object")
            if obj is None:
                return state, gr.update(), "⚠️ 没有可接受的物体（请先确认并处理）", gr.update()
            if state.get("draft_result") is None:
                return state, gr.update(), "⚠️ 请先处理当前物体并检查可视化结果，再决定接受", gr.update()
            state["completed_objects"].append(obj)
            state["completed_results"].append(state["draft_result"])
            state["draft_object"] = None
            state["draft_result"] = None
            state["phase"] = "naming"
            return state, get_display_image(state), build_status_text(state), state.get("last_obj_name", "")

        def h_redo_obj(state):
            """将 draft 物体恢复为可编辑，重新标注后再处理。"""
            obj = state.get("draft_object")
            if obj is None:
                return state, gr.update(), "⚠️ 没有可重标注的待处理物体", gr.update(), gr.update(), gr.update()

            state["current_obj_name"] = obj.get("name", "")
            state["current_camera_view"] = obj.get("camera_view", "main")
            state["current_points"] = {"f0": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": []}
            for fk in ["f0", "f1", "f2", "f3", "f4", "f5"]:
                fidx = int(state.get(f"{fk}_frame_idx", 0))
                p = obj.get("prompts", {}).get(str(fidx))
                if p:
                    state["current_points"][fk] = [
                        {"xy": xy, "label": int(lb)}
                        for xy, lb in zip(p.get("points", []), p.get("labels", []))
                    ]

            state["draft_object"] = None
            state["draft_result"] = None
            state["phase"] = "clicking"
            camera_mode_value = "主视角" if state["current_camera_view"] == "main" else "腕部视角"
            return state, get_display_image(state), build_status_text(state), state.get("current_obj_name", ""), gr.update(), camera_mode_value

        def h_done(state):
            """完成文件：处理全部已接受物体并保存结果，然后加载下一个文件。"""
            # 若仍在点击阶段，不允许直接完成文件
            if state["phase"] == "clicking":
                return (
                    state,
                    gr.update(),
                    "⚠️ 当前仍在标注中。请先点击「完成标注并处理当前物体」，并接受后再完成文件。",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            if state.get("draft_object") is not None:
                return (
                    state,
                    gr.update(),
                    "⚠️ 还有待处理物体。请先处理预览并接受（或重标注）后再完成文件。",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            if not state["completed_objects"]:
                return (state, gr.update(),
                        "⚠️ 请至少接受一个物体后再完成文件",
                        gr.update(), gr.update(), gr.update())

            state["phase"] = "processing"
            idx     = state["file_idx"]
            h5_path = pending[idx]
            objects = state["completed_objects"]

            out_name  = h5_path.stem + "_grounding.hdf5"
            out_path  = output_dir / out_name
            out_video = output_dir / (h5_path.stem + "_grounding.mp4")

            status_lines = [
                f"⏳ 正在处理 {h5_path.name} …",
                f"   物体数: {len(objects)}",
            ]
            for o in objects:
                prompts = o.get("prompts", {})
                used = []
                for fk, lb in [("f0", "0"), ("f1", "1/6"), ("f2", "1/3"), ("f3", "1/2"), ("f4", "2/3"), ("f5", "5/6")]:
                    fidx = int(state.get(f"{fk}_frame_idx", 0))
                    labels = prompts.get(str(fidx), {}).get("labels", [])
                    if labels:
                        p = int(np.sum(np.array(labels, dtype=np.int32) == 1))
                        n = int(np.sum(np.array(labels, dtype=np.int32) == 0))
                        used.append(f"{lb}帧 {len(labels)}点(+{p}/-{n})")
                status_lines.append(f"   • {o['name']}: " + (", ".join(used) if used else "无提示"))

            print(f"\n===== Finalizing {h5_path.name}  ({len(objects)} objects) =====")

            frames = _get_frames(state)
            try:
                # --- aggregate already processed per-object results (no re-run) ---
                fidx, bbox, area, rle, masks = _aggregate_completed_results(state["completed_results"])

                # --- save hdf5 (NO masks) ---
                if out_path.exists():
                    out_path.unlink()
                shutil.copy2(h5_path, out_path)
                append_grounding_to_h5(out_path, objects, fidx, bbox, area, rle)

                # --- render video with mediapy ---
                names = [o["name"] for o in objects]
                render_grounding_video(
                    str(out_video), frames,
                    bbox, masks, names, args.render_fps,
                )

                status_lines.append(f"\n✅ 已保存: {out_path.name}")
                status_lines.append(f"✅ 视频:   {out_video.name}")
                print(f"  saved hdf5 : {out_path}")
                print(f"  saved video: {out_video}")

                # 删除当前文件临时预览视频（避免残留最后一个物体的 preview）
                _cleanup_preview_video(state)

            except Exception as e:
                status_lines.append(f"\n❌ 处理失败: {e}")
                traceback.print_exc()
                state["phase"] = "naming"
                return (state, gr.update(),
                        "\n".join(status_lines),
                        gr.update(), gr.update(), gr.update())

            # --- remember last processed for potential redo ---
            state["last_out_hdf5"] = str(out_path)
            state["last_out_video"] = str(out_video)
            state["last_file_idx"]  = idx

            # --- load next file ---
            state["file_idx"] = idx + 1
            state = load_current_file(state)
            img  = get_display_image(state)
            info = file_info_text(state)

            status_lines.append("\n─────────────────")
            status_lines.append(build_status_text(state))

            return (
                state,
                img,
                "\n".join(status_lines),
                info,
                str(out_video) if out_video.exists() else None,
                state.get("last_obj_name", ""),
            )

        def h_skip(state):
            """跳过当前文件，加载下一个."""
            if state["file_idx"] >= len(pending):
                return (state, gr.update(), "✅ 所有文件已完成",
                        gr.update(), "")
            _cleanup_preview_video(state)
            skipped_path = pending[state["file_idx"]]
            _skipped_files_in_runtime.add(str(skipped_path))
            skipped = skipped_path.name
            print(f"  skipped: {skipped}")
            state["file_idx"] += 1
            state = load_current_file(state)
            return (
                state,
                get_display_image(state),
                f"⏭️ 已跳过 {skipped}\n\n" + build_status_text(state),
                file_info_text(state),
                state.get("last_obj_name", ""),
            )

        def h_redo(state):
            """删除上次处理的输出文件，回退到上次的文件重新标注."""
            if state.get("last_file_idx") is None:
                return (state, gr.update(),
                        "⚠️ 没有可以重做的记录",
                        gr.update(), gr.update(), gr.update())

            # 删除输出文件
            deleted = []
            for p in [state["last_out_hdf5"], state["last_out_video"]]:
                if p and os.path.exists(p):
                    os.remove(p)
                    deleted.append(os.path.basename(p))
            _cleanup_preview_video(state)

            prev_idx = state["last_file_idx"]
            prev_name = pending[prev_idx].name
            print(f"  redo: deleted output for {prev_name}, re-annotating")

            # 回退到上次处理的文件
            state["file_idx"] = prev_idx
            state["last_out_hdf5"] = None
            state["last_out_video"] = None
            state["last_file_idx"]  = None
            state = load_current_file(state)

            msg = f"🔄 已删除: {', '.join(deleted)}\n重新标注: {prev_name}\n\n"
            msg += build_status_text(state)
            return (
                state,
                get_display_image(state),
                msg,
                file_info_text(state),
                None,   # clear video preview
                state.get("last_obj_name", ""),
            )

        # ════════════════════ wire events ════════════════════

        outs_3 = [app_state, image_out, status_box]

        # page load
        demo.load(
            h_load, [app_state],
            [app_state, image_out, info_box, status_box, video_out, obj_name_in, click_mode, frame_mode, camera_mode],
            js="""
            (s) => {
                // 禁用标注图像上的浏览器右键菜单，避免干扰右键打负点
                setTimeout(() => {
                    const root = document.getElementById('sam2-annot-image');
                    if (!root) return;
                    root.addEventListener('contextmenu', (e) => e.preventDefault());
                }, 300);
                return [s];
            }
            """,
        )

        # switch click mode (positive / negative)
        click_mode.change(
            h_set_click_mode, [app_state, click_mode],
            [app_state, status_box],
        )

        # switch annotation frame (0 / 1/6 / 1/3 / 1/2 / 2/3 / 5/6)
        frame_mode.change(
            h_set_frame_mode, [app_state, frame_mode],
            [app_state, image_out, status_box],
        )

        camera_mode.change(
            h_set_camera_mode, [app_state, camera_mode],
            [app_state, status_box],
        )

        # confirm name — press Enter in textbox
        obj_name_in.submit(
            h_confirm_name, [app_state, obj_name_in],
            [app_state, image_out, status_box, obj_name_in, frame_mode],
        )

        # click on image → add point
        image_out.select(h_click, [app_state], outs_3)

        # undo point
        btn_undo_pt.click(h_undo_point,  [app_state], outs_3)

        # process and preview current object
        btn_process_obj.click(
            h_process_obj, [app_state],
            [app_state, image_out, status_box, info_box, video_out, obj_name_in],
        )

        # accept current draft object
        btn_accept_obj.click(
            h_accept_obj, [app_state],
            [app_state, image_out, status_box, obj_name_in],
        )

        # redo current draft object
        btn_redo_obj.click(
            h_redo_obj, [app_state],
            [app_state, image_out, status_box, obj_name_in, frame_mode, camera_mode],
        )

        # finish current file → process all accepted objects
        btn_done.click(
            h_done, [app_state],
            [app_state, image_out, status_box, info_box, video_out, obj_name_in],
        )

        # skip file
        btn_skip.click(
            h_skip, [app_state],
            [app_state, image_out, status_box, info_box, obj_name_in],
        )

        # redo last processed file
        btn_redo.click(
            h_redo, [app_state],
            [app_state, image_out, status_box, info_box, video_out, obj_name_in],
        )

    # ────────────── launch ──────────────

    print(f"Launching Gradio on {args.host}:{args.port} …")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        allowed_paths=[str(output_dir)],
    )


if __name__ == "__main__":
    main()
