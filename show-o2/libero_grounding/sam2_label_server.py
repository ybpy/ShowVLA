"""
SAM2 Video Labeling Tool — Gradio Interactive Version

Usage:
    python sam2_label_server.py \
        --input_dir  /path/to/hdf5_folder \
        --output_dir /path/to/output \
        --model_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
        --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
        --device cuda \
        [--port 7860] [--share]
"""

import os
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


# ═══════════════════════════ RLE / mask utils ═══════════════════════════

def mask_to_rle_json(mask_uint8):
    rle = mask_utils.encode(np.asfortranarray(mask_uint8))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return json.dumps(rle, ensure_ascii=False)


# ═══════════════════════════ SAM2 inference ═══════════════════════════

def run_sam2_on_one_video(predictor, frames_dir, objects, device):
    """
    objects : list of {"name": str, "points": [[x, y], ...]}
              每个 object 的所有 point 都是正点 (label=1).
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

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(frames_dir))

        for obj_idx, obj in enumerate(objects, start=1):
            pts = np.array(obj["points"], dtype=np.float32)
            lbls = np.ones(len(obj["points"]), dtype=np.int32)
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_idx,
                points=pts,
                labels=lbls,
            )

        frame_results = []
        masks_all = []

        for frame_idx_val, obj_ids, mask_logits in predictor.propagate_in_video(state):
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

    # sort by frame index
    frame_results = sorted(frame_results, key=lambda x: x["frame_idx"])
    masks_all = [m for _, m in sorted(
        [(fr["frame_idx"], mk) for fr, mk in zip(frame_results, masks_all)],
        key=lambda t: t[0],
    )]

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
                    points_xy : [K, 2]  float32
                    labels    : [K]     int32 (all ones)
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
            og.create_dataset("points_xy", data=np.array(obj["points"], dtype=np.float32))
            og.create_dataset("labels", data=np.ones(len(obj["points"]), dtype=np.int32))

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
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
        for px, py in obj["points"]:
            dx, dy = int(px * scale), int(py * scale)
            cv2.circle(vis, (dx, dy), 6, color, -1)
            cv2.circle(vis, (dx, dy), 8, color, 2)
        if obj["points"]:
            fx = int(obj["points"][0][0] * scale)
            fy = int(obj["points"][0][1] * scale)
            cv2.putText(vis, obj["name"], (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # 正在标注的物体（坐标同样在原图空间）
    if current_points:
        cidx = len(completed_objects)
        color = COLORS_BGR[cidx % len(COLORS_BGR)]
        for px, py in current_points:
            dx, dy = int(px * scale), int(py * scale)
            cv2.circle(vis, (dx, dy), 6, color, -1)
            cv2.circle(vis, (dx, dy), 8, color, 2)
        if current_obj_name:
            fx = int(current_points[0][0] * scale)
            fy = int(current_points[0][1] * scale)
            cv2.putText(vis, f"{current_obj_name} (labeling...)",
                        (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ═══════════════════════════ Status text builder ═══════════════════════════

def build_status_text(state):
    lines = [f"📌 已标注物体: {len(state['completed_objects'])}"]
    for i, o in enumerate(state["completed_objects"]):
        lines.append(f"  {i + 1}. {o['name']}  ({len(o['points'])} 个点)")

    if state["phase"] == "clicking":
        lines.append(f"\n🔵 正在标注: {state['current_obj_name']}")
        lines.append(f"   已点击: {len(state['current_points'])} 个点")
    elif state["phase"] == "naming":
        if state["completed_objects"]:
            lines.append("\n💡 输入下一个物体名称，或点击「完成并处理」")
        else:
            lines.append("\n💡 请输入物体名称开始标注")
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
            "current_obj_name":    "",
            "current_points":      [],       # [[x,y], ...]
            "phase":               "idle",   # idle | naming | clicking | processing | all_done
            "last_out_hdf5":       None,     # 上次处理输出的 hdf5 路径 (str)
            "last_out_video":      None,     # 上次处理输出的 mp4 路径 (str)
            "last_file_idx":       None,     # 上次处理的 pending 索引
        }

    def _advance_to_next_unprocessed(state):
        """从当前 file_idx 开始，跳过已有输出的文件（处理刷新后的恢复）."""
        idx = state["file_idx"]
        while idx < len(pending):
            out_path = output_dir / (pending[idx].stem + "_grounding.hdf5")
            if out_path.exists() and not args.overwrite:
                print(f"  skip already processed: {pending[idx].name}")
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
        state.update({
            "language_instruction": lang,
            "completed_objects":   [],
            "current_obj_name":    "",
            "current_points":      [],
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
        return (
            f"📁 文件: {p.name}\n"
            f"📝 指令: {lang}\n"
            f"🎞️ 帧数: {n}\n"
            f"📊 进度: {idx + 1} / {len(pending)}"
        )

    def get_display_image(state):
        frames = _get_frames(state)
        if not frames:
            return None
        return draw_annotations_on_frame(
            frames[0],
            state["completed_objects"],
            state["current_points"],
            state["current_obj_name"],
        )

    # ────────────────────── build Gradio UI ──────────────────────

    with gr.Blocks(
        title="SAM2 标注工具",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# 🎯 SAM2 视频物体标注工具\n"
            "**标注流程**: ① 输入物体名称 → ② 确认名称 → "
            "③ 在图片上点击标注点 → ④ 确认物体 → "
            "⑤ 继续添加物体 **或** 完成标注并处理"
        )

        app_state = gr.State(make_state())

        with gr.Row():
            # ---- left column: image + video ----
            with gr.Column(scale=3):
                image_out = gr.Image(
                    label="第一帧 — 点击添加标注点",
                    type="numpy",
                    interactive=False,
                )
                video_out = gr.Video(label="追踪结果预览")

            # ---- right column: info + controls ----
            with gr.Column(scale=2):
                info_box = gr.Textbox(
                    label="📄 文件信息", interactive=False, lines=4,
                )
                status_box = gr.Textbox(
                    label="📋 标注状态", interactive=False, lines=8,
                    value="正在加载 …",
                )

                obj_name_in = gr.Textbox(
                    label="物体名称",
                    placeholder="输入名称，如: red_cup  (回车亦可确认)",
                    interactive=True,
                )

                with gr.Row():
                    btn_name     = gr.Button("✏️ 确认名称，开始标注", variant="primary", size="sm")
                    btn_undo_pt  = gr.Button("↩️ 撤销上一个点",      size="sm")
                    btn_undo_obj = gr.Button("🗑️ 删除上一物体",      size="sm")

                with gr.Row():
                    btn_confirm_obj = gr.Button("✅ 确认当前物体", variant="primary")
                    btn_done        = gr.Button("🚀 完成标注并处理", variant="stop")

                with gr.Row():
                    btn_skip = gr.Button("⏭️ 跳过此文件", size="sm")
                    btn_redo = gr.Button("🔄 删除上次结果，重新标注", size="sm")

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
                "",     # clear name input
            )

        def h_confirm_name(state, name):
            """用户确认物体名称 → 进入点击标注阶段."""
            if state["phase"] == "all_done":
                return state, gr.update(), "✅ 所有文件已完成", ""
            name = name.strip()
            if not name:
                return state, gr.update(), "⚠️ 请输入物体名称！", gr.update()
            for o in state["completed_objects"]:
                if o["name"] == name:
                    return state, gr.update(), f"⚠️ 物体名 '{name}' 已存在，请换一个", gr.update()
            state["current_obj_name"] = name
            state["current_points"]   = []
            state["phase"]            = "clicking"
            return (
                state,
                get_display_image(state),
                build_status_text(state),
                "",   # clear name input
            )

        def h_click(state, evt: gr.SelectData):
            """用户在图片上点击 → 添加一个标注点."""
            if state["phase"] != "clicking":
                return state, gr.update(), "⚠️ 请先输入并确认物体名称"
            # evt.index → [x, y] 是浏览器显示空间的坐标
            # 需要除以 scale 转换回原图像素坐标（用于 SAM2 推理）
            frames = _get_frames(state)
            scale = compute_display_scale(frames[0])
            x_orig = float(evt.index[0]) / scale
            y_orig = float(evt.index[1]) / scale
            state["current_points"].append([x_orig, y_orig])
            return (
                state,
                get_display_image(state),
                build_status_text(state),
            )

        def h_undo_point(state):
            """撤销最后一个标注点."""
            if state["phase"] == "clicking" and state["current_points"]:
                state["current_points"].pop()
            return (
                state,
                get_display_image(state),
                build_status_text(state),
            )

        def h_undo_obj(state):
            """删除最后一个已确认的物体."""
            msg = ""
            if state["completed_objects"]:
                removed = state["completed_objects"].pop()
                msg = f"\n🗑️ 已删除: {removed['name']}"
            else:
                msg = "\n(没有可删除的物体)"
            return (
                state,
                get_display_image(state),
                build_status_text(state) + msg,
            )

        def h_confirm_obj(state):
            """确认当前物体标注 → 可以继续下一个物体."""
            if state["phase"] != "clicking":
                return state, gr.update(), "⚠️ 没有正在标注的物体", gr.update()
            if not state["current_points"]:
                return state, gr.update(), "⚠️ 请至少点击一个标注点", gr.update()

            obj = {
                "name":   state["current_obj_name"],
                "points": list(state["current_points"]),
            }
            state["completed_objects"].append(obj)
            state["current_obj_name"] = ""
            state["current_points"]   = []
            state["phase"]            = "naming"

            s  = build_status_text(state)
            s += f"\n\n✅ 物体 '{obj['name']}' 已确认（{len(obj['points'])} 个点）"
            s += "\n🔄 继续输入下一个物体名称，或点击「完成标注并处理」"
            return (
                state,
                get_display_image(state),
                s,
                "",   # clear name input
            )

        def h_done(state):
            """完成所有标注 → 运行 SAM2 追踪 → 保存结果 → 加载下一个文件."""
            # 如果正在标注且已有点，自动确认
            if state["phase"] == "clicking" and state["current_points"]:
                obj = {
                    "name":   state["current_obj_name"],
                    "points": list(state["current_points"]),
                }
                state["completed_objects"].append(obj)
                state["current_obj_name"] = ""
                state["current_points"]   = []

            if not state["completed_objects"]:
                return (state, gr.update(),
                        "⚠️ 请至少标注一个物体",
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
                status_lines.append(f"   • {o['name']}: {len(o['points'])} 点")

            print(f"\n===== Processing {h5_path.name}  ({len(objects)} objects) =====")

            frames = _get_frames(state)
            try:
                # --- export frames → run SAM2 ---
                with tempfile.TemporaryDirectory(prefix="sam2_frames_") as td:
                    export_frames_to_jpg(frames, Path(td), args.jpg_quality)
                    fidx, bbox, area, rle, masks = run_sam2_on_one_video(
                        predictor, Path(td), objects, args.device,
                    )

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
                "",   # clear name input
            )

        def h_skip(state):
            """跳过当前文件，加载下一个."""
            if state["file_idx"] >= len(pending):
                return (state, gr.update(), "✅ 所有文件已完成",
                        gr.update(), "")
            skipped = pending[state["file_idx"]].name
            print(f"  skipped: {skipped}")
            state["file_idx"] += 1
            state = load_current_file(state)
            return (
                state,
                get_display_image(state),
                f"⏭️ 已跳过 {skipped}\n\n" + build_status_text(state),
                file_info_text(state),
                "",   # clear name input
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
                "",     # clear name input
            )

        # ════════════════════ wire events ════════════════════

        outs_3 = [app_state, image_out, status_box]

        # page load
        demo.load(
            h_load, [app_state],
            [app_state, image_out, info_box, status_box, video_out, obj_name_in],
        )

        # confirm name — button click OR press Enter in textbox
        for trigger in [btn_name.click, obj_name_in.submit]:
            trigger(
                h_confirm_name, [app_state, obj_name_in],
                [app_state, image_out, status_box, obj_name_in],
            )

        # click on image → add point
        image_out.select(h_click, [app_state], outs_3)

        # undo / delete
        btn_undo_pt.click(h_undo_point,  [app_state], outs_3)
        btn_undo_obj.click(h_undo_obj,   [app_state], outs_3)

        # confirm current object
        btn_confirm_obj.click(
            h_confirm_obj, [app_state],
            [app_state, image_out, status_box, obj_name_in],
        )

        # finish annotation → SAM2 processing
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

    print(f"Launching Gradio on port {args.port} …")
    demo.launch(
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(output_dir)],
    )


if __name__ == "__main__":
    main()
