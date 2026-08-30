import h5py
import numpy as np
import cv2
import mediapy as media
from pathlib import Path
import json
from pycocotools import mask as mask_utils

# 复用 sam2_label_server.py 中的辅助函数
def decode_jpeg_object(obj):
    arr = np.asarray(obj, dtype=np.uint8).reshape(-1)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG bytes from HDF5 element.")
    return img_bgr

COLORS_BGR = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
    (255, 255,   0), (255,   0, 255), (  0, 255, 255),
    (128,   0, 255), (255, 128,   0), (  0, 128, 255),
]

def render_grounding_video(video_path, frames_bgr, bbox_xywh, masks,
                           object_names=None, fps=10):
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

    media.write_video(str(video_path), rendered_rgb, fps=fps)

def export_hdf5_to_video(h5_path, out_video_path, frames_key="rgb_comb"):
    print(f"Reading {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        # 读取帧
        ds = f[frames_key]
        frames_bgr = [decode_jpeg_object(ds[i]) for i in range(len(ds))]
        
        if "grounding" not in f:
            print(f"Error: 'grounding' group not found in {h5_path}")
            return

        g = f["grounding"]
        object_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in g["object_names"][()]]
        bbox_xywh = g["bbox_xywh"][()]
        rle_data = g["rle"][()]
        
        T, N = bbox_xywh.shape[:2]
        h, w = frames_bgr[0].shape[:2]
        
        # 从 RLE 恢复 masks
        masks = np.zeros((T, N, h, w), dtype=np.uint8)
        for t in range(T):
            for n in range(N):
                rle_str = rle_data[t, n]
                if rle_str:
                    if isinstance(rle_str, bytes):
                        rle_str = rle_str.decode("utf-8")
                    rle_dict = json.loads(rle_str)
                    # pycocotools 需要 counts 为 bytes
                    if isinstance(rle_dict["counts"], str):
                        rle_dict["counts"] = rle_dict["counts"].encode("utf-8")
                    mask = mask_utils.decode(rle_dict)
                    masks[t, n] = mask

        print(f"Rendering video to {out_video_path}...")
        render_grounding_video(out_video_path, frames_bgr, bbox_xywh, masks, object_names)
        print("Done!")

if __name__ == "__main__":
    h5_file = "/home/hyx/datasets/JAKA_grounding/clutter_put/clutter_put_Doraemon_doll_episode_0003_grounding.hdf5"
    # 修改输出路径到工作目录
    out_video = "./clutter_put_Doraemon_doll_episode_0003_grounding.mp4"
    export_hdf5_to_video(h5_file, out_video)
