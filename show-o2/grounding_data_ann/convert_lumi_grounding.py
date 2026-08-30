#!/usr/bin/env python3
"""
Convert LumiData COCO grounding annotations into RobotGroundingDataset HDF5.

Output layout matches libero_*_regen_split_grounding:
  language_instruction  (from annotations.json info.task)
  rgb_comb              [T] JPEG bytes, each 336x320
  grounding/
    object_names, object_ids, frame_idx, bbox_xywh [T,N,4], area [T,N], rle [T,N]

Sources per episode (exported as separate clips when present):
  - grounding/                              → shelf multi-object, main only (wrist black)
  - grounding_pick_up_target_only/rgb_main  → pickup target (+ wrist if annotated)

Main-view crop/resize matches convert_lumi.py / convert_lumi_data_heatmap.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from export_grounding_video import render_grounding_video

# ── geometry (shared with meta_lumi_data/convert_lumi.py) ────────────────────
MAIN_CAMERA_CROP_BOX = (300, 20, 1280 - 220, 720)  # PIL LTRB, right/bottom exclusive
MAIN_TGT_SIZE = (224, 320)  # H, W
WRIST_TGT_SIZE = (112, 160)
COMB_SIZE = (336, 320)  # H, W
WRIST_OFFSET = (MAIN_TGT_SIZE[0], MAIN_TGT_SIZE[1] // 2)  # (y0, x0) bottom-right

# Same SAM3 category names as openpi/examples/lumi/convert_lumi_data_heatmap.py,
# mapped to natural instruction phrases from Lumi info.task wording.
CATEGORY_DISPLAY_NAMES = {
    "orange juice": "a bottle of orange juice",
    "lay's potato chips": "a mini can of Lay's potato chips",
    "mineral water": "a bottle of mineral water",
    "Coca-Cola": "a can of Coca-Cola",
    "coconut drink": "a carton of coconut drink",
    "Sprite": "a can of Sprite",
    "Garden Wafer Biscuits": "a box of Garden Wafer Biscuits",
    "Oreo Cocoa Crispy Rolls": "a box of Oreo Cocoa Crispy Rolls",
    "grapefruit drink": "a pouch of grapefruit drink",
}


def display_name(category_name: str, task_dir: str = "") -> str:
    """Map SAM3 category_name → natural phrase aligned with info.task."""
    key = category_name.strip()
    if key in CATEGORY_DISPLAY_NAMES:
        return CATEGORY_DISPLAY_NAMES[key]
    for k, v in CATEGORY_DISPLAY_NAMES.items():
        if k.casefold() == key.casefold():
            return v
    return key


RAW_DATA_META = [
    ("get_orange_juice_new",),
    ("get_potato_chips_new",),
    ("get_mineral_water_new",),
    ("get_coca_cola_can",),
    ("get_coconut_drink",),
    ("get_sprite_can",),
    ("get_biscuits",),
    ("get_grapefruit_drink",),
    ("get_oreo_rolls",),
    ("get2_biscuits",),
    ("get2_coca_cola_can",),
    ("get2_coconut_drink",),
    ("get2_grapefruit_drink",),
    ("get2_mineral_water_new",),
    ("get2_orange_juice_new",),
    ("get2_oreo_rolls",),
    ("get2_potato_chips_new",),
    ("get2_sprite_can",),
]


def sanitize_filename_part(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "clip"


def encode_frame_jpeg(frame_rgb: np.ndarray, quality: int = 95) -> np.ndarray:
    assert frame_rgb.dtype == np.uint8 and frame_rgb.ndim == 3
    pil_image = Image.fromarray(frame_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def load_main_rgb(ep_dir: Path, frame_id: int, crop_box=MAIN_CAMERA_CROP_BOX) -> np.ndarray:
    path = ep_dir / "colors" / f"{frame_id:06d}_rgb_main.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        assert img.size == (1280, 720), f"{path}: expected (1280, 720), got {img.size}"
        img = img.crop(crop_box)
        return np.asarray(img.convert("RGB"))


def load_wrist_rgb(ep_dir: Path, frame_id: int) -> Optional[np.ndarray]:
    path = ep_dir / "colors" / f"{frame_id:06d}_rgb_wrist_0.jpg"
    if not path.is_file():
        return None
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def combine_main_wrist(
    main_img: np.ndarray,
    wrist_img: Optional[np.ndarray],
    *,
    black_wrist: bool,
) -> np.ndarray:
    """main → top 224x320; wrist → bottom-right 112x160 (or black)."""
    main_r = np.array(
        Image.fromarray(main_img).resize((MAIN_TGT_SIZE[1], MAIN_TGT_SIZE[0]), Image.BILINEAR)
    )
    comb = np.zeros((COMB_SIZE[0], COMB_SIZE[1], 3), dtype=np.uint8)
    comb[: MAIN_TGT_SIZE[0]] = main_r
    if not black_wrist and wrist_img is not None:
        wrist_r = np.array(
            Image.fromarray(wrist_img).resize((WRIST_TGT_SIZE[1], WRIST_TGT_SIZE[0]), Image.BILINEAR)
        )
        y0, x0 = WRIST_OFFSET
        comb[y0 : y0 + WRIST_TGT_SIZE[0], x0 : x0 + WRIST_TGT_SIZE[1]] = wrist_r
    return comb


def _rle_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if segmentation is None:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        rle = {
            "size": list(segmentation["size"]),
            "counts": segmentation["counts"],
        }
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    if isinstance(segmentation, list):
        # polygon
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(np.uint8)
    raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")


def _mask_to_rle_json(mask: np.ndarray) -> str:
    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_u8)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return json.dumps({"size": list(rle["size"]), "counts": counts})


def _bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)


def warp_main_ann_to_canvas(
    bbox_xywh: Sequence[float],
    segmentation: Any,
    src_h: int,
    src_w: int,
    crop_box=MAIN_CAMERA_CROP_BOX,
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Map one main-camera annotation into 336x320 canvas (top main region only).
    Returns (bbox_xywh[4], mask_canvas, rle_json, area). Empty if no overlap with crop.
    """
    crop_l, crop_t, crop_r, crop_b = crop_box
    crop_w = float(crop_r - crop_l)
    crop_h = float(crop_b - crop_t)

    full_mask = _rle_to_mask(segmentation, src_h, src_w)
    # Intersect with crop
    cropped = full_mask[crop_t:crop_b, crop_l:crop_r]
    if cropped.size == 0 or cropped.max() == 0:
        # fallback: try bbox intersection only
        x, y, w, h = (float(v) for v in bbox_xywh)
        x1 = max(x, float(crop_l))
        y1 = max(y, float(crop_t))
        x2 = min(x + w, float(crop_r))
        y2 = min(y + h, float(crop_b))
        if x2 <= x1 or y2 <= y1:
            empty = np.zeros(COMB_SIZE, dtype=np.uint8)
            return np.zeros(4, dtype=np.float32), empty, "", 0
        # rasterize bbox rect into crop space
        cropped = np.zeros((int(crop_h), int(crop_w)), dtype=np.uint8)
        rx1 = int(round(x1 - crop_l))
        ry1 = int(round(y1 - crop_t))
        rx2 = int(round(x2 - crop_l))
        ry2 = int(round(y2 - crop_t))
        cropped[ry1:ry2, rx1:rx2] = 1

    # resize crop mask → main tgt
    main_mask = cv2.resize(
        cropped.astype(np.uint8),
        (MAIN_TGT_SIZE[1], MAIN_TGT_SIZE[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros(COMB_SIZE, dtype=np.uint8)
    canvas[: MAIN_TGT_SIZE[0]] = main_mask
    if canvas.max() == 0:
        return np.zeros(4, dtype=np.float32), canvas, "", 0

    bx, by, bw, bh = _bbox_from_mask(canvas)
    rle = _mask_to_rle_json(canvas)
    area = int(canvas.sum())
    return np.array([bx, by, bw, bh], dtype=np.float32), canvas, rle, area


def warp_wrist_ann_to_canvas(
    bbox_xywh: Sequence[float],
    segmentation: Any,
    src_h: int,
    src_w: int,
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """Map wrist annotation into bottom-right 112x160 of 336x320 canvas."""
    full_mask = _rle_to_mask(segmentation, src_h, src_w)
    if full_mask.max() == 0:
        x, y, w, h = (float(v) for v in bbox_xywh)
        if w <= 0 or h <= 0:
            empty = np.zeros(COMB_SIZE, dtype=np.uint8)
            return np.zeros(4, dtype=np.float32), empty, "", 0
        full_mask = np.zeros((src_h, src_w), dtype=np.uint8)
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        full_mask[max(0, y1) : min(src_h, y2), max(0, x1) : min(src_w, x2)] = 1

    wrist_mask = cv2.resize(
        full_mask.astype(np.uint8),
        (WRIST_TGT_SIZE[1], WRIST_TGT_SIZE[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros(COMB_SIZE, dtype=np.uint8)
    y0, x0 = WRIST_OFFSET
    canvas[y0 : y0 + WRIST_TGT_SIZE[0], x0 : x0 + WRIST_TGT_SIZE[1]] = wrist_mask
    if canvas.max() == 0:
        return np.zeros(4, dtype=np.float32), canvas, "", 0
    bx, by, bw, bh = _bbox_from_mask(canvas)
    rle = _mask_to_rle_json(canvas)
    return np.array([bx, by, bw, bh], dtype=np.float32), canvas, rle, int(canvas.sum())


def load_coco(path: Path) -> Optional[dict]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    if not coco.get("annotations"):
        return None
    return coco


def _best_ann_per_category(anns: List[dict]) -> Dict[str, dict]:
    """category_name -> best annotation (score, area)."""
    best: Dict[str, dict] = {}
    for ann in anns:
        name = str(ann.get("category_name") or "").strip()
        if not name:
            continue
        key = (float(ann.get("score", 0.0)), float(ann.get("area", 0.0)))
        prev = best.get(name)
        if prev is None:
            best[name] = ann
            continue
        prev_key = (float(prev.get("score", 0.0)), float(prev.get("area", 0.0)))
        if key > prev_key:
            best[name] = ann
    return best


def build_frame_index(coco: dict) -> Tuple[List[int], Dict[int, dict], Dict[int, List[dict]]]:
    """Return sorted frame_ids, image_by_frame, anns_by_frame."""
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    frame_ids = []
    image_by_frame: Dict[int, dict] = {}
    anns_by_frame: Dict[int, List[dict]] = {}
    for image_id, im in images_by_id.items():
        fid = int(im["frame_id"])
        frame_ids.append(fid)
        image_by_frame[fid] = im
        anns_by_frame[fid] = anns_by_image.get(image_id, [])
    frame_ids = sorted(set(frame_ids))
    return frame_ids, image_by_frame, anns_by_frame


def convert_shelf_clip(
    ep_dir: Path,
    coco: dict,
    output_path: Path,
    *,
    overwrite: bool,
    write_preview: bool,
    min_frames: int,
) -> bool:
    """Multi-object main-only grounding → one HDF5 (wrist black)."""
    if output_path.exists() and not overwrite:
        return False

    language = str(coco.get("info", {}).get("task") or "").strip()
    if not language:
        raise ValueError(f"{ep_dir}: grounding info.task missing")

    frame_ids, image_by_frame, anns_by_frame = build_frame_index(coco)
    cat_names = [c["name"] for c in coco.get("categories", []) if c.get("name")]
    if not cat_names:
        cat_names = sorted(
            {str(a["category_name"]) for a in coco["annotations"] if a.get("category_name")}
        )
    if not cat_names:
        return False
    task_dir = ep_dir.parent.name
    display_names = [display_name(n, task_dir) for n in cat_names]
    name_to_idx = {n: i for i, n in enumerate(cat_names)}
    n_obj = len(cat_names)

    list_rgb, list_bbox, list_area, list_rle = [], [], [], []

    for fid in frame_ids:
        im = image_by_frame[fid]
        src_h, src_w = int(im["height"]), int(im["width"])
        try:
            main_img = load_main_rgb(ep_dir, fid)
        except FileNotFoundError:
            continue
        if np.all(main_img == 0):
            continue

        comb = combine_main_wrist(main_img, None, black_wrist=True)
        bbox_row = np.zeros((n_obj, 4), dtype=np.float32)
        area_row = np.zeros((n_obj,), dtype=np.int32)
        rle_row = [""] * n_obj

        best = _best_ann_per_category(anns_by_frame.get(fid, []))
        any_valid = False
        for cat_name, ann in best.items():
            j = name_to_idx.get(cat_name)
            if j is None:
                continue
            bb, _, rle, area = warp_main_ann_to_canvas(
                ann["bbox"], ann.get("segmentation"), src_h, src_w
            )
            if area <= 0:
                continue
            bbox_row[j] = bb
            area_row[j] = area
            rle_row[j] = rle
            any_valid = True

        if not any_valid:
            continue

        list_rgb.append(comb)
        list_bbox.append(bbox_row)
        list_area.append(area_row)
        list_rle.append(rle_row)

    return _finalize_and_write(
        output_path=output_path,
        language=language,
        cat_names=cat_names,
        display_names=display_names,
        list_rgb=list_rgb,
        list_bbox=list_bbox,
        list_area=list_area,
        list_rle=list_rle,
        min_frames=min_frames,
        write_preview=write_preview,
        overwrite=overwrite,
    )


def convert_pickup_clip(
    ep_dir: Path,
    coco_main: dict,
    coco_wrist: Optional[dict],
    output_path: Path,
    *,
    overwrite: bool,
    write_preview: bool,
    min_frames: int,
) -> bool:
    """
    Pickup target clip. If wrist annotations exist, object_names has 2 slots
    (main, wrist) with the same display name — matching libero dual-view pattern.
    Otherwise wrist region is black and N=1.
    """
    if output_path.exists() and not overwrite:
        return False

    language = str(coco_main.get("info", {}).get("task") or "").strip()
    if not language:
        raise ValueError(f"{ep_dir}: pickup info.task missing")

    frame_ids, image_by_frame, anns_by_frame = build_frame_index(coco_main)
    # target category (usually one)
    cat_names = [c["name"] for c in coco_main.get("categories", []) if c.get("name")]
    if not cat_names:
        cat_names = sorted(
            {
                str(a["category_name"])
                for a in coco_main["annotations"]
                if a.get("category_name")
            }
        )
    if not cat_names:
        return False
    task_dir = ep_dir.parent.name
    # Prefer single primary category (task target); if multiple, keep all as main-only cols
    use_wrist = coco_wrist is not None and bool(coco_wrist.get("annotations"))
    if use_wrist and len(cat_names) == 1:
        raw_name = cat_names[0]
        disp = display_name(raw_name, task_dir)
        slot_raw = [raw_name, raw_name]
        slot_disp = [disp, disp]
    else:
        use_wrist = False  # multi-cat pickup: main only
        slot_raw = list(cat_names)
        slot_disp = [display_name(n, task_dir) for n in slot_raw]

    name_to_main_idx = {n: i for i, n in enumerate(cat_names)}
    n_obj = len(slot_disp)

    wrist_by_frame: Dict[int, List[dict]] = {}
    wrist_image_by_frame: Dict[int, dict] = {}
    if use_wrist:
        w_fids, w_ims, w_anns = build_frame_index(coco_wrist)
        wrist_by_frame = w_anns
        wrist_image_by_frame = w_ims

    list_rgb, list_bbox, list_area, list_rle = [], [], [], []

    for fid in frame_ids:
        im = image_by_frame[fid]
        src_h, src_w = int(im["height"]), int(im["width"])
        try:
            main_img = load_main_rgb(ep_dir, fid)
        except FileNotFoundError:
            continue
        if np.all(main_img == 0):
            continue

        wrist_img = load_wrist_rgb(ep_dir, fid) if use_wrist else None
        comb = combine_main_wrist(main_img, wrist_img, black_wrist=not use_wrist)

        bbox_row = np.zeros((n_obj, 4), dtype=np.float32)
        area_row = np.zeros((n_obj,), dtype=np.int32)
        rle_row = [""] * n_obj

        best_main = _best_ann_per_category(anns_by_frame.get(fid, []))
        any_valid = False

        if use_wrist:
            # slot 0 = main, slot 1 = wrist
            raw_name = cat_names[0]
            ann = best_main.get(raw_name)
            if ann is not None:
                bb, _, rle, area = warp_main_ann_to_canvas(
                    ann["bbox"], ann.get("segmentation"), src_h, src_w
                )
                if area > 0:
                    bbox_row[0] = bb
                    area_row[0] = area
                    rle_row[0] = rle
                    any_valid = True
            w_anns = wrist_by_frame.get(fid, [])
            best_w = _best_ann_per_category(w_anns)
            w_ann = best_w.get(raw_name)
            if w_ann is not None:
                w_im = wrist_image_by_frame.get(fid, {})
                wh = int(w_im.get("height", 480))
                ww = int(w_im.get("width", 640))
                bb, _, rle, area = warp_wrist_ann_to_canvas(
                    w_ann["bbox"], w_ann.get("segmentation"), wh, ww
                )
                if area > 0:
                    bbox_row[1] = bb
                    area_row[1] = area
                    rle_row[1] = rle
                    any_valid = True
        else:
            for cat_name, ann in best_main.items():
                j = name_to_main_idx.get(cat_name)
                if j is None or j >= n_obj:
                    continue
                bb, _, rle, area = warp_main_ann_to_canvas(
                    ann["bbox"], ann.get("segmentation"), src_h, src_w
                )
                if area <= 0:
                    continue
                bbox_row[j] = bb
                area_row[j] = area
                rle_row[j] = rle
                any_valid = True

        if not any_valid:
            continue

        list_rgb.append(comb)
        list_bbox.append(bbox_row)
        list_area.append(area_row)
        list_rle.append(rle_row)

    return _finalize_and_write(
        output_path=output_path,
        language=language,
        cat_names=slot_raw,
        display_names=slot_disp,
        list_rgb=list_rgb,
        list_bbox=list_bbox,
        list_area=list_area,
        list_rle=list_rle,
        min_frames=min_frames,
        write_preview=write_preview,
        overwrite=overwrite,
    )


def _finalize_and_write(
    *,
    output_path: Path,
    language: str,
    cat_names: List[str],
    display_names: List[str],
    list_rgb: List[np.ndarray],
    list_bbox: List[np.ndarray],
    list_area: List[np.ndarray],
    list_rle: List[List[str]],
    min_frames: int,
    write_preview: bool,
    overwrite: bool,
) -> bool:
    t = len(list_rgb)
    if t < min_frames:
        return False

    n_obj = len(display_names)
    assert all(b.shape == (n_obj, 4) for b in list_bbox)
    assert len(display_names) == len(cat_names) == n_obj

    bbox = np.stack(list_bbox, axis=0).astype(np.float32)
    area = np.stack(list_area, axis=0).astype(np.int32)
    rle = np.array(list_rle, dtype=object)
    frame_idx = np.arange(t, dtype=np.int32)
    object_ids = np.arange(1, n_obj + 1, dtype=np.int32)

    rgb_bytes = np.empty(t, dtype=object)
    for i, frame in enumerate(list_rgb):
        assert frame.shape == COMB_SIZE + (3,), frame.shape
        rgb_bytes[i] = encode_frame_jpeg(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and overwrite:
        output_path.unlink()

    str_dtype = h5py.string_dtype(encoding="utf-8")
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(output_path, "w") as f:
        f.create_dataset("language_instruction", data=language, dtype=str_dtype)
        f.create_dataset("rgb_comb", data=rgb_bytes, dtype=vlen_uint8)
        g = f.create_group("grounding")
        g.create_dataset("object_ids", data=object_ids)
        name_ds = g.create_dataset("object_names", shape=(n_obj,), dtype=str_dtype)
        for i, name in enumerate(display_names):
            name_ds[i] = name
        g.create_dataset("frame_idx", data=frame_idx)
        g.create_dataset("bbox_xywh", data=bbox)
        g.create_dataset("area", data=area)
        rle_ds = g.create_dataset("rle", shape=rle.shape, dtype=str_dtype)
        for ti in range(t):
            for ni in range(n_obj):
                rle_ds[ti, ni] = rle[ti, ni]

    if write_preview:
        mp4_path = output_path.with_suffix(".mp4")
        # Same visualization as export_grounding_video.py: per-object mask+bbox+label.
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in list_rgb]
        h, w = COMB_SIZE
        masks = np.zeros((t, n_obj, h, w), dtype=np.uint8)
        for ti in range(t):
            for ni in range(n_obj):
                rle_str = rle[ti, ni]
                if not rle_str:
                    continue
                if isinstance(rle_str, bytes):
                    rle_str = rle_str.decode("utf-8")
                rle_dict = json.loads(rle_str)
                if isinstance(rle_dict["counts"], str):
                    rle_dict["counts"] = rle_dict["counts"].encode("utf-8")
                masks[ti, ni] = mask_utils.decode(rle_dict)
        render_grounding_video(
            mp4_path,
            frames_bgr,
            bbox,
            masks,
            object_names=display_names,
            fps=10,
        )

    return True


def process_episode(
    ep_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    write_preview: bool,
    min_frames: int,
) -> List[str]:
    """Convert one episode; return list of written (or existing) hdf5 paths."""
    written: List[str] = []
    task_name = ep_dir.parent.name
    ep_name = ep_dir.name

    shelf_coco = load_coco(ep_dir / "grounding" / "annotations.json")
    if shelf_coco is not None:
        out = output_dir / f"{task_name}_{ep_name}_goto_grounding.hdf5"
        ok = convert_shelf_clip(
            ep_dir,
            shelf_coco,
            out,
            overwrite=overwrite,
            write_preview=write_preview,
            min_frames=min_frames,
        )
        if ok or out.exists():
            # absolute() keeps symlink prefix
            written.append(str(out.absolute()))

    pickup_main = load_coco(
        ep_dir / "grounding_pick_up_target_only" / "rgb_main" / "annotations.json"
    )
    if pickup_main is not None:
        pickup_wrist = load_coco(
            ep_dir / "grounding_pick_up_target_only" / "rgb_wrist_0" / "annotations.json"
        )
        out = output_dir / f"{task_name}_{ep_name}_pickup_grounding.hdf5"
        ok = convert_pickup_clip(
            ep_dir,
            pickup_main,
            pickup_wrist,
            out,
            overwrite=overwrite,
            write_preview=write_preview,
            min_frames=min_frames,
        )
        if ok or out.exists():
            written.append(str(out.absolute()))

    return written


def write_metainfo(datalist: List[str], output_json: Path, dataset_name: str, data_dirs: List[str]):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset_name": dataset_name,
        "data_dirs": data_dirs,
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": len(datalist),
        "datalist": datalist,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"Wrote {output_json} with {len(datalist)} clips")


def main():
    parser = argparse.ArgumentParser(description="Convert Lumi grounding COCO → RobotGrounding HDF5")
    parser.add_argument(
        "--lumi_root",
        type=str,
        default="/home/hyx/LumiData",
        help="Root of raw LumiData",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyx/datasets/Lumi_grounding",
        help="Directory for output HDF5 files",
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        default="/home/hyx/ShowVLA/show-o2/grounding_data_ann/meta_lumi",
        help="Directory for metainfo JSON",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task folder names under lumi_root (default: get*/get2*)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        nargs="*",
        default=None,
        help="Optional episode names filter, e.g. episode_0002",
    )
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_preview", action="store_true")
    parser.add_argument(
        "--meta_name",
        type=str,
        default="lumi_grounding_metainfo.json",
        help="Metainfo filename under meta_dir",
    )
    args = parser.parse_args()

    lumi_root = Path(args.lumi_root)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        task_dirs = [lumi_root / t for t in args.tasks]
    else:
        task_dirs = [lumi_root / t[0] for t in RAW_DATA_META]

    all_eps: List[Path] = []
    used_task_dirs = []
    for td in task_dirs:
        if not td.is_dir():
            print(f"Skip missing task dir: {td}")
            continue
        used_task_dirs.append(str(td.resolve()))
        eps = sorted(td.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
        if args.episodes:
            eps = [e for e in eps if e.name in set(args.episodes)]
        all_eps.extend(eps)

    datalist: List[str] = []
    for ep_dir in tqdm(all_eps, desc="episodes"):
        try:
            paths = process_episode(
                ep_dir,
                output_dir,
                overwrite=args.overwrite,
                write_preview=args.write_preview,
                min_frames=args.min_frames,
            )
            datalist.extend(paths)
        except Exception as e:
            print(f"[ERROR] {ep_dir}: {e}", file=sys.stderr)
            raise

    # de-dup preserve order
    seen = set()
    uniq = []
    for p in datalist:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    # Prefer absolute() over resolve() so symlink prefix is preserved
    # (/home/hyx/datasets -> /hyx_datasets).
    out_dir_str = str(output_dir.absolute())
    write_metainfo(
        uniq,
        meta_dir / args.meta_name,
        dataset_name="Lumi-grounding",
        data_dirs=[out_dir_str],
    )

    # Also split shelf / pickup metas for convenience
    goto = [p for p in uniq if p.endswith("_goto_grounding.hdf5")]
    pickup = [p for p in uniq if p.endswith("_pickup_grounding.hdf5")]
    write_metainfo(
        goto,
        meta_dir / "lumi_goto_grounding_metainfo.json",
        dataset_name="Lumi-goto-grounding",
        data_dirs=[out_dir_str],
    )
    write_metainfo(
        pickup,
        meta_dir / "lumi_pickup_grounding_metainfo.json",
        dataset_name="Lumi-pickup-grounding",
        data_dirs=[out_dir_str],
    )


if __name__ == "__main__":
    main()
