"""
Convert AgiBot World Challenge Manipulation-SimData subtask clips to ShowVLA HDF5.

Each ``label_info.action_config`` entry becomes one HDF5 + MP4 pair.
Language uses English only (``english_action_text``).

Expected raw episode layout:
  {task_dir}/{task_id}/{job_id}/{sn_code}/{episode_id}/
    aligned_joints.h5
    camera/{frame_idx}/head_color.jpg
    camera/{frame_idx}/hand_left_color.jpg
    camera/{frame_idx}/hand_right_color.jpg

Output HDF5 (matches AGIBOTHDF5Handler):
  /language_instruction
  /rgb_comb
  /actions/end/position        [T, 2, 3]  EE xyz in robot base frame
  /actions/end/orientation     [T, 2, 4]  EE quat (wxyz) in robot base frame
  /actions/effector/position   [T, 2]

Sim raw ``action/end/*`` is in large-scene world coordinates. Convert transforms
EE pose into the robot base frame via ``state/robot/{position,orientation}``
(wxyz; falls back to ``parameters/camera/state.json`` robot pose). Without this,
cross-episode origins differ by ~10m and MSE action loss blows up vs Real/Libero.

MP4 name: {cur_episode}_{episode_id}_{clip_id}_{english_action_text}.mp4
HDF5 name: {cur_episode}_{episode_id}_{clip_id}.hdf5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import h5py
import mediapy as media
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

AGIBOT_FPS = 30
MIN_FRAMES = 10
MAIN_TGT_SIZE = (224, 320)
WRIST_TGT_SIZE = (112, 160)
COMB_SIZE = (336, 320)
# Discard clip on horizontal teleport impulses: Δxy jumps this far from a quiet
# previous step (meters). Also catch upward z-pops (dz > soft floor) with
# nontrivial ‖Δxyz‖ after a quiet prev (covers small-xy flips). Post-jump settle
# is allowed; sustained pre-motion and z-drops are kept.
DEFAULT_MAX_OBJECT_TRANS_DIFF = 0.03
DEFAULT_OBJECT_XY_QUIET = 0.005
DEFAULT_OBJECT_ZPOP_MIN_DZ = 0.017
DEFAULT_OBJECT_ZPOP_MIN_XYZ = 0.015
# Cap successful exports per english_action_text (0 = no cap).
DEFAULT_MAX_CLIPS_PER_LANG = 100
# Long-clip temporal downsample: Prefer dropping true-idle frames, then
# uniform subsample. 0 disables. ~300 keeps short-mode Place/Threw and Pull.
# Cap temporal compression at max_downsample_rate (final_len >= ceil(T / rate)).
# Floor compression at min_downsample_rate when downsampling (kept <= floor(T / min_rate)).
DEFAULT_MAX_FRAMES = 0
DEFAULT_MAX_DOWNSAMPLE_RATE = 3.5
DEFAULT_MIN_DOWNSAMPLE_RATE = 2.0
# True idle after world→base. Tiny floors absorb float noise; do not copy
# Real's 5e-4/5e-3 (those treat slow motion as idle).
# trans 1e-7: catches ~6e-8 leftovers; grip stays exact 0 (not transformed).
# ori 1e-5: float32 unit quats yield 2*arccos(q·q)~1e-8 on identical frames.
DEFAULT_IDLE_TRANS_EPS = 1e-7
DEFAULT_IDLE_ORI_EPS = 1e-5
DEFAULT_IDLE_GRIP_EPS = 0.0
# Adjacent-frame RGB jump filter (after temporal downsample).
# Loose when state.json has tracked objects in the clip.
# Medium for a small set of high-false-positive language instructions.
# Strict otherwise (missing state or empty objects).
MAIN_MAX_ADJACENT_DIFF_LOOSE = 6.98
WRIST_MAX_ADJACENT_DIFF_LOOSE = 38.05
MAIN_MAX_ADJACENT_DIFF_MEDIUM = 6.05
WRIST_MAX_ADJACENT_DIFF_MEDIUM = 17.0
MAIN_MAX_ADJACENT_DIFF_STRICT = 5.9
WRIST_MAX_ADJACENT_DIFF_STRICT = 11.6
# Per-instruction medium thresholds (override loose/strict tier for these langs).
MEDIUM_IMAGE_JUMP_LANGS = frozenset({
    "Place the bread slice onto the lettuce slice in the plate on the table with the right arm",
    "Close the freezer door with both arms",
    "Open the door of the microwave oven with the right arm",
    "Pick up the lettuce slice from the box on the table with the right arm",
    "Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm",
    "Place the caviar held in the right arm into the shopping cart",
    "Place the picked bread slice into the plate on the table with the right arm",
})
# Defaults for helpers / CLI aliases (strict).
MAIN_MAX_ADJACENT_DIFF = MAIN_MAX_ADJACENT_DIFF_STRICT
WRIST_MAX_ADJACENT_DIFF = WRIST_MAX_ADJACENT_DIFF_STRICT

HEAD_COLOR = "head_color.jpg"
HAND_LEFT_COLOR = "hand_left_color.jpg"
HAND_RIGHT_COLOR = "hand_right_color.jpg"
STATE_JSON_REL = os.path.join("parameters", "camera", "state.json")
# Skip these Manipulation-SimData task subdirs when discovering under the root.
EXCLUDED_TASK_NAMES = frozenset({
    "pack_moving_objects_from_conveyor",
})
# (episode_id, clip_id) pairs manually excluded from export.
EXCLUDED_CLIPS = frozenset({
    (12095995, 1),  # heat_the_food: Pick up plate with bread (was 1401_12095995_1)
    (12088490, 5),  # make_a_sandwich: Place lettuce onto ham (was 2439_12088490_5)
    (12088565, 5),  # make_a_sandwich: Place lettuce onto ham (was 2463_12088565_5)
})
# make_a_sandwich: if clip6 length > this × clip5 length, discard both clips.
SANDWICH_TASK_NAME = "make_a_sandwich"
SANDWICH_CLIP6_OVER_CLIP5_RATIO = 3.0


def sandwich_clip5_6_length_discard(action_config: list) -> set[int]:
    """Return {5, 6} when sandwich clip6 frames > ratio × clip5 frames."""
    if len(action_config) < 7:
        return set()
    len5 = int(action_config[5]["end_frame"]) - int(action_config[5]["start_frame"])
    len6 = int(action_config[6]["end_frame"]) - int(action_config[6]["start_frame"])
    if len5 > 0 and len6 > SANDWICH_CLIP6_OVER_CLIP5_RATIO * len5:
        return {5, 6}
    return set()


def encode_frames_to_jpeg_bytes(frames):
    """Compress RGB frames back to JPEG byte arrays for HDF5 storage."""
    encoded = np.empty(len(frames), dtype=object)
    for idx, frame in enumerate(frames):
        assert frame.dtype == np.uint8
        pil_image = Image.fromarray(frame)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded[idx] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return encoded


def resize_view(img: np.ndarray, tgt_size: tuple[int, int]) -> np.ndarray:
    """Resize HWC uint8 image to (height, width)."""
    return np.array(
        Image.fromarray(img).resize((tgt_size[1], tgt_size[0]), Image.BILINEAR),
        dtype=np.uint8,
    )


def max_adjacent_frame_diff(frames: list[np.ndarray]) -> float:
    """Return the largest mean absolute RGB diff between consecutive frames."""
    if len(frames) < 2:
        return 0.0

    max_diff = 0.0
    prev = frames[0].astype(np.float32)
    for frame in frames[1:]:
        diff = float(np.mean(np.abs(frame.astype(np.float32) - prev)))
        max_diff = max(max_diff, diff)
        prev = frame.astype(np.float32)
    return max_diff


def has_adjacent_view_jump(
    main_frames: list[np.ndarray],
    wrist_left_frames: list[np.ndarray],
    wrist_right_frames: list[np.ndarray],
    main_threshold: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_threshold: float = WRIST_MAX_ADJACENT_DIFF,
) -> tuple[bool, str | None]:
    """Return True if head or either wrist has an abnormal adjacent-frame jump."""
    main_diff = max_adjacent_frame_diff(main_frames)
    if main_diff > main_threshold:
        return True, f"main_view_jump({main_diff:.1f}>{main_threshold})"

    left_diff = max_adjacent_frame_diff(wrist_left_frames)
    if left_diff > wrist_threshold:
        return True, f"wrist_left_jump({left_diff:.1f}>{wrist_threshold})"

    right_diff = max_adjacent_frame_diff(wrist_right_frames)
    if right_diff > wrist_threshold:
        return True, f"wrist_right_jump({right_diff:.1f}>{wrist_threshold})"

    return False, None


def normalize_lang_for_image_jump(lang: str) -> str:
    return lang.strip().rstrip(".")


def image_jump_thresholds_for_clip(
    language_instruction: str,
    has_tracked_objects: bool,
    main_loose: float = MAIN_MAX_ADJACENT_DIFF_LOOSE,
    wrist_loose: float = WRIST_MAX_ADJACENT_DIFF_LOOSE,
    main_medium: float = MAIN_MAX_ADJACENT_DIFF_MEDIUM,
    wrist_medium: float = WRIST_MAX_ADJACENT_DIFF_MEDIUM,
    main_strict: float = MAIN_MAX_ADJACENT_DIFF_STRICT,
    wrist_strict: float = WRIST_MAX_ADJACENT_DIFF_STRICT,
) -> tuple[float, float]:
    """Pick head/wrist adjacent-diff thresholds for a clip."""
    if normalize_lang_for_image_jump(language_instruction) in MEDIUM_IMAGE_JUMP_LANGS:
        return main_medium, wrist_medium
    if has_tracked_objects:
        return main_loose, wrist_loose
    return main_strict, wrist_strict


def combine_resized_main_wrists(
    main_resized: np.ndarray,
    wrist_left_resized: np.ndarray,
    wrist_right_resized: np.ndarray,
    main_tgt_size=MAIN_TGT_SIZE,
    wrist_tgt_size=WRIST_TGT_SIZE,
    comb_size=COMB_SIZE,
) -> np.ndarray:
    """Combine pre-resized head (top) + left/right wrist (bottom)."""
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]
    assert wrist_tgt_size[1] * 2 == main_tgt_size[1]

    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[: main_tgt_size[0], :] = main_resized
    comb_img[main_tgt_size[0] :, : wrist_tgt_size[1]] = wrist_left_resized
    comb_img[main_tgt_size[0] :, wrist_tgt_size[1] :] = wrist_right_resized
    return comb_img


def combine_main_wrist_views(
    main_img,
    wrist_left_img,
    wrist_right_img,
    main_tgt_size=MAIN_TGT_SIZE,
    wrist_tgt_size=WRIST_TGT_SIZE,
    comb_size=COMB_SIZE,
):
    """Combine head (top) + left/right wrist (bottom) views."""
    return combine_resized_main_wrists(
        resize_view(main_img, main_tgt_size),
        resize_view(wrist_left_img, wrist_tgt_size),
        resize_view(wrist_right_img, wrist_tgt_size),
        main_tgt_size=main_tgt_size,
        wrist_tgt_size=wrist_tgt_size,
        comb_size=comb_size,
    )


def sanitize_lang_for_filename(lang: str, max_len: int = 120) -> str:
    """Keep readable English; strip only filesystem-unsafe characters."""
    slug = lang.strip()
    slug = slug.replace("/", "-").replace("\\", "-").replace("\0", "")
    slug = re.sub(r"\s+", " ", slug).strip(" .")
    if not slug:
        slug = "no_instruction"
    return slug[:max_len].rstrip(" .")


def setup_seed(seed: int):
    np.random.seed(seed)


def discard(msg: str):
    print(f"[Discard!] {msg}", flush=True)


def episode_dir_from_meta(task_dir: str, ep_meta: dict) -> str:
    return os.path.join(
        task_dir,
        str(ep_meta["task_id"]),
        str(ep_meta["job_id"]),
        str(ep_meta["sn_code"]),
        str(ep_meta["episode_id"]),
    )


def load_state_frames(ep_dir: str):
    """Load ``parameters/camera/state.json`` frames, or None if missing."""
    state_path = os.path.join(ep_dir, STATE_JSON_REL)
    if not os.path.isfile(state_path):
        return None
    with open(state_path, "r") as f:
        state = json.load(f)
    return state.get("frames")


def clip_has_tracked_objects(state_frames, start: int, end: int) -> bool:
    """True if any frame in [start, end) lists an object with a pose."""
    if not state_frames:
        return False
    lo = max(start, 0)
    hi = min(end, len(state_frames))
    for i in range(lo, hi):
        for obj in (state_frames[i].get("objects") or {}).values():
            if isinstance(obj, dict) and "pose" in obj:
                return True
    return False


def _pose_translation(pose) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64)[:3, 3]


def _object_delta_xy(state_frames, frame_idx: int, name: str):
    """Horizontal ‖Δxy‖ of ``name`` between ``frame_idx-1`` and ``frame_idx``, or None."""
    if frame_idx < 1 or frame_idx >= len(state_frames):
        return None
    prev_objs = state_frames[frame_idx - 1].get("objects") or {}
    cur_objs = state_frames[frame_idx].get("objects") or {}
    if name not in prev_objs or name not in cur_objs:
        return None
    if "pose" not in prev_objs[name] or "pose" not in cur_objs[name]:
        return None
    delta_xyz = _pose_translation(cur_objs[name]["pose"]) - _pose_translation(
        prev_objs[name]["pose"]
    )
    return float(np.linalg.norm(delta_xyz[:2]))


def find_extreme_object_jump(
    state_frames,
    start: int,
    end: int,
    max_object_trans_diff: float = DEFAULT_MAX_OBJECT_TRANS_DIFF,
    object_xy_quiet: float = DEFAULT_OBJECT_XY_QUIET,
    object_zpop_min_dz: float = DEFAULT_OBJECT_ZPOP_MIN_DZ,
    object_zpop_min_xyz: float = DEFAULT_OBJECT_ZPOP_MIN_XYZ,
):
    """
    Return (object_name, frame_idx, delta_xy_m) for a teleport impulse inside
    [start, end) after a quiet previous step:
      - large horizontal Δxy, or
      - upward z-pop (dz > soft floor) with ‖Δxyz‖ >= soft floor (small-xy flips).
    Residual settle after the impulse is allowed; z-drops are not flagged.
    """
    if not state_frames:
        return None

    worst = None  # (name, score, frame_idx, kind, delta_xy)
    lo = max(start + 1, 1)
    hi = min(end, len(state_frames))
    for i in range(lo, hi):
        prev_objs = state_frames[i - 1].get("objects") or {}
        cur_objs = state_frames[i].get("objects") or {}
        for name, cur in cur_objs.items():
            if name not in prev_objs:
                continue
            if "pose" not in cur or "pose" not in prev_objs[name]:
                continue
            delta_xyz = _pose_translation(cur["pose"]) - _pose_translation(prev_objs[name]["pose"])
            delta_xy = float(np.linalg.norm(delta_xyz[:2]))
            delta_xyz_n = float(np.linalg.norm(delta_xyz))
            dz = float(delta_xyz[2])

            prev_xy = _object_delta_xy(state_frames, i - 1, name)
            if prev_xy is not None and prev_xy >= object_xy_quiet:
                continue

            # Upward pops only; downward drops (place) are kept.
            if delta_xy >= max_object_trans_diff:
                kind = "xy"
            elif dz > object_zpop_min_dz and delta_xyz_n >= object_zpop_min_xyz:
                kind = "zpop"
            else:
                continue

            score = delta_xy if kind == "xy" else delta_xyz_n
            if worst is None or score > worst[1]:
                worst = (name, score, i, kind, delta_xy)
    if worst is None:
        return None
    name, _score, frame_idx, kind, delta_xy = worst
    return name, frame_idx, delta_xy


def _read_rgb_jpg(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_frame_images(camera_dir: str, frame_idx: int):
    """Load head / left / right JPG for one frame (sequential within the frame)."""
    frame_dir = os.path.join(camera_dir, str(frame_idx))
    head_path = os.path.join(frame_dir, HEAD_COLOR)
    left_path = os.path.join(frame_dir, HAND_LEFT_COLOR)
    right_path = os.path.join(frame_dir, HAND_RIGHT_COLOR)
    if not (
        os.path.isfile(head_path)
        and os.path.isfile(left_path)
        and os.path.isfile(right_path)
    ):
        return None
    return (
        _read_rgb_jpg(head_path),
        _read_rgb_jpg(left_path),
        _read_rgb_jpg(right_path),
    )


def load_clip_frames_parallel(
    camera_dir: str,
    start: int,
    end: int,
    max_workers: int = 8,
) -> tuple[list[np.ndarray] | None, str | None]:
    """
    Load and resize-combine all frames in [start, end).

    Parallelizes across frames (each worker reads 3 view JPGs). Same-frame
    3-way threading is slower here due to pool/GIL overhead on small JPEGs.
    """
    if end <= start:
        return [], None

    def _one(frame_idx: int):
        imgs = load_frame_images(camera_dir, frame_idx)
        if imgs is None:
            return frame_idx, None
        head, left, right = imgs
        head_r = resize_view(head, MAIN_TGT_SIZE)
        left_r = resize_view(left, WRIST_TGT_SIZE)
        right_r = resize_view(right, WRIST_TGT_SIZE)
        return frame_idx, combine_resized_main_wrists(head_r, left_r, right_r)

    workers = max(1, min(max_workers, end - start))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, range(start, end)))

    rgb_comb = []
    for frame_idx, comb in results:
        if comb is None:
            return None, f"missing_image@{frame_idx}"
        rgb_comb.append(comb)
    return rgb_comb, None


def load_robot_base_pose_wxyz(proprio_path: str, frame_idx: int = 0):
    """
    Robot base pose in world frame: translation [3] + quat wxyz [4].

    Prefer ``state/robot/*`` in aligned_joints.h5 (``action/robot/*`` is all-zero
    in Sim). Fall back to ``parameters/camera/state.json`` ``frames[i].robot.pose``.
    Returns None if neither source is available.
    """
    with h5py.File(proprio_path, "r") as f:
        n = len(f["action/end/position"])
        idx = int(np.clip(frame_idx, 0, max(n - 1, 0)))
        if "state/robot/position" in f and "state/robot/orientation" in f:
            pos = np.asarray(f["state/robot/position"][idx], dtype=np.float64).reshape(3)
            quat_wxyz = np.asarray(
                f["state/robot/orientation"][idx], dtype=np.float64
            ).reshape(4)
            return pos, quat_wxyz

    ep_dir = os.path.dirname(proprio_path)
    state_path = os.path.join(ep_dir, STATE_JSON_REL)
    if not os.path.isfile(state_path):
        return None
    with open(state_path, "r") as fh:
        state = json.load(fh)
    frames = state.get("frames") or []
    if not frames:
        return None
    idx = int(np.clip(frame_idx, 0, len(frames) - 1))
    pose = (frames[idx].get("robot") or {}).get("pose")
    if pose is None:
        return None
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape != (4, 4):
        return None
    pos = mat[:3, 3].copy()
    quat_wxyz = Rotation.from_matrix(mat[:3, :3]).as_quat(scalar_first=True)
    return pos, np.asarray(quat_wxyz, dtype=np.float64)


def world_ee_to_robot_base(
    end_position: np.ndarray,
    end_orientation_wxyz: np.ndarray,
    robot_pos: np.ndarray,
    robot_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform dual-arm EE pose from world to robot base.

    ``end_position``: [T, 2, 3], ``end_orientation_wxyz``: [T, 2, 4] (wxyz).
    ``robot_*`` is a single base pose (episode-static in Sim).
    """
    pos = np.asarray(end_position, dtype=np.float64)
    ori = np.asarray(end_orientation_wxyz, dtype=np.float64)
    t_shape = pos.shape[:-1]
    R_wb = Rotation.from_quat(
        np.asarray(robot_quat_wxyz, dtype=np.float64).reshape(4),
        scalar_first=True,
    )
    R_bw = R_wb.inv()
    pos_base = R_bw.apply(pos.reshape(-1, 3) - np.asarray(robot_pos, dtype=np.float64).reshape(3))
    pos_base = pos_base.reshape(t_shape + (3,)).astype(np.float32)

    R_we = Rotation.from_quat(ori.reshape(-1, 4), scalar_first=True)
    ori_base = (R_bw * R_we).as_quat(scalar_first=True).reshape(t_shape + (4,))
    # Keep quaternion hemisphere stable vs input (avoid sign flips).
    dots = np.sum(ori_base * ori, axis=-1, keepdims=True)
    ori_base = np.where(dots < 0.0, -ori_base, ori_base).astype(np.float32)
    return pos_base, ori_base


def read_action_arrays(
    proprio_path: str,
    start: int,
    end: int,
    to_robot_base: bool = False,
):
    """
    Slice dual-arm EE actions for half-open [start, end).

    If ``to_robot_base`` is True (Sim), convert world-frame EE pose into the
    robot base frame. RealRobot data is already base-local; leave False there.
    """
    with h5py.File(proprio_path, "r") as f:
        pos = np.asarray(f["action/end/position"][start:end], dtype=np.float32)
        ori = np.asarray(f["action/end/orientation"][start:end], dtype=np.float32)
        if "action/effector/position" in f:
            grip = np.asarray(f["action/effector/position"][start:end], dtype=np.float32)
        else:
            left = np.asarray(f["action/left_effector/position"][start:end], dtype=np.float32)
            right = np.asarray(f["action/right_effector/position"][start:end], dtype=np.float32)
            if left.ndim == 1:
                left = left[:, None]
            if right.ndim == 1:
                right = right[:, None]
            grip = np.concatenate([left[:, :1], right[:, :1]], axis=-1)
        num_frames = len(f["action/end/position"])

    if to_robot_base:
        base = load_robot_base_pose_wxyz(proprio_path, frame_idx=start)
        if base is None:
            return None, None, None, num_frames
        robot_pos, robot_quat = base
        pos, ori = world_ee_to_robot_base(pos, ori, robot_pos, robot_quat)

    return pos, ori, grip, num_frames


def ee_frame_speed(end_position: np.ndarray) -> np.ndarray:
    """Per-frame max dual-arm EE translation speed (m/frame), shape [T]."""
    pos = np.asarray(end_position, dtype=np.float64)
    t = pos.shape[0]
    speed = np.zeros(t, dtype=np.float64)
    if t < 2:
        return speed
    delta = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)  # [T-1, 2]
    speed[1:] = delta.max(axis=-1)
    speed[0] = speed[1]
    return speed


def ee_ori_frame_speed(end_orientation: np.ndarray) -> np.ndarray:
    """Per-frame max dual-arm orientation change (rad/frame), shape [T]."""
    ori = np.asarray(end_orientation, dtype=np.float64)
    t = ori.shape[0]
    speed = np.zeros(t, dtype=np.float64)
    if t < 2:
        return speed
    # Raw Sim quats are slightly non-unit (|q|~1.0006). Then q·q > 1 and
    # clip→arccos(1)=0; after world→base the float32 quat can have q·q < 1
    # and a fake nonzero angle even on identical consecutive poses.
    norms = np.linalg.norm(ori, axis=-1, keepdims=True)
    ori = ori / np.clip(norms, 1e-12, None)
    dots = np.abs(np.sum(ori[1:] * ori[:-1], axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    ang = 2.0 * np.arccos(dots)  # [T-1, 2]
    speed[1:] = ang.max(axis=-1)
    speed[0] = speed[1]
    return speed


def grip_frame_speed(effector_position: np.ndarray) -> np.ndarray:
    """Per-frame max dual-arm |Δgripper| (effector units/frame), shape [T]."""
    grip = np.asarray(effector_position, dtype=np.float64)
    t = grip.shape[0]
    speed = np.zeros(t, dtype=np.float64)
    if t < 2:
        return speed
    delta = np.abs(grip[1:] - grip[:-1])
    if delta.ndim == 1:
        speed[1:] = delta
    else:
        speed[1:] = delta.reshape(delta.shape[0], -1).max(axis=-1)
    speed[0] = speed[1]
    return speed


def true_idle_mask(
    end_position: np.ndarray,
    end_orientation: np.ndarray,
    effector_position: np.ndarray,
    trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
) -> np.ndarray:
    """
    True idle: EE translation / orientation / gripper speeds at or below eps.

    Sim defaults: trans 1e-7 m/frame, ori 1e-5 rad/frame, grip exact 0.
    """
    trans = ee_frame_speed(end_position)
    ori = ee_ori_frame_speed(end_orientation)
    grip = grip_frame_speed(effector_position)
    return (trans <= trans_eps) & (ori <= ori_eps) & (grip <= grip_eps)


def select_downsample_indices(
    idle: np.ndarray,
    max_frames: int,
) -> np.ndarray:
    """
    Choose up to ``max_frames`` indices in [0, T).

    Prefer dropping true-idle frames while always keeping endpoints; if still
    too long, uniformly subsample the survivors.
    """
    idle = np.asarray(idle, dtype=bool)
    t = int(idle.shape[0])
    if max_frames <= 0 or t <= max_frames:
        return np.arange(t, dtype=np.int64)

    idle = idle.copy()
    idle[0] = False
    idle[-1] = False
    keep = np.flatnonzero(~idle).astype(np.int64)

    # If stripping idle removes almost everything, fall back to all frames.
    if keep.size < max(MIN_FRAMES, max_frames // 2):
        keep = np.arange(t, dtype=np.int64)
    elif keep.size <= max_frames:
        return keep

    sel = np.round(np.linspace(0, keep.size - 1, num=max_frames)).astype(np.int64)
    sel[0] = 0
    sel[-1] = keep.size - 1
    sel = np.unique(sel)
    return keep[sel]


def apply_frame_indices(arrays: dict, indices: np.ndarray) -> dict:
    idx = np.asarray(indices, dtype=np.int64)
    rgb = arrays["rgb_comb"]
    if isinstance(rgb, list):
        rgb_out = [rgb[i] for i in idx.tolist()]
    else:
        rgb_out = rgb[idx]
    return {
        "language_instruction": arrays["language_instruction"],
        "rgb_comb": rgb_out,
        "end_position": arrays["end_position"][idx],
        "end_orientation": arrays["end_orientation"][idx],
        "effector_position": arrays["effector_position"][idx],
    }


def target_frames_with_rate_cap(
    num_frames: int,
    max_frames: int,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
) -> int:
    """
    Frames to keep when downsampling.

    Start from ``max_frames``, then clamp so temporal rate ``T/kept`` lies in
    ``[min_downsample_rate, max_downsample_rate]`` when possible:
      kept <= floor(T / min_rate)   # rate >= min
      kept >= ceil(T / max_rate)    # rate <= max
    """
    if max_frames <= 0 or num_frames <= max_frames:
        return num_frames

    target = int(max_frames)
    min_rate = float(min_downsample_rate)
    max_rate = float(max_downsample_rate)

    if min_rate > 0:
        max_keep = int(math.floor(num_frames / min_rate))
        max_keep = max(max_keep, MIN_FRAMES)
        target = min(target, max_keep)
    if max_rate > 0:
        min_keep = int(math.ceil(num_frames / max_rate))
        target = max(target, min_keep)

    target = min(target, num_frames)
    target = max(target, min(MIN_FRAMES, num_frames))
    return target


def maybe_downsample_long_clip(
    arrays: dict,
    max_frames: int,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
):
    """
    If clip length exceeds ``max_frames``, idle-aware temporal downsample.

    Idle frames: EE translation / orientation / gripper speed at or below eps.
    Compression rate T/kept is clamped to [min_downsample_rate, max_downsample_rate].
    Returns (arrays, info_or_None). ``info`` is a short log string when applied.
    """
    if max_frames <= 0:
        return arrays, None
    t0 = len(arrays["rgb_comb"])
    if t0 <= max_frames:
        return arrays, None

    target = target_frames_with_rate_cap(
        t0,
        max_frames,
        max_downsample_rate=max_downsample_rate,
        min_downsample_rate=min_downsample_rate,
    )
    if target >= t0:
        return arrays, None

    idle = true_idle_mask(
        arrays["end_position"],
        arrays["end_orientation"],
        arrays["effector_position"],
        trans_eps=idle_trans_eps,
        ori_eps=idle_ori_eps,
        grip_eps=idle_grip_eps,
    )
    idle_ratio = float(idle.mean()) if t0 else 0.0
    indices = select_downsample_indices(idle, target)
    arrays = apply_frame_indices(arrays, indices)
    t1 = len(arrays["rgb_comb"])
    rate = (t0 / t1) if t1 else 0.0
    info = (
        f"downsample {t0}->{t1} (true_idle_ratio={idle_ratio:.2f}, "
        f"target={target}, rate={rate:.2f}x, max_frames={max_frames}, "
        f"rate_range=[{min_downsample_rate},{max_downsample_rate}], "
        f"eps_t/o/g=[{idle_trans_eps:g},{idle_ori_eps:g},{idle_grip_eps:g}])"
    )
    return arrays, info


def build_subtask_clip(
    ep_dir: str,
    start: int,
    end: int,
    language_instruction: str,
    state_frames=None,
    max_object_trans_diff: float = DEFAULT_MAX_OBJECT_TRANS_DIFF,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
    main_max_adjacent_diff_loose: float = MAIN_MAX_ADJACENT_DIFF_LOOSE,
    wrist_max_adjacent_diff_loose: float = WRIST_MAX_ADJACENT_DIFF_LOOSE,
    main_max_adjacent_diff_medium: float = MAIN_MAX_ADJACENT_DIFF_MEDIUM,
    wrist_max_adjacent_diff_medium: float = WRIST_MAX_ADJACENT_DIFF_MEDIUM,
    main_max_adjacent_diff_strict: float = MAIN_MAX_ADJACENT_DIFF_STRICT,
    wrist_max_adjacent_diff_strict: float = WRIST_MAX_ADJACENT_DIFF_STRICT,
):
    proprio_path = os.path.join(ep_dir, "aligned_joints.h5")
    camera_dir = os.path.join(ep_dir, "camera")
    if not os.path.isfile(proprio_path):
        return None, "missing_proprio"
    if not os.path.isdir(camera_dir):
        return None, "missing_camera"

    pos, ori, grip, num_frames = read_action_arrays(
        proprio_path, start, end, to_robot_base=True
    )
    if start < 0 or end > num_frames or end <= start:
        return None, "bad_frame_range"
    if pos is None:
        return None, "missing_robot_base_pose"

    if state_frames is None:
        state_frames = load_state_frames(ep_dir)

    has_tracked_objects = clip_has_tracked_objects(state_frames, start, end)
    if has_tracked_objects:
        jump = find_extreme_object_jump(
            state_frames,
            start,
            end,
            max_object_trans_diff=max_object_trans_diff,
        )
        if jump is not None:
            name, frame_idx, delta_xy = jump
            return None, f"object_teleport@{name}(frame={frame_idx},Δxy={delta_xy:.4f}m)"

    main_max_adjacent_diff, wrist_max_adjacent_diff = image_jump_thresholds_for_clip(
        language_instruction,
        has_tracked_objects,
        main_loose=main_max_adjacent_diff_loose,
        wrist_loose=wrist_max_adjacent_diff_loose,
        main_medium=main_max_adjacent_diff_medium,
        wrist_medium=wrist_max_adjacent_diff_medium,
        main_strict=main_max_adjacent_diff_strict,
        wrist_strict=wrist_max_adjacent_diff_strict,
    )

    rgb_comb, img_err = load_clip_frames_parallel(camera_dir, start, end)
    if img_err is not None:
        return None, img_err

    if len(rgb_comb) < MIN_FRAMES:
        return None, "too_few_frames"

    if not (len(rgb_comb) == pos.shape[0] == ori.shape[0] == grip.shape[0]):
        return None, "length_mismatch"

    arrays = {
        "language_instruction": language_instruction,
        "rgb_comb": rgb_comb,
        "end_position": pos,
        "end_orientation": ori,
        "effector_position": grip,
    }
    arrays, ds_info = maybe_downsample_long_clip(
        arrays,
        max_frames=max_frames,
        max_downsample_rate=max_downsample_rate,
        min_downsample_rate=min_downsample_rate,
        idle_trans_eps=idle_trans_eps,
        idle_ori_eps=idle_ori_eps,
        idle_grip_eps=idle_grip_eps,
    )
    if len(arrays["rgb_comb"]) < MIN_FRAMES:
        return None, "too_few_frames_after_downsample"

    # Filter on final frames (after downsample) so subsample gaps are caught.
    final_frames = arrays["rgb_comb"]
    main_h = MAIN_TGT_SIZE[0]
    wrist_w = WRIST_TGT_SIZE[1]
    has_jump, jump_reason = has_adjacent_view_jump(
        [f[:main_h, :] for f in final_frames],
        [f[main_h:, :wrist_w] for f in final_frames],
        [f[main_h:, wrist_w:] for f in final_frames],
        main_threshold=main_max_adjacent_diff,
        wrist_threshold=wrist_max_adjacent_diff,
    )
    if has_jump:
        return None, jump_reason

    return arrays, ds_info


def write_clip_hdf5(h5_path: str, arrays: dict, mp4_path: str):
    rgb_comb_bytes = encode_frames_to_jpeg_bytes(arrays["rgb_comb"])
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset(
            "language_instruction",
            data=arrays["language_instruction"],
            dtype=str_dtype,
        )
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        h5_file.create_dataset("rgb_comb", data=rgb_comb_bytes, dtype=vlen_uint8)

        actions = h5_file.create_group("actions")
        end_grp = actions.create_group("end")
        end_grp.create_dataset("position", data=arrays["end_position"])
        end_grp.create_dataset("orientation", data=arrays["end_orientation"])
        effector_grp = actions.create_group("effector")
        effector_grp.create_dataset("position", data=arrays["effector_position"])

    media.write_video(mp4_path, arrays["rgb_comb"], fps=AGIBOT_FPS)


def discover_task_dirs(root_or_task_dir: str) -> list[str]:
    """Return task dirs that contain task_train.json.

    Accepts either a single task directory or the Manipulation-SimData root.
    Task names in ``EXCLUDED_TASK_NAMES`` are skipped when scanning a root.
    """
    root_or_task_dir = os.path.abspath(root_or_task_dir)
    direct_json = os.path.join(root_or_task_dir, "task_train.json")
    if os.path.isfile(direct_json):
        name = os.path.basename(root_or_task_dir.rstrip(os.sep))
        if name in EXCLUDED_TASK_NAMES:
            print(f"Skipping excluded task: {root_or_task_dir}")
            return []
        return [root_or_task_dir]

    task_dirs = []
    for name in sorted(os.listdir(root_or_task_dir)):
        if name in EXCLUDED_TASK_NAMES:
            print(f"Skipping excluded task: {name}")
            continue
        path = os.path.join(root_or_task_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "task_train.json")):
            task_dirs.append(path)
    return task_dirs


def confirm_fresh_export(output_dir: str, metainfo_json_out_path: str) -> bool:
    """If output targets already exist, ask whether to delete and start fresh."""
    output_dir = os.path.abspath(output_dir)
    metainfo_json_out_path = os.path.abspath(metainfo_json_out_path)

    existing = []
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        existing.append(f"output_dir: {output_dir}")
    elif os.path.exists(output_dir) and not os.path.isdir(output_dir):
        existing.append(f"output_dir (not a directory): {output_dir}")
    if os.path.exists(metainfo_json_out_path):
        existing.append(f"metainfo: {metainfo_json_out_path}")

    if not existing:
        return True

    print("The following export targets already exist:")
    for item in existing:
        print(f"  - {item}")
    answer = input("Delete them and re-export from scratch? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return False

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted directory: {output_dir}")
    elif os.path.exists(output_dir):
        os.remove(output_dir)
        print(f"Deleted file: {output_dir}")

    if os.path.exists(metainfo_json_out_path):
        os.remove(metainfo_json_out_path)
        print(f"Deleted metainfo: {metainfo_json_out_path}")
    return True


def convert_task_to_hdf5(
    task_dir: str,
    output_dir: str,
    task_json_path: str | None = None,
    dataset_name: str = "AGIBOT-HDF5-Sim",
    meta_json: dict | None = None,
    max_object_trans_diff: float = DEFAULT_MAX_OBJECT_TRANS_DIFF,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
    main_max_adjacent_diff_loose: float = MAIN_MAX_ADJACENT_DIFF_LOOSE,
    wrist_max_adjacent_diff_loose: float = WRIST_MAX_ADJACENT_DIFF_LOOSE,
    main_max_adjacent_diff_medium: float = MAIN_MAX_ADJACENT_DIFF_MEDIUM,
    wrist_max_adjacent_diff_medium: float = WRIST_MAX_ADJACENT_DIFF_MEDIUM,
    main_max_adjacent_diff_strict: float = MAIN_MAX_ADJACENT_DIFF_STRICT,
    wrist_max_adjacent_diff_strict: float = WRIST_MAX_ADJACENT_DIFF_STRICT,
    max_clips_per_lang: int = DEFAULT_MAX_CLIPS_PER_LANG,
):
    task_dir = os.path.abspath(task_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if task_json_path is None:
        task_json_path = os.path.join(task_dir, "task_train.json")
    with open(task_json_path, "r") as f:
        episodes = json.load(f)

    if meta_json is None:
        meta_json = {
            "dataset_name": dataset_name,
            "task_dirs": [],
            "language_instruction_key": "language_instruction",
            "observation_key": ["rgb_comb"],
            "num_ep": 0,
            "datalist": [],
        }

    # Resume from last *assigned* index (not success count). Discarded clips
    # still consume an index by design; using len(datalist)/num_ep here would
    # rewind and reuse those gaps across task boundaries.
    cur_episode = int(meta_json.get("last_clip_index", meta_json.get("num_ep", 0)))
    if "task_dirs" not in meta_json:
        meta_json["task_dirs"] = []
    if task_dir not in meta_json["task_dirs"]:
        meta_json["task_dirs"].append(task_dir)

    skip_stats = defaultdict(int)
    downsample_count = 0
    lang_export_counts = defaultdict(int)
    for lang, cnt in (meta_json.get("lang_export_counts") or {}).items():
        lang_export_counts[lang] = int(cnt)
    task_name = os.path.basename(task_dir.rstrip(os.sep))
    print(f"Converting {task_dir}")
    print(f"Loaded {len(episodes)} episodes from {task_json_path}")
    if task_name == SANDWICH_TASK_NAME:
        print(
            f"Sandwich rule: discard clip 5&6 when clip6_len > "
            f"{SANDWICH_CLIP6_OVER_CLIP5_RATIO:g}× clip5_len"
        )

    pbar = tqdm(episodes, desc=task_name)
    for ep_meta in pbar:
        episode_id = ep_meta["episode_id"]
        ep_dir = episode_dir_from_meta(task_dir, ep_meta)
        if not os.path.isdir(ep_dir):
            discard(f"episode_id={episode_id}: missing dir {ep_dir}")
            skip_stats["missing_episode_dir"] += 1
            continue

        action_config = ep_meta.get("label_info", {}).get("action_config", [])
        if not action_config:
            discard(f"episode_id={episode_id}: empty action_config")
            skip_stats["empty_action_config"] += 1
            continue

        sandwich_skip_clips = (
            sandwich_clip5_6_length_discard(action_config)
            if task_name == SANDWICH_TASK_NAME
            else set()
        )

        state_frames = load_state_frames(ep_dir)

        for clip_id, clip in enumerate(action_config):
            lang = (clip.get("english_action_text") or "").strip()
            if not lang:
                discard(f"episode_id={episode_id} clip={clip_id}: missing english_action_text")
                skip_stats["missing_english"] += 1
                continue

            if clip_id in sandwich_skip_clips:
                len5 = int(action_config[5]["end_frame"]) - int(action_config[5]["start_frame"])
                len6 = int(action_config[6]["end_frame"]) - int(action_config[6]["start_frame"])
                discard(
                    f"episode_id={episode_id} clip={clip_id}: "
                    f"sandwich_clip5_6_length(clip6={len6}>{SANDWICH_CLIP6_OVER_CLIP5_RATIO:g}×clip5={len5})"
                )
                skip_stats["sandwich_clip5_6_length"] += 1
                continue

            if max_clips_per_lang > 0 and lang_export_counts[lang] >= max_clips_per_lang:
                discard(
                    f"episode_id={episode_id} clip={clip_id}: "
                    f"lang_cap({lang_export_counts[lang]}>={max_clips_per_lang})"
                )
                skip_stats["lang_cap"] += 1
                continue

            if (episode_id, clip_id) in EXCLUDED_CLIPS:
                discard(f"episode_id={episode_id} clip={clip_id}: blacklisted")
                skip_stats["blacklisted"] += 1
                continue

            start = int(clip["start_frame"])
            end = int(clip["end_frame"])
            cur_episode += 1
            lang_slug = sanitize_lang_for_filename(lang)
            h5_path = os.path.join(output_dir, f"{cur_episode}_{episode_id}_{clip_id}.hdf5")
            mp4_path = os.path.join(
                output_dir, f"{cur_episode}_{episode_id}_{clip_id}_{lang_slug}.mp4"
            )

            arrays, reason = build_subtask_clip(
                ep_dir,
                start,
                end,
                lang,
                state_frames=state_frames,
                max_object_trans_diff=max_object_trans_diff,
                max_frames=max_frames,
                max_downsample_rate=max_downsample_rate,
                min_downsample_rate=min_downsample_rate,
                idle_trans_eps=idle_trans_eps,
                idle_ori_eps=idle_ori_eps,
                idle_grip_eps=idle_grip_eps,
                main_max_adjacent_diff_loose=main_max_adjacent_diff_loose,
                wrist_max_adjacent_diff_loose=wrist_max_adjacent_diff_loose,
                main_max_adjacent_diff_medium=main_max_adjacent_diff_medium,
                wrist_max_adjacent_diff_medium=wrist_max_adjacent_diff_medium,
                main_max_adjacent_diff_strict=main_max_adjacent_diff_strict,
                wrist_max_adjacent_diff_strict=wrist_max_adjacent_diff_strict,
            )
            if arrays is None:
                discard(
                    f"episode_id={episode_id} clip={clip_id} "
                    f"[{start},{end}): {reason}"
                )
                skip_stats[reason.split("@")[0].split("(")[0]] += 1
                continue

            ds_note = ""
            if reason and str(reason).startswith("downsample"):
                downsample_count += 1
                ds_note = f" [{reason}]"

            write_clip_hdf5(h5_path, arrays, mp4_path)
            meta_json["datalist"].append(h5_path)
            lang_export_counts[lang] += 1
            meta_json["lang_export_counts"] = dict(lang_export_counts)

            print(
                f"[{cur_episode}] ep={episode_id} clip={clip_id} "
                f"frames={len(arrays['rgb_comb'])} {lang}{ds_note}",
                flush=True,
            )

    meta_json["num_ep"] = len(meta_json["datalist"])
    meta_json["last_clip_index"] = cur_episode
    meta_json["dataset_name"] = dataset_name

    print("\n=== Task conversion summary ===")
    print(f"Clips so far: {meta_json['num_ep']}")
    print(f"Last clip index: {cur_episode}")
    print(f"Datalist size: {len(meta_json['datalist'])}")
    if max_frames > 0:
        print(f"Downsampled long clips: {downsample_count} (max_frames={max_frames}, rate=[{min_downsample_rate},{max_downsample_rate}])")
    print("Skip stats:")
    for reason, cnt in sorted(skip_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt}")
    return meta_json


def convert_to_hdf5(
    task_dir: str,
    output_dir: str,
    metainfo_json_out_path: str,
    task_json_path: str | None = None,
    dataset_name: str = "AGIBOT-HDF5-Sim",
    max_object_trans_diff: float = DEFAULT_MAX_OBJECT_TRANS_DIFF,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
    main_max_adjacent_diff_loose: float = MAIN_MAX_ADJACENT_DIFF_LOOSE,
    wrist_max_adjacent_diff_loose: float = WRIST_MAX_ADJACENT_DIFF_LOOSE,
    main_max_adjacent_diff_medium: float = MAIN_MAX_ADJACENT_DIFF_MEDIUM,
    wrist_max_adjacent_diff_medium: float = WRIST_MAX_ADJACENT_DIFF_MEDIUM,
    main_max_adjacent_diff_strict: float = MAIN_MAX_ADJACENT_DIFF_STRICT,
    wrist_max_adjacent_diff_strict: float = WRIST_MAX_ADJACENT_DIFF_STRICT,
    max_clips_per_lang: int = DEFAULT_MAX_CLIPS_PER_LANG,
):
    output_dir = os.path.abspath(output_dir)
    metainfo_json_out_path = os.path.abspath(metainfo_json_out_path)

    if not confirm_fresh_export(output_dir, metainfo_json_out_path):
        return

    task_dirs = discover_task_dirs(task_dir)
    if not task_dirs:
        raise FileNotFoundError(
            f"No task_train.json found under {task_dir}. "
            "Pass a task directory or the Manipulation-SimData root."
        )
    if task_json_path and len(task_dirs) > 1:
        raise ValueError("--task_json can only be used with a single task directory.")

    print(f"Found {len(task_dirs)} task(s) under {os.path.abspath(task_dir)}")
    print(
        f"Object-teleport filter: xy>={max_object_trans_diff} m from quiet prev, "
        f"or dz>{DEFAULT_OBJECT_ZPOP_MIN_DZ} m with ‖Δxyz‖>={DEFAULT_OBJECT_ZPOP_MIN_XYZ} m "
        f"(skipped when state.json missing)"
    )
    print(
        "EE pose: world → robot base via state/robot "
        "(fallback: state.json robot.pose); action/robot is unused (all-zero in Sim)"
    )
    print(
        "Image-jump filter (after downsample): "
        f"tracked objects → head>{main_max_adjacent_diff_loose} / "
        f"wrist>{wrist_max_adjacent_diff_loose}; "
        f"{len(MEDIUM_IMAGE_JUMP_LANGS)} medium langs → head>{main_max_adjacent_diff_medium} / "
        f"wrist>{wrist_max_adjacent_diff_medium}; "
        f"else → head>{main_max_adjacent_diff_strict} / "
        f"wrist>{wrist_max_adjacent_diff_strict}"
    )
    if max_frames > 0:
        print(
            f"Long-clip downsample: T>{max_frames} → drop true-idle "
            f"(EE/ori/grip speed ≤ {idle_trans_eps:g}/{idle_ori_eps:g}/{idle_grip_eps:g}), "
            f"else uniform; rate in [{min_downsample_rate}, {max_downsample_rate}]x"
        )
    else:
        print("Long-clip downsample: disabled (pass --max_frames N to enable)")
    if max_clips_per_lang > 0:
        print(f"Per-language export cap: {max_clips_per_lang} clips per english_action_text")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metainfo_json_out_path) or ".", exist_ok=True)

    meta_json = {
        "dataset_name": dataset_name,
        "task_dirs": [],
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": 0,
        "datalist": [],
    }
    for one_task_dir in task_dirs:
        meta_json = convert_task_to_hdf5(
            one_task_dir,
            output_dir,
            task_json_path=task_json_path,
            dataset_name=dataset_name,
            meta_json=meta_json,
            max_object_trans_diff=max_object_trans_diff,
            max_frames=max_frames,
            max_downsample_rate=max_downsample_rate,
            min_downsample_rate=min_downsample_rate,
            idle_trans_eps=idle_trans_eps,
            idle_ori_eps=idle_ori_eps,
            idle_grip_eps=idle_grip_eps,
            main_max_adjacent_diff_loose=main_max_adjacent_diff_loose,
            wrist_max_adjacent_diff_loose=wrist_max_adjacent_diff_loose,
            main_max_adjacent_diff_medium=main_max_adjacent_diff_medium,
            wrist_max_adjacent_diff_medium=wrist_max_adjacent_diff_medium,
            main_max_adjacent_diff_strict=main_max_adjacent_diff_strict,
            wrist_max_adjacent_diff_strict=wrist_max_adjacent_diff_strict,
            max_clips_per_lang=max_clips_per_lang,
        )

    with open(metainfo_json_out_path, "w") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)

    print("\n=== Conversion summary ===")
    print(f"Total clips exported: {meta_json['num_ep']}")
    print(f"Metainfo written to {metainfo_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AgiBot Manipulation-SimData subtask clips to ShowVLA HDF5."
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="/datasets3/agibot_world_challenge_2025/Manipulation-SimData",
        help=(
            "Task directory with task_train.json, or Manipulation-SimData root "
            "containing multiple task subdirectories."
        ),
    )
    parser.add_argument(
        "--task_json",
        type=str,
        default="",
        help="Optional path to task_train.json (only for a single task directory).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyx/datasets/AGIBOT-Sim",
        help="Directory for converted HDF5 and MP4 files.",
    )
    parser.add_argument(
        "--metainfo",
        type=str,
        default="./AGIBOT-HDF5-Sim_metainfo.json",
        help="Output metainfo JSON path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="AGIBOT-HDF5-Sim",
        help="Dataset name in metainfo (must match AGIBOTHDF5Handler registry).",
    )
    parser.add_argument(
        "--max_object_trans_diff",
        type=float,
        default=DEFAULT_MAX_OBJECT_TRANS_DIFF,
        help=(
            "Discard clip on teleport impulses after a quiet previous step: "
            "object Δxy at least this far (meters), or upward dz>1.7cm with "
            "nontrivial ‖Δxyz‖. Post-jump settle allowed; z-drops kept."
        ),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "If clip length exceeds this, idle-aware temporal downsample "
            "(prefer dropping frames with EE/ori/grip speed at or below idle eps, "
            "then uniform). 0 disables. Suggested starting point: 300."
        ),
    )
    parser.add_argument(
        "--max_downsample_rate",
        type=float,
        default=DEFAULT_MAX_DOWNSAMPLE_RATE,
        help="Max temporal compression T/kept. Default: 3.",
    )
    parser.add_argument(
        "--min_downsample_rate",
        type=float,
        default=DEFAULT_MIN_DOWNSAMPLE_RATE,
        help=(
            "Min temporal compression T/kept when downsampling "
            "(kept <= floor(T / min_rate)). Default: 1.5."
        ),
    )
    parser.add_argument(
        "--idle_trans_eps",
        type=float,
        default=DEFAULT_IDLE_TRANS_EPS,
        help="True-idle EE translation speed threshold (m/frame). Sim default: 1e-7.",
    )
    parser.add_argument(
        "--idle_ori_eps",
        type=float,
        default=DEFAULT_IDLE_ORI_EPS,
        help="True-idle EE orientation speed threshold (rad/frame). Sim default: 1e-5.",
    )
    parser.add_argument(
        "--idle_grip_eps",
        type=float,
        default=DEFAULT_IDLE_GRIP_EPS,
        help="True-idle gripper speed threshold (effector units/frame). Sim default: 0.",
    )
    parser.add_argument(
        "--main_max_adjacent_diff_loose",
        type=float,
        default=MAIN_MAX_ADJACENT_DIFF_LOOSE,
        help="Head-view jump threshold when clip has tracked objects in state.json (looser).",
    )
    parser.add_argument(
        "--wrist_max_adjacent_diff_loose",
        type=float,
        default=WRIST_MAX_ADJACENT_DIFF_LOOSE,
        help="Wrist-view jump threshold when clip has tracked objects in state.json (looser).",
    )
    parser.add_argument(
        "--main_max_adjacent_diff_medium",
        type=float,
        default=MAIN_MAX_ADJACENT_DIFF_MEDIUM,
        help="Head-view jump threshold for selected medium-tier language instructions.",
    )
    parser.add_argument(
        "--wrist_max_adjacent_diff_medium",
        type=float,
        default=WRIST_MAX_ADJACENT_DIFF_MEDIUM,
        help="Wrist-view jump threshold for selected medium-tier language instructions.",
    )
    parser.add_argument(
        "--main_max_adjacent_diff_strict",
        type=float,
        default=MAIN_MAX_ADJACENT_DIFF_STRICT,
        help=(
            "Head-view jump threshold when state.json is missing or objects is empty "
            "in the clip (stricter)."
        ),
    )
    parser.add_argument(
        "--wrist_max_adjacent_diff_strict",
        type=float,
        default=WRIST_MAX_ADJACENT_DIFF_STRICT,
        help=(
            "Wrist-view jump threshold when state.json is missing or objects is empty "
            "in the clip (stricter)."
        ),
    )
    parser.add_argument(
        "--max_clips_per_lang",
        type=int,
        default=DEFAULT_MAX_CLIPS_PER_LANG,
        help=(
            "Stop exporting clips for a language instruction after this many "
            "successful exports (0 = no cap)."
        ),
    )
    args = parser.parse_args()

    setup_seed(0)
    convert_to_hdf5(
        args.task_dir,
        args.output_dir,
        args.metainfo,
        task_json_path=args.task_json or None,
        dataset_name=args.dataset_name,
        max_object_trans_diff=args.max_object_trans_diff,
        max_frames=args.max_frames,
        max_downsample_rate=args.max_downsample_rate,
        min_downsample_rate=args.min_downsample_rate,
        idle_trans_eps=args.idle_trans_eps,
        idle_ori_eps=args.idle_ori_eps,
        idle_grip_eps=args.idle_grip_eps,
        main_max_adjacent_diff_loose=args.main_max_adjacent_diff_loose,
        wrist_max_adjacent_diff_loose=args.wrist_max_adjacent_diff_loose,
        main_max_adjacent_diff_medium=args.main_max_adjacent_diff_medium,
        wrist_max_adjacent_diff_medium=args.wrist_max_adjacent_diff_medium,
        main_max_adjacent_diff_strict=args.main_max_adjacent_diff_strict,
        wrist_max_adjacent_diff_strict=args.wrist_max_adjacent_diff_strict,
        max_clips_per_lang=args.max_clips_per_lang,
    )
