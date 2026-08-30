"""
Convert AgiBot World Challenge Manipulation-RealRobot subtask clips to ShowVLA HDF5.

Each ``label_info.action_config`` entry becomes one HDF5 + MP4 pair.
Language uses ``action_text`` (RealRobot has no ``english_action_text``).

Expected raw layout under ``data_root``:
  task_info/{task_id}.json
  observations/{task_id}/{episode_id}/videos/
    head_color.mp4
    hand_left_color.mp4
    hand_right_color.mp4
  proprio_stats/{task_id}/{episode_id}/proprio_stats.h5

Differences vs Manipulation-SimData (see convert_agibot_sim_hdf5.py):
  - Split dirs (observations / proprio_stats / task_info) instead of nested
    task_name/task_id/job_id/sn_code/episode_id
  - RGB from AV1 MP4 videos, not per-frame JPG under camera/
  - Proprio file is proprio_stats.h5 (already has action/effector/position [T,2])
  - No sim ``state.json`` object poses → object-teleport filter is skipped
  - Clip language field is ``action_text``
  - Clean language: strip non-arm `` using …`` junk; ``drawer drawer``→``drawer``;
    ``with left/right arm``→``with the … arm`` (keep ``near left/right arm``)

RealRobot-tuned defaults (from ~18k clips + proprio/RGB samples):
  - Clip length median~190, p90~328; Open/Close/Pull often 500–900f
    → max_frames=400, min_downsample_rate=1.5 (Sim 300 / 2.0)
  - Proprio noise ⇒ exact-zero idle almost never → soft idle eps
  - Head adjacent RGB diffs higher (lighting / compression / door)
    → main_max_adjacent_diff=16 (Sim 6.98 discarded ~1/3)

Output HDF5 (matches AGIBOTHDF5Handler / AGIBOT-HDF5-Real):
  /language_instruction
  /rgb_comb
  /actions/end/position        [T, 2, 3]
  /actions/end/orientation     [T, 2, 4]
  /actions/effector/position   [T, 2]

MP4 name: {cur_episode}_{episode_id}_{clip_id}_{action_text}.mp4
HDF5 name: {cur_episode}_{episode_id}_{clip_id}.hdf5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
from tqdm import tqdm

# Legitimate "… using the left/right arm / both arms."; anything else after
# " using " is RealRobot annotation noise (scene props / leaked translate prompts).
_LEGIT_USING_ARM = re.compile(
    r"^(?P<head>.+?)\s+using\s+(?P<arm>the\s+(?:left|right)\s+arm|both\s+arms)\.?$",
    re.IGNORECASE,
)

# Reuse shared clip pipeline helpers from the Sim converter.
from convert_agibot_sim_hdf5 import (
    AGIBOT_FPS,
    MAIN_TGT_SIZE,
    MIN_FRAMES,
    WRIST_TGT_SIZE,
    apply_frame_indices,
    combine_resized_main_wrists,
    confirm_fresh_export,
    discard,
    ee_frame_speed,
    ee_ori_frame_speed,
    grip_frame_speed,
    has_adjacent_view_jump,
    read_action_arrays,
    resize_view,
    sanitize_lang_for_filename,
    select_downsample_indices,
    setup_seed,
    target_frames_with_rate_cap,
    write_clip_hdf5,
)

HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"

# --- RealRobot-tuned hyperparameter defaults ---
DEFAULT_MAX_FRAMES = 400
DEFAULT_MAX_DOWNSAMPLE_RATE = 3.5
DEFAULT_MIN_DOWNSAMPLE_RATE = 1.5
# Full-clip head max-adj sample: p90~11, max~15; Sim 6.98 is too strict.
MAIN_MAX_ADJACENT_DIFF = 16.0
# Wrist max-adj stayed well below Sim's 38; keep the same headroom.
WRIST_MAX_ADJACENT_DIFF = 38.05
# Soft idle: Real EE/ori speeds rarely hit exact 0 (noise ~1e-5–1e-4).
DEFAULT_IDLE_TRANS_EPS = 5e-4
DEFAULT_IDLE_ORI_EPS = 5e-3
DEFAULT_IDLE_GRIP_EPS = 5e-3


def soft_idle_mask(
    end_position: np.ndarray,
    end_orientation: np.ndarray,
    effector_position: np.ndarray,
    trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
) -> np.ndarray:
    """Near-zero EE translation / orientation / gripper speed (Real proprio noise)."""
    trans = ee_frame_speed(end_position)
    ori = ee_ori_frame_speed(end_orientation)
    grip = grip_frame_speed(effector_position)
    return (trans <= trans_eps) & (ori <= ori_eps) & (grip <= grip_eps)


def maybe_downsample_long_clip_real(
    arrays: dict,
    max_frames: int,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
):
    """Idle-aware downsample using soft speed thresholds suited to RealRobot."""
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

    idle = soft_idle_mask(
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
        f"downsample {t0}->{t1} (soft_idle_ratio={idle_ratio:.2f}, "
        f"target={target}, rate={rate:.2f}x, max_frames={max_frames}, "
        f"rate_range=[{min_downsample_rate},{max_downsample_rate}], "
        f"eps_t/o/g=[{idle_trans_eps:g},{idle_ori_eps:g},{idle_grip_eps:g}])"
    )
    return arrays, info


def resolve_data_root(data_root: str) -> str:
    """Accept extracted root or nested Manipulation-RealRobot/ archive folder."""
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, "task_info")) and os.path.isdir(
        os.path.join(data_root, "observations")
    ):
        return data_root
    nested = os.path.join(data_root, "Manipulation-RealRobot")
    if os.path.isdir(os.path.join(nested, "task_info")) and os.path.isdir(
        os.path.join(nested, "observations")
    ):
        return nested
    raise FileNotFoundError(
        f"Cannot find task_info/ + observations/ under {data_root} "
        "(or {data_root}/Manipulation-RealRobot)."
    )


def discover_task_ids(data_root: str, task_ids: list[str] | None = None) -> list[str]:
    """Return task_id strings that have task_info JSON (sorted numerically)."""
    data_root = resolve_data_root(data_root)
    ti_dir = os.path.join(data_root, "task_info")
    found = []
    for name in os.listdir(ti_dir):
        if name.endswith(".json"):
            found.append(name[: -len(".json")])
    found = sorted(found, key=lambda x: int(x) if x.isdigit() else x)
    if task_ids:
        want = set(task_ids)
        missing = sorted(want - set(found), key=lambda x: int(x) if x.isdigit() else x)
        if missing:
            raise FileNotFoundError(f"Missing task_info for task_ids: {missing}")
        found = [t for t in found if t in want]
    return found


def episode_paths(data_root: str, task_id: str, episode_id) -> dict:
    """Resolve observation / proprio paths for one episode."""
    eid = str(episode_id)
    tid = str(task_id)
    obs_dir = os.path.join(data_root, "observations", tid, eid)
    return {
        "obs_dir": obs_dir,
        "video_dir": os.path.join(obs_dir, "videos"),
        "head": os.path.join(obs_dir, "videos", HEAD_COLOR),
        "hand_left": os.path.join(obs_dir, "videos", HAND_LEFT_COLOR),
        "hand_right": os.path.join(obs_dir, "videos", HAND_RIGHT_COLOR),
        "proprio": os.path.join(
            data_root, "proprio_stats", tid, eid, "proprio_stats.h5"
        ),
    }


def read_video_rgb_range(video_path: str, start: int, end: int) -> list[np.ndarray]:
    """
    Decode RGB frames for half-open [start, end) from an MP4 (AV1/H264/...).

    Uses PTS-based frame indices after a backward seek so late clips stay fast.
    """
    if end <= start:
        return []
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else float(AGIBOT_FPS)

        if start > 0:
            t = max(0.0, (start - 5) / fps)
            container.seek(int(t * av.time_base), any_frame=False, backward=True)

        frames: dict[int, np.ndarray] = {}
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            idx = int(round(float(frame.pts * stream.time_base) * fps))
            if idx < start:
                continue
            if idx >= end:
                break
            if idx not in frames:
                frames[idx] = frame.to_ndarray(format="rgb24")
    finally:
        container.close()

    out = []
    for i in range(start, end):
        if i not in frames:
            raise RuntimeError(
                f"missing_frame@{i} in {os.path.basename(video_path)} "
                f"(decoded {len(frames)}/{end - start} in [{start},{end}))"
            )
        out.append(frames[i])
    return out


def load_clip_views(
    paths: dict, start: int, end: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load head / left wrist / right wrist RGB for [start, end) in parallel."""

    def _load(key: str):
        return read_video_rgb_range(paths[key], start, end)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_h = pool.submit(_load, "head")
        fut_l = pool.submit(_load, "hand_left")
        fut_r = pool.submit(_load, "hand_right")
        head = fut_h.result()
        left = fut_l.result()
        right = fut_r.result()
    return head, left, right


def clean_action_text(text: str) -> str:
    """
    Normalize RealRobot ``action_text``.

    - Task 881 Open clips: strip annotation junk after non-arm `` using …``
      (keep ``using the left/right arm`` / ``using both arms``).
    - ``drawer drawer`` → ``drawer`` (task 949 typo).
    - ``with left/right arm`` → ``with the left/right arm``
      (tasks 1645/1968); leave spatial ``near left/right arm`` untouched.
    """
    lang = " ".join((text or "").split()).strip()
    if not lang:
        return lang

    m = _LEGIT_USING_ARM.match(lang)
    if m:
        head = m.group("head").rstrip(" .")
        arm = m.group("arm").lower()
        lang = f"{head} using {arm}."
    else:
        # Non-arm " using …" → annotation pollution; keep the instruction head.
        low = lang.lower()
        idx = low.find(" using ")
        if idx >= 0:
            after = low[idx + len(" using ") :]
            if not re.match(
                r"(the\s+(?:left|right)\s+arm|both\s+arms)\.?\s*$", after
            ):
                lang = lang[:idx].rstrip(" .") + "."

    lang = re.sub(r"\bdrawer\s+drawer\b", "drawer", lang, flags=re.IGNORECASE)
    # Only "with … arm", not "near … arm".
    lang = re.sub(
        r"\bwith (left|right) arm\b",
        r"with the \1 arm",
        lang,
        flags=re.IGNORECASE,
    )
    return lang


def clip_language(clip: dict) -> str:
    """Prefer english_action_text when present; RealRobot uses action_text."""
    raw = clip.get("english_action_text") or clip.get("action_text") or ""
    return clean_action_text(raw)


def build_subtask_clip(
    paths: dict,
    start: int,
    end: int,
    language_instruction: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    main_max_adjacent_diff: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_max_adjacent_diff: float = WRIST_MAX_ADJACENT_DIFF,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
):
    proprio_path = paths["proprio"]
    if not os.path.isfile(proprio_path):
        return None, "missing_proprio"
    for key in ("head", "hand_left", "hand_right"):
        if not os.path.isfile(paths[key]):
            return None, f"missing_video:{key}"

    pos, ori, grip, num_frames = read_action_arrays(proprio_path, start, end)
    if start < 0 or end > num_frames or end <= start:
        return None, "bad_frame_range"

    try:
        head_frames, left_frames, right_frames = load_clip_views(paths, start, end)
    except Exception as e:
        return None, f"video_read({e})"

    if not (
        len(head_frames)
        == len(left_frames)
        == len(right_frames)
        == pos.shape[0]
        == ori.shape[0]
        == grip.shape[0]
    ):
        return None, "length_mismatch"

    rgb_comb = []
    for head, left, right in zip(head_frames, left_frames, right_frames):
        head_r = resize_view(head, MAIN_TGT_SIZE)
        left_r = resize_view(left, WRIST_TGT_SIZE)
        right_r = resize_view(right, WRIST_TGT_SIZE)
        rgb_comb.append(combine_resized_main_wrists(head_r, left_r, right_r))

    if len(rgb_comb) < MIN_FRAMES:
        return None, "too_few_frames"

    arrays = {
        "language_instruction": language_instruction,
        "rgb_comb": rgb_comb,
        "end_position": pos,
        "end_orientation": ori,
        "effector_position": grip,
    }
    arrays, ds_info = maybe_downsample_long_clip_real(
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


def convert_task_to_hdf5(
    data_root: str,
    task_id: str,
    output_dir: str,
    dataset_name: str = "AGIBOT-HDF5-Real",
    meta_json: dict | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    main_max_adjacent_diff: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_max_adjacent_diff: float = WRIST_MAX_ADJACENT_DIFF,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
    max_episodes: int | None = None,
):
    data_root = resolve_data_root(data_root)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    task_json_path = os.path.join(data_root, "task_info", f"{task_id}.json")
    with open(task_json_path, "r") as f:
        episodes = json.load(f)

    if max_episodes is not None and max_episodes > 0:
        episodes = episodes[:max_episodes]

    if meta_json is None:
        meta_json = {
            "dataset_name": dataset_name,
            "task_ids": [],
            "data_root": data_root,
            "language_instruction_key": "language_instruction",
            "observation_key": ["rgb_comb"],
            "num_ep": 0,
            "datalist": [],
        }

    # Resume from last *assigned* index (not success count). Discarded clips
    # still consume an index by design; using len(datalist)/num_ep here would
    # rewind and reuse those gaps across task boundaries.
    cur_episode = int(meta_json.get("last_clip_index", meta_json.get("num_ep", 0)))
    if "task_ids" not in meta_json:
        meta_json["task_ids"] = []
    if task_id not in meta_json["task_ids"]:
        meta_json["task_ids"].append(task_id)
    meta_json["data_root"] = data_root

    skip_stats = defaultdict(int)
    downsample_count = 0
    print(f"Converting task_id={task_id}")
    print(f"Loaded {len(episodes)} episodes from {task_json_path}")

    pbar = tqdm(episodes, desc=f"task_{task_id}")
    for ep_meta in pbar:
        episode_id = ep_meta["episode_id"]
        paths = episode_paths(data_root, task_id, episode_id)
        if not os.path.isdir(paths["obs_dir"]):
            discard(f"episode_id={episode_id}: missing dir {paths['obs_dir']}")
            skip_stats["missing_episode_dir"] += 1
            continue

        action_config = ep_meta.get("label_info", {}).get("action_config", [])
        if not action_config:
            discard(f"episode_id={episode_id}: empty action_config")
            skip_stats["empty_action_config"] += 1
            continue

        for clip_id, clip in enumerate(action_config):
            lang = clip_language(clip)
            if not lang:
                discard(f"episode_id={episode_id} clip={clip_id}: missing action_text")
                skip_stats["missing_action_text"] += 1
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
                paths,
                start,
                end,
                lang,
                max_frames=max_frames,
                max_downsample_rate=max_downsample_rate,
                min_downsample_rate=min_downsample_rate,
                main_max_adjacent_diff=main_max_adjacent_diff,
                wrist_max_adjacent_diff=wrist_max_adjacent_diff,
                idle_trans_eps=idle_trans_eps,
                idle_ori_eps=idle_ori_eps,
                idle_grip_eps=idle_grip_eps,
            )
            if arrays is None:
                discard(
                    f"episode_id={episode_id} clip={clip_id} "
                    f"[{start},{end}): {reason}"
                )
                skip_stats[str(reason).split("@")[0].split("(")[0]] += 1
                continue

            ds_note = ""
            if reason and str(reason).startswith("downsample"):
                downsample_count += 1
                ds_note = f" [{reason}]"

            write_clip_hdf5(h5_path, arrays, mp4_path)
            meta_json["datalist"].append(h5_path)

            print(
                f"[{cur_episode}] task={task_id} ep={episode_id} clip={clip_id} "
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
        print(
            f"Downsampled long clips: {downsample_count} "
            f"(max_frames={max_frames}, rate=[{min_downsample_rate},{max_downsample_rate}])"
        )
    print("Skip stats:")
    for reason, cnt in sorted(skip_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt}")
    return meta_json


def convert_to_hdf5(
    data_root: str,
    output_dir: str,
    metainfo_json_out_path: str,
    task_ids: list[str] | None = None,
    dataset_name: str = "AGIBOT-HDF5-Real",
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_downsample_rate: float = DEFAULT_MAX_DOWNSAMPLE_RATE,
    min_downsample_rate: float = DEFAULT_MIN_DOWNSAMPLE_RATE,
    main_max_adjacent_diff: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_max_adjacent_diff: float = WRIST_MAX_ADJACENT_DIFF,
    idle_trans_eps: float = DEFAULT_IDLE_TRANS_EPS,
    idle_ori_eps: float = DEFAULT_IDLE_ORI_EPS,
    idle_grip_eps: float = DEFAULT_IDLE_GRIP_EPS,
    max_episodes: int | None = None,
    skip_confirm: bool = False,
):
    output_dir = os.path.abspath(output_dir)
    metainfo_json_out_path = os.path.abspath(metainfo_json_out_path)
    data_root = resolve_data_root(data_root)

    if not skip_confirm:
        if not confirm_fresh_export(output_dir, metainfo_json_out_path):
            return
    else:
        # Non-interactive: wipe existing targets if present.
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            shutil.rmtree(output_dir)
            print(f"Deleted directory: {output_dir}")
        if os.path.exists(metainfo_json_out_path):
            os.remove(metainfo_json_out_path)
            print(f"Deleted metainfo: {metainfo_json_out_path}")

    task_id_list = discover_task_ids(data_root, task_ids=task_ids)
    if not task_id_list:
        raise FileNotFoundError(f"No task_info/*.json found under {data_root}")

    print(f"data_root: {data_root}")
    print(f"Found {len(task_id_list)} task(s): {task_id_list}")
    print("Object-teleport filter: disabled (RealRobot has no sim state.json objects)")
    print(
        f"Image-jump filter (after downsample): head adjacent mean-abs>"
        f"{main_max_adjacent_diff} or wrist>{wrist_max_adjacent_diff} → discard"
    )
    if max_frames > 0:
        print(
            f"Long-clip downsample: T>{max_frames} → drop soft-idle "
            f"(EE/ori/grip speed ≤ {idle_trans_eps:g}/{idle_ori_eps:g}/{idle_grip_eps:g}), "
            f"else uniform; rate in [{min_downsample_rate}, {max_downsample_rate}]x"
        )
    else:
        print("Long-clip downsample: disabled (pass --max_frames N to enable)")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metainfo_json_out_path) or ".", exist_ok=True)

    meta_json = {
        "dataset_name": dataset_name,
        "task_ids": [],
        "data_root": data_root,
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": 0,
        "datalist": [],
    }
    for task_id in task_id_list:
        meta_json = convert_task_to_hdf5(
            data_root,
            task_id,
            output_dir,
            dataset_name=dataset_name,
            meta_json=meta_json,
            max_frames=max_frames,
            max_downsample_rate=max_downsample_rate,
            min_downsample_rate=min_downsample_rate,
            main_max_adjacent_diff=main_max_adjacent_diff,
            wrist_max_adjacent_diff=wrist_max_adjacent_diff,
            idle_trans_eps=idle_trans_eps,
            idle_ori_eps=idle_ori_eps,
            idle_grip_eps=idle_grip_eps,
            max_episodes=max_episodes,
        )

    with open(metainfo_json_out_path, "w") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)

    print("\n=== Conversion summary ===")
    print(f"Total clips exported (num_ep): {meta_json['num_ep']}")
    print(f"Metainfo written to {metainfo_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AgiBot Manipulation-RealRobot subtask clips to ShowVLA HDF5."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/datasets3/agibot_world_challenge_2025/Manipulation-RealRobot",
        help="Manipulation-RealRobot root (with task_info/, observations/, proprio_stats/).",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default="",
        help="Optional comma-separated task ids (default: all under task_info/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyx/datasets/AGIBOT-Real",
        help="Directory for converted HDF5 and MP4 files.",
    )
    parser.add_argument(
        "--metainfo",
        type=str,
        default="./AGIBOT-HDF5-Real_metainfo.json",
        help="Output metainfo JSON path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="AGIBOT-HDF5-Real",
        help="Dataset name in metainfo (must match AGIBOTHDF5Handler registry).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "If clip length exceeds this, soft-idle temporal downsample. "
            f"0 disables. RealRobot default: {DEFAULT_MAX_FRAMES} "
            "(clip p90~328; Open/Close often longer)."
        ),
    )
    parser.add_argument(
        "--max_downsample_rate",
        type=float,
        default=DEFAULT_MAX_DOWNSAMPLE_RATE,
        help="Max temporal compression T/kept.",
    )
    parser.add_argument(
        "--min_downsample_rate",
        type=float,
        default=DEFAULT_MIN_DOWNSAMPLE_RATE,
        help=(
            "Min temporal compression T/kept when downsampling. "
            f"RealRobot default: {DEFAULT_MIN_DOWNSAMPLE_RATE} "
            "(gentler than Sim 2.0; exact-zero idle is rare)."
        ),
    )
    parser.add_argument(
        "--main_max_adjacent_diff",
        type=float,
        default=MAIN_MAX_ADJACENT_DIFF,
        help=(
            "Discard clip if head-view adjacent-frame mean abs RGB diff exceeds this. "
            f"RealRobot default: {MAIN_MAX_ADJACENT_DIFF} (Sim 6.98 is too strict)."
        ),
    )
    parser.add_argument(
        "--wrist_max_adjacent_diff",
        type=float,
        default=WRIST_MAX_ADJACENT_DIFF,
        help="Discard clip if either wrist adjacent-frame mean abs RGB diff exceeds this.",
    )
    parser.add_argument(
        "--idle_trans_eps",
        type=float,
        default=DEFAULT_IDLE_TRANS_EPS,
        help="Soft-idle EE translation speed threshold (m/frame).",
    )
    parser.add_argument(
        "--idle_ori_eps",
        type=float,
        default=DEFAULT_IDLE_ORI_EPS,
        help="Soft-idle EE orientation speed threshold (rad/frame).",
    )
    parser.add_argument(
        "--idle_grip_eps",
        type=float,
        default=DEFAULT_IDLE_GRIP_EPS,
        help="Soft-idle gripper speed threshold (effector units/frame).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="If >0, only convert the first N episodes per task (debug).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt; delete existing output_dir/metainfo if present.",
    )
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()] or None

    setup_seed(0)
    convert_to_hdf5(
        args.data_root,
        args.output_dir,
        args.metainfo,
        task_ids=task_ids,
        dataset_name=args.dataset_name,
        max_frames=args.max_frames,
        max_downsample_rate=args.max_downsample_rate,
        min_downsample_rate=args.min_downsample_rate,
        main_max_adjacent_diff=args.main_max_adjacent_diff,
        wrist_max_adjacent_diff=args.wrist_max_adjacent_diff,
        idle_trans_eps=args.idle_trans_eps,
        idle_ori_eps=args.idle_ori_eps,
        idle_grip_eps=args.idle_grip_eps,
        max_episodes=args.max_episodes or None,
        skip_confirm=args.yes,
    )
