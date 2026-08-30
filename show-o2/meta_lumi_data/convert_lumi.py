import os
import argparse
import json
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import mediapy as media
import sys

# Add ShowVLA to path to import datasets_vla.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets_vla.utils import euler_to_rotate6d

# Subtask clips with these labels are skipped during conversion.
SKIP_SUBTASKS = {
    "turn to the second rack",
}

LINEAR_VELOCITY_LIMITS = (-0.125, 0.125)
ANGULAR_VELOCITY_LIMITS = (-0.3, 0.3)

# Default main-view crop in PIL (left, upper, right, lower). Matches convert_lumi_data_subtask.
DEFAULT_CROP_MAIN = (300, 20, 1280 - 220, 720)


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


def combine_main_wrist_views(
    main_img,
    wrist_img,
    main_tgt_size=(224, 320),
    wrist_tgt_size=(112, 160),
    comb_size=(336, 320),
    wrist_at_left=False,
):
    """
    Combine main view (top) and single wrist view (bottom).
    Default: wrist at bottom-right (right-arm layout, matching JAKA right wrist).
    """
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]
    assert wrist_tgt_size[1] * 2 == main_tgt_size[1]

    main_img = np.array(
        Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR)
    )
    wrist_img = np.array(
        Image.fromarray(wrist_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR)
    )

    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[: main_tgt_size[0]] = main_img
    if wrist_at_left:
        comb_img[main_tgt_size[0] :, : wrist_tgt_size[1]] = wrist_img
    else:
        comb_img[main_tgt_size[0] :, wrist_tgt_size[1] :] = wrist_img
    return comb_img


def setup_seed(seed):
    np.random.seed(seed)


def sanitize_filename_part(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def unwrap_eef_rotations(list_eef_pose):
    """Unwrap rotation angles (rpy) to avoid ±pi discontinuities."""
    if len(list_eef_pose) < 2:
        return list_eef_pose
    poses = np.stack(list_eef_pose)
    poses[:, 3:6] = np.unwrap(poses[:, 3:6], axis=0)
    return [poses[i] for i in range(len(poses))]


def parse_eef_pose(eef_pose_r):
    eef_pose = np.array(eef_pose_r, dtype=np.float64)
    eef_pose[:3] /= 1000.0  # xyz: mm -> m
    return eef_pose


def parse_grip(act_grip_r: bool) -> float:
    return 1.0 if act_grip_r else 0.0


def parse_mobile(
    agv_data,
    linear_velocity_limits=None,
    angular_velocity_limits=None,
):
    """Return stored Lumi chassis velocities: linear [v], angular [yaw rate]."""
    linear = np.array([agv_data["linear_velocity"]], dtype=np.float64)
    angular = np.array([agv_data["angular_velocity"]], dtype=np.float64)
    if linear_velocity_limits is not None:
        linear[0] = np.clip(linear[0], *linear_velocity_limits)
    if angular_velocity_limits is not None:
        angular[0] = np.clip(angular[0], *angular_velocity_limits)
    return linear, angular


def load_episode(ep_path):
    with open(os.path.join(ep_path, "data.json")) as f:
        json_data = json.load(f)

    info = json_data["info"]
    steps = json_data["data"]
    task = info["task"]
    total_steps = info["total_steps"]
    fps = info.get("frequency", 20)
    assert len(steps) == total_steps, f"{ep_path}: expected {total_steps} steps, got {len(steps)}"
    return task, total_steps, fps, steps


def load_subtask_clips(ep_path):
    ann_path = os.path.join(ep_path, "subtask_anns.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        anns = json.load(f)
    clips = anns.get("clips")
    if not clips:
        return None
    return clips


def load_step_images(ep_path, step, frame_idx, crop_main):
    colors = step["colors"]
    main_rel = colors.get("rgb_main")
    wrist_rel = colors.get("rgb_wrist_0")
    if main_rel is None or wrist_rel is None:
        raise ValueError(f"{ep_path} frame {frame_idx}: missing rgb_main or rgb_wrist_0")

    main_path = os.path.join(ep_path, main_rel)
    wrist_path = os.path.join(ep_path, wrist_rel)

    with Image.open(main_path) as img:
        assert img.size == (1280, 720), f"{main_path}: expected (1280, 720), got {img.size}"
        img = img.crop(crop_main)
        main_img = np.asarray(img.convert("RGB"))

    with Image.open(wrist_path) as img:
        wrist_img = np.asarray(img.convert("RGB"))

    return main_img, wrist_img


def is_black_image(img: np.ndarray) -> bool:
    return bool(np.all(img == 0))


def save_clip_hdf5(
    h5_path,
    language_instruction,
    list_rgb_comb,
    list_eef,
    list_linear,
    list_angular,
    fps,
    overwrite=False,
):
    if os.path.exists(h5_path) and not overwrite:
        return False

    assert len(list_rgb_comb) == len(list_eef) == len(list_linear) == len(list_angular) > 0

    rgb_comb_bytes = encode_frames_to_jpeg_bytes(list_rgb_comb)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("language_instruction", data=language_instruction, dtype=str_dtype)
        h5_file.create_dataset("rgb_comb", data=rgb_comb_bytes, dtype=vlen_uint8)
        h5_file.create_dataset("eef_xyz_rotate6d_grip", data=np.asarray(list_eef, dtype=np.float64))
        h5_file.create_dataset("agv_linear_velocity", data=np.asarray(list_linear, dtype=np.float64))
        h5_file.create_dataset("agv_angular_velocity", data=np.asarray(list_angular, dtype=np.float64))

    mp4_path = h5_path.replace(".hdf5", ".mp4")
    media.write_video(mp4_path, list_rgb_comb, fps=max(1, int(fps)))
    return True


def convert_frame_range(
    ep_path,
    fps,
    steps,
    list_eef_pose,
    list_grip,
    list_linear,
    list_angular,
    start,
    end,
    crop_main,
    speed_up=1,
):
    """Build absolute-state sequences for frames in [start, end] (inclusive), downsampled."""
    assert start < end, f"{ep_path}: invalid range [{start}, {end}]"
    sampled_indices = list(range(start, end + 1, speed_up))
    assert len(sampled_indices) > 1, f"{ep_path}: too few frames in [{start}, {end}] speed_up={speed_up}"

    list_rgb_comb = []
    list_eef = []
    out_linear = []
    out_angular = []

    for frame_idx in sampled_indices:
        main_img, wrist_img = load_step_images(ep_path, steps[frame_idx], frame_idx, crop_main)
        if is_black_image(main_img) or is_black_image(wrist_img):
            print(f"[{ep_path}] [Skip frame {frame_idx}]: black image")
            continue

        eef_pose = list_eef_pose[frame_idx]
        xyz = eef_pose[:3]
        rotate6d = euler_to_rotate6d(eef_pose[3:6], pattern="xyz")
        eef_xyz_rotate6d_grip = np.concatenate([xyz, rotate6d, [list_grip[frame_idx]]])

        list_rgb_comb.append(combine_main_wrist_views(main_img, wrist_img))
        list_eef.append(eef_xyz_rotate6d_grip)
        out_linear.append(list_linear[frame_idx])
        out_angular.append(list_angular[frame_idx])

    return list_rgb_comb, list_eef, out_linear, out_angular, fps


def convert_episode_clips(
    episode_dir,
    data_dir_basename,
    data_dir,
    output_dir,
    crop_main,
    head_length,
    tail_length,
    speed_up,
    linear_velocity_limits,
    angular_velocity_limits,
    overwrite,
    meta_json,
    cur_episode,
):
    """Convert one raw episode into one or more HDF5 clips. Returns updated cur_episode."""
    task, total_steps, fps, steps = load_episode(episode_dir)
    clips = load_subtask_clips(episode_dir)

    list_eef_pose = unwrap_eef_rotations(
        [parse_eef_pose(step["states"]["eef_pose_r"]) for step in steps]
    )
    list_grip = [parse_grip(step["actions"]["act_grip_r"]) for step in steps]
    list_linear = []
    list_angular = []
    for step in steps:
        linear, angular = parse_mobile(
            step["agv_data"],
            linear_velocity_limits=linear_velocity_limits,
            angular_velocity_limits=angular_velocity_limits,
        )
        list_linear.append(linear)
        list_angular.append(angular)

    rel_path = os.path.relpath(episode_dir, data_dir)
    rel_tag = rel_path.replace(os.sep, "_")

    # Ranges to convert: (task_text, start, end, suffix)
    ranges = []
    if clips is None:
        num_raw_steps = len(list_grip) - 1
        start = 0 + head_length
        end = num_raw_steps - tail_length
        print(f"[0, {num_raw_steps}) -> [{start}, {end})")
        ranges.append((task, start, end, None))
    else:
        print(f"Using {len(clips)} subtask clips from subtask_anns.json")
        for clip in clips:
            clip_id = clip["clip_id"]
            clip_start = int(clip["start_frame"])
            clip_end_frame = int(clip["end_frame"])
            clip_task = clip["subtask"]

            if clip_task.strip().lower() in SKIP_SUBTASKS:
                print(f"Clip {clip_id}: skip filtered subtask '{clip_task}'")
                continue

            if clip_start < 0 or clip_end_frame > total_steps or clip_start >= clip_end_frame:
                raise ValueError(
                    f"{episode_dir} clip {clip_id}: invalid range [{clip_start}, {clip_end_frame})"
                )

            start = clip_start + (head_length if clip_start == 0 else 0)
            end = (clip_end_frame - 1) - (tail_length if clip_end_frame == total_steps else 0)
            print(f"Clip {clip_id}: {clip_task}")
            print(f"[{clip_start}, {clip_end_frame}) -> [{start}, {end})")
            suffix = f"clip_{clip_id}_{sanitize_filename_part(clip_task)}"
            ranges.append((clip_task, start, end, suffix))

    for lang, start, end, suffix in ranges:
        if start >= end:
            print(f"[{episode_dir}] skip empty range [{start}, {end}] for '{lang}'")
            continue

        if suffix is None:
            h5_filename = f"{data_dir_basename}_{rel_tag}.hdf5"
        else:
            h5_filename = f"{data_dir_basename}_{rel_tag}_{suffix}.hdf5"
        h5_path = os.path.join(output_dir, h5_filename)

        if h5_path in meta_json["datalist"]:
            print(f"Warning: h5_path {h5_path} already in datalist. Skipping!")
            continue

        if os.path.exists(h5_path) and not overwrite:
            print(f"Warning: {h5_path} exists. Directly add it to datalist!")
            meta_json["datalist"].append(h5_path)
            cur_episode += 1
            continue

        list_rgb_comb, list_eef, out_linear, out_angular, out_fps = convert_frame_range(
            episode_dir,
            fps,
            steps,
            list_eef_pose,
            list_grip,
            list_linear,
            list_angular,
            start,
            end,
            crop_main,
            speed_up=speed_up,
        )

        if len(list_rgb_comb) <= 1:
            print(f"[{episode_dir}] skip '{lang}': too few valid frames ({len(list_rgb_comb)})")
            continue

        wrote = save_clip_hdf5(
            h5_path,
            lang,
            list_rgb_comb,
            list_eef,
            out_linear,
            out_angular,
            out_fps / speed_up if speed_up > 1 else out_fps,
            overwrite=overwrite,
        )
        if not wrote and not overwrite:
            meta_json["datalist"].append(h5_path)
            cur_episode += 1
            continue

        meta_json["datalist"].append(h5_path)
        cur_episode += 1
        print(
            f"[{cur_episode}] {h5_filename} frames={len(list_rgb_comb)} lang='{lang}'",
            flush=True,
        )

    return cur_episode


def convert_lumi_to_hdf5(
    data_dir,
    output_dir,
    metainfo_json_out_path,
    crop_main,
    speed_up=1,
    head_length=0,
    tail_length=0,
    overwrite=False,
    dataset_name="Lumi-mobile",
    linear_velocity_limits=LINEAR_VELOCITY_LIMITS,
    angular_velocity_limits=ANGULAR_VELOCITY_LIMITS,
):
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(metainfo_json_out_path, "r") as f:
            meta_json = json.load(f)
        cur_episode = meta_json["num_ep"]
    except Exception:
        meta_json = {
            "dataset_name": dataset_name,
            "is_mobile": True,
            "data_dirs": [],
            "language_instruction_key": "language_instruction",
            "observation_key": ["rgb_comb"],
            "num_ep": 0,
            "datalist": [],
        }
        cur_episode = 0

    if "data_dirs" not in meta_json:
        meta_json["data_dirs"] = []

    if data_dir not in meta_json["data_dirs"]:
        meta_json["data_dirs"].append(data_dir)
    else:
        print(f"Warning: data_dir {data_dir} already processed!")

    print(f"Converting {data_dir}")

    list_ep_folders = []
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if d.startswith("episode_"):
                list_ep_folders.append(os.path.join(root, d))
    list_ep_folders.sort(key=lambda p: int(os.path.basename(p).split("_")[-1]))

    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))

    pbar = tqdm(list_ep_folders)
    for ep_cnt, episode_dir in enumerate(pbar):
        pbar.set_description(f"[{ep_cnt + 1}/{len(list_ep_folders)}] {os.path.basename(episode_dir)}")
        cur_episode = convert_episode_clips(
            episode_dir,
            data_dir_basename,
            data_dir,
            output_dir,
            crop_main,
            head_length,
            tail_length,
            speed_up,
            linear_velocity_limits,
            angular_velocity_limits,
            overwrite,
            meta_json,
            cur_episode,
        )

    meta_json["num_ep"] = cur_episode
    with open(metainfo_json_out_path, "w") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Lumi raw episodes to ShowVLA HDF5.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of raw Lumi data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output HDF5 files")
    parser.add_argument("--meta_prefix", type=str, default="Lumi", help="Prefix for metainfo json file")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Lumi-mobile",
        help="Dataset name in metainfo json (must contain 'mobile' for chassis velocities)",
    )
    parser.add_argument("--speed_up", type=int, default=1)
    parser.add_argument("--head_length", type=int, default=0, help="Trim frames at episode/clip start")
    parser.add_argument("--tail_length", type=int, default=0, help="Trim frames at episode/clip end")
    parser.add_argument(
        "--crop_main",
        type=str,
        default=repr(DEFAULT_CROP_MAIN),
        help="Main-view PIL crop (left, upper, right, lower)",
    )
    parser.add_argument(
        "--linear_velocity_limits",
        type=float,
        nargs=2,
        default=list(LINEAR_VELOCITY_LIMITS),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--angular_velocity_limits",
        type=float,
        nargs=2,
        default=list(ANGULAR_VELOCITY_LIMITS),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    setup_seed(0)

    crop_main = eval(args.crop_main)
    assert len(crop_main) == 4, f"crop_main must have 4 ints, got {crop_main}"
    metainfo_json_out_path = f"{args.meta_prefix}_metainfo.json"

    convert_lumi_to_hdf5(
        args.data_dir,
        args.output_dir,
        metainfo_json_out_path,
        crop_main,
        speed_up=args.speed_up,
        head_length=args.head_length,
        tail_length=args.tail_length,
        overwrite=args.overwrite,
        dataset_name=args.dataset_name,
        linear_velocity_limits=tuple(args.linear_velocity_limits),
        angular_velocity_limits=tuple(args.angular_velocity_limits),
    )
