import os
import re
import argparse
import json
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict
from tqdm import tqdm
import mediapy as media
import tensorflow_datasets as tfds

DROID_FPS = 15  # matches DroidHandler
MIN_FRAMES = 20
MAX_FRAMES = 600
MIN_LANG_WORDS = 2
MIN_LANG_CHARS = 5
MAX_LANG_CHARS = 256
MAX_EPISODES_PER_INSTRUCTION = 2
MAX_FRAME_GAP = 10  # discard episode if any kept-segment gap >= this many frames
MAIN_TGT_SIZE = (224, 320)
WRIST_TGT_SIZE = (112, 160)
COMB_SIZE = (336, 320)
MAIN_MAX_ADJACENT_DIFF = 20.0  # main view: normal p95 ~18
WRIST_MAX_ADJACENT_DIFF = 45.0  # wrist view: normal p95 ~45


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
    wrist_frames: list[np.ndarray],
    main_threshold: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_threshold: float = WRIST_MAX_ADJACENT_DIFF,
) -> tuple[bool, str | None]:
    """Return True if main or wrist view has an abnormal adjacent-frame jump."""
    main_diff = max_adjacent_frame_diff(main_frames)
    if main_diff > main_threshold:
        return True, f"main_view_jump({main_diff:.1f}>{main_threshold})"

    wrist_diff = max_adjacent_frame_diff(wrist_frames)
    if wrist_diff > wrist_threshold:
        return True, f"wrist_view_jump({wrist_diff:.1f}>{wrist_threshold})"

    return False, None


def combine_resized_main_wrist(main_resized: np.ndarray, wrist_resized: np.ndarray) -> np.ndarray:
    """Combine pre-resized main (top) and wrist (bottom-right) views."""
    comb_img = np.zeros((COMB_SIZE[0], COMB_SIZE[1], 3), dtype=np.uint8)
    comb_img[: MAIN_TGT_SIZE[0], :] = main_resized
    comb_img[MAIN_TGT_SIZE[0] :, WRIST_TGT_SIZE[1] :] = wrist_resized
    return comb_img


def combine_main_wrist_zero(main_img, wrist_img):
    """Combine exterior main view (top) with wrist (bottom-right); bottom-left is zeros."""
    assert COMB_SIZE[0] == MAIN_TGT_SIZE[0] + WRIST_TGT_SIZE[0]
    assert COMB_SIZE[1] == MAIN_TGT_SIZE[1]
    assert WRIST_TGT_SIZE[1] * 2 == MAIN_TGT_SIZE[1]
    return combine_resized_main_wrist(
        resize_view(main_img, MAIN_TGT_SIZE),
        resize_view(wrist_img, WRIST_TGT_SIZE),
    )


def sanitize_lang_for_filename(lang: str, max_len: int = 80) -> str:
    slug = lang.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    if not slug:
        slug = "no_instruction"
    return slug[:max_len].rstrip("_")


def mp4_path_from_h5(h5_path: str, language_instruction: str) -> str:
    base, _ = os.path.splitext(h5_path)
    lang_slug = sanitize_lang_for_filename(language_instruction)
    return f"{base}_{lang_slug}.mp4"


def setup_seed(seed):
    np.random.seed(seed)


def count_words(text: str) -> int:
    return len(text.strip().split())


def is_valid_language(lang: str) -> bool:
    lang = lang.strip()
    if not lang:
        return False
    if lang.count(" ") == 0:
        return False
    if len(lang) < MIN_LANG_CHARS:
        return False
    if count_words(lang) <= MIN_LANG_WORDS:
        return False
    if len(lang) > MAX_LANG_CHARS:
        return False
    return True


def decode_rlds_image(img_tensor) -> np.ndarray | None:
    try:
        raw = img_tensor.numpy()
    except Exception:
        return None

    if raw is None:
        return None

    if raw.ndim == 3 and raw.dtype == np.uint8:
        if raw.shape[-1] != 3 or raw.size == 0:
            return None
        return raw

    try:
        if raw.ndim == 0:
            raw = raw.item()
        if isinstance(raw, (bytes, bytearray)):
            buf = raw
        else:
            buf = np.asarray(raw, dtype=np.uint8).tobytes()
        if not buf:
            return None
        img = Image.open(BytesIO(buf)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def is_valid_image(img: np.ndarray | None) -> bool:
    return (
        img is not None
        and img.ndim == 3
        and img.shape[-1] == 3
        and img.size > 0
    )


def load_keep_ranges(keep_ranges_path: str) -> dict:
    with open(keep_ranges_path, "r") as f:
        return json.load(f)


def build_kept_step_indices(ranges):
    kept = set()
    for start, end in ranges:
        kept.update(range(int(start), int(end)))
    return kept


def has_large_frame_gap(ranges, max_gap: int = MAX_FRAME_GAP) -> bool:
    """Return True if any gap between kept segments is >= max_gap frames."""
    sorted_ranges = sorted((int(s), int(e)) for s, e in ranges)
    for i in range(len(sorted_ranges) - 1):
        dropped = sorted_ranges[i + 1][0] - sorted_ranges[i][1]
        if dropped >= max_gap:
            return True
    return False


def make_episode_key(episode) -> str:
    em = episode["episode_metadata"]
    recording = em["recording_folderpath"].numpy()
    file_path = em["file_path"].numpy()
    if isinstance(recording, bytes):
        recording = recording.decode()
    if isinstance(file_path, bytes):
        file_path = file_path.decode()
    return f"{recording}--{file_path}"


def is_success_episode(episode) -> bool:
    file_path = episode["episode_metadata"]["file_path"].numpy()
    if isinstance(file_path, bytes):
        file_path = file_path.decode()
    return "success" in file_path


def read_episode_language(episode) -> str | None:
    steps = list(episode["steps"])
    if not steps:
        return None
    lang = steps[0]["language_instruction"].numpy()
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")
    lang = lang.strip()
    if not is_valid_language(lang):
        return None
    return lang


def episode_to_arrays(
    episode,
    kept_indices: set[int],
    main_exterior_key: str,
    main_max_adjacent_diff: float = MAIN_MAX_ADJACENT_DIFF,
    wrist_max_adjacent_diff: float = WRIST_MAX_ADJACENT_DIFF,
):
    steps = list(episode["steps"])
    if not steps:
        return None, "empty_episode"

    lang = read_episode_language(episode)
    if lang is None:
        return None, "bad_language"

    rgb_images = []
    main_frames = []
    wrist_frames = []
    cart_list = []
    grip_list = []

    for t, step in enumerate(steps):
        if t not in kept_indices:
            continue

        obs = step["observation"]
        main_img = decode_rlds_image(obs[main_exterior_key])
        wrist_img = decode_rlds_image(obs["wrist_image_left"])
        if not (is_valid_image(main_img) and is_valid_image(wrist_img)):
            return None, "missing_image"

        wrist_img = wrist_img[::-1, ::-1]
        main_resized = resize_view(main_img, MAIN_TGT_SIZE)
        wrist_resized = resize_view(wrist_img, WRIST_TGT_SIZE)
        main_frames.append(main_resized)
        wrist_frames.append(wrist_resized)
        rgb_images.append(combine_resized_main_wrist(main_resized, wrist_resized))

        cart_list.append(np.asarray(obs["cartesian_position"], dtype=np.float32))
        grip = np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)
        if grip.size == 0:
            return None, "missing_image"
        grip_list.append(grip[:1])

    if len(rgb_images) < MIN_FRAMES:
        return None, "too_few_frames"

    has_jump, jump_reason = has_adjacent_view_jump(
        main_frames,
        wrist_frames,
        main_threshold=main_max_adjacent_diff,
        wrist_threshold=wrist_max_adjacent_diff,
    )
    if has_jump:
        return None, jump_reason

    return {
        "language_instruction": lang,
        "rgb_comb": rgb_images,
        "cartesian_position": np.stack(cart_list, axis=0),
        "gripper_position": np.stack(grip_list, axis=0),
    }, None


def write_episode_hdf5(h5_path: str, arrays: dict):
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

        obs_grp = h5_file.create_group("observation")
        obs_grp.create_dataset("cartesian_position", data=arrays["cartesian_position"])
        obs_grp.create_dataset("gripper_position", data=arrays["gripper_position"])

    mp4_path = mp4_path_from_h5(h5_path, arrays["language_instruction"])
    media.write_video(mp4_path, arrays["rgb_comb"], fps=DROID_FPS)


def convert_droid_split(
    input_dir,
    left_output_dir,
    right_output_dir,
    keep_ranges_path,
    split,
    left_metainfo_path,
    right_metainfo_path,
    main_max_adjacent_diff=MAIN_MAX_ADJACENT_DIFF,
    wrist_max_adjacent_diff=WRIST_MAX_ADJACENT_DIFF,
):
    os.makedirs(left_output_dir, exist_ok=True)
    os.makedirs(right_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(left_metainfo_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(right_metainfo_path) or ".", exist_ok=True)

    keep_ranges = load_keep_ranges(keep_ranges_path)
    print(f"Loaded keep_ranges for {len(keep_ranges)} episodes from {keep_ranges_path}")

    meta_left = {
        "dataset_name": "Droid-Left",
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": 0,
        "datalist": [],
    }
    meta_right = {
        "dataset_name": "Droid-Right",
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": 0,
        "datalist": [],
    }

    lang_counters = defaultdict(int)
    valid_ep_counter = 0
    skip_stats = defaultdict(int)

    builder = tfds.builder_from_directory(input_dir)
    split_info = builder.info.splits[split]
    num_examples = split_info.num_examples
    ds = builder.as_dataset(split=split, shuffle_files=False)

    pbar = tqdm(total=num_examples, desc=f"convert droid/{split}")
    for index, episode in enumerate(ds):
        if not is_success_episode(episode):
            skip_stats["not_success"] += 1
            pbar.update(1)
            continue

        ep_key = make_episode_key(episode)
        if ep_key not in keep_ranges:
            skip_stats["not_in_keep_ranges"] += 1
            pbar.update(1)
            continue

        kept_indices = build_kept_step_indices(keep_ranges[ep_key])
        if not kept_indices:
            skip_stats["empty_keep_ranges"] += 1
            pbar.update(1)
            continue

        if has_large_frame_gap(keep_ranges[ep_key]):
            skip_stats["large_frame_gap"] += 1
            pbar.update(1)
            continue

        if len(kept_indices) > MAX_FRAMES:
            skip_stats["too_many_frames"] += 1
            pbar.update(1)
            continue

        if len(kept_indices) < MIN_FRAMES:
            skip_stats["too_few_frames"] += 1
            pbar.update(1)
            continue

        lang = read_episode_language(episode)
        if lang is None:
            skip_stats["bad_language"] += 1
            pbar.update(1)
            continue

        count = lang_counters[lang]
        if count >= MAX_EPISODES_PER_INSTRUCTION:
            skip_stats["instruction_quota_full"] += 1
            pbar.update(1)
            continue

        subset = "Droid-Left" if valid_ep_counter % 2 == 0 else "Droid-Right"
        main_exterior_key = (
            "exterior_image_1_left" if subset == "Droid-Left" else "exterior_image_2_left"
        )
        arrays, skip_reason = episode_to_arrays(
            episode,
            kept_indices,
            main_exterior_key,
            main_max_adjacent_diff=main_max_adjacent_diff,
            wrist_max_adjacent_diff=wrist_max_adjacent_diff,
        )
        if arrays is None:
            skip_stats[skip_reason] += 1
            pbar.update(1)
            continue

        output_dir = left_output_dir if subset == "Droid-Left" else right_output_dir
        meta_json = meta_left if subset == "Droid-Left" else meta_right

        h5_path = os.path.join(output_dir, f"droid_{split}_{index}.hdf5")
        write_episode_hdf5(h5_path, arrays)

        meta_json["datalist"].append(h5_path)
        meta_json["num_ep"] += 1
        lang_counters[lang] = count + 1
        valid_ep_counter += 1

        print(
            f"[{subset}][{meta_json['num_ep']}][{index}/{num_examples}] "
            f"valid_ep={valid_ep_counter} frames={len(arrays['rgb_comb'])} "
            f"lang_count={lang_counters[lang]}/{MAX_EPISODES_PER_INSTRUCTION} {lang}",
            flush=True,
        )
        pbar.update(1)

    with open(left_metainfo_path, "w") as f:
        json.dump(meta_left, f, indent=4, ensure_ascii=False)
    with open(right_metainfo_path, "w") as f:
        json.dump(meta_right, f, indent=4, ensure_ascii=False)

    print("\n=== Conversion summary ===")
    print(f"Total valid episodes exported: {valid_ep_counter}")
    print(f"Droid-Left episodes: {meta_left['num_ep']}")
    print(f"Droid-Right episodes: {meta_right['num_ep']}")
    print(f"Unique language instructions exported: {len(lang_counters)}")
    print("Skip stats:")
    for reason, cnt in sorted(skip_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DROID RLDS dataset to ShowVLA HDF5 format.")
    parser.add_argument(
        "--input_dir",
        default="/datasets3/droid/1.0.1",
        type=str,
        help="Original DROID RLDS dataset directory (use 1.0.1 for keep_ranges filter).",
    )
    parser.add_argument(
        "--keep_ranges_path",
        default="/datasets3/droid/KarIP/droid/keep_ranges_1_0_1.json",
        type=str,
        help="Idle-action filter ranges from DROID annotations README.",
    )
    parser.add_argument(
        "--left_dir",
        default="/home/hyx/datasets/Droid-Left",
        type=str,
        help="Output directory for Droid-Left HDF5 files.",
    )
    parser.add_argument(
        "--right_dir",
        default="/home/hyx/datasets/Droid-Right",
        type=str,
        help="Output directory for Droid-Right HDF5 files.",
    )
    parser.add_argument(
        "--left_metainfo",
        default="./Droid-Left_metainfo.json",
        type=str,
        help="Metainfo json path for Droid-Left.",
    )
    parser.add_argument(
        "--right_metainfo",
        default="./Droid-Right_metainfo.json",
        type=str,
        help="Metainfo json path for Droid-Right.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train"],
        help="RLDS split to convert (DROID only has train).",
    )
    parser.add_argument(
        "--main_max_adjacent_diff",
        default=MAIN_MAX_ADJACENT_DIFF,
        type=float,
        help="Discard episode if main-view adjacent-frame mean abs diff exceeds this.",
    )
    parser.add_argument(
        "--wrist_max_adjacent_diff",
        default=WRIST_MAX_ADJACENT_DIFF,
        type=float,
        help="Discard episode if wrist-view adjacent-frame mean abs diff exceeds this.",
    )
    args = parser.parse_args()

    setup_seed(0)
    convert_droid_split(
        args.input_dir,
        args.left_dir,
        args.right_dir,
        args.keep_ranges_path,
        args.split,
        args.left_metainfo,
        args.right_metainfo,
        main_max_adjacent_diff=args.main_max_adjacent_diff,
        wrist_max_adjacent_diff=args.wrist_max_adjacent_diff,
    )
