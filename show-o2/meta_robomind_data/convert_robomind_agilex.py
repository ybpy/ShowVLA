import os
import re
import argparse
import json
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import mediapy as media

ROBOMIND_FPS = 30  # matches RobomindHandler freq
MIN_FRAMES = 10
MERGE_MAX_FRAMES = 20
RAW_TAIL_TRIM = 240
# Drop segments with extreme inter-frame jumps (confirmed jump samples ~main≥36, wrist≥60).
DEFAULT_MAX_MAIN_FRAME_DIFF = 35.0
DEFAULT_MAX_WRIST_FRAME_DIFF = 60.0
MAIN_TGT_H = 224
WRIST_TGT_W = 160
FRAME_RE = re.compile(r"(\d+)")
LEFT_TAG_RE = re.compile(r"\[left\]", re.IGNORECASE)
RIGHT_TAG_RE = re.compile(r"\[right\]", re.IGNORECASE)
BOTH_TAG_RE = re.compile(r"\[both\]", re.IGNORECASE)
BLOCK_PREFIX_RE = re.compile(r"^\[block\]\s*", re.IGNORECASE)
MALFORMED_BRACKET_PREFIX_RE = re.compile(r"^\[;\s*")
DOUBLE_ARM_TAG_RE = re.compile(r"\[\[(left|right|both)\]", re.IGNORECASE)
RESIDUAL_ARM_TAG_RE = re.compile(r"\[(?:both|left|right)\]", re.IGNORECASE)
TRAILING_BRACKET_RE = re.compile(r"[\[\]]+\s*$")
EPISODE_TIMESTAMP_RE = re.compile(r"(\d{13,})")
LANGUAGE_TEXT_SUBS = (
    (re.compile(r"\bwifruite\b", re.IGNORECASE), "kiwifruit"),
    (re.compile(r"\bform pot\b", re.IGNORECASE), "from pot"),
    (re.compile(r"\bon the plate on the plate\b", re.IGNORECASE), "on the plate"),
    (re.compile(r"\bmove away the oven\b", re.IGNORECASE), "move away from the oven"),
    (re.compile(r"\bmove the rgg\b", re.IGNORECASE), "move the egg"),
    (re.compile(r"\bgrab the bow\b(?!l)", re.IGNORECASE), "grab the bowl"),
    (re.compile(r"\bon In the bowl\b", re.IGNORECASE), "in the bowl"),
    (re.compile(r"\bmove away from move away from\b", re.IGNORECASE), "move away from"),
    (re.compile(r"\bthe a basket\b", re.IGNORECASE), "the basket"),
    (re.compile(r"\bmove towards the a basket\b", re.IGNORECASE), "move towards the basket"),
    (re.compile(r"\bgrab the a basket\b", re.IGNORECASE), "grab the basket"),
    (re.compile(r"\bmove away from the all items\b", re.IGNORECASE), "move away from all items"),
    (re.compile(r"\btoaster oven\b", re.IGNORECASE), "oven"),
    (re.compile(r"\b(right|left|both)\s+arm\s+approaching\b", re.IGNORECASE), r"\1 arm move towards"),
    (re.compile(r"\b(right|left)\s+arm\s+raise\b", re.IGNORECASE), r"\1 arm raises"),
    (re.compile(r"\b(left|right)\s+arm\s+Put\b"), r"\1 arm put"),
)
NONE_LANGUAGE_RAW_RE = re.compile(r"^\[*none\]*$", re.IGNORECASE)
LONG_LANGUAGE_RAW_WORD_THRESHOLD = 25
THE_WORD_RE = re.compile(r"\bthe\b", re.IGNORECASE)


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


def decode_jpeg_bytes(img_bytes):
    if isinstance(img_bytes, np.ndarray) and img_bytes.dtype == np.uint8:
        raw = img_bytes.tobytes()
    else:
        raw = bytes(img_bytes)
    return np.array(Image.open(BytesIO(raw)))[:, :, ::-1]  # BGR to RGB


def combine_main_wrist_views(
    main_img,
    wrist_0_img,
    wrist_1_img,
    main_tgt_size=(224, 320),
    wrist_tgt_size=(112, 160),
    comb_size=(336, 320),
):
    """Combine front + left wrist + right wrist views (same layout as JAKA)."""
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]
    assert wrist_tgt_size[1] * 2 == main_tgt_size[1]

    main_img = np.array(
        Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR)
    )
    wrist_0_img = np.array(
        Image.fromarray(wrist_0_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR)
    )
    wrist_1_img = np.array(
        Image.fromarray(wrist_1_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR)
    )

    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[: main_tgt_size[0], :] = main_img
    comb_img[main_tgt_size[0] :, : wrist_tgt_size[1]] = wrist_0_img
    comb_img[main_tgt_size[0] :, wrist_tgt_size[1] :] = wrist_1_img
    return comb_img


def normalize_arm_tag_markup(text: str) -> str:
    """Fix malformed annotations like [[right] -> [right]."""
    return DOUBLE_ARM_TAG_RE.sub(r"[\1]", text)


def strip_residual_bracket_artifacts(text: str) -> str:
    text = MALFORMED_BRACKET_PREFIX_RE.sub("", text)
    text = RESIDUAL_ARM_TAG_RE.sub("", text)
    text = TRAILING_BRACKET_RE.sub("", text)
    text = re.sub(r"^\[+", "", text)
    text = re.sub(r"\]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_language_text(text: str) -> str:
    text = normalize_arm_tag_markup(text.strip())
    text = re.sub(r",(?=\S)", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = BLOCK_PREFIX_RE.sub("", text)
    text = MALFORMED_BRACKET_PREFIX_RE.sub("", text)
    text = TRAILING_BRACKET_RE.sub("", text).strip()
    for pattern, repl in LANGUAGE_TEXT_SUBS:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


ARM_TAG_RE = re.compile(r"\[(?:both|left|right)\]", re.IGNORECASE)
ARM_TAG_SEPARATORS = set(",;:.!?)]")


def insert_semicolon_before_arm_tags(desc: str) -> str:
    """Insert ';' before [left]/[right]/[both] when not at start and lacking a separator."""
    parts = []
    last = 0
    for match in ARM_TAG_RE.finditer(desc):
        if match.start() > 0:
            prev_char = desc[match.start() - 1]
            if not (prev_char.isspace() or prev_char in ARM_TAG_SEPARATORS):
                parts.append(desc[last : match.start()])
                parts.append("; ")
                last = match.start()
        parts.append(desc[last : match.end()])
        last = match.end()
    parts.append(desc[last:])
    return "".join(parts)


def format_step_description(desc: str) -> str:
    desc = normalize_arm_tag_markup(desc.strip())
    desc = re.sub(r",(?=\S)", ", ", desc)
    desc = re.sub(r"\s+", " ", desc)
    desc = BLOCK_PREFIX_RE.sub("", desc)
    desc = insert_semicolon_before_arm_tags(desc)
    desc = re.sub(r"\[both\]", "both arms ", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\[left\]", "left arm ", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\[right\]", "right arm ", desc, flags=re.IGNORECASE)
    desc = strip_residual_bracket_artifacts(desc)
    for pattern, repl in LANGUAGE_TEXT_SUBS:
        desc = pattern.sub(repl, desc)
    return re.sub(r"\s+", " ", desc).strip()


def is_malformed_language_instruction(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    if "[" in cleaned or "]" in cleaned:
        return True
    return is_none_language(cleaned)


def discard(msg: str):
    print(f"[Discard!] {msg}", flush=True)


def max_consecutive_frame_mae(frames) -> float:
    """Max mean-absolute-error between consecutive RGB frames."""
    if len(frames) < 2:
        return 0.0
    max_mae = 0.0
    prev = np.asarray(frames[0], dtype=np.float32)
    for i in range(1, len(frames)):
        cur = np.asarray(frames[i], dtype=np.float32)
        mae = float(np.mean(np.abs(cur - prev)))
        if mae > max_mae:
            max_mae = mae
        prev = cur
    return max_mae


def find_extreme_image_jump(
    rgb_comb,
    max_main_frame_diff: float = DEFAULT_MAX_MAIN_FRAME_DIFF,
    max_wrist_frame_diff: float = DEFAULT_MAX_WRIST_FRAME_DIFF,
):
    """
    Return (view_name, max_mae) if any camera view has an ultra-high frame jump.
    Checks main (top) and left/right wrist (bottom) regions of rgb_comb.
    """
    if not rgb_comb or len(rgb_comb) < 2:
        return None

    views = (
        ("main", [f[:MAIN_TGT_H] for f in rgb_comb], max_main_frame_diff),
        ("left_wrist", [f[MAIN_TGT_H:, :WRIST_TGT_W] for f in rgb_comb], max_wrist_frame_diff),
        ("right_wrist", [f[MAIN_TGT_H:, WRIST_TGT_W:] for f in rgb_comb], max_wrist_frame_diff),
    )
    worst = None
    for name, crops, thr in views:
        mae = max_consecutive_frame_mae(crops)
        if mae >= thr and (worst is None or mae > worst[1]):
            worst = (name, mae)
    return worst


def count_words(text: str) -> int:
    return len(text.strip().split())


def is_none_language(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return True
    if NONE_LANGUAGE_RAW_RE.match(raw):
        return True
    cleaned = clean_language_text(raw).lower()
    return not cleaned or cleaned == "none"


def compress_long_language_raw(lang: str) -> str:
    if count_words(lang) <= LONG_LANGUAGE_RAW_WORD_THRESHOLD:
        return lang
    compressed = THE_WORD_RE.sub("", lang)
    return re.sub(r"\s+", " ", compressed).strip()


def parse_frame_idx(frame_name: str) -> int:
    return int(FRAME_RE.search(frame_name).group(1))


def episode_id_from_src(src_path: str, data_dir: str) -> str:
    rel_path = os.path.relpath(os.path.dirname(src_path), data_dir)
    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))
    return f"{data_dir_basename}/{rel_path.replace(os.sep, '/')}"


def episode_timestamp(episode_id: str):
    match = EPISODE_TIMESTAMP_RE.search(episode_id)
    return match.group(1) if match else None


def read_language_raw(src_f: h5py.File) -> str:
    raw = src_f["language_raw"][0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return clean_language_text(raw)


def step_arm_side(desc: str):
    """Return 'left' or 'right' if step uses a single arm tag; otherwise None."""
    if BOTH_TAG_RE.search(desc):
        return None
    has_left = LEFT_TAG_RE.search(desc) is not None
    has_right = RIGHT_TAG_RE.search(desc) is not None
    if has_left and has_right:
        return None
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def is_mergeable_step_segment(seg: dict) -> bool:
    num_seg_frames = seg["end"] - seg["start"] + 1
    if num_seg_frames >= MERGE_MAX_FRAMES:
        return False
    return step_arm_side(seg["raw_desc"]) is not None


def merge_adjacent_step_segments(segments):
    """Merge adjacent short single-arm steps into one segment."""
    if not segments:
        return []

    merged = []
    idx = 0
    while idx < len(segments):
        seg = segments[idx]
        if not is_mergeable_step_segment(seg):
            merged.append(
                {
                    "step_idx": seg["step_idx"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "language_instruction": format_step_description(seg["raw_desc"]),
                }
            )
            idx += 1
            continue

        group = [seg]
        arm = step_arm_side(seg["raw_desc"])
        next_idx = idx + 1
        while next_idx < len(segments):
            candidate = segments[next_idx]
            if not is_mergeable_step_segment(candidate):
                break
            if step_arm_side(candidate["raw_desc"]) != arm:
                break
            group.append(candidate)
            next_idx += 1

        if len(group) == 1:
            merged.append(
                {
                    "step_idx": group[0]["step_idx"],
                    "start": group[0]["start"],
                    "end": group[0]["end"],
                    "language_instruction": format_step_description(group[0]["raw_desc"]),
                }
            )
        else:
            merged.append(
                {
                    "step_idx": group[0]["step_idx"],
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "language_instruction": ". ".join(
                        format_step_description(item["raw_desc"]) for item in group
                    ),
                }
            )
        idx = next_idx
    return merged


def build_step_segments(steps, num_frames: int, merge_steps: bool = False):
    """Split a trajectory into step segments using external annotations."""
    segments = []
    prev_end = None

    for step_idx, step in enumerate(steps):
        start = parse_frame_idx(step["start_frame"])
        end = parse_frame_idx(step["end_frame"])
        if start == end:
            if prev_end is None:
                prev_end = end
                discard(f"step {step_idx} start==end with no previous step")
                continue
            start = prev_end
        prev_end = end

        desc = clean_language_text(step["step_description"])
        if is_none_language(desc):
            discard(f"step {step_idx} [none]")
            continue

        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))
        if end < start:
            discard(f"step {step_idx} invalid frame range [{start}, {end}]")
            continue

        num_seg_frames = end - start + 1
        if num_seg_frames < MIN_FRAMES and not merge_steps:
            discard(f"step {step_idx} too short ({num_seg_frames} frames)")
            continue

        segments.append(
            {
                "step_idx": step_idx,
                "start": start,
                "end": end,
                "raw_desc": desc,
            }
        )

    if merge_steps:
        segments = merge_adjacent_step_segments(segments)

    filtered_segments = []
    for seg in segments:
        num_seg_frames = seg["end"] - seg["start"] + 1
        if num_seg_frames < MIN_FRAMES:
            discard(f"step {seg['step_idx']} too short ({num_seg_frames} frames)")
            continue
        language_instruction = seg.get("language_instruction") or format_step_description(seg["raw_desc"])
        if is_malformed_language_instruction(language_instruction):
            discard(f"step {seg['step_idx']} malformed language: {language_instruction!r}")
            continue
        filtered_segments.append(
            {
                "step_idx": seg["step_idx"],
                "start": seg["start"],
                "end": seg["end"],
                "language_instruction": language_instruction,
            }
        )
    return filtered_segments


def slice_trajectory_arrays(src_f: h5py.File, start: int, end: int):
    front = src_f["observations/rgb_images/camera_front"]
    left_wrist = src_f["observations/rgb_images/camera_left_wrist"]
    right_wrist = src_f["observations/rgb_images/camera_right_wrist"]
    ee_left = src_f["puppet/end_effector_left"][()]
    ee_right = src_f["puppet/end_effector_right"][()]

    rgb_comb = []
    for i in range(start, end + 1):
        main_img = decode_jpeg_bytes(front[i])
        wrist_0_img = decode_jpeg_bytes(left_wrist[i])
        wrist_1_img = decode_jpeg_bytes(right_wrist[i])
        rgb_comb.append(combine_main_wrist_views(main_img, wrist_0_img, wrist_1_img))

    return {
        "rgb_comb": rgb_comb,
        "end_effector_left": ee_left[start : end + 1],
        "end_effector_right": ee_right[start : end + 1],
    }


def build_whole_trajectory_segment(src_f: h5py.File, language_instruction: str):
    front = src_f["observations/rgb_images/camera_front"]
    left_wrist = src_f["observations/rgb_images/camera_left_wrist"]
    right_wrist = src_f["observations/rgb_images/camera_right_wrist"]
    ee_left = src_f["puppet/end_effector_left"]
    ee_right = src_f["puppet/end_effector_right"]
    num_frames = min(len(front), len(left_wrist), len(right_wrist), len(ee_left), len(ee_right))

    language_instruction = clean_language_text(language_instruction)
    if is_malformed_language_instruction(language_instruction):
        return None

    language_instruction = compress_long_language_raw(language_instruction)
    remaining_frames = num_frames - RAW_TAIL_TRIM
    if remaining_frames < MIN_FRAMES:
        discard(
            f"{src_f.filename}: language after trimming last {RAW_TAIL_TRIM} frames "
            f"too short ({remaining_frames} frames)"
        )
        return None

    end = remaining_frames - 1
    arrays = slice_trajectory_arrays(src_f, 0, end)
    arrays["language_instruction"] = language_instruction
    arrays["step_idx"] = None
    return arrays


def build_output_segments(src_path: str, data_dir: str, annotation_index: dict, merge_steps: bool = False):
    with h5py.File(src_path, "r") as src_f:
        front = src_f["observations/rgb_images/camera_front"]
        left_wrist = src_f["observations/rgb_images/camera_left_wrist"]
        right_wrist = src_f["observations/rgb_images/camera_right_wrist"]
        ee_left = src_f["puppet/end_effector_left"]
        ee_right = src_f["puppet/end_effector_right"]
        language_raw = read_language_raw(src_f)

        num_frames = min(len(front), len(left_wrist), len(right_wrist), len(ee_left), len(ee_right))
        if num_frames < MIN_FRAMES:
            discard(f"{src_path}: trajectory too short ({num_frames} frames)")
            return []

        episode_id = episode_id_from_src(src_path, data_dir)
        annotation = lookup_annotation(episode_id, annotation_index)
        if annotation is not None:
            step_segments = build_step_segments(annotation["steps"], num_frames, merge_steps)
            outputs = []
            for seg in step_segments:
                arrays = slice_trajectory_arrays(src_f, seg["start"], seg["end"])
                arrays["language_instruction"] = seg["language_instruction"]
                arrays["step_idx"] = seg["step_idx"]
                arrays["start"] = seg["start"]
                arrays["end"] = seg["end"]
                outputs.append(arrays)
            if outputs:
                return outputs

            discard(f"{src_path}: no valid step segments after filtering")
            return []

        if is_none_language(language_raw):
            discard(f"{src_path}: filtered language_raw: {language_raw!r}")
            return []

        word_count = count_words(language_raw)
        if word_count <= 2:
            discard(
                f"{src_path}: language_raw too short "
                f"({word_count} words): {language_raw!r}"
            )
            return []

        fallback = build_whole_trajectory_segment(src_f, language_raw)
        if fallback is None:
            discard(f"{src_path}: failed to build whole-trajectory segment")
            return []
        return [fallback]


def load_annotation_json(annotation_json_path: str):
    if not annotation_json_path:
        return {"exact": {}, "lower": {}, "timestamp": {}}
    with open(annotation_json_path, "r") as f:
        entries = json.load(f)
    exact = {}
    lower = {}
    timestamp = {}
    for item in entries:
        response = item["response"]
        episode_id = item["id"]
        exact[episode_id] = response
        lower[episode_id.lower()] = response
        ts = episode_timestamp(episode_id)
        if ts is not None and ts not in timestamp:
            timestamp[ts] = response
    return {"exact": exact, "lower": lower, "timestamp": timestamp}


def lookup_annotation(episode_id: str, annotation_index: dict):
    if episode_id in annotation_index["exact"]:
        return annotation_index["exact"][episode_id]
    lowered = episode_id.lower()
    if lowered in annotation_index["lower"]:
        return annotation_index["lower"][lowered]
    ts = episode_timestamp(episode_id)
    if ts is not None and ts in annotation_index["timestamp"]:
        return annotation_index["timestamp"][ts]
    return None


def write_segment_hdf5(h5_path: str, arrays: dict):
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
        puppet_grp = h5_file.create_group("puppet")
        puppet_grp.create_dataset("end_effector_left", data=arrays["end_effector_left"])
        puppet_grp.create_dataset("end_effector_right", data=arrays["end_effector_right"])

    if arrays.get("step_idx") is not None:
        base, _ = os.path.splitext(h5_path)
        mp4_path = f"{base}_{arrays['start']:04d}_{arrays['end']:04d}.mp4"
    else:
        mp4_path = h5_path.replace(".hdf5", ".mp4")
    media.write_video(mp4_path, arrays["rgb_comb"], fps=ROBOMIND_FPS)


def setup_seed(seed):
    np.random.seed(seed)


def convert_robomind_to_hdf5(
    data_dir,
    output_dir,
    metainfo_json_out_path,
    annotation_json_path="",
    overwrite=False,
    dataset_name="robomind-agilex",
    merge_steps=False,
    max_main_frame_diff=DEFAULT_MAX_MAIN_FRAME_DIFF,
    max_wrist_frame_diff=DEFAULT_MAX_WRIST_FRAME_DIFF,
):
    os.makedirs(output_dir, exist_ok=True)
    annotation_index = load_annotation_json(annotation_json_path)

    try:
        with open(metainfo_json_out_path, "r") as f:
            meta_json = json.load(f)
        cur_episode = meta_json["num_ep"]
    except OSError:
        meta_json = {
            "dataset_name": dataset_name,
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
    print(
        f"extreme jump filter: main_mae>={max_main_frame_diff}, "
        f"wrist_mae>={max_wrist_frame_diff}"
    )
    if merge_steps:
        print(f"merge_steps enabled: merge adjacent steps < {MERGE_MAX_FRAMES} frames with single [left]/[right]")
    if annotation_json_path:
        print(
            f"Loaded {len(annotation_index['exact'])} external language annotations "
            f"from {annotation_json_path}"
        )

    list_trajectory_files = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn == "trajectory.hdf5":
                list_trajectory_files.append(os.path.join(root, fn))
    list_trajectory_files.sort()

    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))

    pbar = tqdm(list_trajectory_files)
    for ep_cnt, src_path in enumerate(pbar):
        episode_dir = os.path.abspath(os.path.dirname(src_path))
        print(f"\n{episode_dir}", flush=True)

        rel_path = os.path.relpath(os.path.dirname(src_path), data_dir)
        rel_slug = rel_path.replace(os.sep, "_")
        pbar.set_description(f"[{ep_cnt + 1}/{len(list_trajectory_files)}] {rel_path}")

        segments = build_output_segments(src_path, data_dir, annotation_index, merge_steps)
        if not segments:
            continue

        for seg in segments:
            if seg["step_idx"] is None:
                h5_filename = f"{data_dir_basename}_{rel_slug}.hdf5"
            else:
                h5_filename = f"{data_dir_basename}_{rel_slug}_step{seg['step_idx']:02d}.hdf5"
            h5_path = os.path.join(output_dir, h5_filename)

            if h5_path in meta_json["datalist"]:
                continue

            jump = find_extreme_image_jump(
                seg["rgb_comb"],
                max_main_frame_diff=max_main_frame_diff,
                max_wrist_frame_diff=max_wrist_frame_diff,
            )
            if jump is not None:
                view_name, mae = jump
                discard(
                    f"{h5_filename}: extreme image jump on {view_name} "
                    f"(max consecutive MAE={mae:.1f})"
                )
                continue

            if os.path.exists(h5_path) and not overwrite:
                meta_json["datalist"].append(h5_path)
                cur_episode += 1
                continue

            write_segment_hdf5(h5_path, seg)
            meta_json["datalist"].append(h5_path)
            cur_episode += 1
            num_frames = len(seg["rgb_comb"])
            print(
                f"[{cur_episode}][{ep_cnt + 1}/{len(list_trajectory_files)}] "
                f"frames={num_frames} {seg['language_instruction']}",
                flush=True,
            )

    meta_json["num_ep"] = cur_episode
    with open(metainfo_json_out_path, "w") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)

    print(f"Done. episodes={cur_episode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboMIND agilex h5 data to ShowVLA HDF5 format.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets2/hyx_data/RoboMIND/benchmark1_1/h5_agilex_3rgb",
        help="Directory of raw RoboMIND trajectory.hdf5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyx/datasets/RoboMIND/benchmark1_1/h5_agilex_3rgb",
        help="Directory for converted HDF5 files.",
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default="/home/hyx/RoboMIND/static/language_description_annotation_json/h5_agilex_3rgb.json",
        help="External step-level language annotation JSON.",
    )
    parser.add_argument(
        "--meta_prefix",
        type=str,
        default="robomind-agilex",
        help="Prefix for metainfo json file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="robomind-agilex",
        help="Dataset name in metainfo json (must match RobomindHandler).",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--merge_steps",
        action="store_true",
        default=False,
        help=(
            "Merge adjacent steps shorter than 20 frames that use only [left] or only [right] "
            "into one segment; joined language uses '. '."
        ),
    )
    parser.add_argument(
        "--max_main_frame_diff",
        type=float,
        default=DEFAULT_MAX_MAIN_FRAME_DIFF,
        help="Discard segment if main-view consecutive-frame MAE reaches this value.",
    )
    parser.add_argument(
        "--max_wrist_frame_diff",
        type=float,
        default=DEFAULT_MAX_WRIST_FRAME_DIFF,
        help="Discard segment if either wrist-view consecutive-frame MAE reaches this value.",
    )
    args = parser.parse_args()

    setup_seed(0)
    metainfo_json_out_path = f"{args.meta_prefix}_metainfo.json"

    convert_robomind_to_hdf5(
        args.data_dir,
        args.output_dir,
        metainfo_json_out_path,
        args.annotation_json,
        args.overwrite,
        args.dataset_name,
        args.merge_steps,
        args.max_main_frame_diff,
        args.max_wrist_frame_diff,
    )
