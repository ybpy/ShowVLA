import os
import re
import argparse
import json
import h5py
import numpy as np
from collections import defaultdict
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import mediapy as media

ROBOMIND_FPS = 30  # matches RobomindHandler freq
MIN_FRAMES = 10
RAW_TAIL_TRIM = 20
FRAME_RE = re.compile(r"(\d+)")
DISCARD_LANGUAGE_RAW_RES = (
    re.compile(r"^place .+ on the table$", re.IGNORECASE),
    re.compile(r"^place .+ in the table$", re.IGNORECASE),
    re.compile(r"^insert .+ from the vase$", re.IGNORECASE),
    re.compile(r"^place the trash in the trash can$", re.IGNORECASE),
)
LANGUAGE_RAW_REPLACEMENTS = (
    ("pike up the red pepper from the table", "pick up the red pepper from the table"),
    ("pick up the flowers from the vase", "pick up the flowers"),
    ("pick up the flowers from the table", "pick up the flowers"),
)
MAIN_TGT_SIZE = (224, 320)
WRIST_TGT_SIZE = (112, 160)
COMB_SIZE = (336, 320)
DEFAULT_MAX_PER_TASK_FOLDER = 100
DEFAULT_MAX_PER_LANGUAGE = 300


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
    return np.array(Image.open(BytesIO(raw)))


def combine_main_zero_bottom(
    main_img,
    main_tgt_size=MAIN_TGT_SIZE,
    wrist_tgt_size=WRIST_TGT_SIZE,
    comb_size=COMB_SIZE,
):
    """Combine main view on top; bottom wrist region is zeros (same layout as Bridge)."""
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]

    main_img = np.array(
        Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR)
    )
    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[: main_tgt_size[0]] = main_img
    return comb_img


def normalize_language(lang: str) -> str:
    lang = lang.strip()
    lang = re.sub(r",(?=\S)", ", ", lang)
    lang = re.sub(r"\s+", " ", lang)
    for old, new in LANGUAGE_RAW_REPLACEMENTS:
        if lang.lower() == old:
            lang = new
            break
    return lang


def normalize_step_description(desc: str) -> str:
    desc = desc.strip()
    desc = re.sub(r"\s+", " ", desc)
    return desc


def format_step_description(desc: str) -> str:
    return normalize_language(normalize_step_description(desc))


def discard(msg: str):
    print(f"[Discard!] {msg}", flush=True)


def limit_skip(msg: str):
    print(f"[Limit!] {msg}", flush=True)


def task_folder_from_rel_path(rel_path: str) -> str:
    return rel_path.split(os.sep)[0]


def read_h5_language_instruction(h5_path: str) -> str:
    with h5py.File(h5_path, "r") as h5_file:
        lang = h5_file["language_instruction"][()]
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")
    return str(lang)


def task_folder_from_h5_filename(h5_filename: str, data_dir_basename: str) -> str:
    prefix = f"{data_dir_basename}_"
    if not h5_filename.startswith(prefix):
        return ""
    slug = h5_filename[len(prefix) :]
    if slug.endswith(".hdf5"):
        slug = slug[: -len(".hdf5")]
    slug = re.sub(r"_step\d+$", "", slug)
    if "_success_episodes" in slug:
        return slug.split("_success_episodes", 1)[0]
    return slug.split("_", 1)[0]


def rebuild_export_counters(datalist, data_dir_basename: str):
    task_folder_counts = defaultdict(int)
    language_counts = defaultdict(int)
    for h5_path in datalist:
        if not os.path.exists(h5_path):
            continue
        h5_filename = os.path.basename(h5_path)
        task_folder = task_folder_from_h5_filename(h5_filename, data_dir_basename)
        if not task_folder:
            continue
        language = read_h5_language_instruction(h5_path)
        task_folder_counts[task_folder] += 1
        language_counts[language] += 1
    return task_folder_counts, language_counts


def can_export_segment(task_folder: str, language: str, counters, max_per_task_folder: int, max_per_language: int):
    task_folder_counts, language_counts = counters
    if max_per_task_folder > 0 and task_folder_counts[task_folder] >= max_per_task_folder:
        return False, "task_folder", task_folder_counts[task_folder]
    if max_per_language > 0 and language_counts[language] >= max_per_language:
        return False, "language", language_counts[language]
    return True, None, None


def record_export(task_folder: str, language: str, counters):
    task_folder_counts, language_counts = counters
    task_folder_counts[task_folder] += 1
    language_counts[language] += 1


def count_words(text: str) -> int:
    return len(text.strip().split())


def is_none_step(desc: str) -> bool:
    cleaned = desc.strip().lower()
    return cleaned in ("[none]", "none")


def parse_frame_idx(frame_name: str) -> int:
    return int(FRAME_RE.search(frame_name).group(1))


def episode_id_from_src(src_path: str, data_dir: str) -> str:
    rel_path = os.path.relpath(os.path.dirname(src_path), data_dir)
    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))
    return f"{data_dir_basename}/{rel_path.replace(os.sep, '/')}"


def read_language_raw(src_f: h5py.File) -> str:
    raw = src_f["language_raw"][0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return normalize_language(raw)


def is_discarded_language_raw(lang: str) -> bool:
    return any(pattern.match(lang.strip()) for pattern in DISCARD_LANGUAGE_RAW_RES)


def build_step_segments(steps, num_frames: int):
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

        desc = normalize_step_description(step["step_description"])
        if is_none_step(desc):
            discard(f"step {step_idx} [none]")
            continue

        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))
        if end < start:
            discard(f"step {step_idx} invalid frame range [{start}, {end}]")
            continue

        num_seg_frames = end - start + 1
        if num_seg_frames < MIN_FRAMES:
            discard(f"step {step_idx} too short ({num_seg_frames} frames)")
            continue

        segments.append(
            {
                "step_idx": step_idx,
                "start": start,
                "end": end,
                "language_instruction": format_step_description(desc),
            }
        )
    return segments


def slice_trajectory_arrays(src_f: h5py.File, start: int, end: int):
    camera_top = src_f["observations/rgb_images/camera_top"]
    ee = src_f["puppet/end_effector"][()]
    joint_pos = src_f["puppet/joint_position"][()]

    rgb_comb = []
    for i in range(start, end + 1):
        img = decode_jpeg_bytes(camera_top[i])
        rgb_comb.append(combine_main_zero_bottom(img))

    return {
        "rgb_comb": rgb_comb,
        "end_effector": ee[start : end + 1],
        "joint_position": joint_pos[start : end + 1],
    }


def build_output_segments(src_path: str, data_dir: str, annotation_by_id: dict):
    with h5py.File(src_path, "r") as src_f:
        camera_top = src_f["observations/rgb_images/camera_top"]
        ee = src_f["puppet/end_effector"]
        joint_pos = src_f["puppet/joint_position"]
        language_raw = read_language_raw(src_f)
        if is_discarded_language_raw(language_raw):
            discard(f"{src_path}: filtered language_raw: {language_raw!r}")
            return []

        num_frames = min(len(camera_top), len(ee), len(joint_pos))
        if num_frames < MIN_FRAMES:
            discard(f"{src_path}: trajectory too short ({num_frames} frames)")
            return []

        episode_id = episode_id_from_src(src_path, data_dir)
        annotation = annotation_by_id.get(episode_id)
        if annotation is not None:
            step_segments = build_step_segments(annotation["steps"], num_frames)
            outputs = []
            for seg in step_segments:
                arrays = slice_trajectory_arrays(src_f, seg["start"], seg["end"])
                arrays["language_instruction"] = seg["language_instruction"]
                arrays["step_idx"] = seg["step_idx"]
                arrays["start"] = seg["start"]
                arrays["end"] = seg["end"]
                outputs.append(arrays)
            if not outputs:
                discard(f"{src_path}: no valid step segments after filtering")
            return outputs

        if count_words(language_raw) <= 2:
            discard(
                f"{src_path}: language_raw too short "
                f"({count_words(language_raw)} words): {language_raw!r}"
            )
            return []

        remaining_frames = num_frames - RAW_TAIL_TRIM
        if remaining_frames < MIN_FRAMES:
            discard(
                f"{src_path}: language_raw after trimming last {RAW_TAIL_TRIM} frames "
                f"too short ({remaining_frames} frames)"
            )
            return []

        end = remaining_frames - 1
        arrays = slice_trajectory_arrays(src_f, 0, end)
        arrays["language_instruction"] = language_raw
        arrays["step_idx"] = None
        return [arrays]


def load_annotation_json(annotation_json_path: str):
    if not annotation_json_path:
        return {}
    with open(annotation_json_path, "r") as f:
        entries = json.load(f)
    return {item["id"]: item["response"] for item in entries}


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
        puppet_grp.create_dataset("end_effector", data=arrays["end_effector"])
        puppet_grp.create_dataset("joint_position", data=arrays["joint_position"])

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
    dataset_name="robomind-ur",
    max_per_task_folder=DEFAULT_MAX_PER_TASK_FOLDER,
    max_per_language=DEFAULT_MAX_PER_LANGUAGE,
):
    os.makedirs(output_dir, exist_ok=True)
    annotation_by_id = load_annotation_json(annotation_json_path)

    try:
        with open(metainfo_json_out_path, "r") as f:
            meta_json = json.load(f)
        cur_episode = meta_json["num_ep"]
        print(f"{metainfo_json_out_path} exists!\ncur_episode={cur_episode}", flush=True)
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
    if annotation_json_path:
        print(f"Loaded {len(annotation_by_id)} external language annotations from {annotation_json_path}")
    print(
        f"Export limits: max_per_task_folder={max_per_task_folder}, "
        f"max_per_language={max_per_language}",
        flush=True,
    )

    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))
    export_limits = meta_json.get("export_limits")
    if (
        export_limits
        and export_limits.get("max_per_task_folder") == max_per_task_folder
        and export_limits.get("max_per_language") == max_per_language
        and export_limits.get("data_dir_basename") == data_dir_basename
    ):
        task_folder_counts = defaultdict(int, export_limits.get("task_folder_counts", {}))
        language_counts = defaultdict(int, export_limits.get("language_counts", {}))
        print(
            f"Restored export counters from metainfo: "
            f"task_folders={len(task_folder_counts)}, languages={len(language_counts)}",
            flush=True,
        )
    else:
        print("Rebuilding export counters from existing datalist...", flush=True)
        task_folder_counts, language_counts = rebuild_export_counters(
            meta_json.get("datalist", []), data_dir_basename
        )
        print(
            f"Rebuilt export counters: "
            f"task_folders={len(task_folder_counts)}, languages={len(language_counts)}",
            flush=True,
        )
    export_counters = (task_folder_counts, language_counts)

    list_trajectory_files = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn == "trajectory.hdf5":
                list_trajectory_files.append(os.path.join(root, fn))
    list_trajectory_files.sort()

    pbar = tqdm(list_trajectory_files)
    for ep_cnt, src_path in enumerate(pbar):
        episode_dir = os.path.abspath(os.path.dirname(src_path))
        print(f"\n{episode_dir}", flush=True)

        rel_path = os.path.relpath(os.path.dirname(src_path), data_dir)
        rel_slug = rel_path.replace(os.sep, "_")
        pbar.set_description(f"[{ep_cnt + 1}/{len(list_trajectory_files)}] {rel_path}")

        segments = build_output_segments(src_path, data_dir, annotation_by_id)
        if not segments:
            continue

        task_folder = task_folder_from_rel_path(rel_path)

        for seg in segments:
            if seg["step_idx"] is None:
                h5_filename = f"{data_dir_basename}_{rel_slug}.hdf5"
            else:
                h5_filename = f"{data_dir_basename}_{rel_slug}_step{seg['step_idx']:02d}.hdf5"
            h5_path = os.path.join(output_dir, h5_filename)
            language = seg["language_instruction"]

            if h5_path in meta_json["datalist"]:
                continue

            if os.path.exists(h5_path) and not overwrite:
                meta_json["datalist"].append(h5_path)
                cur_episode += 1
                record_export(task_folder, language, export_counters)
                continue

            allowed, limit_type, current_count = can_export_segment(
                task_folder,
                language,
                export_counters,
                max_per_task_folder,
                max_per_language,
            )
            if not allowed:
                if limit_type == "task_folder":
                    limit_skip(
                        f"{src_path}: task folder {task_folder!r} reached limit "
                        f"({current_count}/{max_per_task_folder})"
                    )
                else:
                    limit_skip(
                        f"{src_path}: language {language!r} reached limit "
                        f"({current_count}/{max_per_language})"
                    )
                continue

            write_segment_hdf5(h5_path, seg)
            meta_json["datalist"].append(h5_path)
            cur_episode += 1
            record_export(task_folder, language, export_counters)
            num_frames = len(seg["rgb_comb"])
            print(
                f"[{cur_episode}][{ep_cnt + 1}/{len(list_trajectory_files)}] "
                f"frames={num_frames} {language}",
                flush=True,
            )

    meta_json["num_ep"] = cur_episode
    meta_json["export_limits"] = {
        "max_per_task_folder": max_per_task_folder,
        "max_per_language": max_per_language,
        "data_dir_basename": data_dir_basename,
        "task_folder_counts": dict(task_folder_counts),
        "language_counts": dict(language_counts),
    }
    with open(metainfo_json_out_path, "w") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)

    print(f"Done. episodes={cur_episode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboMIND UR h5 data to ShowVLA HDF5 format.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/hyx/RoboMIND/benchmark1_1/h5_ur_1rgb",
        help="Directory of raw RoboMIND trajectory.hdf5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyx/datasets/RoboMIND/benchmark1_1/h5_ur_1rgb",
        help="Directory for converted HDF5 files.",
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default="/home/hyx/RoboMIND/static/language_description_annotation_json/h5_ur_1rgb.json",
        help="External step-level language annotation JSON.",
    )
    parser.add_argument(
        "--meta_prefix",
        type=str,
        default="robomind-ur",
        help="Prefix for metainfo json file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="robomind-ur",
        help="Dataset name in metainfo json (must match RobomindHandler).",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--max_per_task_folder",
        type=int,
        default=DEFAULT_MAX_PER_TASK_FOLDER,
        help="Max exported segments per task folder (0 disables).",
    )
    parser.add_argument(
        "--max_per_language",
        type=int,
        default=DEFAULT_MAX_PER_LANGUAGE,
        help="Max exported segments per language instruction (0 disables).",
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
        args.max_per_task_folder,
        args.max_per_language,
    )
